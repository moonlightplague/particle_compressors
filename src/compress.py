"""Compression-stage orchestration for fieldwise lossy codecs."""

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.constants import POSITION_FIELDS, VELOCITY_FIELDS
from src.lattice_layout import (
    DenseLatticeLayout,
    IDENTITY_TRANSFORM,
    LATTICE_LAYOUT_NAME,
    LatticeLayoutUnavailable,
    POSITION_RESIDUAL_TRANSFORM,
    infer_dense_lattice_layout,
    position_transform_guard,
)
from src.manifest import update_compressed_size_metrics
from src.models import CanonicalOrder
from src.raw_codecs import compress_integer_raw, compress_lossy_raw
from src.runtime import (
    read_raw,
    require_output_path,
    resolve_field_workers,
    write_json,
)
from src.shaped_codecs import compress_shaped_lossy_raw


@dataclass(frozen=True)
class CompressionSettings:
    lossy_codec: str
    force: bool
    sort_requested: bool = False
    sort_by_id: bool = False
    lattice_requested: bool = False
    lattice_min_occupancy: float = 0.8
    lattice_axis_search: bool = True
    field_workers: int = 1

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "CompressionSettings":
        lossy_codec = str(args.lossy_compressor)
        if lossy_codec not in ("szo", "sz3", "sperr", "qoz"):
            raise RuntimeError(
                "--lossy-compressor must be one of: szo, sz3, sperr, qoz."
            )
        sort_requested = bool(getattr(args, "sort", False))
        lattice_requested = bool(getattr(args, "lattice_layout", False))
        lattice_min_occupancy = float(
            getattr(args, "lattice_min_occupancy", 0.8)
        )
        if not 0.0 < lattice_min_occupancy <= 1.0:
            raise RuntimeError("--lattice-min-occupancy must be in (0, 1].")
        field_workers = int(getattr(args, "field_workers", 1))
        if field_workers < 0:
            raise RuntimeError("--field-workers must be non-negative.")
        return cls(
            lossy_codec=lossy_codec,
            force=bool(args.force),
            sort_requested=sort_requested,
            sort_by_id=sort_requested or lattice_requested,
            lattice_requested=lattice_requested,
            lattice_min_occupancy=lattice_min_occupancy,
            lattice_axis_search=bool(
                getattr(args, "lattice_axis_search", True)
            ),
            field_workers=field_workers,
        )


@dataclass(frozen=True)
class LossyCompressionJob:
    """Pickle-friendly description of one independent codec invocation."""

    codec: str
    raw_path: str
    dtype: str
    compressed_path: str
    field_name: str
    count: int
    abs_error_bound: float
    force: bool
    encoded_shape: Optional[Tuple[int, int, int]] = None
    axis_search: bool = False


def _compress_lossy_job(job: LossyCompressionJob) -> Dict[str, Any]:
    if job.encoded_shape is None:
        return compress_lossy_raw(
            job.codec,
            job.raw_path,
            job.dtype,
            job.compressed_path,
            job.field_name,
            job.count,
            job.abs_error_bound,
            job.force,
        )
    return compress_shaped_lossy_raw(
        job.codec,
        job.raw_path,
        job.dtype,
        job.compressed_path,
        job.field_name,
        job.count,
        job.abs_error_bound,
        job.force,
        job.encoded_shape,
        job.axis_search,
    )


class CompressionPipeline:
    """Compress all fields while maintaining one particle row order."""

    def __init__(
        self,
        args: argparse.Namespace,
        manifest: Dict[str, Any],
        raw_paths: Dict[str, str],
    ) -> None:
        self.args = args
        self.manifest = manifest
        self.raw_paths = raw_paths
        self.settings = CompressionSettings.from_args(args)
        self.work_dir = Path(args.work_dir).resolve()
        self.preprocessed_dir = self.work_dir / "preprocessed"
        self.artifacts = manifest["artifacts"]["compressed"]
        self.compressed_fields = manifest["compressed_fields"]
        self.count = int(manifest["count"])
        self.lattice: Optional[DenseLatticeLayout] = None
        self._lattice_ordered_values: Dict[str, np.ndarray] = {}
        self.field_workers = resolve_field_workers(
            self.settings.field_workers,
        )
        self.lossy_fields_seconds = 0.0

    def run(self) -> Dict[str, Any]:
        started = time.perf_counter()
        canonical_order = self._select_canonical_order()
        self._prepare_lattice_layout(canonical_order)
        self._record_ordering(canonical_order)
        self._compress_id(canonical_order)
        if self.lattice is None:
            self._compress_positions(canonical_order)
            self._compress_velocities(canonical_order)
        else:
            self._compress_lattice_fields(canonical_order)
        self._finalize(started)
        return self.manifest

    def _select_canonical_order(self) -> CanonicalOrder:
        if not self.settings.sort_by_id:
            return CanonicalOrder()
        id_dtype = np.dtype(self.manifest["fields"]["id"]["dtype"])
        particle_ids = read_raw(
            self.raw_paths["id"],
            id_dtype,
            self.count,
        )
        order = np.argsort(particle_ids, kind="stable")
        order_path = self.preprocessed_dir / "id_sort_order.i64.raw"
        require_output_path(order_path, self.settings.force)
        order.astype(np.int64, copy=False).tofile(order_path)
        self.raw_paths["id_sort_order"] = str(order_path)
        return CanonicalOrder(
            mapping="id_sorted",
            field="id",
            artifact="id_sort_order",
            artifact_dtype="int64",
            values=order.astype(np.intp, copy=False),
        )

    def _prepare_lattice_layout(self, order: CanonicalOrder) -> None:
        if not self.settings.lattice_requested:
            return
        common = {
            "requested": True,
            "minimum_occupancy": self.settings.lattice_min_occupancy,
            "axis_search": self.settings.lattice_axis_search,
        }
        if order.values is None or order.field != "id":
            self.manifest["lattice_layout"] = {
                **common,
                "enabled": False,
                "reason": "lattice layout requires the canonical ID order",
            }
            return
        root_attrs = self.manifest.get("root_attrs", {})
        side_payload = root_attrs.get("nsidemesh")
        if not isinstance(side_payload, dict) or "value" not in side_payload:
            self.manifest["lattice_layout"] = {
                **common,
                "enabled": False,
                "reason": "root attribute 'nsidemesh' is unavailable",
            }
            return
        try:
            side = int(side_payload["value"])
            id_dtype = np.dtype(self.manifest["fields"]["id"]["dtype"])
            sorted_ids = read_raw(
                self.raw_paths["id"],
                id_dtype,
                self.count,
            )[order.values]
            sorted_positions = {
                logical: read_raw(
                    self.raw_paths[logical],
                    np.dtype("float32"),
                    self.count,
                )[order.values]
                for logical in POSITION_FIELDS
            }
            self.lattice = infer_dense_lattice_layout(
                sorted_ids,
                sorted_positions,
                side,
                self.settings.lattice_min_occupancy,
            )
        except (
            LatticeLayoutUnavailable,
            TypeError,
            ValueError,
            OverflowError,
        ) as exc:
            self.manifest["lattice_layout"] = {
                **common,
                "enabled": False,
                "reason": str(exc),
            }
            return
        self._lattice_ordered_values = {
            "id": sorted_ids,
            **sorted_positions,
        }
        self.manifest["lattice_layout"] = {
            **common,
            **self.lattice.manifest_metadata(),
        }

    def _record_ordering(self, order: CanonicalOrder) -> None:
        self.manifest["ordering"] = {
            "reconstructed_rows": {
                "mapping": order.mapping,
                "original_row_order_restored": not order.is_reordered,
                "canonical_field": order.field,
                "temporary_permutation_artifact": order.artifact,
                "temporary_permutation_dtype": order.artifact_dtype,
            },
            "id": {"mapping": order.mapping},
        }
        self.manifest["particle_sort"] = {
            "requested": (
                self.settings.sort_requested
                or self.settings.lattice_requested
            ),
            "enabled": order.field == "id",
            "key": "id" if order.field == "id" else None,
            "direction": "ascending" if order.field == "id" else None,
            "stable": bool(order.field == "id"),
        }

    def _compress_id(self, order: CanonicalOrder) -> None:
        dtype = self.manifest["fields"]["id"]["dtype"]
        raw_path = self._ordered_raw_path("id", dtype, order)
        self.compressed_fields["id"] = compress_integer_raw(
            self.args.lossless,
            raw_path,
            dtype,
            self.artifacts["id"],
            "id",
            self.count,
            self.settings.force,
        )

    def _compress_positions(self, order: CanonicalOrder) -> None:
        jobs = []
        for logical in POSITION_FIELDS:
            raw_path = self._ordered_raw_path(logical, "float32", order)
            jobs.append(LossyCompressionJob(
                self.settings.lossy_codec,
                raw_path,
                "float32",
                self.artifacts[logical],
                logical,
                self.count,
                float(
                    self.manifest["field_error_bounds"][logical][
                        "compressor_abs"
                    ]
                ),
                self.settings.force,
            ))
        for logical, field in zip(
            POSITION_FIELDS,
            self._compress_jobs(jobs),
        ):
            self.compressed_fields[logical] = field
        self.manifest["ordering"]["positions"] = {
            "mapping": order.mapping
        }

    def _prepare_lattice_positions(
        self,
        order: CanonicalOrder,
    ) -> Tuple[List[LossyCompressionJob], List[Dict[str, Any]]]:
        assert self.lattice is not None
        jobs: List[LossyCompressionJob] = []
        field_updates: List[Dict[str, Any]] = []
        for logical in POSITION_FIELDS:
            values = self._ordered_values(logical, "float32", order)
            requested_bound = float(
                self.manifest["field_error_bounds"][logical][
                    "compressor_abs"
                ]
            )
            dense, measured_error, wrap_offsets = self.lattice.encode_field(
                values,
                logical,
                position_residual=True,
            )
            guard = position_transform_guard(values, measured_error)
            transform = POSITION_RESIDUAL_TRANSFORM
            compressor_bound = requested_bound - guard
            if compressor_bound <= 0.0:
                dense, measured_error, wrap_offsets = self.lattice.encode_field(
                    values,
                    logical,
                    position_residual=False,
                )
                guard = 0.0
                compressor_bound = requested_bound
                transform = IDENTITY_TRANSFORM

            dense_path = (
                self.preprocessed_dir
                / f"{logical}.lattice.{dense.dtype.name}.raw"
            )
            require_output_path(dense_path, self.settings.force)
            dense.tofile(dense_path)
            self.raw_paths[f"{logical}_lattice"] = str(dense_path)
            jobs.append(LossyCompressionJob(
                self.settings.lossy_codec,
                str(dense_path),
                str(dense.dtype),
                self.artifacts[logical],
                logical,
                self.count,
                compressor_bound,
                self.settings.force,
                self.lattice.shape,
                self.settings.lattice_axis_search,
            ))
            updates: Dict[str, Any] = {
                "spatial_layout": LATTICE_LAYOUT_NAME,
                "lattice_transform": transform,
                "requested_compressor_abs": requested_bound,
                "transform_roundoff_guard": guard,
                "transform_roundtrip_max_abs": measured_error,
            }
            if wrap_offsets is not None:
                wrap_raw_path = (
                    self.preprocessed_dir
                    / f"{logical}.lattice-wrap.int8.raw"
                )
                require_output_path(wrap_raw_path, self.settings.force)
                wrap_offsets.tofile(wrap_raw_path)
                wrap_compressed_path = (
                    self.work_dir
                    / "compressed"
                    / f"{logical}.lattice-wrap.pco"
                )
                wrap_field = compress_integer_raw(
                    "pcodec",
                    str(wrap_raw_path),
                    "int8",
                    str(wrap_compressed_path),
                    f"{logical}_lattice_wrap",
                    self.count,
                    self.settings.force,
                )
                updates["lattice_wrap_field"] = wrap_field
                self.raw_paths[f"{logical}_lattice_wrap"] = str(
                    wrap_raw_path
                )
                self.artifacts[f"{logical}_lattice_wrap"] = str(
                    wrap_compressed_path
                )
            field_updates.append(updates)
        return jobs, field_updates

    def _compress_velocities(self, order: CanonicalOrder) -> None:
        jobs = []
        for logical in VELOCITY_FIELDS:
            dtype = self.manifest["fields"][logical]["dtype"]
            raw_path = self._ordered_raw_path(logical, dtype, order)
            jobs.append(LossyCompressionJob(
                self.settings.lossy_codec,
                raw_path,
                dtype,
                self.artifacts[logical],
                logical,
                self.count,
                float(self.manifest["field_error_bounds"][logical]["abs"]),
                self.settings.force,
            ))
        for logical, field in zip(
            VELOCITY_FIELDS,
            self._compress_jobs(jobs),
        ):
            self.compressed_fields[logical] = field
        self.manifest["ordering"]["velocities"] = {
            "mapping": order.mapping
        }

    def _prepare_lattice_velocities(
        self,
        order: CanonicalOrder,
    ) -> List[LossyCompressionJob]:
        assert self.lattice is not None
        jobs: List[LossyCompressionJob] = []
        for logical in VELOCITY_FIELDS:
            dtype = self.manifest["fields"][logical]["dtype"]
            values = self._ordered_values(logical, dtype, order)
            dense, _, _ = self.lattice.encode_field(
                values,
                logical,
                position_residual=False,
            )
            dense_path = (
                self.preprocessed_dir
                / f"{logical}.lattice.{dense.dtype.name}.raw"
            )
            require_output_path(dense_path, self.settings.force)
            dense.tofile(dense_path)
            self.raw_paths[f"{logical}_lattice"] = str(dense_path)
            jobs.append(LossyCompressionJob(
                self.settings.lossy_codec,
                str(dense_path),
                str(dense.dtype),
                self.artifacts[logical],
                logical,
                self.count,
                float(self.manifest["field_error_bounds"][logical]["abs"]),
                self.settings.force,
                self.lattice.shape,
                self.settings.lattice_axis_search,
            ))
        return jobs

    def _compress_lattice_fields(self, order: CanonicalOrder) -> None:
        position_jobs, position_updates = self._prepare_lattice_positions(order)
        velocity_jobs = self._prepare_lattice_velocities(order)
        results = self._compress_jobs(position_jobs + velocity_jobs)
        position_results = results[: len(POSITION_FIELDS)]
        velocity_results = results[len(POSITION_FIELDS) :]

        for logical, field, updates in zip(
            POSITION_FIELDS,
            position_results,
            position_updates,
        ):
            field.update(updates)
            self.compressed_fields[logical] = field
        for logical, field in zip(VELOCITY_FIELDS, velocity_results):
            field.update(
                {
                    "spatial_layout": LATTICE_LAYOUT_NAME,
                    "lattice_transform": IDENTITY_TRANSFORM,
                }
            )
            self.compressed_fields[logical] = field
        self.manifest["ordering"]["positions"] = {
            "mapping": order.mapping,
            "spatial_layout": LATTICE_LAYOUT_NAME,
        }
        self.manifest["ordering"]["velocities"] = {
            "mapping": order.mapping,
            "spatial_layout": LATTICE_LAYOUT_NAME,
        }

    def _compress_jobs(
        self,
        jobs: List[LossyCompressionJob],
    ) -> List[Dict[str, Any]]:
        started = time.perf_counter()
        workers = min(self.field_workers, len(jobs))
        if workers == 1:
            results = [_compress_lossy_job(job) for job in jobs]
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                results = list(executor.map(_compress_lossy_job, jobs))
        self.lossy_fields_seconds += time.perf_counter() - started
        return results

    def _ordered_raw_path(
        self,
        logical: str,
        dtype: str,
        order: CanonicalOrder,
    ) -> str:
        source_path = self.raw_paths[logical]
        if not order.is_reordered:
            return source_path
        if order.values is None:
            raise RuntimeError("Canonical order metadata is incomplete.")

        data_type = np.dtype(dtype)
        cached = self._lattice_ordered_values.pop(logical, None)
        ordered_path = _reorder_raw(
            source_path,
            dtype,
            self.preprocessed_dir
            / f"{logical}.{order.mapping}.{data_type.name}.raw",
            self.count,
            order.values,
            self.settings.force,
            values=cached,
        )
        self.raw_paths[f"{logical}_canonical_ordered"] = ordered_path
        return ordered_path

    def _ordered_values(
        self,
        logical: str,
        dtype: str,
        order: CanonicalOrder,
    ) -> np.ndarray:
        cached = self._lattice_ordered_values.pop(logical, None)
        if cached is not None:
            return cached
        values = read_raw(self.raw_paths[logical], np.dtype(dtype), self.count)
        if not order.is_reordered:
            return values
        if order.values is None:
            raise RuntimeError("Canonical order metadata is incomplete.")
        return np.ascontiguousarray(values[order.values])

    def _finalize(self, started: float) -> None:
        self.manifest["format_version"] = 8 if self.lattice is not None else 3
        timing = self.manifest.setdefault("timing", {})
        timing["lossy_fields_wall_seconds"] = self.lossy_fields_seconds
        timing["compress_wall_seconds"] = time.perf_counter() - started
        self.manifest.setdefault("runtime", {})[
            "compression_field_workers"
        ] = self.field_workers
        update_compressed_size_metrics(self.manifest, self.work_dir)
        write_json(
            self.work_dir / "manifest.json",
            self.manifest,
            force=True,
        )


def _reorder_raw(
    raw_path: str,
    dtype: str,
    output_path: Path,
    count: int,
    order: np.ndarray,
    force: bool,
    values: Optional[np.ndarray] = None,
) -> str:
    require_output_path(output_path, force)
    if values is None:
        source = read_raw(raw_path, np.dtype(dtype), count)
        values = source[order]
    elif values.ndim != 1 or values.size != count:
        raise RuntimeError(
            f"Cached ordered field expected {count} values, got {values.shape}."
        )
    np.ascontiguousarray(values).tofile(output_path)
    return str(output_path)


def compress(
    args: argparse.Namespace,
    manifest: Dict[str, Any],
    raw_paths: Dict[str, str],
) -> Dict[str, Any]:
    return CompressionPipeline(args, manifest, raw_paths).run()
