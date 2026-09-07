"""Compression-stage orchestration for fieldwise lossy codecs."""

import argparse
import math
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
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
    infer_complete_lattice_layout,
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
        if lossy_codec not in ("szo", "sz3", "sperr", "qoz", "tthresh"):
            raise RuntimeError(
                "--lossy-compressor must be one of: szo, sz3, sperr, qoz, "
                "tthresh."
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
    relative_error_bound: Optional[float] = None


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
            relative_error_bound=job.relative_error_bound,
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
        relative_error_bound=job.relative_error_bound,
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
        self.dense_merged_id_base = self._dense_merged_id_base()
        self.lattice: Optional[DenseLatticeLayout] = None
        self._lattice_ordered_values: Dict[str, np.ndarray] = {}
        self.field_workers = resolve_field_workers(
            self.settings.field_workers,
        )
        self.lossy_fields_seconds = 0.0
        self.canonical_order_seconds = 0.0
        self.lattice_prepare_seconds = 0.0
        self.id_compress_seconds = 0.0
        self.lattice_field_prepare_seconds = 0.0

    def run(self) -> Dict[str, Any]:
        started = time.perf_counter()
        stage_started = time.perf_counter()
        canonical_order = self._select_canonical_order()
        self.canonical_order_seconds = time.perf_counter() - stage_started
        stage_started = time.perf_counter()
        self._prepare_lattice_layout(canonical_order)
        self.lattice_prepare_seconds = time.perf_counter() - stage_started
        self._record_ordering(canonical_order)
        stage_started = time.perf_counter()
        self._compress_id(canonical_order)
        self.id_compress_seconds = time.perf_counter() - stage_started
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
        if self.dense_merged_id_base is None:
            particle_ids = read_raw(
                self.raw_paths["id"],
                id_dtype,
                self.count,
            )
            order = np.argsort(particle_ids, kind="stable")
            algorithm = "stable_argsort"
        else:
            order = _inverse_dense_id_order(
                self.raw_paths["id"],
                id_dtype,
                self.count,
                self.dense_merged_id_base,
            )
            algorithm = "dense_inverse_permutation"

        artifact = None
        artifact_dtype = None
        persist_order = not (
            self.dense_merged_id_base is not None
            and not bool(getattr(self.args, "metrics", False))
        )
        if persist_order:
            order_path = self.preprocessed_dir / "id_sort_order.i64.raw"
            require_output_path(order_path, self.settings.force)
            order.astype(np.int64, copy=False).tofile(order_path)
            self.raw_paths["id_sort_order"] = str(order_path)
            artifact = "id_sort_order"
            artifact_dtype = "int64"
        self.manifest.setdefault("runtime", {})[
            "canonical_order_algorithm"
        ] = algorithm
        return CanonicalOrder(
            mapping="id_sorted",
            field="id",
            artifact=artifact,
            artifact_dtype=artifact_dtype,
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
            if self._can_use_complete_lattice_fast_path(side):
                self.lattice = self._prepare_complete_lattice_layout(
                    order,
                    side,
                )
                self.manifest.setdefault("runtime", {})[
                    "complete_lattice_fast_path"
                ] = True
                self.manifest["lattice_layout"] = {
                    **common,
                    **self.lattice.manifest_metadata(),
                }
                return
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

    def _dense_merged_id_base(self) -> Optional[int]:
        merge = self.manifest.get("merge", {})
        if not isinstance(merge, dict) or not merge.get("enabled", False):
            return None
        overlap = merge.get("id_overlap_check", {})
        if not isinstance(overlap, dict) or overlap.get("status") != "passed":
            return None
        try:
            count = int(overlap["count"])
            minimum = int(overlap["minimum"])
            maximum = int(overlap["maximum"])
        except (KeyError, TypeError, ValueError, OverflowError):
            return None
        if count != self.count or maximum - minimum + 1 != count:
            return None
        return minimum

    def _can_use_complete_lattice_fast_path(self, side: int) -> bool:
        if self.dense_merged_id_base not in (0, 1):
            return False
        if side <= 0 or side**3 != self.count:
            return False
        position_stats = self.manifest.get("preprocess", {}).get(
            "positions",
            {},
        )
        for logical in POSITION_FIELDS:
            stats = position_stats.get(logical, {})
            try:
                minimum = float(stats["min_in_compressor_units"])
                maximum = float(stats["max_in_compressor_units"])
            except (KeyError, TypeError, ValueError, OverflowError):
                return False
            if (
                not math.isfinite(minimum)
                or not math.isfinite(maximum)
                or minimum < 0.0
                or maximum >= 1.0
            ):
                return False
        return True

    def _prepare_complete_lattice_layout(
        self,
        order: CanonicalOrder,
        side: int,
    ) -> DenseLatticeLayout:
        if order.values is None or self.dense_merged_id_base is None:
            raise LatticeLayoutUnavailable(
                "complete lattice optimization requires a dense ID order"
            )
        sample_count = min(self.count, 200_000)
        sample_indices = (
            np.arange(self.count, dtype=np.intp)
            if sample_count == self.count
            else np.linspace(
                0,
                self.count - 1,
                sample_count,
                dtype=np.intp,
            )
        )
        source_indices = order.values[sample_indices]
        id_dtype = np.dtype(self.manifest["fields"]["id"]["dtype"])
        sampled_ids = (
            sample_indices.astype(np.uint64)
            + np.uint64(self.dense_merged_id_base)
        ).astype(id_dtype, copy=False)
        sampled_positions = {}
        for logical in POSITION_FIELDS:
            source = np.memmap(
                self.raw_paths[logical],
                dtype=np.dtype("float32"),
                mode="r",
                shape=(self.count,),
            )
            sampled_positions[logical] = np.asarray(source[source_indices])
        return infer_complete_lattice_layout(
            sampled_ids,
            sampled_positions,
            side,
            self.dense_merged_id_base,
            self.dense_merged_id_base + self.count - 1,
        )

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
                relative_error_bound=self._relative_error_bound(logical),
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
        prepared = [
            self._prepare_lattice_position(logical, order)
            for logical in POSITION_FIELDS
        ]
        return (
            [job for job, _ in prepared],
            [updates for _, updates in prepared],
        )

    def _prepare_lattice_position(
        self,
        logical: str,
        order: CanonicalOrder,
    ) -> Tuple[LossyCompressionJob, Dict[str, Any]]:
        assert self.lattice is not None
        values = self._ordered_values(logical, "float32", order)
        requested_bound = float(
            self.manifest["field_error_bounds"][logical]["compressor_abs"]
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
        job = LossyCompressionJob(
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
            relative_error_bound=self._relative_error_bound(logical),
        )
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
            self.raw_paths[f"{logical}_lattice_wrap"] = str(wrap_raw_path)
            self.artifacts[f"{logical}_lattice_wrap"] = str(
                wrap_compressed_path
            )
        return job, updates

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
                relative_error_bound=self._relative_error_bound(logical),
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
        return [
            self._prepare_lattice_velocity(logical, order)
            for logical in VELOCITY_FIELDS
        ]

    def _prepare_lattice_velocity(
        self,
        logical: str,
        order: CanonicalOrder,
    ) -> LossyCompressionJob:
        assert self.lattice is not None
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
        return LossyCompressionJob(
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
            relative_error_bound=self._relative_error_bound(logical),
        )

    def _relative_error_bound(self, logical: str) -> Optional[float]:
        value = self.manifest["field_error_bounds"][logical].get("relative")
        return None if value is None else float(value)

    def _compress_lattice_fields(self, order: CanonicalOrder) -> None:
        prepare_started = time.perf_counter()
        assert self.lattice is not None
        if self.lattice.implicit_full_lattice and self.field_workers > 1:
            with ThreadPoolExecutor(
                max_workers=min(self.field_workers, 6)
            ) as executor:
                position_futures = [
                    executor.submit(
                        self._prepare_lattice_position,
                        logical,
                        order,
                    )
                    for logical in POSITION_FIELDS
                ]
                velocity_futures = [
                    executor.submit(
                        self._prepare_lattice_velocity,
                        logical,
                        order,
                    )
                    for logical in VELOCITY_FIELDS
                ]
                position_prepared = [
                    future.result() for future in position_futures
                ]
                velocity_jobs = [
                    future.result() for future in velocity_futures
                ]
            position_jobs = [job for job, _ in position_prepared]
            position_updates = [
                updates for _, updates in position_prepared
            ]
            self.manifest.setdefault("runtime", {})[
                "parallel_lattice_field_preparation"
            ] = True
        else:
            position_jobs, position_updates = (
                self._prepare_lattice_positions(order)
            )
            velocity_jobs = self._prepare_lattice_velocities(order)
        self.lattice_field_prepare_seconds = (
            time.perf_counter() - prepare_started
        )
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
        output_path = (
            self.preprocessed_dir
            / f"{logical}.{order.mapping}.{data_type.name}.raw"
        )
        if logical == "id" and self.dense_merged_id_base is not None:
            ordered_path = _write_dense_ids(
                output_path,
                data_type,
                self.count,
                self.dense_merged_id_base,
                self.settings.force,
            )
            self.raw_paths[f"{logical}_canonical_ordered"] = ordered_path
            return ordered_path
        cached = self._lattice_ordered_values.pop(logical, None)
        ordered_path = _reorder_raw(
            source_path,
            dtype,
            output_path,
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
        values = np.memmap(
            self.raw_paths[logical],
            dtype=np.dtype(dtype),
            mode="r",
            shape=(self.count,),
        )
        if not order.is_reordered:
            return values
        if order.values is None:
            raise RuntimeError("Canonical order metadata is incomplete.")
        return np.ascontiguousarray(values[order.values])

    def _finalize(self, started: float) -> None:
        self.manifest["format_version"] = 8 if self.lattice is not None else 3
        timing = self.manifest.setdefault("timing", {})
        timing["canonical_order_wall_seconds"] = self.canonical_order_seconds
        timing["lattice_prepare_wall_seconds"] = self.lattice_prepare_seconds
        timing["id_compress_wall_seconds"] = self.id_compress_seconds
        if self.lattice is not None:
            timing["lattice_field_prepare_wall_seconds"] = (
                self.lattice_field_prepare_seconds
            )
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
        source = np.memmap(
            raw_path,
            dtype=np.dtype(dtype),
            mode="r",
            shape=(count,),
        )
        values = source[order]
    elif values.ndim != 1 or values.size != count:
        raise RuntimeError(
            f"Cached ordered field expected {count} values, got {values.shape}."
        )
    np.ascontiguousarray(values).tofile(output_path)
    return str(output_path)


def _inverse_dense_id_order(
    raw_path: str,
    dtype: np.dtype,
    count: int,
    id_base: int,
) -> np.ndarray:
    """Build sorted-row source indices in O(N) for a dense unique ID span."""

    source = np.memmap(
        raw_path,
        dtype=dtype,
        mode="r",
        shape=(count,),
    )
    order = np.empty(count, dtype=np.intp)
    chunk_rows = 4_194_304
    for start in range(0, count, chunk_rows):
        end = min(count, start + chunk_rows)
        values = source[start:end]
        if dtype.kind == "u":
            offsets = values.astype(np.uint64, copy=False) - np.uint64(id_base)
        else:
            offsets = values.astype(np.int64, copy=False) - id_base
        if np.any(offsets < 0) or np.any(offsets >= count):
            raise RuntimeError(
                "Merged dense-ID metadata does not match the exported IDs."
            )
        order[offsets.astype(np.intp, copy=False)] = np.arange(
            start,
            end,
            dtype=np.intp,
        )
    return order


def _write_dense_ids(
    output_path: Path,
    dtype: np.dtype,
    count: int,
    id_base: int,
    force: bool,
) -> str:
    require_output_path(output_path, force)
    chunk_rows = 4_194_304
    with output_path.open("wb") as output:
        for start in range(0, count, chunk_rows):
            end = min(count, start + chunk_rows)
            np.arange(
                id_base + start,
                id_base + end,
                dtype=dtype,
            ).tofile(output)
    return str(output_path)


def compress(
    args: argparse.Namespace,
    manifest: Dict[str, Any],
    raw_paths: Dict[str, str],
) -> Dict[str, Any]:
    return CompressionPipeline(args, manifest, raw_paths).run()
