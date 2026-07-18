"""Compression-stage orchestration.

Codec implementations live in :mod:`src.raw_codecs` and :mod:`src.lcp_codec`.
This module coordinates row ordering and records the resulting manifest.
"""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from src.constants import (
    LCP_CHUNK_CONTAINER,
    MAX_INT32_ORDER_VALUES,
    POSITION_FIELDS,
    VELOCITY_FIELDS,
)
from src.lcp_codec import (
    compress_chunked_lcp_triplet,
    compress_lcp_triplet,
    compress_lcp_triplet_batch,
    read_lcp_permutation,
    reorder_raw,
    velocity_order_bits,
)
from src.manifest import update_compressed_size_metrics
from src.models import CanonicalOrder, ToolPaths
from src.raw_codecs import (
    compress_integer_raw,
    compress_lossy_raw,
    compress_pcodec_raw,
    compress_pysz_raw,
    compress_szo_raw,
    pad_codec_input,
)
from src.runtime import (
    read_raw,
    require_output_path,
    resolve_lcp_chunk_workers,
    write_json,
)


@dataclass(frozen=True)
class CompressionSettings:
    position_codec: str
    velocity_codec: str
    velocity_chunk_size: int
    configured_chunk_workers: int
    effective_chunk_workers: int
    force: bool
    sort_requested: bool = False
    sort_by_id: bool = False

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "CompressionSettings":
        position_codec = getattr(args, "pos_compressor", "lcp")
        velocity_codec = args.vel_compressor
        sort_requested = bool(getattr(args, "sort", False))
        sort_by_id = (
            sort_requested
            and position_codec != "lcp"
            and velocity_codec != "lcp"
        )
        chunk_size = int(getattr(args, "vel_chunk_size", 0))
        configured_workers = int(getattr(args, "vel_chunk_workers", 0))
        _validate_chunk_configuration(
            position_codec,
            velocity_codec,
            chunk_size,
            configured_workers,
        )
        return cls(
            position_codec=position_codec,
            velocity_codec=velocity_codec,
            sort_requested=sort_requested,
            sort_by_id=sort_by_id,
            velocity_chunk_size=chunk_size,
            configured_chunk_workers=configured_workers,
            effective_chunk_workers=resolve_lcp_chunk_workers(
                configured_workers
            ),
            force=bool(args.force),
        )


class CompressionPipeline:
    """Coordinate codecs while maintaining one canonical particle row order."""

    def __init__(
        self,
        args: argparse.Namespace,
        manifest: Dict[str, Any],
        raw_paths: Dict[str, str],
        tools: ToolPaths,
    ) -> None:
        self.args = args
        self.manifest = manifest
        self.raw_paths = raw_paths
        self.tools = tools
        self.settings = CompressionSettings.from_args(args)
        self.work_dir = Path(args.work_dir).resolve()
        self.preprocessed_dir = self.work_dir / "preprocessed"
        self.artifacts = manifest["artifacts"]["compressed"]
        self.compressed_fields = manifest["compressed_fields"]
        self.count = int(manifest["count"])

    def run(self) -> Dict[str, Any]:
        started = time.perf_counter()
        canonical_order = self._select_canonical_order()
        self._record_ordering(canonical_order)
        self._compress_id(canonical_order)
        self._compress_positions(canonical_order)
        self._compress_velocities(canonical_order)
        self._finalize(started)
        return self.manifest

    def _select_canonical_order(self) -> CanonicalOrder:
        if self.settings.position_codec == "lcp":
            return self._compress_canonical_positions()
        if self.settings.velocity_codec == "lcp":
            return self._compress_canonical_velocities()
        if self.settings.sort_by_id:
            return self._id_sorted_order()
        return CanonicalOrder()

    def _id_sorted_order(self) -> CanonicalOrder:
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

    def _compress_canonical_positions(self) -> CanonicalOrder:
        order_path = self.preprocessed_dir / "order.i32.raw"
        self.raw_paths["position_order"] = str(order_path)
        compressed_path = self.artifacts["positions"]
        abs_error_bound = float(
            self.manifest["error_bounds"]["positions_lcp_abs"]
        )
        compress_lcp_triplet(
            self.tools,
            self._raw_triplet(POSITION_FIELDS),
            compressed_path,
            self.count,
            abs_error_bound,
            order_path,
            self.settings.force,
        )
        self.compressed_fields["positions"] = self._lcp_field_metadata(
            "positions",
            POSITION_FIELDS,
            compressed_path,
            abs_error_bound,
        )
        return CanonicalOrder(
            mapping="lcp_position_sorted",
            field="positions",
            artifact="position_order",
            artifact_dtype="int32",
            values=read_lcp_permutation(str(order_path), self.count),
        )

    def _compress_canonical_velocities(self) -> CanonicalOrder:
        order_path = self.preprocessed_dir / "velocity_order.i32.raw"
        self.raw_paths["velocity_order"] = str(order_path)
        compress_lcp_triplet(
            self.tools,
            self._raw_triplet(VELOCITY_FIELDS, suffix="_lcp"),
            self.artifacts["velocities"],
            self.count,
            float(self.manifest["error_bounds"]["velocities_lcp_abs"]),
            order_path,
            self.settings.force,
        )
        return CanonicalOrder(
            mapping="lcp_velocity_sorted",
            field="velocities",
            artifact="velocity_order",
            artifact_dtype="int32",
            values=read_lcp_permutation(str(order_path), self.count),
        )

    def _record_ordering(self, order: CanonicalOrder) -> None:
        reconstructed_rows = {
            "mapping": order.mapping,
            "original_row_order_restored": not order.is_reordered,
            "canonical_field": order.field,
            "canonical_lcp_field": (
                order.field
                if order.field in ("positions", "velocities")
                else None
            ),
            "lcp_permutation_stored": False,
            "temporary_permutation_artifact": order.artifact,
            "temporary_permutation_dtype": order.artifact_dtype,
        }
        id_ordering: Dict[str, Any] = {"mapping": order.mapping}
        if order.field == "positions":
            reconstructed_rows["position_permutation_stored"] = False
            id_ordering["replaces_lcp_position_order"] = True
        elif order.field == "velocities":
            reconstructed_rows["velocity_permutation_stored"] = False
            id_ordering["replaces_lcp_velocity_order"] = True
        self.manifest["ordering"] = {
            "reconstructed_rows": reconstructed_rows,
            "id": id_ordering,
        }
        self.manifest["particle_sort"] = {
            "requested": self.settings.sort_requested,
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
        if self.settings.position_codec == "lcp":
            self.manifest["ordering"]["positions"] = {
                "mapping": order.mapping
            }
            return

        for logical in POSITION_FIELDS:
            raw_path = self._ordered_raw_path(logical, "float32", order)
            self.compressed_fields[logical] = compress_lossy_raw(
                self.settings.position_codec,
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
            )
        self.manifest["ordering"]["positions"] = {"mapping": order.mapping}

    def _compress_velocities(self, order: CanonicalOrder) -> None:
        if self.settings.velocity_codec == "lcp":
            self._compress_lcp_velocities(order)
            return

        for logical in VELOCITY_FIELDS:
            dtype = self.manifest["fields"][logical]["dtype"]
            raw_path = self._ordered_raw_path(logical, dtype, order)
            self.compressed_fields[logical] = compress_lossy_raw(
                self.settings.velocity_codec,
                raw_path,
                dtype,
                self.artifacts[logical],
                logical,
                self.count,
                float(self.manifest["field_error_bounds"][logical]["abs"]),
                self.settings.force,
            )
        self.manifest["ordering"]["velocities"] = {"mapping": order.mapping}

    def _compress_lcp_velocities(self, order: CanonicalOrder) -> None:
        if self.settings.position_codec == "lcp":
            self._compress_secondary_lcp_velocities(order)

        compressed_path = self.artifacts["velocities"]
        chunk_size = (
            self.settings.velocity_chunk_size
            if self.settings.position_codec == "lcp"
            else 0
        )
        self.compressed_fields["velocities"] = self._lcp_field_metadata(
            "velocities",
            VELOCITY_FIELDS,
            compressed_path,
            float(self.manifest["error_bounds"]["velocities_lcp_abs"]),
            chunk_size=chunk_size,
        )
        if self.settings.position_codec == "lcp":
            self.manifest["ordering"]["velocities"] = {
                "mapping": (
                    "lcp_velocity_sorted_index_to_lcp_position_sorted_row"
                ),
                "field": "velocity_order",
                "index_scope": (
                    "chunk_local"
                    if self.settings.velocity_chunk_size
                    else "global"
                ),
                "chunk_size": self.settings.velocity_chunk_size,
            }
        else:
            self.manifest["ordering"]["velocities"] = {
                "mapping": order.mapping
            }

    def _compress_secondary_lcp_velocities(
        self,
        order: CanonicalOrder,
    ) -> None:
        if order.values is None:
            raise RuntimeError(
                "Position-ordered LCP velocities require a canonical order."
            )

        def reorder_velocity(logical: str) -> Tuple[str, str]:
            return logical, reorder_raw(
                self.raw_paths[f"{logical}_lcp"],
                "float32",
                self.preprocessed_dir
                / f"{logical}.{order.mapping}.float32.raw",
                self.count,
                order.values,
                self.settings.force,
            )

        with ThreadPoolExecutor(max_workers=len(VELOCITY_FIELDS)) as executor:
            ordered_paths = dict(
                executor.map(reorder_velocity, VELOCITY_FIELDS)
            )
        for logical, path in ordered_paths.items():
            self.raw_paths[f"{logical}_canonical_ordered"] = path

        order_path = self.preprocessed_dir / "velocity_order.i32.raw"
        self.raw_paths["velocity_order"] = str(order_path)
        chunk_metadata = self._run_secondary_velocity_compressor(
            ordered_paths,
            order_path,
        )
        order_field = compress_integer_raw(
            self.args.lossless,
            str(order_path),
            "int32",
            self.artifacts["velocity_order"],
            "velocity_order",
            self.count,
            self.settings.force,
        )
        order_field.update(
            self._velocity_order_metadata(order_field, chunk_metadata)
        )
        self.compressed_fields["velocity_order"] = order_field

    def _run_secondary_velocity_compressor(
        self,
        ordered_paths: Dict[str, str],
        order_path: Path,
    ) -> Optional[Dict[str, int]]:
        started = time.perf_counter()
        inputs = tuple(ordered_paths[field] for field in VELOCITY_FIELDS)
        abs_error_bound = float(
            self.manifest["error_bounds"]["velocities_lcp_abs"]
        )
        chunk_metadata = None
        if self.settings.velocity_chunk_size:
            chunk_metadata = compress_chunked_lcp_triplet(
                self.tools,
                inputs,
                self.artifacts["velocities"],
                self.count,
                self.settings.velocity_chunk_size,
                abs_error_bound,
                order_path,
                self.settings.force,
                self.settings.effective_chunk_workers,
            )
        else:
            compress_lcp_triplet(
                self.tools,
                inputs,
                self.artifacts["velocities"],
                self.count,
                abs_error_bound,
                order_path,
                self.settings.force,
            )
        self.manifest.setdefault("timing", {})[
            "velocity_lcp_compress_wall_seconds"
        ] = time.perf_counter() - started
        return chunk_metadata

    def _velocity_order_metadata(
        self,
        order_field: Dict[str, Any],
        chunk_metadata: Optional[Dict[str, int]],
    ) -> Dict[str, Any]:
        common = {
            "uncompressed_storage_dtype": "int32",
            "compressed_bits_per_particle": (
                8.0 * float(order_field["bytes"]) / self.count
                if self.count
                else 0.0
            ),
        }
        if chunk_metadata is not None:
            return {
                **chunk_metadata,
                **common,
                "index_scope": "chunk_local",
            }
        return {
            **common,
            "chunk_size": 0,
            "chunk_count": 1,
            "index_scope": "global",
            "order_bits_per_particle": velocity_order_bits(self.count),
        }

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
        ordered_path = reorder_raw(
            source_path,
            dtype,
            self.preprocessed_dir
            / f"{logical}.{order.mapping}.{data_type.name}.raw",
            self.count,
            order.values,
            self.settings.force,
        )
        key = (
            "id_canonical_ordered"
            if logical == "id"
            else f"{logical}_canonical_ordered"
        )
        self.raw_paths[key] = ordered_path
        return ordered_path

    def _raw_triplet(
        self,
        fields: Tuple[str, str, str],
        suffix: str = "",
    ) -> Tuple[str, str, str]:
        return tuple(self.raw_paths[f"{field}{suffix}"] for field in fields)

    def _lcp_field_metadata(
        self,
        field_name: str,
        source_fields: Tuple[str, str, str],
        compressed_path: str,
        abs_error_bound: float,
        chunk_size: int = 0,
    ) -> Dict[str, Any]:
        path = Path(compressed_path)
        metadata = {
            "field": field_name,
            "codec": "lcp",
            "dtype": "float32",
            "source_dtypes": {
                logical: self.manifest["fields"][logical]["dtype"]
                for logical in source_fields
            },
            "count": self.count,
            "abs_error_bound": abs_error_bound,
            "path": str(path),
            "bytes": path.stat().st_size,
        }
        if field_name == "velocities":
            metadata.update(
                {
                    "chunk_size": chunk_size,
                    "chunk_count": (
                        (self.count + chunk_size - 1) // chunk_size
                        if chunk_size
                        else 1
                    ),
                    "container": (
                        LCP_CHUNK_CONTAINER
                        if chunk_size
                        else "native_lcp"
                    ),
                }
            )
        return metadata

    def _finalize(self, started: float) -> None:
        chunk_size = self.settings.velocity_chunk_size
        self.manifest["velocity_chunking"] = {
            "enabled": bool(chunk_size),
            "chunk_size": chunk_size,
            "chunk_count": (
                (self.count + chunk_size - 1) // chunk_size
                if chunk_size
                else 1
            ),
            "configured_workers": self.settings.configured_chunk_workers,
            "effective_workers": (
                self.settings.effective_chunk_workers
                if chunk_size
                else 1
            ),
        }
        self.manifest["format_version"] = 4 if chunk_size else 3
        self.manifest.setdefault("timing", {})["compress_wall_seconds"] = (
            time.perf_counter() - started
        )
        update_compressed_size_metrics(self.manifest, self.work_dir)
        write_json(
            self.work_dir / "manifest.json",
            self.manifest,
            force=True,
        )


def _validate_chunk_configuration(
    position_codec: str,
    velocity_codec: str,
    chunk_size: int,
    workers: int,
) -> None:
    if chunk_size < 0:
        raise RuntimeError("--vel-chunk-size must be non-negative.")
    if chunk_size > MAX_INT32_ORDER_VALUES:
        raise RuntimeError(
            "--vel-chunk-size cannot exceed 2^31 when using int32 order indices."
        )
    if workers < 0:
        raise RuntimeError("--vel-chunk-workers must be non-negative.")
    if chunk_size and not (
        position_codec == "lcp" and velocity_codec == "lcp"
    ):
        raise RuntimeError(
            "--vel-chunk-size is only supported when both position and "
            "velocity compressors are lcp."
        )


def compress(
    args: argparse.Namespace,
    manifest: Dict[str, Any],
    raw_paths: Dict[str, str],
    tools: ToolPaths,
) -> Dict[str, Any]:
    return CompressionPipeline(args, manifest, raw_paths, tools).run()


# Backwards-compatible name used by earlier tests and integrations.
pysz_encoded_values = pad_codec_input


__all__ = [
    "CompressionPipeline",
    "CompressionSettings",
    "compress",
    "compress_chunked_lcp_triplet",
    "compress_integer_raw",
    "compress_lcp_triplet",
    "compress_lcp_triplet_batch",
    "compress_lossy_raw",
    "compress_pcodec_raw",
    "compress_pysz_raw",
    "compress_szo_raw",
    "pysz_encoded_values",
    "read_lcp_permutation",
    "reorder_raw",
    "velocity_order_bits",
]
