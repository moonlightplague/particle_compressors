"""Compression-stage orchestration.

Codec implementations live in :mod:`src.raw_codecs`, :mod:`src.lcp_codec`,
and :mod:`src.xnyzip_codec`. This module coordinates row ordering and records
the resulting manifest.
"""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from src.cli import validate_compressor_combination
from src.constants import (
    LCP_CHUNK_CONTAINER,
    MAX_INT32_ORDER_VALUES,
    POSITION_FIELDS,
    VELOCITY_FIELDS,
    XNYZIP_CHUNK_CONTAINER,
)
from src.field_export import (
    export_ordered_triplet_for_xnyzip,
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
    resolve_velocity_chunk_workers,
    write_json,
)
from src.xnyzip_codec import (
    XNYZIP_CURVE,
    XNYZIP_DIRECT_THRESHOLD,
    XNYZIP_ORDER_DTYPE,
    XNYZIP_QUANTIZER,
    XNYZIP_STORAGE_MODE,
    compress_chunked_xnyzip_triplet,
    compress_xnyzip_triplet,
    read_xnyzip_permutation,
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
        validate_compressor_combination(position_codec, velocity_codec)
        sort_requested = bool(getattr(args, "sort", False))
        sort_by_id = (
            sort_requested
            and position_codec not in ("lcp", "xnyzip")
            and velocity_codec not in ("lcp", "xnyzip")
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
            effective_chunk_workers=resolve_velocity_chunk_workers(
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
        if self.settings.position_codec == "xnyzip":
            return self._compress_canonical_xnyzip_positions()
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

    def _compress_canonical_xnyzip_positions(self) -> CanonicalOrder:
        order_path = self.preprocessed_dir / "order.u64.raw"
        self.raw_paths["position_order"] = str(order_path)
        compressed_path = self.artifacts["positions"]
        l2_error_bound = float(
            self.manifest["error_bounds"]["positions_xnyzip_abs"]
        )
        order = compress_xnyzip_triplet(
            self.tools,
            self.raw_paths["positions_xnyzip"],
            compressed_path,
            self.count,
            l2_error_bound,
            order_path,
            self.settings.force,
        )
        if order is None:
            order = read_xnyzip_permutation(str(order_path), self.count)
        self.compressed_fields["positions"] = (
            self._xnyzip_field_metadata(
                "positions",
                POSITION_FIELDS,
                compressed_path,
                l2_error_bound,
            )
        )
        return CanonicalOrder(
            mapping="xnyzip_position_sorted",
            field="positions",
            artifact="position_order",
            artifact_dtype=str(XNYZIP_ORDER_DTYPE),
            values=order,
        )

    def _record_ordering(self, order: CanonicalOrder) -> None:
        reconstructed_rows = {
            "mapping": order.mapping,
            "original_row_order_restored": not order.is_reordered,
            "canonical_field": order.field,
            "canonical_lcp_field": (
                "positions"
                if (
                    order.field == "positions"
                    and self.settings.position_codec == "lcp"
                )
                else None
            ),
            "lcp_permutation_stored": False,
            "temporary_permutation_artifact": order.artifact,
            "temporary_permutation_dtype": order.artifact_dtype,
        }
        id_ordering: Dict[str, Any] = {"mapping": order.mapping}
        if order.field == "positions":
            reconstructed_rows["position_permutation_stored"] = False
            if self.settings.position_codec == "lcp":
                id_ordering["replaces_lcp_position_order"] = True
            else:
                reconstructed_rows["canonical_xnyzip_field"] = "positions"
                reconstructed_rows["xnyzip_permutation_stored"] = False
                id_ordering["replaces_xnyzip_position_order"] = True
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
        if self.settings.position_codec in ("lcp", "xnyzip"):
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
        if self.settings.velocity_codec == "xnyzip":
            self._compress_xnyzip_velocities(order)
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

    def _compress_xnyzip_velocities(self, order: CanonicalOrder) -> None:
        if order.values is None:
            raise RuntimeError(
                "Position-ordered XnYZip velocities require a canonical "
                "order."
            )

        interleaved_path, interleaved_metadata = (
            export_ordered_triplet_for_xnyzip(
                {
                    logical: self.raw_paths[f"{logical}_xnyzip"]
                    for logical in VELOCITY_FIELDS
                },
                VELOCITY_FIELDS,
                self.preprocessed_dir
                / "velocities.xnyzip.canonical.f32.raw",
                self.count,
                order.values,
                self.settings.force,
            )
        )
        self.raw_paths["velocities_xnyzip"] = interleaved_path

        order_path = self.preprocessed_dir / "velocity_order.u64.raw"
        self.raw_paths["velocity_order"] = str(order_path)
        l2_error_bound = float(
            self.manifest["error_bounds"]["velocities_xnyzip_abs"]
        )
        started = time.perf_counter()
        chunk_metadata = None
        if self.settings.velocity_chunk_size:
            chunk_metadata = compress_chunked_xnyzip_triplet(
                self.tools,
                interleaved_path,
                self.artifacts["velocities"],
                self.count,
                self.settings.velocity_chunk_size,
                l2_error_bound,
                order_path,
                self.settings.force,
                self.settings.effective_chunk_workers,
            )
        else:
            compress_xnyzip_triplet(
                self.tools,
                interleaved_path,
                self.artifacts["velocities"],
                self.count,
                l2_error_bound,
                order_path,
                self.settings.force,
            )
        self.manifest.setdefault("timing", {})[
            "velocity_xnyzip_compress_wall_seconds"
        ] = time.perf_counter() - started

        velocity_field = self._xnyzip_field_metadata(
            "velocities",
            VELOCITY_FIELDS,
            self.artifacts["velocities"],
            l2_error_bound,
            chunk_size=self.settings.velocity_chunk_size,
        )
        velocity_field["preprocessed_interleaved"] = interleaved_metadata
        self.compressed_fields["velocities"] = velocity_field

        order_field = compress_integer_raw(
            self.args.lossless,
            str(order_path),
            str(XNYZIP_ORDER_DTYPE),
            self.artifacts["velocity_order"],
            "velocity_order",
            self.count,
            self.settings.force,
        )
        order_metadata = {
            "uncompressed_storage_dtype": str(XNYZIP_ORDER_DTYPE),
            "compressed_bits_per_particle": (
                8.0 * float(order_field["bytes"]) / self.count
                if self.count
                else 0.0
            ),
            "chunk_size": self.settings.velocity_chunk_size,
            "chunk_count": (
                (self.count + self.settings.velocity_chunk_size - 1)
                // self.settings.velocity_chunk_size
                if self.settings.velocity_chunk_size
                else 1
            ),
            "index_scope": (
                "chunk_local"
                if self.settings.velocity_chunk_size
                else "global"
            ),
            "order_bits_per_particle": velocity_order_bits(
                self.settings.velocity_chunk_size or self.count
            ),
            "order_mapping": (
                "xnyzip_velocity_sorted_index_to_"
                "xnyzip_position_sorted_row"
            ),
        }
        if chunk_metadata is not None:
            order_metadata.update(chunk_metadata)
        order_field.update(order_metadata)
        self.compressed_fields["velocity_order"] = order_field
        self.manifest["ordering"]["velocities"] = {
            "mapping": (
                "xnyzip_velocity_sorted_index_to_"
                "xnyzip_position_sorted_row"
            ),
            "field": "velocity_order",
            "index_scope": order_metadata["index_scope"],
            "chunk_size": self.settings.velocity_chunk_size,
        }

    def _compress_lcp_velocities(self, order: CanonicalOrder) -> None:
        self._compress_secondary_lcp_velocities(order)

        compressed_path = self.artifacts["velocities"]
        self.compressed_fields["velocities"] = self._lcp_field_metadata(
            "velocities",
            VELOCITY_FIELDS,
            compressed_path,
            float(self.manifest["error_bounds"]["velocities_lcp_abs"]),
            chunk_size=self.settings.velocity_chunk_size,
        )
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
    ) -> Tuple[str, str, str]:
        return tuple(self.raw_paths[field] for field in fields)

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

    def _xnyzip_field_metadata(
        self,
        field_name: str,
        source_fields: Tuple[str, str, str],
        compressed_path: str,
        l2_error_bound: float,
        chunk_size: int = 0,
    ) -> Dict[str, Any]:
        path = Path(compressed_path)
        metadata = {
            "field": field_name,
            "codec": "xnyzip",
            "dtype": "float32",
            "source_dtypes": {
                logical: self.manifest["fields"][logical]["dtype"]
                for logical in source_fields
            },
            "count": self.count,
            "input_layout": "triplet_interleaved",
            "interleaved_fields": list(source_fields),
            "native_order_dtype": str(XNYZIP_ORDER_DTYPE),
            "error_bound_norm": "l2",
            "l2_error_bound": l2_error_bound,
            "quantizer": XNYZIP_QUANTIZER,
            "curve": XNYZIP_CURVE,
            "storage_mode": XNYZIP_STORAGE_MODE,
            "direct_threshold": XNYZIP_DIRECT_THRESHOLD,
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
                        XNYZIP_CHUNK_CONTAINER
                        if chunk_size
                        else "native_xnyzip"
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
        if (
            self.settings.velocity_codec == "xnyzip"
            and chunk_size
        ):
            self.manifest["format_version"] = 6
        elif (
            self.settings.position_codec == "xnyzip"
            or self.settings.velocity_codec == "xnyzip"
        ):
            self.manifest["format_version"] = 5
        else:
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
    if (
        position_codec == "lcp"
        and velocity_codec == "lcp"
        and chunk_size > MAX_INT32_ORDER_VALUES
    ):
        raise RuntimeError(
            "--vel-chunk-size cannot exceed 2^31 when using int32 order indices."
        )
    if workers < 0:
        raise RuntimeError("--vel-chunk-workers must be non-negative.")
    chunked_pair = (
        position_codec == velocity_codec
        and position_codec in ("lcp", "xnyzip")
    )
    if chunk_size and not chunked_pair:
        raise RuntimeError(
            "--vel-chunk-size is only supported when both position and "
            "velocity compressors are lcp or both are xnyzip."
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
    "compress_chunked_xnyzip_triplet",
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
