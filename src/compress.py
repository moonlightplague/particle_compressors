"""Compression-stage orchestration.

Codec implementations live in :mod:`src.raw_codecs`, :mod:`src.lcp_codec`,
and :mod:`src.xnyzip_codec`. This module coordinates row ordering and records
the resulting manifest.
"""

import argparse
import math
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
from src.huffman_encode import huffman_encode_file
from src.lattice_layout import (
    DenseLatticeLayout,
    IDENTITY_TRANSFORM,
    LATTICE_LAYOUT_NAME,
    LatticeLayoutUnavailable,
    POSITION_RESIDUAL_TRANSFORM,
    infer_complete_lattice_layout,
    infer_dense_lattice_layout,
    infer_lattice_id_mapping,
    position_transform_guard,
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
    SZO_LORENZO_1D_PROFILE,
    compress_integer_raw,
    compress_lattice_hilbert_ids,
    compress_lossy_raw,
    compress_pcodec_raw,
    compress_pysz_raw,
    compress_szo_raw,
    pad_codec_input,
)
from src.structured_layout import (
    HYBRID_VELOCITY_LAYOUT,
    StructuredParticleLayout,
    hybrid_velocity_order,
    make_structured_layout,
)
from src.shaped_codecs import compress_shaped_lossy_raw
from src.runtime import (
    read_raw,
    require_output_path,
    resolve_field_workers,
    resolve_velocity_chunk_workers,
    write_json,
)
from src.xnyzip_codec import (
    XNYZIP_CURVE,
    XNYZIP_DIRECT_THRESHOLD,
    XNYZIP_HILBERT_CURVE,
    XNYZIP_ORDER_DTYPE,
    XNYZIP_QUANTIZER,
    XNYZIP_STORAGE_MODE,
    compress_chunked_xnyzip_triplet,
    compress_xnyzip_triplet,
    read_xnyzip_permutation,
    run_xnyzip_decompress,
)


@dataclass(frozen=True)
class CompressionSettings:
    position_codec: str
    velocity_codec: str
    velocity_chunk_size: int
    configured_chunk_workers: int
    effective_chunk_workers: int
    blockwise_order: bool
    force: bool
    sort_requested: bool = False
    sort_by_id: bool = False
    lattice_requested: bool = False
    lattice_layout: bool = False
    lattice_velocity_only: bool = False
    lattice_min_occupancy: float = 0.8
    lattice_axis_search: bool = True
    structure_aware_requested: bool = False
    structure_aware: bool = False
    structure_velocity_cell_bits: int = 7
    field_workers: int = 1

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "CompressionSettings":
        position_codec = getattr(args, "pos_compressor", "lcp")
        velocity_codec = args.vel_compressor
        validate_compressor_combination(position_codec, velocity_codec)
        sort_requested = bool(getattr(args, "sort", False))
        lattice_requested = bool(getattr(args, "lattice_layout", False))
        structure_aware_requested = bool(
            getattr(args, "xnyzip_structure_aware", False)
        )
        structure_velocity_cell_bits = int(
            getattr(args, "xnyzip_velocity_cell_bits", 7)
        )
        if not 1 <= structure_velocity_cell_bits <= 10:
            raise RuntimeError(
                "--xnyzip-velocity-cell-bits must be in [1, 10]."
            )
        fieldwise_pair = (
            position_codec not in ("lcp", "xnyzip")
            and velocity_codec not in ("lcp", "xnyzip")
        )
        lattice_velocity_only = (
            lattice_requested
            and position_codec in ("lcp", "xnyzip")
            and velocity_codec not in ("lcp", "xnyzip")
        )
        # Lattice position residuals use floating-point arithmetic. Preserve
        # source bits by using flat streams for lossless field combinations.
        lattice_layout = lattice_requested and "pcodec" not in (
            position_codec, velocity_codec
        ) and (
            fieldwise_pair or lattice_velocity_only
        )
        structure_aware = (
            structure_aware_requested
            and position_codec == "xnyzip"
            and velocity_codec in ("szo", "xnyzip")
        )
        if lattice_requested and structure_aware_requested:
            raise RuntimeError(
                "--xnyzip-structure-aware cannot be combined with "
                "--lattice-layout."
            )
        sort_by_id = (
            (sort_requested or lattice_layout)
            and fieldwise_pair
        )
        lattice_min_occupancy = float(
            getattr(args, "lattice_min_occupancy", 0.8)
        )
        if not 0.0 < lattice_min_occupancy <= 1.0:
            raise RuntimeError("--lattice-min-occupancy must be in (0, 1].")
        chunk_size = int(getattr(args, "vel_chunk_size", 0))
        configured_workers = int(getattr(args, "vel_chunk_workers", 0))
        blockwise_order = bool(getattr(args, "blockwise_ord", False))
        field_workers = int(getattr(args, "field_workers", 1))
        if field_workers < 0:
            raise RuntimeError("--field-workers must be non-negative.")
        _validate_chunk_configuration(
            position_codec,
            velocity_codec,
            chunk_size,
            configured_workers,
            blockwise_order,
        )
        return cls(
            position_codec=position_codec,
            velocity_codec=velocity_codec,
            sort_requested=sort_requested,
            sort_by_id=sort_by_id,
            lattice_requested=lattice_requested,
            lattice_layout=lattice_layout,
            lattice_velocity_only=lattice_velocity_only,
            lattice_min_occupancy=lattice_min_occupancy,
            lattice_axis_search=bool(
                getattr(args, "lattice_axis_search", True)
            ),
            structure_aware_requested=structure_aware_requested,
            structure_aware=structure_aware,
            structure_velocity_cell_bits=structure_velocity_cell_bits,
            velocity_chunk_size=chunk_size,
            configured_chunk_workers=configured_workers,
            effective_chunk_workers=resolve_velocity_chunk_workers(
                configured_workers
            ),
            blockwise_order=blockwise_order,
            force=bool(args.force),
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
    szo_profile: Optional[str] = None


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
            **({"szo_profile": job.szo_profile} if job.szo_profile else {}),
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
        self.dense_merged_id_base = self._dense_merged_id_base()
        self.lattice: Optional[DenseLatticeLayout] = None
        self.structured_layout: Optional[StructuredParticleLayout] = None
        self._lattice_ordered_values: Dict[str, np.ndarray] = {}
        self.field_workers = resolve_field_workers(
            self.settings.field_workers,
        )
        self.lossy_field_workers_used = 0
        self.lossy_fields_seconds = 0.0
        self.canonical_order_seconds = 0.0
        self.lattice_prepare_seconds = 0.0
        self.structure_prepare_seconds = 0.0
        self.hybrid_velocity_order_seconds = 0.0
        self.id_compress_seconds = 0.0
        self.lattice_field_prepare_seconds = 0.0

    def run(self) -> Dict[str, Any]:
        started = time.perf_counter()
        stage_started = time.perf_counter()
        self._prepare_structure_aware_layout()
        self.structure_prepare_seconds = time.perf_counter() - stage_started
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
        if self.lattice is not None:
            if self.settings.lattice_velocity_only:
                self._compress_positions(canonical_order)
                self._compress_lattice_velocities(canonical_order)
            else:
                self._compress_lattice_fields(canonical_order)
        elif (
            self.settings.position_codec not in ("lcp", "xnyzip")
            and self.settings.velocity_codec not in ("lcp", "xnyzip")
        ):
            self._compress_flat_fieldwise_fields(canonical_order)
        else:
            self._compress_positions(canonical_order)
            self._compress_velocities(canonical_order)
        self._finalize(started)
        return self.manifest

    def _prepare_structure_aware_layout(self) -> None:
        if not self.settings.structure_aware_requested:
            return
        try:
            if not self.settings.structure_aware:
                raise LatticeLayoutUnavailable(
                    "requires XnYZip positions and SZO or XnYZip velocities"
                )
            payload = self.manifest.get("root_attrs", {}).get("nsidemesh", {})
            if "value" not in payload:
                raise LatticeLayoutUnavailable(
                    "root attribute 'nsidemesh' is unavailable"
                )
            side = int(payload["value"])
            make_structured_layout(
                side, 0, (0, 1, 2), self.settings.structure_velocity_cell_bits
            )
            ids = np.memmap(
                self.raw_paths["id"], mode="r",
                dtype=self.manifest["fields"]["id"]["dtype"], shape=(self.count,),
            )
            sample = np.linspace(
                0, self.count - 1, min(self.count, 200_000), dtype=np.intp
            )
            positions = {
                logical: np.memmap(
                    self.raw_paths[logical], mode="r", dtype="float32",
                    shape=(self.count,),
                )[sample]
                for logical in POSITION_FIELDS
            }
            base, axes = infer_lattice_id_mapping(
                ids[sample], positions, side, int(ids.min()), int(ids.max())
            )
            self.structured_layout = make_structured_layout(
                side, base, axes, self.settings.structure_velocity_cell_bits
            )
            self.manifest["structured_layout"] = {
                "requested": True, **self.structured_layout.metadata(),
            }
        except LatticeLayoutUnavailable as exc:
            self.manifest["structured_layout"] = {
                "requested": True, "enabled": False, "reason": str(exc),
            }

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
            "field_scope": (
                "velocities"
                if self.settings.lattice_velocity_only
                else "positions_and_velocities"
            ),
        }
        if not self.settings.lattice_layout:
            self.manifest["lattice_layout"] = {
                **common,
                "enabled": False,
                "reason": (
                    "lattice layout requires fieldwise SZ3 or SZO "
                    "velocities and either native LCP/XnYZip or fieldwise "
                    "SZ3/SZO positions"
                ),
            }
            return
        expected_order_field = (
            "positions" if self.settings.lattice_velocity_only else "id"
        )
        if order.values is None or order.field != expected_order_field:
            self.manifest["lattice_layout"] = {
                **common,
                "enabled": False,
                "reason": (
                    "velocity-only lattice layout requires the native "
                    "position-compressor order"
                    if self.settings.lattice_velocity_only
                    else "lattice layout requires the canonical ID order"
                ),
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
            if (
                not self.settings.lattice_velocity_only
                and self._can_use_complete_lattice_fast_path(side)
            ):
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
            ordered_ids = read_raw(
                self.raw_paths["id"],
                id_dtype,
                self.count,
            )[order.values]
            ordered_positions = {
                logical: read_raw(
                    self.raw_paths[logical],
                    np.dtype("float32"),
                    self.count,
                )[order.values]
                for logical in POSITION_FIELDS
            }
            self.lattice = infer_dense_lattice_layout(
                ordered_ids,
                ordered_positions,
                side,
                self.settings.lattice_min_occupancy,
            )
        except (LatticeLayoutUnavailable, TypeError, ValueError, OverflowError) as exc:
            self.manifest["lattice_layout"] = {
                **common,
                "enabled": False,
                "reason": str(exc),
            }
            return
        self._lattice_ordered_values["id"] = ordered_ids
        if not self.settings.lattice_velocity_only:
            self._lattice_ordered_values.update(ordered_positions)
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
                minimum_key = (
                    "min_in_compressor_units"
                    if "min_in_compressor_units" in stats
                    else "min_in_lcp_units"
                )
                maximum_key = (
                    "max_in_compressor_units"
                    if "max_in_compressor_units" in stats
                    else "max_in_lcp_units"
                )
                minimum = float(stats[minimum_key])
                maximum = float(stats[maximum_key])
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
        validation_bound = float(
            self.manifest["error_bounds"]["positions_xnyzip_abs"]
        )
        l2_error_bound = validation_bound
        curve = XNYZIP_HILBERT_CURVE if self.structured_layout else XNYZIP_CURVE
        quantizer = XNYZIP_QUANTIZER
        max_attempts = 6
        for attempt in range(max_attempts):
            options = {"curve": curve} if self.structured_layout else {}
            if quantizer != XNYZIP_QUANTIZER:
                options["quantizer"] = quantizer
            order = compress_xnyzip_triplet(
                self.tools, self.raw_paths["positions_xnyzip"], compressed_path,
                self.count, l2_error_bound, order_path,
                self.settings.force if attempt == 0 else True, **options,
            )
            if order is None:
                order = read_xnyzip_permutation(str(order_path), self.count)
            maximum = self._measure_xnyzip_position_error(
                order, l2_error_bound, quantizer
            )
            if math.isfinite(maximum) and maximum <= validation_bound:
                break
            if quantizer == XNYZIP_QUANTIZER:
                # TO can select a negative boundary node that the unsigned
                # block encoder corrupts. Try cube before adjusting precision.
                quantizer = "cube"
                continue
            # Native quantization spends the full bound, leaving no room for
            # float32 shift/recovery rounding. Reserve at least 1%, or twice
            # the measured excess. Always validate against the ORIGINAL budget.
            margin = max(
                0.01 * validation_bound, 2 * (maximum - validation_bound)
            )
            next_bound = l2_error_bound - margin
            if (
                attempt + 1 == max_attempts
                or not math.isfinite(next_bound)
                or next_bound <= 0
            ):
                raise RuntimeError(
                    "XnYZip positions failed L2 validation after codec retries: "
                    f"observed {maximum:.9g}, allowed {validation_bound:.9g}. "
                    "Try a larger position bound or another position compressor."
                )
            l2_error_bound = next_bound

        # The native decoder takes its scale from the command line. Persist
        # the accepted scale everywhere it is consumed or reported.
        self.manifest["error_bounds"]["positions_xnyzip_abs"] = l2_error_bound
        vector_bounds = self.manifest.get("field_error_bounds", {}).get(
            "positions_xnyzip"
        )
        if vector_bounds is not None:
            vector_bounds["compressor_abs"] = l2_error_bound
            vector_bounds["codec_l2_safety_margin"] = validation_bound - l2_error_bound
        if self.structured_layout is not None and quantizer != XNYZIP_QUANTIZER:
            self.manifest["structured_layout"][
                "position_quantizer_fallback"
            ] = "to_roundtrip_failed"
        self.compressed_fields["positions"] = (
            self._xnyzip_field_metadata(
                "positions",
                POSITION_FIELDS,
                compressed_path,
                l2_error_bound,
            )
        )
        self.compressed_fields["positions"]["curve"] = curve
        self.compressed_fields["positions"]["quantizer"] = quantizer
        self.compressed_fields["positions"]["validated_max_l2_error"] = maximum
        self.compressed_fields["positions"]["validation_l2_bound"] = validation_bound
        self.compressed_fields["positions"]["compression_attempts"] = attempt + 1
        if self.structured_layout is None:
            for key in POSITION_FIELDS:
                decoded_path = self.raw_paths.pop(f"{key}_structured_decoded", None)
                if decoded_path is not None:
                    Path(decoded_path).unlink()
        return CanonicalOrder(
            mapping="xnyzip_position_sorted",
            field="positions",
            artifact="position_order",
            artifact_dtype=str(XNYZIP_ORDER_DTYPE),
            values=order,
        )

    def _measure_xnyzip_position_error(
        self, order: np.ndarray, bound: float, quantizer: str = XNYZIP_QUANTIZER,
    ) -> float:
        """Measure every decoded position in float64 before accepting its order."""

        decoded_paths = {
            logical: str(self.preprocessed_dir / f"{logical}.structured-decoded.f32.raw")
            for logical in POSITION_FIELDS
        }
        self.raw_paths.update({f"{key}_structured_decoded": value
                               for key, value in decoded_paths.items()})
        run_xnyzip_decompress(
            self.tools, self.artifacts["positions"], decoded_paths, POSITION_FIELDS,
            self.count, bound, self.preprocessed_dir / "structured-decoded.interleaved.f32.raw", True,
            quantizer=quantizer)
        source = {key: np.memmap(self.raw_paths[key], mode="r", dtype="float32", shape=(self.count,))
                  for key in POSITION_FIELDS}
        decoded = {key: np.memmap(path, mode="r", dtype="float32", shape=(self.count,))
                   for key, path in decoded_paths.items()}
        maximum_squared = 0.0
        for start in range(0, self.count, 1_048_576):
            end = min(self.count, start + 1_048_576)
            squared = np.zeros(end - start, dtype=np.float64)
            for key in POSITION_FIELDS:
                difference = (decoded[key][start:end].astype(np.float64)
                              - source[key][order[start:end]].astype(np.float64))
                squared += difference * difference
            if not np.all(np.isfinite(squared)):
                return math.inf
            maximum_squared = max(maximum_squared, float(squared.max(initial=0.0)))
        return math.sqrt(maximum_squared)

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
            "requested": (
                self.settings.sort_requested
                or self.settings.lattice_layout
            ),
            "enabled": order.field == "id",
            "key": "id" if order.field == "id" else None,
            "direction": "ascending" if order.field == "id" else None,
            "stable": bool(order.field == "id"),
        }

    def _compress_id(self, order: CanonicalOrder) -> None:
        dtype = self.manifest["fields"]["id"]["dtype"]
        raw_path = self._ordered_raw_path("id", dtype, order)
        if self.structured_layout is not None:
            self.compressed_fields["id"] = compress_lattice_hilbert_ids(
                read_raw(raw_path, np.dtype(dtype), self.count), dtype,
                self.artifacts["id"], "id", self.structured_layout, self.settings.force)
            return
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
        jobs = []
        for logical in POSITION_FIELDS:
            dtype = (
                self.manifest["fields"][logical]["dtype"]
                if self.settings.position_codec == "pcodec"
                else "float32"
            )
            raw_path = self._ordered_raw_path(logical, dtype, order)
            jobs.append(LossyCompressionJob(
                self.settings.position_codec,
                raw_path,
                dtype,
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
        self.manifest["ordering"]["positions"] = {"mapping": order.mapping}

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
                self.settings.position_codec,
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
        if self.structured_layout is not None and self.settings.velocity_codec == "szo":
            self._compress_structured_velocities(order)
            return
        if self.settings.velocity_codec == "lcp":
            self._compress_lcp_velocities(order)
            return
        if self.settings.velocity_codec == "xnyzip":
            self._compress_xnyzip_velocities(order)
            return
        jobs = []
        for logical in VELOCITY_FIELDS:
            dtype = self.manifest["fields"][logical]["dtype"]
            raw_path = self._ordered_raw_path(logical, dtype, order)
            jobs.append(LossyCompressionJob(
                self.settings.velocity_codec,
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
        self.manifest["ordering"]["velocities"] = {"mapping": order.mapping}

    def _structured_velocity_source_order(self, order: CanonicalOrder) -> np.ndarray:
        assert self.structured_layout is not None and order.values is not None
        started = time.perf_counter()
        decoded_paths = {
            logical: self.raw_paths[f"{logical}_structured_decoded"]
            for logical in POSITION_FIELDS
        }
        ids = read_raw(
            self.raw_paths["id_canonical_ordered"],
            np.dtype(self.manifest["fields"]["id"]["dtype"]), self.count,
        )
        decoded = {
            logical: read_raw(path, np.dtype("float32"), self.count)
            for logical, path in decoded_paths.items()
        }
        permutation = hybrid_velocity_order(ids, decoded, self.structured_layout)
        source_order = order.values[permutation]
        del permutation, ids, decoded
        self.hybrid_velocity_order_seconds = time.perf_counter() - started
        return source_order

    def _compress_structured_velocities(self, order: CanonicalOrder) -> None:
        source_order = self._structured_velocity_source_order(order)
        jobs = []
        for logical in VELOCITY_FIELDS:
            dtype = self.manifest["fields"][logical]["dtype"]
            path = (
                self.preprocessed_dir / f"{logical}.hybrid.{np.dtype(dtype).name}.raw"
            )
            require_output_path(path, self.settings.force)
            values = read_raw(self.raw_paths[logical], np.dtype(dtype), self.count)
            values[source_order].tofile(path)
            del values
            self.raw_paths[f"{logical}_hybrid"] = str(path)
            jobs.append(LossyCompressionJob(
                "szo", str(path), dtype, self.artifacts[logical], logical, self.count,
                float(self.manifest["field_error_bounds"][logical]["abs"]),
                self.settings.force, szo_profile=SZO_LORENZO_1D_PROFILE,
            ))
        del source_order
        for logical, field in zip(VELOCITY_FIELDS, self._compress_jobs(jobs)):
            field["spatial_layout"] = HYBRID_VELOCITY_LAYOUT
            self.compressed_fields[logical] = field
        self.manifest["ordering"]["velocities"] = {
            "mapping": HYBRID_VELOCITY_LAYOUT,
            "reconstructed_mapping": order.mapping,
        }

    def _compress_flat_fieldwise_fields(
        self,
        order: CanonicalOrder,
    ) -> None:
        jobs: List[LossyCompressionJob] = []
        logical_fields = (*POSITION_FIELDS, *VELOCITY_FIELDS)
        for logical in logical_fields:
            is_position = logical in POSITION_FIELDS
            dtype = (
                "float32"
                if is_position and self.settings.position_codec != "pcodec"
                else self.manifest["fields"][logical]["dtype"]
            )
            raw_path = self._ordered_raw_path(logical, dtype, order)
            bound_key = "compressor_abs" if is_position else "abs"
            jobs.append(
                LossyCompressionJob(
                    self.settings.position_codec
                    if is_position
                    else self.settings.velocity_codec,
                    raw_path,
                    dtype,
                    self.artifacts[logical],
                    logical,
                    self.count,
                    float(
                        self.manifest["field_error_bounds"][logical][bound_key]
                    ),
                    self.settings.force,
                )
            )
        for logical, field in zip(logical_fields, self._compress_jobs(jobs)):
            self.compressed_fields[logical] = field
        self.manifest["ordering"]["positions"] = {"mapping": order.mapping}
        self.manifest["ordering"]["velocities"] = {"mapping": order.mapping}

    def _compress_jobs(
        self,
        jobs: List[LossyCompressionJob],
    ) -> List[Dict[str, Any]]:
        started = time.perf_counter()
        workers = min(self.field_workers, len(jobs))
        self.lossy_field_workers_used = max(
            self.lossy_field_workers_used,
            workers,
        )
        if workers == 1:
            results = [_compress_lossy_job(job) for job in jobs]
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                results = list(executor.map(_compress_lossy_job, jobs))
        self.lossy_fields_seconds += time.perf_counter() - started
        return results

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
                self.settings.velocity_codec,
                str(dense_path),
                str(dense.dtype),
                self.artifacts[logical],
                logical,
                self.count,
                float(
                    self.manifest["field_error_bounds"][logical]["abs"]
                ),
                self.settings.force,
                self.lattice.shape,
                self.settings.lattice_axis_search,
            ))
        return jobs

    def _compress_lattice_fields(self, order: CanonicalOrder) -> None:
        prepare_started = time.perf_counter()
        position_jobs, position_updates = self._prepare_lattice_positions(order)
        velocity_jobs = self._prepare_lattice_velocities(order)
        self.lattice_field_prepare_seconds = (
            time.perf_counter() - prepare_started
        )
        results = self._compress_jobs(position_jobs + velocity_jobs)
        position_results = results[: len(POSITION_FIELDS)]
        velocity_results = results[len(POSITION_FIELDS):]

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

    def _compress_lattice_velocities(self, order: CanonicalOrder) -> None:
        prepare_started = time.perf_counter()
        jobs = self._prepare_lattice_velocities(order)
        self.lattice_field_prepare_seconds = (
            time.perf_counter() - prepare_started
        )
        for logical, field in zip(VELOCITY_FIELDS, self._compress_jobs(jobs)):
            field.update(
                {
                    "spatial_layout": LATTICE_LAYOUT_NAME,
                    "lattice_transform": IDENTITY_TRANSFORM,
                }
            )
            self.compressed_fields[logical] = field
        self.manifest["ordering"]["velocities"] = {
            "mapping": order.mapping,
            "spatial_layout": LATTICE_LAYOUT_NAME,
        }

    def _compress_xnyzip_velocities(self, order: CanonicalOrder) -> None:
        if order.values is None:
            raise RuntimeError(
                "Position-ordered XnYZip velocities require a canonical "
                "order."
            )
        input_mapping = order.mapping
        source_order = order.values
        if self.structured_layout is not None:
            input_mapping = HYBRID_VELOCITY_LAYOUT
            source_order = self._structured_velocity_source_order(order)
        velocity_mapping = (
            f"xnyzip_velocity_sorted_index_to_{input_mapping}_row"
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
                source_order,
                self.settings.force,
            )
        )
        del source_order
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
        if self.structured_layout is not None:
            velocity_field["spatial_layout"] = HYBRID_VELOCITY_LAYOUT
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
            "order_mapping": velocity_mapping,
        }
        if chunk_metadata is not None:
            order_metadata.update(chunk_metadata)
        order_field.update(order_metadata)
        self.compressed_fields["velocity_order"] = order_field
        self.manifest["ordering"]["velocities"] = {
            "mapping": velocity_mapping,
            "field": "velocity_order",
            "index_scope": order_metadata["index_scope"],
            "chunk_size": self.settings.velocity_chunk_size,
        }
        if self.structured_layout is not None:
            self.manifest["ordering"]["velocities"]["reconstructed_mapping"] = order.mapping

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
                "block_local_packed"
                if self.settings.blockwise_order
                else (
                    "chunk_local"
                    if self.settings.velocity_chunk_size
                    else "global"
                )
            ),
            "chunk_size": self.settings.velocity_chunk_size,
            "applied_during_lcp_decompression": (
                self.settings.blockwise_order
            ),
        }
        if self.settings.blockwise_order:
            self.manifest["ordering"]["velocities"][
                "block_id_field"
            ] = "velocity_block_ids"

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

        order_dtype = "uint32" if self.settings.blockwise_order else "int32"
        order_path = (
            self.preprocessed_dir
            / f"velocity_order.{np.dtype(order_dtype).name}.raw"
        )
        self.raw_paths["velocity_order"] = str(order_path)
        block_id_path = (
            self.preprocessed_dir / "velocity_block_ids.raw"
            if self.settings.blockwise_order
            else None
        )
        if block_id_path is not None:
            self.raw_paths["velocity_block_ids"] = str(block_id_path)
        chunk_metadata = self._run_secondary_velocity_compressor(
            ordered_paths,
            order_path,
            block_id_path,
        )
        if block_id_path is not None:
            self._compress_blockwise_velocity_sidecars(
                order_path,
                block_id_path,
            )
            return

        order_field = compress_integer_raw(
            self.args.lossless,
            str(order_path),
            order_dtype,
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
        block_id_path: Optional[Path] = None,
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
                block_id_path,
            )
        self.manifest.setdefault("timing", {})[
            "velocity_lcp_compress_wall_seconds"
        ] = time.perf_counter() - started
        return chunk_metadata

    def _compress_blockwise_velocity_sidecars(
        self,
        order_path: Path,
        block_id_path: Path,
    ) -> None:
        order_bytes = order_path.stat().st_size
        if order_bytes % np.dtype("uint32").itemsize:
            raise RuntimeError(
                "LCP blockwise order file contains a partial uint32 word."
            )
        order_word_count = order_bytes // np.dtype("uint32").itemsize
        order_field = compress_integer_raw(
            self.args.lossless,
            str(order_path),
            "uint32",
            self.artifacts["velocity_order"],
            "velocity_order",
            order_word_count,
            self.settings.force,
        )
        order_field.update(
            {
                "order_encoding": "lcp_blockwise_packed",
                "word_bits": 32,
                "packed_word_count": order_word_count,
                "particle_count": self.count,
                "uncompressed_bytes": order_bytes,
                "index_scope": "block_local_packed",
                "chunk_size": 0,
                "chunk_count": 1,
                "applied_during_lcp_decompression": True,
                "block_id_field": "velocity_block_ids",
                "compressed_bits_per_particle": (
                    8.0 * float(order_field["bytes"]) / self.count
                    if self.count
                    else 0.0
                ),
            }
        )
        self.compressed_fields["velocity_order"] = order_field

        huffman_path = (
            self.preprocessed_dir / "velocity_block_ids.huffman.u8.raw"
        )
        huffman_started = time.perf_counter()
        huffman_metadata = huffman_encode_file(
            block_id_path,
            huffman_path,
            self.settings.force,
            expected_count=self.count,
        )
        self.manifest.setdefault("timing", {})[
            "velocity_block_id_huffman_encode_wall_seconds"
        ] = time.perf_counter() - huffman_started
        self.raw_paths["velocity_block_ids_huffman"] = str(huffman_path)

        block_id_field = compress_integer_raw(
            self.args.lossless,
            str(huffman_path),
            "uint8",
            self.artifacts["velocity_block_ids"],
            "velocity_block_ids",
            huffman_path.stat().st_size,
            self.settings.force,
        )
        block_id_field.update(
            {
                "preprocessor": huffman_metadata,
                "decoded_dtype": huffman_metadata["symbol_dtype"],
                "decoded_count": self.count,
                "decoded_bytes": block_id_path.stat().st_size,
                "applied_during_lcp_decompression": True,
            }
        )
        self.compressed_fields["velocity_block_ids"] = block_id_field

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
        output_path = (
            self.preprocessed_dir
            / f"{logical}.{order.mapping}.{data_type.name}.raw"
        )
        if (
            logical == "id"
            and order.field == "id"
            and self.dense_merged_id_base is not None
        ):
            ordered_path = _write_dense_ids(
                output_path,
                data_type,
                self.count,
                self.dense_merged_id_base,
                self.settings.force,
            )
            self.raw_paths["id_canonical_ordered"] = ordered_path
            return ordered_path
        cached = self._lattice_ordered_values.pop(logical, None)
        ordered_path = reorder_raw(
            source_path,
            dtype,
            output_path,
            self.count,
            order.values,
            self.settings.force,
            values=cached,
        )
        key = (
            "id_canonical_ordered"
            if logical == "id"
            else f"{logical}_canonical_ordered"
        )
        self.raw_paths[key] = ordered_path
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
        if self.structured_layout is not None:
            self.manifest["format_version"] = 9
        elif self.lattice is not None:
            self.manifest["format_version"] = 8
        elif self.settings.blockwise_order:
            self.manifest["format_version"] = 7
        elif (
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
        timing = self.manifest.setdefault("timing", {})
        timing["canonical_order_wall_seconds"] = self.canonical_order_seconds
        if self.structured_layout is not None:
            timing["structure_prepare_wall_seconds"] = self.structure_prepare_seconds
            timing["hybrid_velocity_order_wall_seconds"] = self.hybrid_velocity_order_seconds
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
        ] = self.lossy_field_workers_used
        update_compressed_size_metrics(self.manifest, self.work_dir)
        write_json(
            self.work_dir / "manifest.json",
            self.manifest,
            force=True,
        )


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


def _validate_chunk_configuration(
    position_codec: str,
    velocity_codec: str,
    chunk_size: int,
    workers: int,
    blockwise_order: bool = False,
) -> None:
    if blockwise_order and (
        position_codec != "lcp" or velocity_codec != "lcp"
    ):
        raise RuntimeError(
            "--blockwise-ord requires --pos-compressor lcp and "
            "--vel-compressor lcp."
        )
    if blockwise_order and chunk_size:
        raise RuntimeError(
            "--blockwise-ord cannot be combined with --vel-chunk-size."
        )
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
        (position_codec == "lcp" and velocity_codec == "lcp")
        or (
            position_codec in ("lcp", "xnyzip")
            and velocity_codec == "xnyzip"
        )
    )
    if chunk_size and not chunked_pair:
        raise RuntimeError(
            "--vel-chunk-size is only supported for lcp velocities with "
            "lcp positions or xnyzip velocities with lcp/xnyzip positions."
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
    "_inverse_dense_id_order",
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
