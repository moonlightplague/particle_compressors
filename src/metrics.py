"""Roundtrip quality metrics and human-readable reporting."""

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import h5py
import numpy as np

from src.constants import LOGICAL_ORDER, POSITION_FIELDS, VELOCITY_FIELDS
from src.manifest import (
    compressed_sizes,
    order_dtype_from_manifest,
    position_compressor_from_manifest,
    velocity_compressor_from_manifest,
)
from src.runtime import read_raw


@dataclass
class MetricAccumulator:
    """Numerically accumulate error statistics for one field."""

    count: int = 0
    sum_squared_error: float = 0.0
    sum_absolute_error: float = 0.0
    max_absolute_error: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf

    def update(self, original: np.ndarray, reconstructed: np.ndarray) -> None:
        original64 = original.astype(np.float64, copy=False)
        reconstructed64 = reconstructed.astype(np.float64, copy=False)
        difference = reconstructed64 - original64
        absolute_difference = np.abs(difference)

        self.count += int(original.size)
        self.sum_squared_error += float(np.dot(difference, difference))
        self.sum_absolute_error += float(absolute_difference.sum())
        self.max_absolute_error = max(
            self.max_absolute_error,
            float(absolute_difference.max(initial=0.0)),
        )
        if original.size:
            self.minimum = min(self.minimum, float(original64.min()))
            self.maximum = max(self.maximum, float(original64.max()))

    def finalize(self) -> Dict[str, Any]:
        if self.count == 0:
            raise RuntimeError("Cannot finalize metrics for zero elements.")

        mse = self.sum_squared_error / self.count
        rmse = math.sqrt(mse)
        value_range = self.maximum - self.minimum
        if mse == 0:
            psnr = math.inf
        elif value_range == 0:
            psnr = -math.inf
        else:
            psnr = (
                20.0 * math.log10(value_range)
                - 10.0 * math.log10(mse)
            )
        nrmse = (
            0.0
            if value_range == 0 and rmse == 0
            else (rmse / value_range if value_range else math.inf)
        )
        return {
            "count": self.count,
            "min": self.minimum,
            "max": self.maximum,
            "range": value_range,
            "max_absolute_error": self.max_absolute_error,
            "mean_absolute_error": self.sum_absolute_error / self.count,
            "mse": mse,
            "mmse": mse,
            "rmse": rmse,
            "nrmse": nrmse,
            "psnr": psnr,
        }


def report_field_dtype(report: Mapping[str, Any], logical: str) -> np.dtype:
    field = report["fields"][logical]
    dtype = field.get("dtype", field.get("original_dtype"))
    if dtype is None:
        raise RuntimeError(f"Missing dtype for field {logical!r} in report.")
    return np.dtype(dtype)


def report_count(report: Mapping[str, Any]) -> int:
    if "count" in report:
        return int(report["count"])
    sizes = report.get("sizes", {})
    if "selected_particle_count" in sizes:
        return int(sizes["selected_particle_count"])
    first_field = report["fields"][LOGICAL_ORDER[0]]
    return int(first_field.get("count", 0))


def original_bytes_for_fields(
    report: Mapping[str, Any],
    fields: Iterable[str],
) -> int:
    count = report_count(report)
    return int(
        sum(report_field_dtype(report, field).itemsize * count for field in fields)
    )


def compressed_bytes_with_prefixes(
    components: Mapping[str, int],
    prefixes: Iterable[str],
) -> int:
    prefix_tuple = tuple(prefixes)
    return sum(
        int(size)
        for path, size in components.items()
        if path.startswith(prefix_tuple)
    )


def compression_ratio(original_bytes: int, compressed_bytes: int) -> float:
    return (
        original_bytes / compressed_bytes
        if compressed_bytes
        else math.inf
    )


def component_compression_ratios(
    report: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    components = report.get("sizes", {}).get(
        "compressed_components_bytes",
        {},
    )
    position_order_bytes = compressed_bytes_with_prefixes(
        components,
        ("compressed/order.",),
    )
    position_bytes = (
        int(components.get("compressed/positions.lcp", 0))
        + position_order_bytes
        + compressed_bytes_with_prefixes(
            components,
            ("compressed/x.", "compressed/y.", "compressed/z."),
        )
    )
    velocity_lcp_bytes = int(
        components.get("compressed/velocities.lcp", 0)
    )
    velocity_order_bytes = compressed_bytes_with_prefixes(
        components,
        ("compressed/velocity_order.",),
    )
    count = report_count(report)
    order_bytes = int(order_dtype_from_manifest(report).itemsize * count)

    entries = {
        "x": _field_size_entry(report, components, "x"),
        "y": _field_size_entry(report, components, "y"),
        "z": _field_size_entry(report, components, "z"),
        "xyz": _size_entry(
            original_bytes_for_fields(report, POSITION_FIELDS),
            position_bytes,
        ),
        "order": _size_entry(order_bytes, position_order_bytes),
        "velocity_order": _size_entry(order_bytes, velocity_order_bytes),
        "id": _size_entry(
            original_bytes_for_fields(report, ("id",)),
            compressed_bytes_with_prefixes(components, ("compressed/id.",)),
        ),
        "vx": _field_size_entry(report, components, "vx"),
        "vy": _field_size_entry(report, components, "vy"),
        "vz": _field_size_entry(report, components, "vz"),
        "vxyz": _size_entry(
            original_bytes_for_fields(report, VELOCITY_FIELDS),
            velocity_lcp_bytes + velocity_order_bytes,
        ),
    }
    return entries


def field_group_compression_ratios(
    report: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Return combined size ratios for positions, IDs, and velocities."""

    components = report.get("sizes", {}).get(
        "compressed_components_bytes",
        {},
    )
    component_ratios = component_compression_ratios(report)
    velocity_bytes = compressed_bytes_with_prefixes(
        components,
        (
            "compressed/velocities.lcp",
            "compressed/velocity_order.",
            "compressed/vx.",
            "compressed/vy.",
            "compressed/vz.",
        ),
    )
    return {
        "positions": component_ratios["xyz"],
        "id": component_ratios["id"],
        "velocities": _size_entry(
            original_bytes_for_fields(report, VELOCITY_FIELDS),
            velocity_bytes,
        ),
    }


def print_component_summary(report: Mapping[str, Any]) -> None:
    ratios = component_compression_ratios(report)
    print("component_CR:")

    fieldwise_triplets = (
        position_compressor_from_manifest(report) != "lcp"
        and velocity_compressor_from_manifest(report) != "lcp"
    )
    names = list(POSITION_FIELDS) if fieldwise_triplets else ["xyz"]
    if ratios["order"]["compressed_bytes"] > 0:
        names.append("order")
    names.append("id")
    if ratios["vxyz"]["compressed_bytes"] > 0:
        names.append("vxyz")
        if ratios["velocity_order"]["compressed_bytes"] > 0:
            names.append("velocity_order")
    else:
        names.extend(VELOCITY_FIELDS)

    for name in names:
        entry = ratios[name]
        ratio = entry["compression_ratio"]
        ratio_label = "inf" if math.isinf(ratio) else f"{ratio:.6g}"
        includes_order = (
            name == "xyz" and ratios["order"]["compressed_bytes"] > 0
        ) or (
            name == "vxyz"
            and ratios["velocity_order"]["compressed_bytes"] > 0
        )
        note = " includes order sidecar" if includes_order else ""
        print(
            f"  {name}: CR={ratio_label}, "
            f"original_bytes={entry['original_bytes']}, "
            f"compressed_bytes={entry['compressed_bytes']}{note}"
        )


def print_summary(metrics: Mapping[str, Any], metrics_path: Path) -> None:
    print(f"metrics_json = {metrics_path}")
    print("Compressor Configuration: ")
    print(f"Lossless: {metrics["compressors"]["lossless"]}")
    print(f"Positions: {metrics["compressors"]["positions"]}")
    print(f"Velocities: {metrics["compressors"]["velocities"]}")
    sizes = metrics["sizes"]
    print(
        "payload_CR = "
        f"{sizes.get('payload_compression_ratio', math.nan):.6g}"
    )
    print(
        "compressed_total_bytes = "
        f"{sizes.get('compressed_total_bytes', 0)}"
    )
    print_component_summary(metrics)

    for logical in LOGICAL_ORDER:
        field = metrics["fields"][logical]
        bound = metrics["error_bound_consistency"][logical]
        display_field = (
            field.get("fixed_point_units", field)
            if logical in POSITION_FIELDS
            else field
        )
        units = (
            "lcp_units"
            if logical in POSITION_FIELDS and display_field is field
            else "source_units"
        )
        psnr = display_field["psnr"]
        psnr_label = (
            "inf"
            if math.isinf(psnr) and psnr > 0
            else f"{psnr:.6g}"
        )
        print(
            f"{logical}: "
            f"max_abs={display_field['max_absolute_error']:.6g}, "
            f"mse={display_field['mse']:.6g}, "
            f"psnr={psnr_label}, units={units}, "
            f"bound_ok={bound['satisfied']}"
        )


def comparison_order_for_reconstructed_rows(
    original: h5py.File,
    reconstructed: h5py.File,
    manifest: Mapping[str, Any],
    count: int,
) -> Tuple[Optional[np.ndarray], str]:
    row_order = manifest.get("ordering", {}).get("reconstructed_rows", {})
    if row_order.get("original_row_order_restored", True):
        return None, "original_row"

    artifact = (
        row_order.get("temporary_permutation_artifact")
        or "position_order"
    )
    raw_order_path = (
        manifest.get("artifacts", {})
        .get("preprocessed", {})
        .get(artifact)
    )
    if raw_order_path and Path(raw_order_path).is_file():
        permutation_dtype = np.dtype(
            row_order.get("temporary_permutation_dtype", "int32")
        )
        order = read_raw(
            raw_order_path,
            permutation_dtype,
            count,
        ).astype(np.intp, copy=False)
        _validate_comparison_order(order, count)
        return order, f"temporary_{artifact}"

    return _align_rows_by_particle_id(
        original,
        reconstructed,
        manifest,
        count,
    )


def compute_metrics(
    original_h5: Path,
    reconstructed_h5: Path,
    manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    started = time.perf_counter()
    count = int(manifest["count"])
    metrics: Dict[str, Any] = {
        "fields": {},
        "error_bound_consistency": {},
        "compressors": dict(manifest.get("compressors", {})),
        "particle_sort": dict(manifest.get("particle_sort", {})),
        "sizes": dict(manifest.get("sizes", {})),
        "timing": dict(manifest.get("timing", {})),
        "order_dtype": str(order_dtype_from_manifest(manifest)),
    }

    with h5py.File(original_h5, "r") as original, h5py.File(
        reconstructed_h5,
        "r",
    ) as reconstructed:
        comparison_order, comparison_source = (
            comparison_order_for_reconstructed_rows(
                original,
                reconstructed,
                manifest,
                count,
            )
        )
        metrics["row_comparison"] = {
            "reconstructed_order": (
                manifest.get("ordering", {})
                .get("reconstructed_rows", {})
                .get("mapping", "original_row")
            ),
            "alignment_source": comparison_source,
        }
        metrics["fields"] = _compute_field_metrics(
            original,
            reconstructed,
            manifest,
            comparison_order,
            count,
        )

    metrics["error_bound_consistency"] = _evaluate_error_bounds(
        metrics["fields"],
        manifest,
    )
    _update_final_size_metrics(metrics, manifest)
    metrics["timing"]["metrics_wall_seconds"] = time.perf_counter() - started
    return metrics


def _compute_field_metrics(
    original: h5py.File,
    reconstructed: h5py.File,
    manifest: Mapping[str, Any],
    comparison_order: Optional[np.ndarray],
    count: int,
) -> Dict[str, Dict[str, Any]]:
    results = {}
    scale = float(manifest["position_scale"]["value"])
    for logical in LOGICAL_ORDER:
        field = manifest["fields"][logical]
        original_dataset = original[field["h5_path"]]
        reconstructed_dataset = reconstructed[field["h5_path"]]
        original_values = original_dataset[:count]
        if comparison_order is not None:
            original_values = original_values[comparison_order]
        reconstructed_values = reconstructed_dataset[:count]

        accumulator = MetricAccumulator()
        fixed_point_accumulator = None
        if logical in POSITION_FIELDS and np.issubdtype(
            original_dataset.dtype,
            np.integer,
        ):
            fixed_point_accumulator = MetricAccumulator()
            fixed_point_accumulator.update(
                original_values,
                reconstructed_values,
            )

        if logical in POSITION_FIELDS:
            accumulator.update(
                original_values.astype(np.float64) / scale,
                reconstructed_values.astype(np.float64) / scale,
            )
        else:
            accumulator.update(original_values, reconstructed_values)

        field_metrics = accumulator.finalize()
        field_metrics["original_dtype"] = str(original_dataset.dtype)
        field_metrics["reconstructed_dtype"] = str(
            reconstructed_dataset.dtype
        )
        if logical == "id":
            field_metrics["exact_match"] = bool(
                np.array_equal(original_values, reconstructed_values)
            )
        if fixed_point_accumulator is not None:
            field_metrics["fixed_point_units"] = (
                fixed_point_accumulator.finalize()
            )
        results[logical] = field_metrics
    return results


def _evaluate_error_bounds(
    field_metrics: Mapping[str, Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    results = {}
    for logical in LOGICAL_ORDER:
        target = _error_bound_target(logical, manifest)
        observed = float(field_metrics[logical]["max_absolute_error"])
        effective_bound = float(target["effective_final_abs_bound"])
        value_range = target.get("range_for_relative")
        target["observed_relative_error"] = (
            observed / float(value_range)
            if value_range not in (None, 0.0)
            else (0.0 if observed == 0.0 else math.inf)
        )
        tolerance = 1e-12 + 1e-6 * max(1.0, effective_bound)
        target["observed_max_absolute_error"] = observed
        target["satisfied"] = bool(observed <= effective_bound + tolerance)
        results[logical] = target
    return results


def _error_bound_target(
    logical: str,
    manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    field_bound = manifest.get("field_error_bounds", {}).get(logical, {})
    if logical in POSITION_FIELDS:
        return _position_error_target(logical, field_bound, manifest)
    if logical in VELOCITY_FIELDS:
        return _velocity_error_target(logical, field_bound, manifest)
    id_bound = float(manifest["error_bounds"]["id_sz3_abs"])
    return {
        "mode": field_bound.get("mode", "lossless"),
        "relative_error_bound": field_bound.get("relative"),
        "range_for_relative": field_bound.get("range"),
        "range_units": field_bound.get("range_units", "source_units"),
        "requested_abs_bound": float(field_bound.get("abs", id_bound)),
        "compressor_abs_eb": id_bound,
        "effective_final_abs_bound": id_bound,
    }


def _position_error_target(
    logical: str,
    field_bound: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    fallback = float(manifest["error_bounds"]["positions_lcp_abs"])
    cast = float(
        manifest.get("preprocess", {})
        .get("positions", {})
        .get(logical, {})
        .get("preprocess_cast_max_abs_in_lcp_units", 0.0)
    )
    scale = float(manifest["position_scale"]["value"])
    dtype = np.dtype(manifest["fields"][logical]["dtype"])
    rounding = 0.5 / scale if np.issubdtype(dtype, np.integer) else 0.0
    requested = float(field_bound.get("abs", fallback))
    compressor_bound = float(field_bound.get("compressor_abs", fallback))
    effective = (
        requested
        if field_bound.get("mode") == "relative"
        else compressor_bound + cast + rounding
    )
    return {
        "mode": field_bound.get("mode", "absolute"),
        "relative_error_bound": field_bound.get("relative"),
        "range_for_relative": field_bound.get("range"),
        "range_units": field_bound.get("range_units", "lcp_units"),
        "requested_abs_bound": requested,
        "compressor_abs_eb": compressor_bound,
        "preprocess_cast_allowance": cast,
        "recombine_rounding_allowance": rounding,
        "effective_final_abs_bound": effective,
    }


def _velocity_error_target(
    logical: str,
    field_bound: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    fallback = float(
        manifest["error_bounds"].get(
            "velocities_lcp_abs",
            manifest["error_bounds"]["velocities_sz3_abs"],
        )
    )
    requested = float(field_bound.get("abs", fallback))
    compressor_bound = float(field_bound.get("compressor_abs", requested))
    is_lcp = (
        manifest.get("compressors", {}).get("velocities", "sz3") == "lcp"
    )
    cast = (
        float(
            manifest.get("preprocess", {})
            .get("velocities", {})
            .get(logical, {})
            .get("preprocess_cast_max_abs", 0.0)
        )
        if is_lcp
        else 0.0
    )
    effective = (
        compressor_bound + cast
        if is_lcp and field_bound.get("mode") != "relative"
        else requested
    )
    return {
        "mode": field_bound.get("mode", "absolute"),
        "relative_error_bound": field_bound.get("relative"),
        "range_for_relative": field_bound.get("range"),
        "range_units": field_bound.get("range_units", "source_units"),
        "requested_abs_bound": requested,
        "compressor_abs_eb": compressor_bound,
        "preprocess_cast_allowance": cast,
        "effective_final_abs_bound": effective,
    }


def _update_final_size_metrics(
    metrics: Dict[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    components = compressed_sizes(_work_dir_from_manifest(manifest))
    compressed_total = int(sum(components.values()))
    selected_payload = int(
        manifest["sizes"]["selected_original_payload_bytes"]
    )
    sizes = metrics["sizes"]
    sizes["compressed_components_bytes"] = components
    sizes["compressed_total_bytes"] = compressed_total
    sizes["selected_particle_count"] = int(manifest["count"])
    sizes["input_h5_file_bytes"] = int(
        manifest.get("input_h5_file_bytes", 0)
    )
    sizes["limit"] = manifest.get("limit")
    sizes["payload_compression_ratio"] = (
        selected_payload / compressed_total
        if compressed_total
        else math.inf
    )
    if "input_h5_file_bytes" in manifest:
        sizes["h5_file_to_compressed_ratio"] = (
            int(manifest["input_h5_file_bytes"]) / compressed_total
            if compressed_total
            else math.inf
        )


def _work_dir_from_manifest(manifest: Mapping[str, Any]) -> Path:
    compressed = manifest.get("artifacts", {}).get("compressed", {})
    artifact_path = compressed.get("positions")
    if artifact_path is None:
        artifact_path = next(iter(compressed.values()), None)
    if artifact_path is None:
        return Path(".")
    return Path(artifact_path).resolve().parents[1]


def _validate_comparison_order(order: np.ndarray, count: int) -> None:
    if count and (int(order.min()) < 0 or int(order.max()) >= count):
        raise RuntimeError(
            "Temporary LCP canonical order is outside the original row range."
        )
    if np.unique(order).size != count:
        raise RuntimeError(
            "Temporary LCP canonical order is not a permutation."
        )


def _align_rows_by_particle_id(
    original: h5py.File,
    reconstructed: h5py.File,
    manifest: Mapping[str, Any],
    count: int,
) -> Tuple[np.ndarray, str]:
    id_path = manifest["fields"]["id"]["h5_path"]
    original_ids = original[id_path][:count]
    reconstructed_ids = reconstructed[id_path][:count]
    original_sort = np.argsort(original_ids, kind="stable")
    reconstructed_sort = np.argsort(reconstructed_ids, kind="stable")
    if not np.array_equal(
        original_ids[original_sort],
        reconstructed_ids[reconstructed_sort],
    ):
        raise RuntimeError(
            "Reconstructed particle IDs do not match the original ID set."
        )
    if np.unique(original_ids).size != count:
        raise RuntimeError(
            "Cannot align reordered rows for metrics after temporary files "
            "were removed because particle IDs are not unique."
        )
    order = np.empty(count, dtype=np.intp)
    order[reconstructed_sort] = original_sort
    return order, "particle_id"


def _field_size_entry(
    report: Mapping[str, Any],
    components: Mapping[str, int],
    logical: str,
) -> Dict[str, Any]:
    return _size_entry(
        original_bytes_for_fields(report, (logical,)),
        compressed_bytes_with_prefixes(
            components,
            (f"compressed/{logical}.",),
        ),
    )


def _size_entry(
    original_bytes: int,
    compressed_bytes: int,
) -> Dict[str, Any]:
    return {
        "original_bytes": original_bytes,
        "compressed_bytes": compressed_bytes,
        "compression_ratio": compression_ratio(
            original_bytes,
            compressed_bytes,
        ),
    }
