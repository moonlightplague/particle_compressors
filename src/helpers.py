"""Compatibility facade for the original helper API.

New code should import from the focused modules directly.  These re-exports
keep existing integrations working while responsibilities live in
``constants``, ``manifest``, ``metrics``, ``models``, and ``runtime``.
"""

import math
import re
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np

from src.constants import (
    FIELD_ALIASES,
    LCP_CHUNK_BATCH_VALUES,
    LCP_CHUNK_CONTAINER,
    LCP_CHUNK_ENTRY,
    LCP_CHUNK_HEADER,
    LCP_CHUNK_MAGIC,
    LOGICAL_ORDER,
    MIN_CODEC_VALUES,
    POSITION_FIELDS,
    VELOCITY_FIELDS,
)
from src.manifest import (
    compressed_sizes,
    order_dtype_from_manifest,
    update_compressed_size_metrics,
)
from src.metrics import (
    comparison_order_for_reconstructed_rows,
    compressed_bytes_with_prefixes,
    compression_ratio,
    component_compression_ratios,
    compute_metrics,
    original_bytes_for_fields,
    print_component_summary,
    print_summary,
    report_count,
    report_field_dtype,
)
from src.models import ToolPaths
from src.runtime import (
    json_size_bytes,
    load_pcodec,
    load_pysz,
    load_pyszo,
    read_json,
    read_raw,
    repo_root,
    require_output_path,
    resolve_lcp_chunk_workers,
    run_command,
    write_json,
)


# Deprecated aliases retained for callers written against the initial layout.
PYSZ_MIN_VALUES = MIN_CODEC_VALUES
SZO_MIN_VALUES = MIN_CODEC_VALUES


def parse_tool_stdout(stdout: str) -> Dict[str, Any]:
    patterns = {
        "reported_compression_ratio": (
            r"compression ratio\s*=\s*([0-9.eE+-]+)"
        ),
        "reported_compression_time_seconds": (
            r"compression time\s*=\s*([0-9.eE+-]+)"
        ),
        "reported_decompression_time_seconds": (
            r"decompression time\s*=\s*([0-9.eE+-]+)"
        ),
    }
    parsed = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, stdout)
        if match:
            parsed[key] = float(match.group(1))
    return parsed


def empty_metric_acc() -> Dict[str, Any]:
    return {
        "count": 0,
        "sum_squared_error": 0.0,
        "sum_absolute_error": 0.0,
        "max_absolute_error": 0.0,
        "min": math.inf,
        "max": -math.inf,
    }


def update_metric_acc(
    accumulator: Dict[str, Any],
    original: np.ndarray,
    reconstructed: np.ndarray,
) -> None:
    original64 = original.astype(np.float64, copy=False)
    reconstructed64 = reconstructed.astype(np.float64, copy=False)
    difference = reconstructed64 - original64
    absolute_difference = np.abs(difference)
    accumulator["count"] += int(original.size)
    accumulator["sum_squared_error"] += float(
        np.dot(difference, difference)
    )
    accumulator["sum_absolute_error"] += float(absolute_difference.sum())
    accumulator["max_absolute_error"] = max(
        accumulator["max_absolute_error"],
        float(absolute_difference.max(initial=0.0)),
    )
    if original.size:
        accumulator["min"] = min(
            accumulator["min"],
            float(original64.min()),
        )
        accumulator["max"] = max(
            accumulator["max"],
            float(original64.max()),
        )


def finalize_metric_acc(accumulator: Mapping[str, Any]) -> Dict[str, Any]:
    count = int(accumulator["count"])
    if count == 0:
        raise RuntimeError("Cannot finalize metrics for zero elements.")
    mse = float(accumulator["sum_squared_error"]) / count
    rmse = math.sqrt(mse)
    value_range = float(accumulator["max"] - accumulator["min"])
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
        "count": count,
        "min": float(accumulator["min"]),
        "max": float(accumulator["max"]),
        "range": value_range,
        "max_absolute_error": float(
            accumulator["max_absolute_error"]
        ),
        "mean_absolute_error": (
            float(accumulator["sum_absolute_error"]) / count
        ),
        "mse": mse,
        "mmse": mse,
        "rmse": rmse,
        "nrmse": nrmse,
        "psnr": psnr,
    }


def manifest_path_from_manifest(manifest: Mapping[str, Any]) -> str:
    compressed = manifest.get("artifacts", {}).get("compressed", {})
    artifact_path = compressed.get("positions")
    if artifact_path is None:
        artifact_path = next(iter(compressed.values()), None)
    if artifact_path is None:
        return "."
    return str(Path(artifact_path).resolve().parents[1])


__all__ = [
    "FIELD_ALIASES",
    "LCP_CHUNK_BATCH_VALUES",
    "LCP_CHUNK_CONTAINER",
    "LCP_CHUNK_ENTRY",
    "LCP_CHUNK_HEADER",
    "LCP_CHUNK_MAGIC",
    "LOGICAL_ORDER",
    "POSITION_FIELDS",
    "PYSZ_MIN_VALUES",
    "SZO_MIN_VALUES",
    "ToolPaths",
    "VELOCITY_FIELDS",
    "comparison_order_for_reconstructed_rows",
    "compressed_bytes_with_prefixes",
    "compression_ratio",
    "component_compression_ratios",
    "compressed_sizes",
    "compute_metrics",
    "empty_metric_acc",
    "finalize_metric_acc",
    "json_size_bytes",
    "load_pcodec",
    "load_pysz",
    "load_pyszo",
    "order_dtype_from_manifest",
    "original_bytes_for_fields",
    "parse_tool_stdout",
    "print_component_summary",
    "print_summary",
    "read_json",
    "read_raw",
    "repo_root",
    "require_output_path",
    "resolve_lcp_chunk_workers",
    "report_count",
    "report_field_dtype",
    "run_command",
    "update_compressed_size_metrics",
    "update_metric_acc",
    "write_json",
]
