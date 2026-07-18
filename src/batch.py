"""Directory-batch discovery, aggregation, and console reporting."""

import argparse
import math
import os
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from src.constants import POSITION_FIELDS


MAX_AUTOMATIC_FILE_WORKERS = 16


@dataclass(frozen=True)
class BatchFileResult:
    """Result returned by one independently executed file pipeline."""

    input_h5: str
    work_dir: str
    wall_seconds: float
    console_output: str = ""
    report_path: Optional[str] = None
    report: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    @property
    def succeeded(self) -> bool:
        return self.error is None and self.report is not None


def discover_h5_files(directory: Path) -> List[Path]:
    """Return direct ``.h5`` children in deterministic filename order."""

    return sorted(
        (path.resolve() for path in directory.glob("*.h5") if path.is_file()),
        key=lambda path: path.name,
    )


def resolve_file_workers(configured: int, file_count: int) -> int:
    """Resolve a bounded process count for a directory batch."""

    if configured < 0:
        raise RuntimeError("File workers must be non-negative.")
    if file_count < 1:
        raise RuntimeError("Cannot resolve workers for an empty file batch.")
    if configured:
        return min(configured, file_count)
    return min(
        file_count,
        MAX_AUTOMATIC_FILE_WORKERS,
        os.cpu_count() or 1,
    )


def args_for_batch_file(
    args: argparse.Namespace,
    input_h5: Path,
    batch_work_dir: Path,
) -> argparse.Namespace:
    """Copy CLI arguments and isolate one file's package directory."""

    file_args = argparse.Namespace(**vars(args))
    file_args.input_h5 = str(input_h5)
    file_args.work_dir = str(batch_work_dir / input_h5.name)
    return file_args


def build_batch_metrics(
    input_directory: Path,
    command: str,
    workers: int,
    results: Sequence[BatchFileResult],
    batch_wall_seconds: float,
) -> Dict[str, Any]:
    """Build byte-weighted compression and end-to-end timing totals."""

    files = [_file_metrics(result) for result in results]
    successful = [result for result in results if result.succeeded]
    failed = [result for result in results if not result.succeeded]

    original_bytes = sum(
        int(
            result.report.get("sizes", {}).get(
                "selected_original_payload_bytes",
                0,
            )
        )
        for result in successful
        if result.report is not None
    )
    compressed_sizes = [
        int(result.report["sizes"]["compressed_total_bytes"])
        for result in successful
        if result.report is not None
        and "compressed_total_bytes" in result.report.get("sizes", {})
    ]
    compressed_bytes = sum(compressed_sizes)
    complete_batch_compression = (
        bool(successful)
        and not failed
        and len(compressed_sizes) == len(successful)
    )
    total_ratio = (
        _compression_ratio(original_bytes, compressed_bytes)
        if complete_batch_compression
        else None
    )

    file_seconds = [result.wall_seconds for result in successful]
    ratios = [
        float(result.report["sizes"]["payload_compression_ratio"])
        for result in successful
        if result.report is not None
        and "payload_compression_ratio" in result.report.get("sizes", {})
    ]
    stage_totals = _sum_stage_timings(successful)
    total_file_seconds = sum(file_seconds)
    particle_count = sum(
        _report_particle_count(result.report)
        for result in successful
        if result.report is not None
    )

    return {
        "input_directory": str(input_directory),
        "command": command,
        "workers": workers,
        "summary": {
            "discovered_files": len(results),
            "successful_files": len(successful),
            "failed_files": len(failed),
            "total_particle_count": particle_count,
        },
        "sizes": {
            "selected_original_payload_bytes_total": original_bytes,
            "compressed_total_bytes": (
                compressed_bytes if complete_batch_compression else None
            ),
            "payload_compression_ratio": total_ratio,
        },
        "timing": {
            "batch_wall_seconds": batch_wall_seconds,
            "file_wall_seconds_total": total_file_seconds,
            "effective_parallel_speedup": (
                total_file_seconds / batch_wall_seconds
                if batch_wall_seconds
                else math.inf
            ),
            "payload_throughput_mib_per_second": (
                original_bytes / (1024 * 1024) / batch_wall_seconds
                if batch_wall_seconds
                else math.inf
            ),
            "stage_seconds_total": stage_totals,
        },
        "statistics": {
            "per_file_payload_compression_ratio": _summary_statistics(ratios),
            "per_file_wall_seconds": _summary_statistics(file_seconds),
        },
        "files": files,
    }


def print_batch_summary(
    metrics: Mapping[str, Any],
    metrics_path: Path,
) -> None:
    """Print the most useful aggregate compression and timing statistics."""

    summary = metrics["summary"]
    sizes = metrics["sizes"]
    timing = metrics["timing"]
    cr_stats = metrics["statistics"]["per_file_payload_compression_ratio"]
    time_stats = metrics["statistics"]["per_file_wall_seconds"]

    print("Batch Summary:")
    print(f"batch_metrics_json = {metrics_path}")
    print(
        "files = "
        f"{summary['successful_files']}/{summary['discovered_files']} successful"
    )
    print(f"workers = {metrics['workers']}")
    print(f"particles_total = {summary['total_particle_count']}")
    ratio = sizes["payload_compression_ratio"]
    print(
        "payload_CR_total = "
        f"{_format_number(ratio) if ratio is not None else 'n/a'}"
    )
    print(
        "original_payload_bytes_total = "
        f"{sizes['selected_original_payload_bytes_total']}"
    )
    compressed_bytes = sizes["compressed_total_bytes"]
    print(
        "compressed_total_bytes = "
        f"{compressed_bytes if compressed_bytes is not None else 'n/a'}"
    )
    print(
        "batch_wall_seconds = "
        f"{_format_number(timing['batch_wall_seconds'])}"
    )
    print(
        "file_wall_seconds_total = "
        f"{_format_number(timing['file_wall_seconds_total'])}"
    )
    print(
        "effective_parallel_speedup = "
        f"{_format_number(timing['effective_parallel_speedup'])}x"
    )
    print(
        "payload_throughput_mib_per_second = "
        f"{_format_number(timing['payload_throughput_mib_per_second'])}"
    )
    for name in (
        "preprocess_wall_seconds",
        "compress_wall_seconds",
        "decompress_and_recombine_wall_seconds",
        "metrics_wall_seconds",
    ):
        value = timing["stage_seconds_total"].get(name)
        if value is not None:
            print(f"{name}_total = {_format_number(value)}")
    if cr_stats:
        print(
            "per_file_CR = "
            f"min={_format_number(cr_stats['min'])}, "
            f"median={_format_number(cr_stats['median'])}, "
            f"max={_format_number(cr_stats['max'])}"
        )
    if time_stats:
        print(
            "per_file_wall_seconds = "
            f"min={_format_number(time_stats['min'])}, "
            f"mean={_format_number(time_stats['mean'])}, "
            f"max={_format_number(time_stats['max'])}"
        )


def _file_metrics(result: BatchFileResult) -> Dict[str, Any]:
    entry: Dict[str, Any] = {
        "input_h5": result.input_h5,
        "work_dir": result.work_dir,
        "status": "success" if result.succeeded else "failed",
        "wall_seconds": result.wall_seconds,
    }
    if result.report_path is not None:
        entry["report_path"] = result.report_path
    if result.error is not None:
        entry["error"] = result.error
    if result.report is None:
        return entry

    sizes = result.report.get("sizes", {})
    entry.update(
        {
            "particle_count": _report_particle_count(result.report),
            "selected_original_payload_bytes": int(
                sizes.get("selected_original_payload_bytes", 0)
            ),
            "compressed_total_bytes": sizes.get("compressed_total_bytes"),
            "payload_compression_ratio": sizes.get(
                "payload_compression_ratio"
            ),
            "timing": dict(result.report.get("timing", {})),
        }
    )
    quality_metrics = _file_quality_metrics(result.report)
    if quality_metrics:
        entry["quality_metrics"] = quality_metrics
    return entry


def _file_quality_metrics(
    report: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Extract the displayed reconstruction metrics for each report field."""

    quality_metrics: Dict[str, Dict[str, Any]] = {}
    fields = report.get("fields", {})
    if not isinstance(fields, Mapping):
        return quality_metrics

    for logical, field in fields.items():
        if not isinstance(field, Mapping):
            continue
        display_field = (
            field.get("fixed_point_units", field)
            if logical in POSITION_FIELDS
            else field
        )
        if not isinstance(display_field, Mapping) or not all(
            name in display_field
            for name in ("max_absolute_error", "mse", "psnr")
        ):
            continue
        quality_metrics[str(logical)] = {
            "max_abs": display_field["max_absolute_error"],
            "mse": display_field["mse"],
            "psnr": display_field["psnr"],
            "units": (
                "lcp_units"
                if logical in POSITION_FIELDS and display_field is field
                else "source_units"
            ),
        }
    return quality_metrics


def _sum_stage_timings(
    results: Sequence[BatchFileResult],
) -> Dict[str, float]:
    totals: Dict[str, float] = {}
    for result in results:
        if result.report is None:
            continue
        for name, value in result.report.get("timing", {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                totals[name] = totals.get(name, 0.0) + float(value)
    return dict(sorted(totals.items()))


def _report_particle_count(report: Mapping[str, Any]) -> int:
    if "count" in report:
        return int(report["count"])
    return int(report.get("sizes", {}).get("selected_particle_count", 0))


def _summary_statistics(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {}
    return {
        "min": min(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def _compression_ratio(
    original_bytes: int,
    compressed_bytes: int,
) -> float:
    return (
        original_bytes / compressed_bytes
        if compressed_bytes
        else math.inf
    )


def _format_number(value: float) -> str:
    if math.isinf(value):
        return "inf"
    return f"{value:.6g}"
