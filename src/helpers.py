import json
import time
import subprocess
import re
import sys
import math
import ctypes
import importlib.util
import hashlib
import struct
import h5py
import numpy as np

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


FIELD_ALIASES = {
    "id": ("id", "particle_id", "pid"),
    "x": ("x", "posx", "position_x"),
    "y": ("y", "posy", "position_y"),
    "z": ("z", "posz", "position_z"),
    "vx": ("vx", "velx", "velocity_x"),
    "vy": ("vy", "vely", "velocity_y"),
    "vz": ("vz", "velz", "velocity_z"),
}
LOGICAL_ORDER = ("id", "x", "y", "z", "vx", "vy", "vz")
POSITION_FIELDS = ("x", "y", "z")
VELOCITY_FIELDS = ("vx", "vy", "vz")

PYSZ_MIN_VALUES = 10_000
SZO_MIN_VALUES = 10_000

LCP_CHUNK_MAGIC = b"LCPCHK2\0"
LCP_CHUNK_CONTAINER = "chunked_lcp_v2"
LCP_CHUNK_HEADER = struct.Struct("<8sQQQ")
LCP_CHUNK_ENTRY = struct.Struct("<QQ")
LCP_CHUNK_BATCH_VALUES = 1 << 20


@dataclass(frozen=True)
class ToolPaths:
    lcp: Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]

def require_output_path(path: Path, force: bool) -> None:
    if path.exists() and not force:
        raise RuntimeError(f"{path} already exists. Use --force to overwrite pipeline outputs.")
    path.parent.mkdir(parents=True, exist_ok=True)

def write_json(path: Path, payload: Mapping[str, Any], force: bool = True) -> None:
    require_output_path(path, force)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

def json_size_bytes(payload: Mapping[str, Any]) -> int:
    return len((json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"))

def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_tool_stdout(stdout: str) -> Dict[str, Any]:
    parsed: Dict[str, Any] = {}
    patterns = {
        "reported_compression_ratio": r"compression ratio\s*=\s*([0-9.eE+-]+)",
        "reported_compression_time_seconds": r"compression time\s*=\s*([0-9.eE+-]+)",
        "reported_decompression_time_seconds": r"decompression time\s*=\s*([0-9.eE+-]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, stdout)
        if match:
            parsed[key] = float(match.group(1))
    return parsed

def run_command(argv: List[str]):
    start = time.perf_counter()
    proc = subprocess.run(argv, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "Command failed with exit code "
            f"{proc.returncode}: {' '.join(argv)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )


def load_pcodec() -> Tuple[Any, Any]:
    try:
        from pcodec import ChunkConfig, standalone

        return standalone, ChunkConfig
    except ImportError as first_error:
        pco_python = repo_root() / "tools" / "pcodec" / "pco_python"
        if pco_python.is_dir() and str(pco_python) not in sys.path:
            sys.path.insert(0, str(pco_python))
        try:
            from pcodec import ChunkConfig, standalone

            return standalone, ChunkConfig
        except ImportError as second_error:
            raise RuntimeError(
                "Could not import pcodec. Build/install the Python extension from the submodule with "
                "`python -m pip install -e tools/pcodec/pco_python`."
            ) from second_error or first_error

def load_pysz() -> Tuple[Any, Any, Any]:
    try:
        from pysz import sz, szConfig, szErrorBoundMode

        return sz, szConfig, szErrorBoundMode
    except ImportError as exc:
        raise RuntimeError("Could not import pysz. Install it with `python -m pip install pysz`.") from exc


def load_pyszo() -> Tuple[Any, Any, Any, Any]:
    try:
        spec = importlib.util.find_spec("pyszo")
        if spec is not None and spec.submodule_search_locations:
            package_dir = Path(next(iter(spec.submodule_search_locations)))
            for library_name in ("libzstd.so", "libzstd.dylib"):
                bundled_zstd = package_dir / library_name
                if bundled_zstd.is_file():
                    ctypes.CDLL(str(bundled_zstd), mode=ctypes.RTLD_GLOBAL)
                    break
        from pyszo import szo, szoAlgorithm, szoConfig, szoErrorBoundMode

        return szo, szoConfig, szoErrorBoundMode, szoAlgorithm
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "Could not import pyszo. Build/install the local binding with "
            "`python -m pip install -e tools/SZo/tools/pyszo`."
        ) from exc


def read_raw(path: str, dtype: np.dtype, count: int) -> np.ndarray:
    data = np.fromfile(path, dtype=dtype, count=count)
    if data.size != count:
        raise RuntimeError(f"Unexpected EOF reading {path}; expected {count}, got {data.size}.")
    return data


def encode_integers_for_szo(values: np.ndarray) -> Tuple[np.ndarray, str]:
    source_dtype = np.dtype(values.dtype)
    if source_dtype.kind not in ("i", "u"):
        raise RuntimeError(f"SZO lossless encoding requires an integer dtype, got {source_dtype}.")

    if source_dtype.kind == "u" and source_dtype.itemsize in (4, 8):
        unsigned_dtype = np.dtype(f"uint{source_dtype.itemsize * 8}")
        signed_dtype = np.dtype(f"int{source_dtype.itemsize * 8}")
        native = np.ascontiguousarray(values.astype(unsigned_dtype, copy=False))
        return native.view(signed_dtype), "reinterpret_unsigned"

    encoded_dtype = np.dtype("int32") if source_dtype.itemsize <= 4 else np.dtype("int64")
    return np.ascontiguousarray(values, dtype=encoded_dtype), "cast"


def decode_integers_from_szo(
    values: np.ndarray,
    source_dtype: np.dtype,
    transform: str,
) -> np.ndarray:
    source_dtype = np.dtype(source_dtype)
    values = np.ascontiguousarray(values)
    if transform == "reinterpret_unsigned":
        unsigned_dtype = np.dtype(f"uint{source_dtype.itemsize * 8}")
        decoded = values.view(unsigned_dtype)
    elif transform == "cast":
        decoded = values
    else:
        raise RuntimeError(f"Unsupported SZO integer transform: {transform!r}.")
    return np.ascontiguousarray(decoded, dtype=source_dtype)


def integer_stream_sha256(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def compressed_sizes(work_dir: Path) -> Dict[str, int]:
    manifest = work_dir / "manifest.json"
    compressed = work_dir / "compressed"
    sizes: Dict[str, int] = {}
    if manifest.exists():
        sizes["manifest.json"] = manifest.stat().st_size
    if compressed.exists():
        for path in sorted(compressed.rglob("*")):
            if path.is_file():
                sizes[str(path.relative_to(work_dir))] = path.stat().st_size
    return sizes

def update_compressed_size_metrics(manifest: Dict[str, Any], work_dir: Path) -> None:
    components = compressed_sizes(work_dir)
    for _ in range(10):
        sizes = manifest.setdefault("sizes", {})
        sizes["compressed_components_bytes"] = dict(components)
        sizes["compressed_total_bytes"] = int(sum(components.values()))
        selected = int(sizes["selected_original_payload_bytes"])
        compressed = int(sizes["compressed_total_bytes"])
        sizes["payload_compression_ratio"] = selected / compressed if compressed else math.inf
        if "input_h5_file_bytes" in manifest:
            sizes["h5_file_to_compressed_ratio"] = (
                int(manifest["input_h5_file_bytes"]) / compressed if compressed else math.inf
            )

        manifest_bytes = json_size_bytes(manifest)
        if components.get("manifest.json") == manifest_bytes:
            break
        components["manifest.json"] = manifest_bytes

def order_dtype_from_manifest(manifest: Mapping[str, Any]) -> np.dtype:
    field = manifest.get("compressed_fields", {}).get("order", {})
    dtype = field.get("dtype", manifest.get("order_dtype", "int64"))
    dt = np.dtype(dtype)
    if dt not in (np.dtype("int32"), np.dtype("int64")):
        raise RuntimeError(f"Unsupported LCP order dtype in manifest: {dt}.")
    return dt





#printer
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

def original_bytes_for_fields(report: Mapping[str, Any], fields: Iterable[str]) -> int:
    count = report_count(report)
    return int(sum(report_field_dtype(report, logical).itemsize * count for logical in fields))

def compressed_bytes_with_prefixes(components: Mapping[str, int], prefixes: Iterable[str]) -> int:
    total = 0
    prefix_tuple = tuple(prefixes)
    for path, size in components.items():
        if path.startswith(prefix_tuple):
            total += int(size)
    return total

def compression_ratio(original_bytes: int, compressed_bytes: int) -> float:
    return original_bytes / compressed_bytes if compressed_bytes else math.inf

def component_compression_ratios(report: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    components = report.get("sizes", {}).get("compressed_components_bytes", {})
    xyz_compressed = (
        int(components.get("compressed/positions.lcp", 0))
        + compressed_bytes_with_prefixes(components, ("compressed/order.",))
        + compressed_bytes_with_prefixes(
            components,
            ("compressed/x.", "compressed/y.", "compressed/z."),
        )
    )
    velocity_lcp_compressed = int(components.get("compressed/velocities.lcp", 0))
    velocity_order_compressed = compressed_bytes_with_prefixes(
        components, ("compressed/velocity_order.",)
    )
    entries = {
        "xyz": {
            "original_bytes": original_bytes_for_fields(report, POSITION_FIELDS),
            "compressed_bytes": xyz_compressed,
        },
        "order": {
            "original_bytes": int(order_dtype_from_manifest(report).itemsize * report_count(report)),
            "compressed_bytes": compressed_bytes_with_prefixes(components, ("compressed/order.",)),
        },
        "velocity_order": {
            "original_bytes": int(order_dtype_from_manifest(report).itemsize * report_count(report)),
            "compressed_bytes": velocity_order_compressed,
        },
        "id": {
            "original_bytes": original_bytes_for_fields(report, ("id",)),
            "compressed_bytes": compressed_bytes_with_prefixes(components, ("compressed/id.",)),
        },
        "vx": {
            "original_bytes": original_bytes_for_fields(report, ("vx",)),
            "compressed_bytes": compressed_bytes_with_prefixes(components, ("compressed/vx.",)),
        },
        "vy": {
            "original_bytes": original_bytes_for_fields(report, ("vy",)),
            "compressed_bytes": compressed_bytes_with_prefixes(components, ("compressed/vy.",)),
        },
        "vz": {
            "original_bytes": original_bytes_for_fields(report, ("vz",)),
            "compressed_bytes": compressed_bytes_with_prefixes(components, ("compressed/vz.",)),
        },
        "vxyz": {
            "original_bytes": original_bytes_for_fields(report, VELOCITY_FIELDS),
            "compressed_bytes": velocity_lcp_compressed + velocity_order_compressed,
        },
    }
    for entry in entries.values():
        entry["compression_ratio"] = compression_ratio(entry["original_bytes"], entry["compressed_bytes"])
    return entries

def print_component_summary(report: Mapping[str, Any]) -> None:
    ratios = component_compression_ratios(report)
    print("component_CR:")
    velocity_lcp = ratios["vxyz"]["compressed_bytes"] > 0
    names = ["xyz"]
    if ratios["order"]["compressed_bytes"] > 0:
        names.append("order")
    names.append("id")
    if velocity_lcp:
        names.append("vxyz")
        if ratios["velocity_order"]["compressed_bytes"] > 0:
            names.append("velocity_order")
    else:
        names.extend(("vx", "vy", "vz"))
    for name in names:
        entry = ratios[name]
        ratio = entry["compression_ratio"]
        ratio_s = "inf" if math.isinf(ratio) else f"{ratio:.6g}"
        includes_order = (
            name == "xyz" and ratios["order"]["compressed_bytes"] > 0
        ) or (
            name == "vxyz" and ratios["velocity_order"]["compressed_bytes"] > 0
        )
        note = " includes order sidecar" if includes_order else ""
        print(
            f"  {name}: CR={ratio_s}, "
            f"original_bytes={entry['original_bytes']}, "
            f"compressed_bytes={entry['compressed_bytes']}{note}"
        )


def print_summary(metrics: Mapping[str, Any], metrics_path: Path) -> None:
    sizes = metrics["sizes"]
    print(f"metrics_json = {metrics_path}")
    print(f"payload_CR = {sizes.get('payload_compression_ratio', math.nan):.6g}")
    print(f"compressed_total_bytes = {sizes.get('compressed_total_bytes', 0)}")
    print_component_summary(metrics)
    for logical in LOGICAL_ORDER:
        field = metrics["fields"][logical]
        eb = metrics["error_bound_consistency"][logical]
        display_field = field.get("fixed_point_units", field) if logical in POSITION_FIELDS else field
        if logical in POSITION_FIELDS and display_field is field:
            units = "lcp_units"
        else:
            units = "source_units"
        psnr = display_field["psnr"]
        psnr_s = "inf" if math.isinf(psnr) and psnr > 0 else f"{psnr:.6g}"
        print(
            f"{logical}: max_abs={display_field['max_absolute_error']:.6g}, "
            f"mse={display_field['mse']:.6g}, psnr={psnr_s}, units={units}, "
            f"bound_ok={eb['satisfied']}"
        )



#compute metrics
def empty_metric_acc() -> Dict[str, Any]:
    return {
        "count": 0,
        "sum_squared_error": 0.0,
        "sum_absolute_error": 0.0,
        "max_absolute_error": 0.0,
        "min": math.inf,
        "max": -math.inf,
    }

def update_metric_acc(acc: Dict[str, Any], orig: np.ndarray, recon: np.ndarray) -> None:
    orig64 = orig.astype(np.float64, copy=False)
    recon64 = recon.astype(np.float64, copy=False)
    diff = recon64 - orig64
    abs_diff = np.abs(diff)
    acc["count"] += int(orig.size)
    acc["sum_squared_error"] += float(np.dot(diff, diff))
    acc["sum_absolute_error"] += float(abs_diff.sum())
    acc["max_absolute_error"] = max(acc["max_absolute_error"], float(abs_diff.max(initial=0.0)))
    if orig.size:
        acc["min"] = min(acc["min"], float(orig64.min(initial=acc["min"])))
        acc["max"] = max(acc["max"], float(orig64.max(initial=acc["max"])))

def manifest_path_from_manifest(manifest: Mapping[str, Any]) -> str:
    compressed = manifest.get("artifacts", {}).get("compressed", {})
    artifact_path = compressed.get("positions")
    if not artifact_path:
        artifact_path = next(iter(compressed.values()), None)
    if not artifact_path:
        return "."
    return str(Path(artifact_path).resolve().parents[1])

def finalize_metric_acc(acc: Mapping[str, Any]) -> Dict[str, Any]:
    count = int(acc["count"])
    if count == 0:
        raise RuntimeError("Cannot finalize metrics for zero elements.")
    mse = float(acc["sum_squared_error"] / count)
    rmse = math.sqrt(mse)
    value_range = float(acc["max"] - acc["min"])
    if mse == 0:
        psnr = math.inf
    elif value_range == 0:
        psnr = -math.inf
    else:
        psnr = 20.0 * math.log10(value_range) - 10.0 * math.log10(mse)
    nrmse = 0.0 if value_range == 0 and rmse == 0 else (rmse / value_range if value_range else math.inf)
    return {
        "count": count,
        "min": float(acc["min"]),
        "max": float(acc["max"]),
        "range": value_range,
        "max_absolute_error": float(acc["max_absolute_error"]),
        "mean_absolute_error": float(acc["sum_absolute_error"] / count),
        "mse": mse,
        "mmse": mse,
        "rmse": rmse,
        "nrmse": nrmse,
        "psnr": psnr,
    }


def comparison_order_for_reconstructed_rows(
    original: h5py.File,
    recon: h5py.File,
    manifest: Mapping[str, Any],
    count: int,
) -> Tuple[Optional[np.ndarray], str]:
    row_order = manifest.get("ordering", {}).get("reconstructed_rows", {})
    if row_order.get("original_row_order_restored", True):
        return None, "original_row"

    order_artifact = row_order.get("temporary_permutation_artifact") or "position_order"
    raw_order_path = (
        manifest.get("artifacts", {})
        .get("preprocessed", {})
        .get(order_artifact)
    )
    if raw_order_path and Path(raw_order_path).is_file():
        order = read_raw(raw_order_path, np.dtype("int32"), count).astype(np.intp, copy=False)
        if count and (int(order.min()) < 0 or int(order.max()) >= count):
            raise RuntimeError("Temporary LCP canonical order is outside the original row range.")
        if np.unique(order).size != count:
            raise RuntimeError("Temporary LCP canonical order is not a permutation.")
        return order, f"temporary_{order_artifact}"

    id_path = manifest["fields"]["id"]["h5_path"]
    original_ids = original[id_path][:count]
    reconstructed_ids = recon[id_path][:count]
    original_sort = np.argsort(original_ids, kind="stable")
    reconstructed_sort = np.argsort(reconstructed_ids, kind="stable")
    if not np.array_equal(original_ids[original_sort], reconstructed_ids[reconstructed_sort]):
        raise RuntimeError("Reconstructed particle IDs do not match the original ID set.")
    if np.unique(original_ids).size != count:
        raise RuntimeError(
            "Cannot align reordered rows for metrics after temporary files were removed because particle IDs are not unique."
        )
    order = np.empty(count, dtype=np.intp)
    order[reconstructed_sort] = original_sort
    return order, "particle_id"

def compute_metrics(
    original_h5: Path,
    reconstructed_h5: Path,
    manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    count = int(manifest["count"])
    scale = float(manifest["position_scale"]["value"])
    metrics: Dict[str, Any] = {
        "fields": {},
        "error_bound_consistency": {},
        "sizes": dict(manifest.get("sizes", {})),
        "timing": dict(manifest.get("timing", {})),
        "order_dtype": str(order_dtype_from_manifest(manifest)),
    }
    metrics_start = time.perf_counter()
    with h5py.File(original_h5, "r") as original, h5py.File(reconstructed_h5, "r") as recon:
        comparison_order, comparison_source = comparison_order_for_reconstructed_rows(
            original, recon, manifest, count
        )
        metrics["row_comparison"] = {
            "reconstructed_order": manifest.get("ordering", {})
            .get("reconstructed_rows", {})
            .get("mapping", "original_row"),
            "alignment_source": comparison_source,
        }
        for logical in LOGICAL_ORDER:
            field = manifest["fields"][logical]
            original_dset = original[field["h5_path"]]
            recon_dset = recon[field["h5_path"]]
            acc = empty_metric_acc()
            fixed_acc = empty_metric_acc() if logical in POSITION_FIELDS and np.issubdtype(original_dset.dtype, np.integer) else None
            exact = True

            orig = original_dset[:count]
            if comparison_order is not None:
                orig = orig[comparison_order]
            dec = recon_dset[:count]
            if logical == "id":
                exact = bool(np.array_equal(orig, dec))
                update_metric_acc(acc, orig, dec)
            elif logical in POSITION_FIELDS:
                if fixed_acc is not None:
                    update_metric_acc(fixed_acc, orig, dec)
                update_metric_acc(acc, orig.astype(np.float64) / scale, dec.astype(np.float64) / scale)
            else:
                update_metric_acc(acc, orig, dec)

            field_metrics = finalize_metric_acc(acc)
            field_metrics["original_dtype"] = str(original_dset.dtype)
            field_metrics["reconstructed_dtype"] = str(recon_dset.dtype)
            if logical == "id":
                field_metrics["exact_match"] = exact
            if fixed_acc is not None:
                field_metrics["fixed_point_units"] = finalize_metric_acc(fixed_acc)
            metrics["fields"][logical] = field_metrics

    pos_eb = float(manifest["error_bounds"]["positions_lcp_abs"])
    vel_eb = float(
        manifest["error_bounds"].get(
            "velocities_lcp_abs",
            manifest["error_bounds"]["velocities_sz3_abs"],
        )
    )
    velocity_compressor = manifest.get("compressors", {}).get("velocities", "sz3")
    id_eb = float(manifest["error_bounds"]["id_sz3_abs"])
    field_error_bounds = manifest.get("field_error_bounds", {})
    preprocess = manifest.get("preprocess", {}).get("positions", {})
    for logical in LOGICAL_ORDER:
        field_bound = field_error_bounds.get(logical, {})
        if logical in POSITION_FIELDS:
            cast = float(preprocess.get(logical, {}).get("preprocess_cast_max_abs_in_lcp_units", 0.0))
            rounding = 0.0
            dtype = np.dtype(manifest["fields"][logical]["dtype"])
            if np.issubdtype(dtype, np.integer):
                rounding = 0.5 / scale
            requested_abs = float(field_bound.get("abs", pos_eb))
            compressor_abs = float(field_bound.get("compressor_abs", pos_eb))
            if field_bound.get("mode") == "relative":
                effective = requested_abs
            else:
                effective = compressor_abs + cast + rounding
            target = {
                "mode": field_bound.get("mode", "absolute"),
                "relative_error_bound": field_bound.get("relative"),
                "range_for_relative": field_bound.get("range"),
                "range_units": field_bound.get("range_units", "lcp_units"),
                "requested_abs_bound": requested_abs,
                "compressor_abs_eb": compressor_abs,
                "preprocess_cast_allowance": cast,
                "recombine_rounding_allowance": rounding,
                "effective_final_abs_bound": effective,
            }
        elif logical in VELOCITY_FIELDS:
            requested_abs = float(field_bound.get("abs", vel_eb))
            compressor_abs = float(field_bound.get("compressor_abs", requested_abs))
            cast = 0.0
            if velocity_compressor == "lcp":
                cast = float(
                    manifest.get("preprocess", {})
                    .get("velocities", {})
                    .get(logical, {})
                    .get("preprocess_cast_max_abs", 0.0)
                )
            effective = requested_abs
            if velocity_compressor == "lcp" and field_bound.get("mode") != "relative":
                effective = compressor_abs + cast
            target = {
                "mode": field_bound.get("mode", "absolute"),
                "relative_error_bound": field_bound.get("relative"),
                "range_for_relative": field_bound.get("range"),
                "range_units": field_bound.get("range_units", "source_units"),
                "requested_abs_bound": requested_abs,
                "compressor_abs_eb": compressor_abs,
                "preprocess_cast_allowance": cast,
                "effective_final_abs_bound": effective,
            }
        else:
            target = {
                "mode": field_bound.get("mode", "lossless"),
                "relative_error_bound": field_bound.get("relative"),
                "range_for_relative": field_bound.get("range"),
                "range_units": field_bound.get("range_units", "source_units"),
                "requested_abs_bound": float(field_bound.get("abs", id_eb)),
                "compressor_abs_eb": id_eb,
                "effective_final_abs_bound": id_eb,
            }
        max_abs = float(metrics["fields"][logical]["max_absolute_error"])
        effective_bound = float(target["effective_final_abs_bound"])
        value_range = target.get("range_for_relative")
        target["observed_relative_error"] = (
            max_abs / float(value_range) if value_range not in (None, 0.0) else (0.0 if max_abs == 0.0 else math.inf)
        )
        tolerance = 1e-12 + 1e-6 * max(1.0, effective_bound)
        target["observed_max_absolute_error"] = max_abs
        target["satisfied"] = bool(max_abs <= effective_bound + tolerance)
        metrics["error_bound_consistency"][logical] = target

    final_components = compressed_sizes(Path(manifest_path_from_manifest(manifest)))
    compressed_total = int(sum(final_components.values()))
    selected_payload = int(manifest["sizes"]["selected_original_payload_bytes"])
    metrics["sizes"]["compressed_components_bytes"] = final_components
    metrics["sizes"]["compressed_total_bytes"] = compressed_total
    metrics["sizes"]["selected_particle_count"] = count
    metrics["sizes"]["input_h5_file_bytes"] = int(manifest.get("input_h5_file_bytes", 0))
    metrics["sizes"]["limit"] = manifest.get("limit")
    metrics["sizes"]["payload_compression_ratio"] = selected_payload / compressed_total if compressed_total else math.inf
    if "input_h5_file_bytes" in manifest:
        metrics["sizes"]["h5_file_to_compressed_ratio"] = (
            int(manifest["input_h5_file_bytes"]) / compressed_total if compressed_total else math.inf
        )
    metrics["timing"]["metrics_wall_seconds"] = time.perf_counter() - metrics_start
    return metrics



def print_summary(metrics: Mapping[str, Any], metrics_path: Path) -> None:
    sizes = metrics["sizes"]
    print(f"metrics_json = {metrics_path}")
    print(f"payload_CR = {sizes.get('payload_compression_ratio', math.nan):.6g}")
    print(f"compressed_total_bytes = {sizes.get('compressed_total_bytes', 0)}")
    print_component_summary(metrics)
    for logical in LOGICAL_ORDER:
        field = metrics["fields"][logical]
        eb = metrics["error_bound_consistency"][logical]
        display_field = field.get("fixed_point_units", field) if logical in POSITION_FIELDS else field
        if logical in POSITION_FIELDS and display_field is field:
            units = "lcp_units"
        else:
            units = "source_units"
        psnr = display_field["psnr"]
        psnr_s = "inf" if math.isinf(psnr) and psnr > 0 else f"{psnr:.6g}"
        print(
            f"{logical}: max_abs={display_field['max_absolute_error']:.6g}, "
            f"mse={display_field['mse']:.6g}, psnr={psnr_s}, units={units}, "
            f"bound_ok={eb['satisfied']}"
        )
