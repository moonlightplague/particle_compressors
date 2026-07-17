"""Convert HDF5 particle datasets into compressor-ready raw streams."""

import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import h5py
import numpy as np

from src.constants import POSITION_FIELDS
from src.models import PositionScale
from src.runtime import read_raw, require_output_path


def update_numeric_stats(
    stats: Dict[str, Any],
    values: np.ndarray,
) -> None:
    if values.size == 0:
        raise RuntimeError("Not a valid input array.")
    values64 = values.astype(np.float64, copy=False)
    stats["float_min"] = min(stats["float_min"], float(values64.min()))
    stats["float_max"] = max(stats["float_max"], float(values64.max()))


def update_numeric_stats_int(
    stats: Dict[str, Any],
    values: np.ndarray,
) -> None:
    if values.size == 0:
        raise RuntimeError("Not a valid input array.")
    stats["int_min"] = min(stats["int_min"], int(values.min()))
    stats["int_max"] = max(stats["int_max"], int(values.max()))


def finalize_numeric_stats(stats: Mapping[str, float]) -> Dict[str, float]:
    float_minimum = float(stats["float_min"])
    float_maximum = float(stats["float_max"])
    if math.isinf(float_minimum) or math.isinf(float_maximum):
        raise RuntimeError(
            "Cannot compute relative error bound for an empty field."
        )
    result = {
        "float_min": float_minimum,
        "float_max": float_maximum,
        "float_range": float_maximum - float_minimum,
    }
    if "int_min" not in stats or "int_max" not in stats:
        return result

    integer_minimum = float(stats["int_min"])
    integer_maximum = float(stats["int_max"])
    if math.isinf(integer_minimum) or math.isinf(integer_maximum):
        raise RuntimeError(
            "Cannot compute relative error bound for an empty field."
        )
    result.update(
        {
            "int_min": integer_minimum,
            "int_max": integer_maximum,
            "int_range": integer_maximum - integer_minimum,
        }
    )
    return result


def resolve_position_scale(
    h5: h5py.File,
    mode: str,
    attr_name: str,
    explicit_value: Optional[float],
    position_dtype: np.dtype,
) -> PositionScale:
    if mode == "value":
        if explicit_value is None:
            raise RuntimeError(
                "--position-scale value requires --position-scale-value."
            )
        return PositionScale("value", _positive_scale(explicit_value))
    if mode == "attr":
        if attr_name not in h5.attrs:
            raise RuntimeError(
                "--position-scale attr requested, but root attr "
                f"{attr_name!r} is missing."
            )
        value = np.asarray(h5.attrs[attr_name]).item()
        return PositionScale(
            "attr",
            _positive_scale(value),
            attr=attr_name,
        )
    if mode == "raw":
        return PositionScale("raw", 1.0)
    if mode != "auto":
        raise RuntimeError(f"Unsupported position scale mode: {mode}")

    if np.issubdtype(position_dtype, np.integer) and attr_name in h5.attrs:
        value = float(np.asarray(h5.attrs[attr_name]).item())
        if value > 0:
            return PositionScale("auto_attr", value, attr=attr_name)
    return PositionScale("auto_raw", 1.0)


def get_selected_count(
    h5: h5py.File,
    fields: Mapping[str, str],
    limit: Optional[int],
) -> int:
    sizes = {logical: int(h5[path].shape[0]) for logical, path in fields.items()}
    unique_sizes = set(sizes.values())
    if len(unique_sizes) != 1:
        raise RuntimeError(
            f"Particle fields do not have the same length: {sizes}"
        )
    count = unique_sizes.pop()
    if limit is not None:
        if limit <= 0:
            raise RuntimeError("--limit must be positive.")
        count = min(count, limit)
    return count


def export_positions_for_lcp(
    h5: h5py.File,
    fields: Mapping[str, str],
    output_dir: Path,
    count: int,
    scale: PositionScale,
    force: bool,
) -> Tuple[Dict[str, str], Dict[str, Dict[str, float]]]:
    paths = {}
    statistics = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    for logical in POSITION_FIELDS:
        dataset = h5[fields[logical]]
        output = output_dir / f"{logical}.f32.raw"
        require_output_path(output, force)

        source = dataset[:count]
        scaled64 = source.astype(np.float64, copy=False) / scale.value
        scaled32 = scaled64.astype(np.float32)
        scaled32.tofile(output)
        cast_error = float(
            np.abs(scaled32.astype(np.float64) - scaled64).max(initial=0.0)
        )
        range_stats = {
            "float_min": math.inf,
            "float_max": -math.inf,
            "int_min": math.inf,
            "int_max": -math.inf,
        }
        update_numeric_stats(range_stats, scaled64)
        update_numeric_stats_int(range_stats, source)
        field_stats = finalize_numeric_stats(range_stats)
        paths[logical] = str(output)
        statistics[logical] = {
            "scale": scale.value,
            "preprocess_cast_max_abs_in_lcp_units": cast_error,
            "preprocess_cast_max_abs_in_original_fixed_point_units": (
                cast_error * scale.value
            ),
            "min_in_lcp_units": field_stats["float_min"],
            "max_in_lcp_units": field_stats["float_max"],
            "range_in_lcp_units": field_stats["float_range"],
            "min_in_int_units": field_stats["int_min"],
            "max_in_int_units": field_stats["int_max"],
            "range_in_int_units": field_stats["int_range"],
        }
    return paths, statistics


def export_positions_for_xnyzip(
    position_paths: Mapping[str, str],
    output: Path,
    count: int,
    force: bool,
) -> Tuple[str, Dict[str, Any]]:
    require_output_path(output, force)
    interleaved = np.memmap(
        output,
        dtype=np.float32,
        mode="w+",
        shape=(count, 3),
    )
    for axis, logical in enumerate(POSITION_FIELDS):
        values = np.memmap(
            position_paths[logical],
            dtype=np.float32,
            mode="r",
        )
        if values.size != count:
            raise RuntimeError(
                f"XnYZip position export expected {count} values in "
                f"{position_paths[logical]}, got {values.size}."
            )
        interleaved[:, axis] = values
    interleaved.flush()
    del interleaved
    return str(output), {
        "dtype": "float32",
        "shape": [count, 3],
        "layout": "xyz_interleaved",
        "file_bytes": count * 3 * np.dtype(np.float32).itemsize,
    }


def export_float_field(
    h5: h5py.File,
    dataset_path: str,
    output: Path,
    count: int,
    force: bool,
) -> Tuple[str, Dict[str, float]]:
    require_output_path(output, force)
    dtype = np.dtype(h5[dataset_path].dtype)
    if dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise RuntimeError(
            "Lossy velocity compression expected float32/float64, "
            f"got {dtype} for {dataset_path}."
        )
    source = h5[dataset_path][:count].astype(dtype, copy=False)
    source.tofile(output)
    range_stats = {"float_min": math.inf, "float_max": -math.inf}
    update_numeric_stats(range_stats, source)
    return str(output), finalize_numeric_stats(range_stats)


def export_float32_for_lcp(
    raw_path: str,
    source_dtype: np.dtype,
    output: Path,
    count: int,
    force: bool,
) -> Tuple[str, float]:
    source_dtype = np.dtype(source_dtype)
    if source_dtype == np.dtype("float32"):
        return raw_path, 0.0
    require_output_path(output, force)
    source = read_raw(raw_path, source_dtype, count)
    encoded = source.astype(np.float32)
    cast_error = np.abs(
        encoded.astype(np.float64) - source.astype(np.float64, copy=False)
    )
    encoded.tofile(output)
    return str(output), float(cast_error.max(initial=0.0))


def export_id_for_pcodec(
    h5: h5py.File,
    dataset_path: str,
    output: Path,
    count: int,
    force: bool,
) -> Tuple[str, Dict[str, Any]]:
    require_output_path(output, force)
    dataset = h5[dataset_path]
    source_dtype = np.dtype(dataset.dtype)
    if not np.issubdtype(source_dtype, np.integer):
        raise RuntimeError(
            f"pcodec ID export expected an integer dtype, got {source_dtype} "
            f"for {dataset_path}."
        )
    source = dataset[:count]
    source.astype(source_dtype, copy=False).tofile(output)
    return str(output), {
        "source_dtype": str(source_dtype),
        "min": int(source.min()) if source.size else None,
        "max": int(source.max()) if source.size else None,
        "pcodec_dtype": str(source_dtype),
    }


def _positive_scale(value: Any) -> float:
    scale = float(value)
    if scale <= 0:
        raise RuntimeError("Position scale must be positive.")
    return scale

