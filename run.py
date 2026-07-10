#!/usr/bin/env python3
"""Round-trip HDF5 particle files through LCP, pcodec, and pysz/SZ3.

The input HDF5 file is expected to contain particle fields equivalent to:
id, x, y, z, vx, vy, vz.  This repository's LCP executable only accepts the
three coordinate arrays, so this pipeline routes x/y/z through LCP, routes
id and the LCP order sidecar through lossless pcodec, and routes vx/vy/vz
through the pysz Python API for SZ3.
"""

from __future__ import annotations

from tools.cli import build_parser
import argparse
import importlib.metadata
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import h5py
import numpy as np


LOGICAL_ORDER = ("id", "x", "y", "z", "vx", "vy", "vz")
POSITION_FIELDS = ("x", "y", "z")
VELOCITY_FIELDS = ("vx", "vy", "vz")
FIELD_ALIASES = {
    "id": ("id", "particle_id", "pid"),
    "x": ("x", "posx", "position_x"),
    "y": ("y", "posy", "position_y"),
    "z": ("z", "posz", "position_z"),
    "vx": ("vx", "velx", "velocity_x"),
    "vy": ("vy", "vely", "velocity_y"),
    "vz": ("vz", "velz", "velocity_z"),
}
PYSZ_MIN_VALUES = 10_000


@dataclass(frozen=True)
class ToolPaths:
    lcp: Path


@dataclass(frozen=True)
class PositionScale:
    mode: str
    value: float
    attr: Optional[str] = None


@dataclass(frozen=True)
class ErrorBoundSelection:
    mode: str
    abs_by_field: Dict[str, float]
    relative: Optional[float] = None
    compressor_abs: Optional[float] = None


class PipelineError(RuntimeError):
    pass


def package_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


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
            raise PipelineError(
                "Could not import pcodec. Build/install the Python extension from the submodule with "
                "`python -m pip install -e tools/pcodec/pco_python`."
            ) from second_error or first_error


def load_pysz() -> Tuple[Any, Any, Any]:
    try:
        from pysz import sz, szConfig, szErrorBoundMode

        return sz, szConfig, szErrorBoundMode
    except ImportError as exc:
        raise PipelineError("Could not import pysz. Install it with `python -m pip install pysz`.") from exc


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_work_dir(input_h5: Path, abs_eb: float, limit: Optional[int]) -> Path:
    suffix = f"eb{abs_eb:g}"
    if limit is not None:
        suffix += f"_n{limit}"
    return Path("particle_pipeline_runs") / f"{input_h5.name}.{suffix}"


def default_work_dir_for_args(args: argparse.Namespace) -> Path:
    if args.pos_rel_eb is not None or args.vel_rel_eb is not None:
        labels: List[str] = []
        if args.pos_rel_eb is not None:
            labels.append(f"posrel{args.pos_rel_eb:g}")
        if args.vel_rel_eb is not None:
            labels.append(f"velrel{args.vel_rel_eb:g}")
        suffix = "_".join(labels)
    elif args.rel_eb is not None:
        suffix = f"rel{args.rel_eb:g}"
    else:
        suffix = f"eb{args.abs_eb:g}"
    if args.limit is not None:
        suffix += f"_n{args.limit}"
    return Path("particle_pipeline_runs") / f"{Path(args.input_h5).name}.{suffix}"


def find_executable(name: str, explicit: Optional[str]) -> Path:
    candidates: List[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    root = repo_root()
    if name == "lcp":
        candidates.extend(
            [
                root / "build" / "tools" / "sz3" / "lcp",
                root / "attributes" / "bin" / "lcp",
            ]
        )
    elif name == "sz3":
        candidates.extend(
            [
                root / "build" / "tools" / "sz3" / "sz3",
                root / "attributes" / "bin" / "sz3",
            ]
        )
    found = shutil.which(name)
    if found:
        candidates.append(Path(found))

    for path in candidates:
        if path.is_file() and os.access(path, os.X_OK):
            return path.resolve()
    raise PipelineError(f"Could not find executable for {name!r}. Use --{name} PATH.")


def as_jsonable_attr(value: Any) -> Dict[str, Any]:
    arr = np.asarray(value)
    if arr.shape == ():
        payload: Any = arr.item()
    else:
        payload = arr.tolist()
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8", errors="surrogateescape")
    return {"dtype": str(arr.dtype), "shape": list(arr.shape), "value": payload}


def restore_attr(payload: Mapping[str, Any]) -> Any:
    dtype = np.dtype(payload["dtype"])
    value = payload["value"]
    shape = tuple(payload.get("shape", []))
    if dtype.kind == "S" and isinstance(value, str):
        value = value.encode("utf-8", errors="surrogateescape")
    arr = np.asarray(value, dtype=dtype)
    if shape:
        return arr.reshape(shape)
    return arr[()]


def collect_attrs(obj: Any) -> Dict[str, Dict[str, Any]]:
    return {name: as_jsonable_attr(obj.attrs[name]) for name in obj.attrs.keys()}


def apply_attrs(obj: Any, attrs: Mapping[str, Mapping[str, Any]]) -> None:
    for name, payload in attrs.items():
        obj.attrs[name] = restore_attr(payload)


def resolve_fields(h5: h5py.File) -> Dict[str, str]:
    available: Dict[str, str] = {}

    def visit(name: str, obj: Any) -> None:
        if isinstance(obj, h5py.Dataset):
            available[name.split("/")[-1].lower()] = name

    h5.visititems(visit)
    resolved: Dict[str, str] = {}
    for logical, aliases in FIELD_ALIASES.items():
        for alias in aliases:
            if alias.lower() in available:
                resolved[logical] = available[alias.lower()]
                break
        if logical not in resolved:
            raise PipelineError(
                f"Could not find dataset for logical field {logical!r}; "
                f"tried aliases {aliases}."
            )
    return resolved


def dataset_nbytes(dtype: np.dtype, count: int) -> int:
    return int(np.dtype(dtype).itemsize * count)


def update_numeric_stats(stats: Dict[str, float], values: np.ndarray) -> None:
    if values.size == 0:
        return
    values64 = values.astype(np.float64, copy=False)
    stats["min"] = min(stats["min"], float(values64.min(initial=stats["min"])))
    stats["max"] = max(stats["max"], float(values64.max(initial=stats["max"])))


def finalize_numeric_stats(stats: Dict[str, float]) -> Dict[str, float]:
    if math.isinf(stats["min"]) or math.isinf(stats["max"]):
        raise PipelineError("Cannot compute relative error bound for an empty field.")
    value_range = float(stats["max"] - stats["min"])
    return {"min": float(stats["min"]), "max": float(stats["max"]), "range": value_range}


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


def run_command(argv: List[str]) -> Dict[str, Any]:
    start = time.perf_counter()
    proc = subprocess.run(argv, text=True, capture_output=True)
    elapsed = time.perf_counter() - start
    record: Dict[str, Any] = {
        "argv": argv,
        "wall_seconds": elapsed,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }
    record.update(parse_tool_stdout(proc.stdout))
    if proc.returncode != 0:
        raise PipelineError(
            "Command failed with exit code "
            f"{proc.returncode}: {' '.join(argv)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    return record


def require_output_path(path: Path, force: bool) -> None:
    if path.exists() and not force:
        raise PipelineError(f"{path} already exists. Use --force to overwrite pipeline outputs.")
    path.parent.mkdir(parents=True, exist_ok=True)


def prepare_output_dir(path: Path, force: bool) -> None:
    if path.exists():
        if not force:
            raise PipelineError(f"{path} already exists. Use --force to overwrite pipeline outputs.")
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    path.mkdir(parents=True, exist_ok=True)


def chunk_slices(count: int, chunk_size: int) -> Iterable[slice]:
    for start in range(0, count, chunk_size):
        yield slice(start, min(start + chunk_size, count))


def get_selected_count(h5: h5py.File, fields: Mapping[str, str], limit: Optional[int]) -> int:
    sizes = {logical: int(h5[path].shape[0]) for logical, path in fields.items()}
    unique_sizes = set(sizes.values())
    if len(unique_sizes) != 1:
        raise PipelineError(f"Particle fields do not have the same length: {sizes}")
    count = unique_sizes.pop()
    if limit is not None:
        if limit <= 0:
            raise PipelineError("--limit must be positive.")
        count = min(count, limit)
    return count


def resolve_position_scale(
    h5: h5py.File,
    mode: str,
    attr_name: str,
    explicit_value: Optional[float],
    pos_dtype: np.dtype,
) -> PositionScale:
    if mode == "value":
        if explicit_value is None:
            raise PipelineError("--position-scale value requires --position-scale-value.")
        return PositionScale(mode="value", value=float(explicit_value))

    if mode == "attr":
        if attr_name not in h5.attrs:
            raise PipelineError(f"--position-scale attr requested, but root attr {attr_name!r} is missing.")
        return PositionScale(mode="attr", value=float(np.asarray(h5.attrs[attr_name]).item()), attr=attr_name)

    if mode == "raw":
        return PositionScale(mode="raw", value=1.0)

    if mode != "auto":
        raise PipelineError(f"Unsupported position scale mode: {mode}")

    if np.issubdtype(pos_dtype, np.integer) and attr_name in h5.attrs:
        value = float(np.asarray(h5.attrs[attr_name]).item())
        if value > 0:
            return PositionScale(mode="auto_attr", value=value, attr=attr_name)
    return PositionScale(mode="auto_raw", value=1.0)


def export_positions_for_lcp(
    h5: h5py.File,
    fields: Mapping[str, str],
    out_dir: Path,
    count: int,
    chunk_size: int,
    scale: PositionScale,
    force: bool,
) -> Tuple[Dict[str, str], Dict[str, Dict[str, float]]]:
    out_paths: Dict[str, str] = {}
    stats: Dict[str, Dict[str, float]] = {}
    out_dir.mkdir(parents=True, exist_ok=True)

    for logical in POSITION_FIELDS:
        dataset = h5[fields[logical]]
        out_path = out_dir / f"{logical}.f32.raw"
        require_output_path(out_path, force)
        max_cast_abs = 0.0
        max_cast_fixed = 0.0
        range_stats = {"min": math.inf, "max": -math.inf}
        with out_path.open("wb") as out:
            for slc in chunk_slices(count, chunk_size):
                source = dataset[slc]
                source64 = source.astype(np.float64, copy=False)
                scaled64 = source64 / scale.value
                scaled32 = scaled64.astype(np.float32)
                scaled32.tofile(out)
                update_numeric_stats(range_stats, scaled64)
                if source.size:
                    cast_abs = np.abs(scaled32.astype(np.float64) - scaled64)
                    local_max = float(cast_abs.max(initial=0.0))
                    max_cast_abs = max(max_cast_abs, local_max)
                    max_cast_fixed = max(max_cast_fixed, local_max * scale.value)
        out_paths[logical] = str(out_path)
        field_stats = finalize_numeric_stats(range_stats)
        stats[logical] = {
            "scale": scale.value,
            "preprocess_cast_max_abs_in_lcp_units": max_cast_abs,
            "preprocess_cast_max_abs_in_original_fixed_point_units": max_cast_fixed,
            "min_in_lcp_units": field_stats["min"],
            "max_in_lcp_units": field_stats["max"],
            "range_in_lcp_units": field_stats["range"],
        }
    return out_paths, stats


def export_float_for_pysz(
    h5: h5py.File,
    dataset_path: str,
    out_path: Path,
    count: int,
    chunk_size: int,
    force: bool,
) -> Tuple[str, Dict[str, float]]:
    require_output_path(out_path, force)
    dtype = np.dtype(h5[dataset_path].dtype)
    if dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise PipelineError(f"pysz velocity export expected float32/float64, got {dtype} for {dataset_path}.")
    range_stats = {"min": math.inf, "max": -math.inf}
    with out_path.open("wb") as out:
        for slc in chunk_slices(count, chunk_size):
            source = h5[dataset_path][slc].astype(dtype, copy=False)
            source.tofile(out)
            update_numeric_stats(range_stats, source)
    field_stats = finalize_numeric_stats(range_stats)
    return str(out_path), field_stats


def export_id_for_pcodec(
    h5: h5py.File,
    dataset_path: str,
    out_path: Path,
    count: int,
    chunk_size: int,
    force: bool,
) -> Tuple[str, Dict[str, Any]]:
    require_output_path(out_path, force)
    dataset = h5[dataset_path]
    source_dtype = np.dtype(dataset.dtype)
    if not np.issubdtype(source_dtype, np.integer):
        raise PipelineError(f"pcodec id export expected an integer dtype, got {source_dtype} for {dataset_path}.")
    min_id: Optional[int] = None
    max_id: Optional[int] = None
    with out_path.open("wb") as out:
        for slc in chunk_slices(count, chunk_size):
            source = dataset[slc]
            local_min = int(source.min()) if source.size else 0
            local_max = int(source.max()) if source.size else 0
            min_id = local_min if min_id is None else min(min_id, local_min)
            max_id = local_max if max_id is None else max(max_id, local_max)
            source.astype(source_dtype, copy=False).tofile(out)
    return str(out_path), {"source_dtype": str(source_dtype), "min": min_id, "max": max_id, "pcodec_dtype": str(source_dtype)}


def effective_part_size(count: int, requested: int) -> int:
    if requested < 0:
        raise PipelineError("Part size must be non-negative.")
    if requested == 0:
        return count
    return max(1, requested)


def validate_error_bound(value: float, label: str) -> float:
    value = float(value)
    if value < 0.0:
        raise PipelineError(f"{label} must be non-negative.")
    return value


def select_relative_or_absolute(
    args: argparse.Namespace,
    prefix: str,
    fields: Iterable[str],
    ranges: Mapping[str, float],
    default_abs: float,
    compressor_abs: Optional[float] = None,
) -> ErrorBoundSelection:
    specific_rel = getattr(args, f"{prefix}_rel_eb")
    specific_abs = getattr(args, f"{prefix}_abs_eb")
    if specific_rel is not None and specific_abs is not None:
        raise PipelineError(f"--{prefix.replace('_', '-')}-rel-eb and --{prefix.replace('_', '-')}-abs-eb cannot both be set.")

    if specific_rel is not None:
        rel = validate_error_bound(specific_rel, f"--{prefix.replace('_', '-')}-rel-eb")
        abs_by_field = {field: rel * float(ranges[field]) for field in fields}
        return ErrorBoundSelection("relative", abs_by_field, relative=rel, compressor_abs=compressor_abs)

    if specific_abs is not None:
        abs_eb = validate_error_bound(specific_abs, f"--{prefix.replace('_', '-')}-abs-eb")
        return ErrorBoundSelection("absolute", {field: abs_eb for field in fields}, compressor_abs=compressor_abs)

    if args.rel_eb is not None:
        rel = validate_error_bound(args.rel_eb, "--rel-eb")
        abs_by_field = {field: rel * float(ranges[field]) for field in fields}
        return ErrorBoundSelection("relative", abs_by_field, relative=rel, compressor_abs=compressor_abs)

    abs_eb = validate_error_bound(default_abs, "--abs-eb")
    return ErrorBoundSelection("absolute", {field: abs_eb for field in fields}, compressor_abs=compressor_abs)


def serialize_error_bound_selection(
    selection: ErrorBoundSelection,
    fields: Iterable[str],
    ranges: Mapping[str, float],
    range_units: str,
) -> Dict[str, Dict[str, Any]]:
    return {
        field: {
            "mode": selection.mode,
            "abs": float(selection.abs_by_field[field]),
            "relative": selection.relative,
            "range": float(ranges[field]),
            "range_units": range_units,
            "compressor_abs": float(
                selection.compressor_abs if selection.compressor_abs is not None else selection.abs_by_field[field]
            ),
        }
        for field in fields
    }


def compress_pcodec_raw_parts(
    raw_path: str,
    dtype: str,
    cmp_dir: Path,
    field_name: str,
    count: int,
    part_size: int,
    force: bool,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    standalone, ChunkConfig = load_pcodec()
    part_dir = cmp_dir / f"{field_name}.pcodecparts"
    prepare_output_dir(part_dir, force)
    dt = np.dtype(dtype)
    records: List[Dict[str, Any]] = []
    parts: List[Dict[str, Any]] = []
    start = 0
    for part_index, chunk in enumerate(raw_chunk_reader(raw_path, dt, count, part_size)):
        part_cmp = part_dir / f"part{part_index:06d}.pco"
        require_output_path(part_cmp, force)
        contiguous = np.ascontiguousarray(chunk)
        start_time = time.perf_counter()
        payload = standalone.simple_compress(contiguous, ChunkConfig())
        elapsed = time.perf_counter() - start_time
        part_cmp.write_bytes(payload)
        compressed_bytes = len(payload)
        parts.append(
            {
                "index": part_index,
                "start": start,
                "count": int(chunk.size),
                "path": str(part_cmp),
                "bytes": compressed_bytes,
            }
        )
        records.append(
            {
                "api": "pcodec.standalone.simple_compress",
                "field": field_name,
                "part": part_index,
                "count": int(chunk.size),
                "input_bytes": int(chunk.nbytes),
                "output_bytes": compressed_bytes,
                "reported_compression_ratio": chunk.nbytes / compressed_bytes if compressed_bytes else math.inf,
                "wall_seconds": elapsed,
            }
        )
        start += int(chunk.size)

    if start != count:
        raise PipelineError(f"pcodec part compression for {field_name} saw {start} values, expected {count}.")

    return (
        {
            "field": field_name,
            "codec": "pcodec",
            "dtype": str(dt),
            "part_size": part_size,
            "parts_dir": str(part_dir),
            "parts": parts,
        },
        records,
    )


def decompress_pcodec_raw_parts(
    segment: Mapping[str, Any],
    out_path: str,
    force: bool,
) -> List[Dict[str, Any]]:
    standalone, _ = load_pcodec()
    dt = np.dtype(segment["dtype"])
    records: List[Dict[str, Any]] = []
    out = Path(out_path)
    require_output_path(out, force)
    with out.open("wb") as out_file:
        for part in segment["parts"]:
            part_index = int(part["index"])
            part_count = int(part["count"])
            payload = Path(part["path"]).read_bytes()
            start_time = time.perf_counter()
            data = standalone.simple_decompress(payload)
            elapsed = time.perf_counter() - start_time
            if data is None:
                raise PipelineError(f"pcodec part decompression for {segment['field']} part {part_index} returned no data.")
            data = np.asarray(data)
            if data.size != part_count:
                raise PipelineError(
                    f"pcodec part decompression for {segment['field']} part {part_index} "
                    f"returned {data.size} values, expected {part_count}."
                )
            if np.dtype(data.dtype) != dt:
                raise PipelineError(
                    f"pcodec part decompression for {segment['field']} part {part_index} "
                    f"returned dtype {data.dtype}, expected {dt}."
                )
            data.tofile(out_file)
            records.append(
                {
                    "api": "pcodec.standalone.simple_decompress",
                    "field": segment["field"],
                    "part": part_index,
                    "count": int(data.size),
                    "input_bytes": len(payload),
                    "output_bytes": int(data.nbytes),
                    "wall_seconds": elapsed,
                }
            )
    return records


def pysz_encoded_chunk(chunk: np.ndarray) -> Tuple[np.ndarray, int]:
    if chunk.size >= PYSZ_MIN_VALUES:
        return np.ascontiguousarray(chunk), int(chunk.size)
    encoded_count = PYSZ_MIN_VALUES
    padded = np.empty(encoded_count, dtype=chunk.dtype)
    padded[: chunk.size] = chunk
    fill_value = chunk[-1] if chunk.size else np.asarray(0, dtype=chunk.dtype)
    padded[chunk.size :] = fill_value
    return padded, encoded_count


def compress_pysz_raw_parts(
    raw_path: str,
    dtype: str,
    cmp_dir: Path,
    field_name: str,
    count: int,
    abs_eb: float,
    part_size: int,
    force: bool,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    PyszSZ, PyszConfig, PyszErrorBoundMode = load_pysz()
    part_dir = cmp_dir / f"{field_name}.pyszparts"
    prepare_output_dir(part_dir, force)
    dt = np.dtype(dtype)
    if dt not in (np.dtype("float32"), np.dtype("float64")):
        raise PipelineError(f"pysz velocity compression expected float32/float64, got {dt} for {field_name}.")
    records: List[Dict[str, Any]] = []
    parts: List[Dict[str, Any]] = []
    start = 0
    for part_index, chunk in enumerate(raw_chunk_reader(raw_path, dt, count, part_size)):
        part_cmp = part_dir / f"part{part_index:06d}.psz"
        require_output_path(part_cmp, force)
        encoded, encoded_count = pysz_encoded_chunk(chunk)
        config = PyszConfig(encoded.shape)
        config.errorBoundMode = PyszErrorBoundMode.ABS
        config.absErrorBound = float(abs_eb)
        start_time = time.perf_counter()
        try:
            compressed, reported_ratio = PyszSZ.compress(encoded, config)
        except Exception as exc:
            raise PipelineError(
                f"pysz compression failed for {field_name} part {part_index} "
                f"with {chunk.size} values encoded as {encoded_count} values."
            ) from exc
        elapsed = time.perf_counter() - start_time
        compressed = np.ascontiguousarray(compressed, dtype=np.uint8)
        compressed.tofile(part_cmp)
        compressed_bytes = int(compressed.size)
        parts.append(
            {
                "index": part_index,
                "start": start,
                "count": int(chunk.size),
                "encoded_count": encoded_count,
                "path": str(part_cmp),
                "bytes": compressed_bytes,
            }
        )
        records.append(
            {
                "api": "pysz.sz.compress",
                "field": field_name,
                "part": part_index,
                "count": int(chunk.size),
                "encoded_count": encoded_count,
                "input_bytes": int(chunk.nbytes),
                "encoded_input_bytes": int(encoded.nbytes),
                "output_bytes": compressed_bytes,
                "reported_compression_ratio": float(reported_ratio),
                "effective_compression_ratio": chunk.nbytes / compressed_bytes if compressed_bytes else math.inf,
                "wall_seconds": elapsed,
            }
        )
        start += int(chunk.size)

    if start != count:
        raise PipelineError(f"pysz part compression for {field_name} saw {start} values, expected {count}.")

    return (
        {
            "field": field_name,
            "codec": "pysz",
            "dtype": str(dt),
            "abs_error_bound": abs_eb,
            "part_size": part_size,
            "parts_dir": str(part_dir),
            "parts": parts,
        },
        records,
    )


def decompress_pysz_raw_parts(
    segment: Mapping[str, Any],
    out_path: str,
    force: bool,
) -> List[Dict[str, Any]]:
    PyszSZ, _, _ = load_pysz()
    dt = np.dtype(segment["dtype"])
    records: List[Dict[str, Any]] = []
    out = Path(out_path)
    require_output_path(out, force)
    with out.open("wb") as out_file:
        for part in segment["parts"]:
            part_index = int(part["index"])
            part_count = int(part["count"])
            encoded_count = int(part.get("encoded_count", part_count))
            compressed = np.fromfile(part["path"], dtype=np.uint8)
            start_time = time.perf_counter()
            try:
                data, _ = PyszSZ.decompress(compressed, dt, (encoded_count,))
            except Exception as exc:
                raise PipelineError(f"pysz decompression failed for {segment['field']} part {part_index}.") from exc
            elapsed = time.perf_counter() - start_time
            data = np.asarray(data, dtype=dt)
            if data.size < part_count:
                raise PipelineError(
                    f"pysz part decompression for {segment['field']} part {part_index} "
                    f"returned {data.size} values, expected at least {part_count}."
                )
            data[:part_count].tofile(out_file)
            records.append(
                {
                    "api": "pysz.sz.decompress",
                    "field": segment["field"],
                    "part": part_index,
                    "count": part_count,
                    "encoded_count": encoded_count,
                    "input_bytes": int(compressed.size),
                    "output_bytes": int(part_count * dt.itemsize),
                    "wall_seconds": elapsed,
                }
            )
    return records


def make_manifest(
    input_h5: Path,
    h5: h5py.File,
    fields: Mapping[str, str],
    count: int,
    limit: Optional[int],
    pos_scale: PositionScale,
    pos_eb: float,
    vel_eb: float,
    id_eb: float,
    field_error_bounds: Mapping[str, Mapping[str, Any]],
    tools: ToolPaths,
) -> Dict[str, Any]:
    datasets: Dict[str, Dict[str, Any]] = {}
    for logical, h5_path in fields.items():
        dset = h5[h5_path]
        datasets[logical] = {
            "h5_path": h5_path,
            "dtype": str(dset.dtype),
            "shape": list(dset.shape),
            "selected_shape": [count],
            "attrs": collect_attrs(dset),
        }
    attrs = collect_attrs(h5)
    if limit is not None and "npart" in attrs:
        attrs["npart"] = as_jsonable_attr(np.asarray(count, dtype=np.asarray(h5.attrs["npart"]).dtype))
    return {
        "format_version": 1,
        "input_h5": str(input_h5),
        "input_h5_file_bytes": input_h5.stat().st_size,
        "count": count,
        "limit": limit,
        "fields": datasets,
        "root_attrs": attrs,
        "position_scale": {"mode": pos_scale.mode, "value": pos_scale.value, "attr": pos_scale.attr},
        "error_bounds": {
            "positions_lcp_abs": pos_eb,
            "velocities_sz3_abs": vel_eb,
            "id_sz3_abs": id_eb,
        },
        "field_error_bounds": field_error_bounds,
        "tools": {
            "lcp": str(tools.lcp),
            "pcodec": package_version("pcodec"),
            "pysz": package_version("pysz"),
        },
    }


def write_json(path: Path, payload: Mapping[str, Any], force: bool = True) -> None:
    require_output_path(path, force)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def json_size_bytes(payload: Mapping[str, Any]) -> int:
    return len((json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def preprocess_and_compress(args: argparse.Namespace) -> Dict[str, Any]:
    input_h5 = Path(args.input_h5).resolve()
    if not input_h5.is_file():
        raise PipelineError(f"Input HDF5 file does not exist: {input_h5}")
    tools = ToolPaths(
        lcp=find_executable("lcp", args.lcp),
    )
    work_dir = Path(args.work_dir).resolve()
    pre_dir = work_dir / "preprocessed"
    cmp_dir = work_dir / "compressed"
    work_dir.mkdir(parents=True, exist_ok=True)
    if args.force and cmp_dir.exists():
        shutil.rmtree(cmp_dir)
    pre_dir.mkdir(parents=True, exist_ok=True)
    cmp_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    commands: Dict[str, Any] = {}
    preprocess_stats: Dict[str, Any] = {}
    raw_paths: Dict[str, str] = {}

    with h5py.File(input_h5, "r") as h5:
        fields = resolve_fields(h5)
        count = get_selected_count(h5, fields, args.limit)
        first_pos_dtype = np.dtype(h5[fields["x"]].dtype)
        pos_scale = resolve_position_scale(
            h5,
            args.position_scale,
            args.position_scale_attr,
            args.position_scale_value,
            first_pos_dtype,
        )

        pos_paths, pos_stats = export_positions_for_lcp(
            h5, fields, pre_dir, count, args.chunk_size, pos_scale, args.force
        )
        raw_paths.update(pos_paths)
        preprocess_stats["positions"] = pos_stats

        id_dtype = np.dtype(h5[fields["id"]].dtype)
        id_path, id_stats = export_id_for_pcodec(
            h5, fields["id"], pre_dir / f"id.{id_dtype.name}.raw", count, args.chunk_size, args.force
        )
        raw_paths["id"] = id_path
        preprocess_stats["id"] = id_stats

        velocity_stats: Dict[str, Dict[str, float]] = {}
        for logical in VELOCITY_FIELDS:
            source_dtype = str(np.dtype(h5[fields[logical]].dtype))
            out_path = pre_dir / f"{logical}.{source_dtype}.raw"
            raw_paths[logical], velocity_stats[logical] = export_float_for_pysz(
                h5, fields[logical], out_path, count, args.chunk_size, args.force
            )
        preprocess_stats["velocities"] = velocity_stats

        position_ranges = {logical: pos_stats[logical]["range_in_lcp_units"] for logical in POSITION_FIELDS}
        velocity_ranges = {logical: velocity_stats[logical]["range"] for logical in VELOCITY_FIELDS}

        pos_selection_base = select_relative_or_absolute(
            args, "pos", POSITION_FIELDS, position_ranges, args.abs_eb
        )
        if pos_selection_base.mode == "relative":
            pos_compressor_bounds: List[float] = []
            for logical in POSITION_FIELDS:
                dtype = np.dtype(h5[fields[logical]].dtype)
                rounding = 0.5 / pos_scale.value if np.issubdtype(dtype, np.integer) else 0.0
                cast = float(pos_stats[logical]["preprocess_cast_max_abs_in_lcp_units"])
                pos_compressor_bounds.append(max(0.0, pos_selection_base.abs_by_field[logical] - cast - rounding))
            pos_eb = float(min(pos_compressor_bounds))
        else:
            pos_eb = float(min(pos_selection_base.abs_by_field.values()))
        pos_selection = ErrorBoundSelection(
            pos_selection_base.mode,
            pos_selection_base.abs_by_field,
            relative=pos_selection_base.relative,
            compressor_abs=pos_eb,
        )
        vel_selection = select_relative_or_absolute(
            args, "vel", VELOCITY_FIELDS, velocity_ranges, args.abs_eb
        )
        vel_eb = float(max(vel_selection.abs_by_field.values()))
        id_eb = validate_error_bound(args.id_abs_eb, "--id-abs-eb")

        field_error_bounds = serialize_error_bound_selection(
            pos_selection, POSITION_FIELDS, position_ranges, "lcp_units"
        )
        field_error_bounds.update(
            serialize_error_bound_selection(vel_selection, VELOCITY_FIELDS, velocity_ranges, "source_units")
        )
        field_error_bounds["id"] = {
            "mode": "lossless",
            "abs": id_eb,
            "relative": None,
            "range": float(id_stats["max"] - id_stats["min"]) if id_stats["min"] is not None else None,
            "range_units": "source_units",
            "compressor_abs": 0.0,
        }

        manifest = make_manifest(
            input_h5,
            h5,
            fields,
            count,
            args.limit,
            pos_scale,
            pos_eb,
            vel_eb,
            id_eb,
            field_error_bounds,
            tools,
        )

        selected_payload_bytes = sum(
            dataset_nbytes(h5[fields[logical]].dtype, count) for logical in LOGICAL_ORDER
        )

    manifest["artifacts"] = {
        "preprocessed": raw_paths,
        "compressed": {
            "positions": str(cmp_dir / "positions.lcp"),
            "order": str(cmp_dir / "order.pcodecparts"),
            "id": str(cmp_dir / "id.pcodecparts"),
            "vx": str(cmp_dir / "vx.pyszparts"),
            "vy": str(cmp_dir / "vy.pyszparts"),
            "vz": str(cmp_dir / "vz.pyszparts"),
        },
    }
    requested_part_size = args.part_size if args.part_size is not None else args.sz3_block_size
    manifest["part_size"] = effective_part_size(manifest["count"], requested_part_size)
    manifest["order_dtype"] = "int32"
    manifest["compressed_segments"] = {}
    manifest["preprocess"] = preprocess_stats
    manifest["sizes"] = {"selected_original_payload_bytes": selected_payload_bytes}
    write_json(work_dir / "manifest.json", manifest, force=True)

    order_raw = pre_dir / "order.i32.raw"
    require_output_path(order_raw, args.force)
    raw_paths["order"] = str(order_raw)

    lcp_cmp = cmp_dir / "positions.lcp"
    require_output_path(lcp_cmp, args.force)
    commands["lcp_compress_positions"] = run_command(
        [
            str(tools.lcp),
            "-i",
            raw_paths["x"],
            raw_paths["y"],
            raw_paths["z"],
            "-z",
            str(lcp_cmp),
            "-1",
            str(manifest["count"]),
            "-eb",
            str(manifest["error_bounds"]["positions_lcp_abs"]),
            "-ord",
            "32",
            str(order_raw),
        ]
    )

    part_size = int(manifest["part_size"])
    manifest["compressed_segments"]["order"], commands["pcodec_compress_lcp_order"] = compress_pcodec_raw_parts(
        str(order_raw),
        "int32",
        cmp_dir,
        "order",
        manifest["count"],
        part_size,
        args.force,
    )

    manifest["compressed_segments"]["id"], commands["pcodec_compress_id"] = compress_pcodec_raw_parts(
        raw_paths["id"],
        manifest["fields"]["id"]["dtype"],
        cmp_dir,
        "id",
        manifest["count"],
        part_size,
        args.force,
    )

    for logical in VELOCITY_FIELDS:
        dtype = manifest["fields"][logical]["dtype"]
        manifest["compressed_segments"][logical], commands[f"pysz_compress_{logical}"] = compress_pysz_raw_parts(
            raw_paths[logical],
            dtype,
            cmp_dir,
            logical,
            manifest["count"],
            manifest["field_error_bounds"][logical]["abs"],
            part_size,
            args.force,
        )

    manifest["commands"] = {"compress": commands}
    manifest["timing"] = {"preprocess_and_compress_wall_seconds": time.perf_counter() - t0}
    update_compressed_size_metrics(manifest, work_dir)
    write_json(work_dir / "manifest.json", manifest, force=True)
    return manifest


def decompress_package(args: argparse.Namespace) -> Dict[str, Any]:
    work_dir = Path(args.work_dir).resolve()
    manifest_path = work_dir / "manifest.json"
    if not manifest_path.is_file():
        raise PipelineError(f"Missing manifest: {manifest_path}")
    manifest = read_json(manifest_path)
    tools = ToolPaths(
        lcp=find_executable("lcp", args.lcp or manifest.get("tools", {}).get("lcp")),
    )

    count = int(manifest["count"])
    dec_dir = work_dir / "decompressed"
    dec_dir.mkdir(parents=True, exist_ok=True)
    output_h5 = Path(args.output_h5).resolve() if args.output_h5 else work_dir / "reconstructed.h5"
    require_output_path(output_h5, args.force)

    t0 = time.perf_counter()
    commands: Dict[str, Any] = {}
    order_dtype = order_dtype_from_manifest(manifest)
    dec_paths = {
        "x": str(dec_dir / "x.f32.raw"),
        "y": str(dec_dir / "y.f32.raw"),
        "z": str(dec_dir / "z.f32.raw"),
        "order": str(dec_dir / f"order.{order_dtype.name}.raw"),
        "id": str(dec_dir / f"id.{np.dtype(manifest['fields']['id']['dtype']).name}.raw"),
        "vx": str(dec_dir / f"vx.{manifest['fields']['vx']['dtype']}.raw"),
        "vy": str(dec_dir / f"vy.{manifest['fields']['vy']['dtype']}.raw"),
        "vz": str(dec_dir / f"vz.{manifest['fields']['vz']['dtype']}.raw"),
    }
    for path in dec_paths.values():
        require_output_path(Path(path), args.force)

    artifacts = manifest["artifacts"]["compressed"]
    commands["lcp_decompress_positions"] = run_command(
        [
            str(tools.lcp),
            "-z",
            artifacts["positions"],
            "-o",
            dec_paths["x"],
            dec_paths["y"],
            dec_paths["z"],
            "-1",
            str(count),
            "-eb",
            str(manifest["error_bounds"]["positions_lcp_abs"]),
        ]
    )

    segments = manifest.get("compressed_segments")
    if not segments:
        raise PipelineError("Manifest does not contain compressed_segments for the Python compressor pipeline.")
    commands["pcodec_decompress_lcp_order"] = decompress_pcodec_raw_parts(
        segments["order"], dec_paths["order"], args.force
    )
    commands["pcodec_decompress_id"] = decompress_pcodec_raw_parts(
        segments["id"], dec_paths["id"], args.force
    )
    for logical in VELOCITY_FIELDS:
        commands[f"pysz_decompress_{logical}"] = decompress_pysz_raw_parts(
            segments[logical], dec_paths[logical], args.force
        )

    recombine_start = time.perf_counter()
    recombine_h5(manifest, dec_paths, output_h5, args.chunk_size)
    recombine_seconds = time.perf_counter() - recombine_start

    manifest.setdefault("commands", {})["decompress"] = commands
    manifest.setdefault("timing", {})["decompress_and_recombine_wall_seconds"] = time.perf_counter() - t0
    manifest["timing"]["recombine_h5_wall_seconds"] = recombine_seconds
    manifest["artifacts"]["decompressed"] = dec_paths
    manifest["artifacts"]["reconstructed_h5"] = str(output_h5)
    manifest.setdefault("sizes", {})["reconstructed_h5_file_bytes"] = output_h5.stat().st_size
    update_compressed_size_metrics(manifest, work_dir)
    write_json(manifest_path, manifest, force=True)
    return manifest


def raw_chunk_reader(path: str, dtype: np.dtype, count: int, chunk_size: int) -> Iterable[np.ndarray]:
    with Path(path).open("rb") as f:
        remaining = count
        while remaining:
            n = min(chunk_size, remaining)
            data = np.fromfile(f, dtype=dtype, count=n)
            if data.size != n:
                raise PipelineError(f"Unexpected EOF reading {path}; expected {n}, got {data.size}.")
            yield data
            remaining -= n


def create_dataset(h5: h5py.File, path: str, dtype: np.dtype, count: int) -> h5py.Dataset:
    parent_path, _, name = path.rpartition("/")
    group = h5
    if parent_path:
        group = h5.require_group(parent_path)
    return group.create_dataset(name, shape=(count,), dtype=dtype)


def recombine_h5(manifest: Mapping[str, Any], dec_paths: Mapping[str, str], output_h5: Path, chunk_size: int) -> None:
    count = int(manifest["count"])
    scale = float(manifest["position_scale"]["value"])
    order_dtype = order_dtype_from_manifest(manifest)
    order = np.fromfile(dec_paths["order"], dtype=order_dtype, count=count)
    if order.size != count:
        raise PipelineError(f"Unexpected EOF reading {dec_paths['order']}; expected {count}, got {order.size}.")
    if count and (int(order.min()) < 0 or int(order.max()) >= count):
        raise PipelineError("LCP order sidecar is not a valid index range for this particle count.")
    order_index = order.astype(np.intp, copy=False)

    with h5py.File(output_h5, "w") as out:
        apply_attrs(out, manifest.get("root_attrs", {}))

        for logical in LOGICAL_ORDER:
            field = manifest["fields"][logical]
            target_dtype = np.dtype(field["dtype"])
            dset = create_dataset(out, field["h5_path"], target_dtype, count)
            apply_attrs(dset, field.get("attrs", {}))

            offset = 0
            if logical == "id":
                for chunk in raw_chunk_reader(dec_paths[logical], target_dtype, count, chunk_size):
                    dset[offset : offset + chunk.size] = chunk
                    offset += chunk.size
            elif logical in POSITION_FIELDS:
                info = np.iinfo(target_dtype) if np.issubdtype(target_dtype, np.integer) else None
                decoded = np.fromfile(dec_paths[logical], dtype=np.float32, count=count)
                if decoded.size != count:
                    raise PipelineError(
                        f"Unexpected EOF reading {dec_paths[logical]}; expected {count}, got {decoded.size}."
                    )
                values64 = decoded.astype(np.float64) * scale
                if info is not None:
                    values64 = np.rint(values64)
                    values64 = np.clip(values64, info.min, info.max)
                converted = values64.astype(target_dtype)
                restored = np.empty(count, dtype=target_dtype)
                restored[order_index] = converted
                dset[:] = restored
            else:
                for chunk in raw_chunk_reader(dec_paths[logical], target_dtype, count, chunk_size):
                    dset[offset : offset + chunk.size] = chunk
                    offset += chunk.size


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


def finalize_metric_acc(acc: Mapping[str, Any]) -> Dict[str, Any]:
    count = int(acc["count"])
    if count == 0:
        raise PipelineError("Cannot finalize metrics for zero elements.")
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


def empty_metric_acc() -> Dict[str, Any]:
    return {
        "count": 0,
        "sum_squared_error": 0.0,
        "sum_absolute_error": 0.0,
        "max_absolute_error": 0.0,
        "min": math.inf,
        "max": -math.inf,
    }


def compute_metrics(
    original_h5: Path,
    reconstructed_h5: Path,
    manifest: Mapping[str, Any],
    chunk_size: int,
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
        for logical in LOGICAL_ORDER:
            field = manifest["fields"][logical]
            original_dset = original[field["h5_path"]]
            recon_dset = recon[field["h5_path"]]
            acc = empty_metric_acc()
            fixed_acc = empty_metric_acc() if logical in POSITION_FIELDS and np.issubdtype(original_dset.dtype, np.integer) else None
            exact = True

            for slc in chunk_slices(count, chunk_size):
                orig = original_dset[slc]
                dec = recon_dset[slc]
                if logical == "id":
                    exact = exact and bool(np.array_equal(orig, dec))
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
    vel_eb = float(manifest["error_bounds"]["velocities_sz3_abs"])
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
            target = {
                "mode": field_bound.get("mode", "absolute"),
                "relative_error_bound": field_bound.get("relative"),
                "range_for_relative": field_bound.get("range"),
                "range_units": field_bound.get("range_units", "source_units"),
                "requested_abs_bound": requested_abs,
                "compressor_abs_eb": float(field_bound.get("compressor_abs", requested_abs)),
                "effective_final_abs_bound": requested_abs,
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


def manifest_path_from_manifest(manifest: Mapping[str, Any]) -> str:
    compressed = manifest.get("artifacts", {}).get("compressed", {})
    positions = compressed.get("positions")
    if not positions:
        return "."
    return str(Path(positions).resolve().parents[1])


def maybe_clean_raw(work_dir: Path) -> None:
    for name in ("preprocessed", "decompressed"):
        path = work_dir / name
        if path.exists():
            shutil.rmtree(path)

def report_field_dtype(report: Mapping[str, Any], logical: str) -> np.dtype:
    field = report["fields"][logical]
    dtype = field.get("dtype", field.get("original_dtype"))
    if dtype is None:
        raise PipelineError(f"Missing dtype for field {logical!r} in report.")
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


def order_dtype_from_manifest(manifest: Mapping[str, Any]) -> np.dtype:
    segment = manifest.get("compressed_segments", {}).get("order", {})
    dtype = segment.get("dtype", manifest.get("order_dtype", "int64"))
    dt = np.dtype(dtype)
    if dt not in (np.dtype("int32"), np.dtype("int64")):
        raise PipelineError(f"Unsupported LCP order dtype in manifest: {dt}.")
    return dt


def compression_ratio(original_bytes: int, compressed_bytes: int) -> float:
    return original_bytes / compressed_bytes if compressed_bytes else math.inf


def component_compression_ratios(report: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    components = report.get("sizes", {}).get("compressed_components_bytes", {})
    xyz_compressed = int(components.get("compressed/positions.lcp", 0)) + compressed_bytes_with_prefixes(
        components, ("compressed/order.",)
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
    }
    for entry in entries.values():
        entry["compression_ratio"] = compression_ratio(entry["original_bytes"], entry["compressed_bytes"])
    return entries


def print_component_summary(report: Mapping[str, Any]) -> None:
    ratios = component_compression_ratios(report)
    print("component_CR:")
    for name in ("xyz", "order", "id", "vx", "vy", "vz"):
        entry = ratios[name]
        ratio = entry["compression_ratio"]
        ratio_s = "inf" if math.isinf(ratio) else f"{ratio:.6g}"
        note = " includes order sidecar" if name == "xyz" else ""
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


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "compress":
            if not args.work_dir:
                args.work_dir = str(default_work_dir_for_args(args))
            manifest = preprocess_and_compress(args)
            print(f"package_dir = {Path(args.work_dir).resolve()}")
            print(f"manifest = {Path(args.work_dir).resolve() / 'manifest.json'}")
            print(f"payload_CR = {manifest['sizes']['payload_compression_ratio']:.6g}")
            print_component_summary(manifest)
            return 0

        if args.command == "decompress":
            manifest = decompress_package(args)
            if args.clean_raw:
                maybe_clean_raw(Path(args.work_dir).resolve())
            print(f"reconstructed_h5 = {manifest['artifacts']['reconstructed_h5']}")
            return 0

        if args.command == "roundtrip":
            if not args.work_dir:
                args.work_dir = str(default_work_dir_for_args(args))
            manifest = preprocess_and_compress(args)
            manifest = decompress_package(args)
            input_h5 = Path(args.input_h5).resolve()
            output_h5 = Path(manifest["artifacts"]["reconstructed_h5"]).resolve()
            metrics = compute_metrics(input_h5, output_h5, manifest, args.chunk_size)
            metrics_path = Path(args.metrics_json).resolve() if args.metrics_json else Path(args.work_dir).resolve() / "metrics.json"
            write_json(metrics_path, metrics, force=True)
            if args.clean_raw:
                maybe_clean_raw(Path(args.work_dir).resolve())
            print_summary(metrics, metrics_path)
            return 0

        parser.error(f"Unknown command: {args.command}")
    except PipelineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
