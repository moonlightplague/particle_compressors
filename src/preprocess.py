import argparse
import shutil
import h5py
import math
import importlib.metadata
import numpy as np

from typing import Any, Dict, Iterable, Mapping, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass

import src.helpers as hp


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


def update_numeric_stats(stats: Dict[str, Any], values: np.ndarray) -> None:
    if values.size == 0:
        raise RuntimeError("Not a valid input array.")
    values64 = values.astype(np.float64, copy=False)
    stats["float_min"] = min(stats["float_min"], float(values64.min(initial=stats["float_min"])))
    stats["float_max"] = max(stats["float_max"], float(values64.max(initial=stats["float_max"])))

def update_numeric_stats_int(stats: Dict[str, Any], values: np.ndarray) -> None:
    if values.size == 0:
        raise RuntimeError("Not a valid input array.")
    values32 = values.astype(np.int32, copy=False)
    stats["int_min"] = min(stats["int_min"], int(np.min(values32)))
    stats["int_max"] = max(stats["int_max"], int(np.max(values32)))


def finalize_numeric_stats(stats: Dict[str, float]) -> Dict[str, float]:
    if math.isinf(stats["float_min"]) or math.isinf(stats["float_max"]):
        raise RuntimeError("Cannot compute relative error bound for an empty field.")
    value_range_float = float(stats["float_max"] - stats["float_min"])
    if "int_min" not in stats or "int_max" not in stats:
        return {"float_min": float(stats["float_min"]), "float_max": float(stats["float_max"]), "float_range": value_range_float}
    if math.isinf(stats["int_min"]) or math.isinf(stats["int_max"]):
        raise RuntimeError("Cannot compute relative error bound for an empty field.")
    value_range_int = float(stats["int_max"] - stats["int_min"])
    return {"float_min": float(stats["float_min"]), "float_max": float(stats["float_max"]), "float_range": value_range_float,
            "int_min": float(stats["int_min"]), "int_max": float(stats["int_max"]), "int_range": value_range_int}


def resolve_fields(h5: h5py.File) -> Dict[str, str]:
    available: Dict[str, str] = {}

    def visit(name: str, obj: Any) -> None:
        if isinstance(obj, h5py.Dataset):
            available[name.split("/")[-1].lower()] = name

    h5.visititems(visit)
    resolved: Dict[str, str] = {}
    for logical, aliases in hp.FIELD_ALIASES.items():
        for alias in aliases:
            if alias.lower() in available:
                resolved[logical] = available[alias.lower()]
                break
        if logical not in resolved:
            raise RuntimeError(
                f"Could not find dataset for logical field {logical!r}; "
                f"tried aliases {aliases}."
            )
    return resolved


def resolve_position_scale(
    h5: h5py.File,
    mode: str,
    attr_name: str,
    explicit_value: Optional[float],
    pos_dtype: np.dtype,
) -> PositionScale:
    if mode == "value":
        if explicit_value is None:
            raise RuntimeError("--position-scale value requires --position-scale-value.")
        return PositionScale(mode="value", value=float(explicit_value))

    if mode == "attr":
        if attr_name not in h5.attrs:
            raise RuntimeError(f"--position-scale attr requested, but root attr {attr_name!r} is missing.")
        return PositionScale(mode="attr", value=float(np.asarray(h5.attrs[attr_name]).item()), attr=attr_name)

    if mode == "raw":
        return PositionScale(mode="raw", value=1.0)

    if mode != "auto":
        raise RuntimeError(f"Unsupported position scale mode: {mode}")

    if np.issubdtype(pos_dtype, np.integer) and attr_name in h5.attrs:
        value = float(np.asarray(h5.attrs[attr_name]).item())
        if value > 0:
            return PositionScale(mode="auto_attr", value=value, attr=attr_name)
    return PositionScale(mode="auto_raw", value=1.0)


def get_selected_count(h5: h5py.File, fields: Mapping[str, str], limit: Optional[int]) -> int:
    sizes = {logical: int(h5[path].shape[0]) for logical, path in fields.items()}
    unique_sizes = set(sizes.values())
    if len(unique_sizes) != 1:
        raise RuntimeError(f"Particle fields do not have the same length: {sizes}")
    count = unique_sizes.pop()
    if limit is not None:
        if limit <= 0:
            raise RuntimeError("--limit must be positive.")
        count = min(count, limit)
    return count


def export_positions_for_lcp(
    h5: h5py.File,
    fields: Mapping[str, str],
    out_dir: Path,
    count: int,
    scale: PositionScale,
    force: bool,
) -> Tuple[Dict[str, str], Dict[str, Dict[str, float]]]:
    out_paths: Dict[str, str] = {}
    stats: Dict[str, Dict[str, float]] = {}
    out_dir.mkdir(parents=True, exist_ok=True)

    for logical in hp.POSITION_FIELDS:
        dataset = h5[fields[logical]]
        out_path = out_dir / f"{logical}.f32.raw"
        out_path_int = out_dir / f"{logical}.i32.raw"
        hp.require_output_path(out_path, force)
        max_cast_abs = 0.0
        max_cast_fixed = 0.0
        range_stats = {"float_min": math.inf, "float_max": -math.inf,
                       "int_min": math.inf, "int_max": -math.inf}
        source = dataset[:count]
        source.tofile(out_path_int)
        source64 = source.astype(np.float64, copy=False)
        scaled64 = source64 / scale.value
        scaled32 = scaled64.astype(np.float32)
        scaled32.tofile(out_path)
        update_numeric_stats(range_stats, scaled64)
        update_numeric_stats_int(range_stats, source)
        if source.size:
            cast_abs = np.abs(scaled32.astype(np.float64) - scaled64)
            max_cast_abs = float(cast_abs.max(initial=0.0))
            max_cast_fixed = max_cast_abs * scale.value
        out_paths[logical] = str(out_path)
        out_paths[f"{logical}_int"]=str(out_path_int)
        field_stats = finalize_numeric_stats(range_stats)
        stats[logical] = {
            "scale": scale.value,
            "preprocess_cast_max_abs_in_lcp_units": max_cast_abs,
            "preprocess_cast_max_abs_in_original_fixed_point_units": max_cast_fixed,
            "min_in_lcp_units": field_stats["float_min"],
            "max_in_lcp_units": field_stats["float_max"],
            "range_in_lcp_units": field_stats["float_range"],
            "min_in_int_units": field_stats["int_min"],
            "max_in_int_units": field_stats["int_max"],
            "range_in_int_units": field_stats["int_range"],
        }
    return out_paths, stats


def export_positions_for_xnyzip(
    position_paths: Mapping[str, str],
    out_path: Path,
    count: int,
    force: bool,
) -> Tuple[str, Dict[str, Any]]:
    hp.require_output_path(out_path, force)
    interleaved = np.memmap(out_path, dtype=np.float32, mode="w+", shape=(count, 3))
    for axis, logical in enumerate(hp.POSITION_FIELDS):
        values = np.memmap(position_paths[logical], dtype=np.float32, mode="r")
        if values.size != count:
            raise RuntimeError(
                f"XnYZip position export expected {count} values in {position_paths[logical]}, "
                f"got {values.size}."
            )
        interleaved[:, axis] = values
    interleaved.flush()
    del interleaved
    return str(out_path), {
        "dtype": "float32",
        "shape": [count, 3],
        "layout": "xyz_interleaved",
        "file_bytes": count * 3 * np.dtype(np.float32).itemsize,
    }


def export_float_for_pysz(
    h5: h5py.File,
    dataset_path: str,
    out_path: Path,
    count: int,
    force: bool,
) -> Tuple[str, Dict[str, float]]:
    hp.require_output_path(out_path, force)
    dtype = np.dtype(h5[dataset_path].dtype)
    if dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise RuntimeError(f"pysz velocity export expected float32/float64, got {dtype} for {dataset_path}.")
    range_stats = {"float_min": math.inf, "float_max": -math.inf}
    source = h5[dataset_path][:count].astype(dtype, copy=False)
    source.tofile(out_path)
    update_numeric_stats(range_stats, source)
    field_stats = finalize_numeric_stats(range_stats)
    return str(out_path), field_stats


def export_float32_for_lcp(
    raw_path: str,
    source_dtype: np.dtype,
    out_path: Path,
    count: int,
    force: bool,
) -> Tuple[str, float]:
    source_dtype = np.dtype(source_dtype)
    if source_dtype == np.dtype("float32"):
        return raw_path, 0.0
    hp.require_output_path(out_path, force)
    source = hp.read_raw(raw_path, source_dtype, count)
    encoded = source.astype(np.float32)
    cast_error = np.abs(encoded.astype(np.float64) - source.astype(np.float64, copy=False))
    encoded.tofile(out_path)
    return str(out_path), float(cast_error.max(initial=0.0))


def export_id_for_pcodec(
    h5: h5py.File,
    dataset_path: str,
    out_path: Path,
    count: int,
    force: bool,
) -> Tuple[str, Dict[str, Any]]:
    hp.require_output_path(out_path, force)
    dataset = h5[dataset_path]
    source_dtype = np.dtype(dataset.dtype)
    if not np.issubdtype(source_dtype, np.integer):
        raise RuntimeError(f"pcodec id export expected an integer dtype, got {source_dtype} for {dataset_path}.")
    source = dataset[:count]
    min_id = int(source.min()) if source.size else None
    max_id = int(source.max()) if source.size else None
    source.astype(source_dtype, copy=False).tofile(out_path)
    return str(out_path), {"source_dtype": str(source_dtype), "min": min_id, "max": max_id, "pcodec_dtype": str(source_dtype)}


def validate_error_bound(value: float, label: str) -> float:
    value = float(value)
    if value < 0.0:
        raise RuntimeError(f"{label} must be non-negative.")
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
        raise RuntimeError(f"--{prefix.replace('_', '-')}-rel-eb and --{prefix.replace('_', '-')}-abs-eb cannot both be set.")

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


def as_jsonable_attr(value: Any) -> Dict[str, Any]:
    arr = np.asarray(value)
    if arr.shape == ():
        payload: Any = arr.item()
    else:
        payload = arr.tolist()
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8", errors="surrogateescape")
    return {"dtype": str(arr.dtype), "shape": list(arr.shape), "value": payload}


def collect_attrs(obj: Any) -> Dict[str, Dict[str, Any]]:
    return {name: as_jsonable_attr(obj.attrs[name]) for name in obj.attrs.keys()}


def package_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def make_manifest(
    input_h5: Path,
    h5: h5py.File,
    fields: Mapping[str, str],
    count: int,
    limit: Optional[int],
    pos_scale: PositionScale,
    pos_eb: float,
    xnyzip_eb: float,
    vel_eb: float,
    id_eb: float,
    field_error_bounds: Mapping[str, Mapping[str, Any]],
    tools: hp.ToolPaths,
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
        "format_version": 2,
        "input_h5": str(input_h5),
        "input_h5_file_bytes": input_h5.stat().st_size,
        "count": count,
        "limit": limit,
        "fields": datasets,
        "root_attrs": attrs,
        "position_scale": {"mode": pos_scale.mode, "value": pos_scale.value, "attr": pos_scale.attr},
        "error_bounds": {
            "positions_lcp_abs": pos_eb,
            "positions_xnyzip_abs": xnyzip_eb,
            "velocities_sz3_abs": vel_eb,
            "id_sz3_abs": id_eb,
        },
        "field_error_bounds": field_error_bounds,
        "tools": {
            "lcp": str(tools.lcp),
            "pcodec": package_version("pcodec"),
            "pysz": package_version("pysz"),
            "pyszo": package_version("pyszo"),
        },
    }


def preprocess(args: argparse.Namespace):
    if args.lossless == "szo" and not 0.0 <= float(args.szo_abs_eb) < 1.0:
        raise RuntimeError("--szo-abs-eb must be at least 0 and less than 1.")
    input_h5 = Path(args.input_h5).resolve()
    if not input_h5.is_file():
        raise RuntimeError(f"Input HDF5 file does not exist: {input_h5}")
    tools = hp.ToolPaths(
        lcp=Path(args.lcp),
    )
    work_dir = Path(args.work_dir).resolve()
    pre_dir = work_dir / "preprocessed"
    cmp_dir = work_dir / "compressed"
    work_dir.mkdir(parents=True, exist_ok=True)
    if args.force and cmp_dir.exists():
        shutil.rmtree(cmp_dir)
    pre_dir.mkdir(parents=True, exist_ok=True)
    cmp_dir.mkdir(parents=True, exist_ok=True)

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
            h5, fields, pre_dir, count, pos_scale, args.force
        )
        raw_paths.update(pos_paths)
        preprocess_stats["positions"] = pos_stats

        xnyzip_path, xnyzip_stats = export_positions_for_xnyzip(
            pos_paths, pre_dir / "positions.xnyzip.bin", count, args.force
        )
        raw_paths["positions_xnyzip"] = xnyzip_path
        preprocess_stats["positions_xnyzip"] = xnyzip_stats

        id_dtype = np.dtype(h5[fields["id"]].dtype)
        id_path, id_stats = export_id_for_pcodec(
            h5, fields["id"], pre_dir / f"id.{id_dtype.name}.raw", count, args.force
        )
        raw_paths["id"] = id_path
        preprocess_stats["id"] = id_stats

        velocity_stats: Dict[str, Dict[str, float]] = {}
        for logical in hp.VELOCITY_FIELDS:
            source_dtype = np.dtype(h5[fields[logical]].dtype)
            out_path = pre_dir / f"{logical}.{source_dtype.name}.raw"
            raw_paths[logical], velocity_stats[logical] = export_float_for_pysz(
                h5, fields[logical], out_path, count, args.force
            )
            velocity_stats[logical]["preprocess_cast_max_abs"] = 0.0
            if args.vel_compressor == "lcp":
                lcp_path, cast_error = export_float32_for_lcp(
                    raw_paths[logical],
                    source_dtype,
                    pre_dir / f"{logical}.lcp.f32.raw",
                    count,
                    args.force,
                )
                raw_paths[f"{logical}_lcp"] = lcp_path
                velocity_stats[logical]["preprocess_cast_max_abs"] = cast_error
        preprocess_stats["velocities"] = velocity_stats

        position_ranges = {logical: pos_stats[logical]["range_in_lcp_units"] for logical in hp.POSITION_FIELDS}
        velocity_ranges = {logical: velocity_stats[logical]["float_range"] for logical in hp.VELOCITY_FIELDS}
        position_diagonal = math.sqrt(sum(value * value for value in position_ranges.values()))

        pos_selection_base = select_relative_or_absolute(
            args, "pos", hp.POSITION_FIELDS, position_ranges, args.abs_eb
        )
        position_preprocess_errors: Dict[str, float] = {}
        for logical in hp.POSITION_FIELDS:
            dtype = np.dtype(h5[fields[logical]].dtype)
            rounding = 0.5 / pos_scale.value if np.issubdtype(dtype, np.integer) else 0.0
            cast = float(pos_stats[logical]["preprocess_cast_max_abs_in_lcp_units"])
            position_preprocess_errors[logical] = cast + rounding
        if pos_selection_base.mode == "relative":
            pos_compressor_bounds = [
                max(0.0, pos_selection_base.abs_by_field[logical] - position_preprocess_errors[logical])
                for logical in hp.POSITION_FIELDS
            ]
            pos_eb = float(min(pos_compressor_bounds))
            xnyzip_requested_eb = float(pos_selection_base.relative * position_diagonal)
            xnyzip_preprocess_error = math.sqrt(
                sum(value * value for value in position_preprocess_errors.values())
            )
            xnyzip_eb = max(0.0, xnyzip_requested_eb - xnyzip_preprocess_error)
        else:
            pos_eb = float(min(pos_selection_base.abs_by_field.values()))
            xnyzip_requested_eb = pos_eb
            xnyzip_preprocess_error = 0.0
            xnyzip_eb = pos_eb
        pos_selection = ErrorBoundSelection(
            pos_selection_base.mode,
            pos_selection_base.abs_by_field,
            relative=pos_selection_base.relative,
            compressor_abs=pos_eb,
        )
        vel_selection_base = select_relative_or_absolute(
            args, "vel", hp.VELOCITY_FIELDS, velocity_ranges, args.abs_eb
        )
        if args.vel_compressor == "lcp":
            velocity_compressor_bounds = [
                max(
                    0.0,
                    vel_selection_base.abs_by_field[logical]
                    - float(velocity_stats[logical]["preprocess_cast_max_abs"]),
                )
                for logical in hp.VELOCITY_FIELDS
            ]
            vel_eb = float(min(velocity_compressor_bounds))
            vel_selection = ErrorBoundSelection(
                vel_selection_base.mode,
                vel_selection_base.abs_by_field,
                relative=vel_selection_base.relative,
                compressor_abs=vel_eb,
            )
        else:
            vel_eb = float(max(vel_selection_base.abs_by_field.values()))
            vel_selection = vel_selection_base
        id_eb = validate_error_bound(args.id_abs_eb, "--id-abs-eb")

        field_error_bounds = serialize_error_bound_selection(
            pos_selection, hp.POSITION_FIELDS, position_ranges, "lcp_units"
        )
        field_error_bounds["positions_xnyzip"] = {
            "mode": pos_selection_base.mode,
            "abs": xnyzip_requested_eb,
            "relative": pos_selection_base.relative,
            "range": position_diagonal,
            "range_units": "lcp_units_bbox_diagonal",
            "compressor_abs": xnyzip_eb,
            "preprocess_l2_max_abs": xnyzip_preprocess_error,
        }
        field_error_bounds.update(
            serialize_error_bound_selection(vel_selection, hp.VELOCITY_FIELDS, velocity_ranges, "source_units")
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
            xnyzip_eb,
            vel_eb,
            id_eb,
            field_error_bounds,
            tools,
        )
        manifest["compressors"] = {
            "positions": args.pos_compressor,
            "velocities": args.vel_compressor,
            "lossless": args.lossless,
        }
        if args.vel_compressor == "lcp":
            manifest["error_bounds"]["velocities_lcp_abs"] = vel_eb
        if args.lossless == "szo":
            manifest["error_bounds"]["szo_integer_abs"] = float(args.szo_abs_eb)

        selected_payload_bytes = sum(
            int(np.dtype(h5[fields[logical]].dtype).itemsize * count) for logical in hp.LOGICAL_ORDER
        )

    lossless_extension = "szo" if args.lossless == "szo" else "pco"
    compressed_artifacts = {"id": str(cmp_dir / f"id.{lossless_extension}")}
    if args.pos_compressor == "lcp":
        compressed_artifacts["positions"] = str(cmp_dir / "positions.lcp")
    else:
        compressed_artifacts.update(
            {
                "x": str(cmp_dir / "x.psz"),
                "y": str(cmp_dir / "y.psz"),
                "z": str(cmp_dir / "z.psz"),
            }
        )
    if args.vel_compressor == "lcp":
        compressed_artifacts["velocities"] = str(cmp_dir / "velocities.lcp")
        if args.pos_compressor == "lcp":
            compressed_artifacts["velocity_order"] = str(
                cmp_dir / f"velocity_order.{lossless_extension}"
            )
    else:
        compressed_artifacts.update(
            {
                "vx": str(cmp_dir / "vx.psz"),
                "vy": str(cmp_dir / "vy.psz"),
                "vz": str(cmp_dir / "vz.psz"),
            }
        )
    manifest["artifacts"] = {
        "preprocessed": raw_paths,
        "compressed": compressed_artifacts,
    }
    manifest["order_dtype"] = "int32"
    manifest["compressed_fields"] = {}
    manifest["preprocess"] = preprocess_stats
    manifest["sizes"] = {"selected_original_payload_bytes": selected_payload_bytes}
    hp.write_json(work_dir / "manifest.json", manifest, force=True)

    return manifest, raw_paths, tools
