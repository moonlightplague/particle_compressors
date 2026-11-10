"""A small QPET-inspired frontend for XnYZip positions and SZO velocities.

Use conservative uniform velocity bounds, then validate the actual decoded
values and losslessly store any offending velocity vectors in canonical order.
The existing codecs do not need point-wise error-bound support.
"""

import argparse
import io
import math
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import h5py
import numpy as np
import zstandard as zstd

from src.constants import VELOCITY_FIELDS
from src.models import CanonicalOrder
from src.raw_codecs import decompress_lossy_raw
from src.runtime import require_output_path


QOI_CHOICES = ("speed", "kinetic-energy", "vx-square", "vy-square", "vz-square")
QOI_DEFAULTS = {"quantity": None, "abs_eb": None, "rel_eb": None, "mass": 1.0}
CHUNK_ROWS = 262_144
CORRECTION_FORMAT = "qoi_velocity_rows_v1"


def validate_qoi_args(args: argparse.Namespace) -> None:
    quantity = getattr(args, "qoi", None)
    if quantity == "none":
        return
    absolute = getattr(args, "qoi_abs_eb", None)
    relative = getattr(args, "qoi_rel_eb", None)
    if quantity is None:
        if absolute is not None or relative is not None:
            raise RuntimeError("QoI error bounds require --qoi. See QOI.md.")
        return
    if quantity not in QOI_CHOICES:
        raise RuntimeError(f"Unknown QoI: {quantity}. See QOI.md.")
    if (absolute is None) == (relative is None):
        raise RuntimeError("Choose exactly one of --qoi-abs-eb and --qoi-rel-eb.")
    for label, value in (("--qoi-abs-eb", absolute), ("--qoi-rel-eb", relative)):
        if value is not None and (not math.isfinite(value) or value < 0):
            raise RuntimeError(f"{label} must be finite and non-negative.")
    mass = getattr(args, "qoi_mass", 1.0)
    if not math.isfinite(mass) or mass <= 0:
        raise RuntimeError("--qoi-mass must be finite and positive.")
    if args.pos_compressor != "xnyzip" or args.vel_compressor != "szo":
        raise RuntimeError("The QoI frontend requires XnYZip positions and SZO velocities.")
    if getattr(args, "lattice_layout", False) or getattr(args, "xnyzip_structure_aware", False):
        raise RuntimeError("The QoI frontend currently requires flat velocities; disable lattice/structure-aware layouts.")


def evaluate_qoi(quantity: str, values: Mapping[str, np.ndarray], mass: float = 1.0) -> np.ndarray:
    """Evaluate the documented QoI in float64, including during validation."""
    with np.errstate(over="ignore", invalid="ignore"):
        if quantity.endswith("-square") and quantity in QOI_CHOICES:
            return np.square(np.asarray(values[quantity[:2]], dtype=np.float64))
        vx, vy, vz = (np.asarray(values[key], dtype=np.float64) for key in VELOCITY_FIELDS)
        if quantity == "speed":
            return np.hypot(np.hypot(vx, vy), vz)
        if quantity == "kinetic-energy":
            return (0.5 * mass) * (vx * vx + vy * vy + vz * vz)
    raise RuntimeError(f"Unknown QoI: {quantity}.")


def uniform_velocity_bound(quantity: str, tolerance: float, max_speed: float,
                           max_square: float, mass: float) -> float:
    """Sufficient per-component error bound, using stable quadratic roots."""
    if tolerance == 0:
        return 0.0
    if quantity == "speed":
        return tolerance / math.sqrt(3)
    if quantity == "kinetic-energy":
        twice_energy = 2 * (tolerance / mass)
        if twice_energy == 0:
            return 0.0  # Underflow: retaining the source values is sufficient.
        denominator = math.hypot(max_speed, math.sqrt(twice_energy)) + max_speed
        return (twice_energy / denominator) / math.sqrt(3)
    magnitude = math.sqrt(max_square)
    return tolerance / (math.hypot(magnitude, math.sqrt(tolerance)) + magnitude)


def _raw_velocities(manifest: Mapping[str, Any], paths: Mapping[str, str]) -> Dict[str, np.ndarray]:
    return {
        key: np.memmap(paths[key], mode="r", dtype=manifest["fields"][key]["dtype"],
                       shape=(int(manifest["count"]),))
        for key in VELOCITY_FIELDS
    }


def prepare_qoi(args: argparse.Namespace, manifest: Dict[str, Any], paths: Mapping[str, str]) -> None:
    """Resolve the QoI tolerance and tighten only the compressor's bounds."""
    quantity = getattr(args, "qoi", None)
    if quantity in (None, "none"):
        return
    started = time.perf_counter()
    source = _raw_velocities(manifest, paths)
    mass = float(args.qoi_mass)
    minimum, maximum, max_speed = math.inf, -math.inf, 0.0
    for start in range(0, int(manifest["count"]), CHUNK_ROWS):
        values = {key: value[start:start + CHUNK_ROWS] for key, value in source.items()}
        if any(not np.isfinite(value).all() for value in values.values()):
            raise RuntimeError("The QoI frontend requires finite velocity values.")
        q = evaluate_qoi(quantity, values, mass)
        if not np.isfinite(q).all():
            raise RuntimeError("Original QoI values overflow float64; choose different units or mass.")
        minimum, maximum = min(minimum, float(q.min())), max(maximum, float(q.max()))
        if quantity == "kinetic-energy":
            max_speed = max(max_speed, float(evaluate_qoi("speed", values).max()))
    value_range = maximum - minimum
    tolerance = (float(args.qoi_abs_eb) if args.qoi_abs_eb is not None
                 else float(args.qoi_rel_eb) * value_range)
    if not math.isfinite(value_range) or not math.isfinite(tolerance):
        raise RuntimeError("The QoI range/tolerance must fit in float64.")
    bound = uniform_velocity_bound(quantity, tolerance, max_speed, maximum, mass)
    if not math.isfinite(bound):
        raise RuntimeError("The derived velocity bound must fit in float64; choose different units.")
    affected = (quantity[:2],) if quantity.endswith("-square") else VELOCITY_FIELDS
    requested, selected = {}, {}
    for key in VELOCITY_FIELDS:
        field = manifest["field_error_bounds"][key]
        requested[key] = float(field["abs"])
        selected[key] = min(requested[key], bound) if key in affected else requested[key]
        field["compressor_abs"] = selected[key]
    manifest["qoi"] = {
        "version": 1, "state": "prepared", "quantity": quantity,
        "scope": "per_particle", "evaluation_dtype": "float64", "mass": mass,
        "tolerance": {
            "mode": "absolute" if args.qoi_abs_eb is not None else "range_relative",
            "requested": float(args.qoi_abs_eb if args.qoi_abs_eb is not None else args.qoi_rel_eb),
            "absolute": tolerance, "original_min": minimum,
            "original_max": maximum, "original_range": value_range,
        },
        "bound_strategy": "conservative_uniform",
        "requested_velocity_abs_bounds": requested,
        "compressor_velocity_abs_bounds": selected,
    }
    manifest.setdefault("timing", {})["qoi_prepare_wall_seconds"] = time.perf_counter() - started


def _record_dtype(manifest: Mapping[str, Any]) -> np.dtype:
    return np.dtype([("row", "<u8")] + [
        (key, np.dtype(manifest["fields"][key]["dtype"]).newbyteorder("<"))
        for key in VELOCITY_FIELDS
    ])


def validate_and_store_qoi(manifest: Dict[str, Any], paths: Mapping[str, str],
                           order: CanonicalOrder, work_dir: Path, force: bool) -> None:
    """Decode once at encode time; keep exact vectors for any failing rows."""
    qoi = manifest.get("qoi")
    if qoi is None:
        return
    started = time.perf_counter()
    count = int(manifest["count"])
    source = _raw_velocities(manifest, paths)
    correction_path = work_dir / "compressed" / "qoi.corrections.zst"
    require_output_path(correction_path, force)
    record_dtype = _record_dtype(manifest)
    corrected_count, qoi_failures, data_failures = 0, 0, 0
    max_before, max_after, squared_after = 0.0, 0.0, 0.0
    tolerance = float(qoi["tolerance"]["absolute"])
    with tempfile.TemporaryDirectory(prefix="qoi-validation-", dir=work_dir / "preprocessed") as temp:
        decoded_paths = {}
        for key in VELOCITY_FIELDS:
            decoded_paths[key] = str(Path(temp) / f"{key}.raw")
            decompress_lossy_raw(manifest["compressed_fields"][key], decoded_paths[key], True)
        decoded = _raw_velocities(manifest, decoded_paths)
        with correction_path.open("wb") as output, zstd.ZstdCompressor().stream_writer(output) as writer:
            for start in range(0, count, CHUNK_ROWS):
                end = min(count, start + CHUNK_ROWS)
                rows = slice(start, end)
                source_rows = order.values[rows] if order.values is not None else rows
                original = {key: value[source_rows] for key, value in source.items()}
                actual = {key: value[rows] for key, value in decoded.items()}
                expected_q = evaluate_qoi(qoi["quantity"], original, qoi["mass"])
                actual_q = evaluate_qoi(qoi["quantity"], actual, qoi["mass"])
                with np.errstate(over="ignore", invalid="ignore"):
                    error = np.abs(expected_q - actual_q)
                    bad_qoi = ~np.isfinite(error) | (error > tolerance)
                    bad_data = np.zeros(end - start, dtype=bool)
                    for key in VELOCITY_FIELDS:
                        delta = np.abs(original[key].astype(np.float64) - actual[key].astype(np.float64))
                        bad_data |= ~np.isfinite(delta) | (delta > qoi["requested_velocity_abs_bounds"][key])
                bad = bad_qoi | bad_data
                qoi_failures += int(np.count_nonzero(bad_qoi))
                data_failures += int(np.count_nonzero(bad_data))
                corrected_count += int(np.count_nonzero(bad))
                max_before = max(max_before, float(np.where(np.isfinite(error), error, math.inf).max()))
                remaining = np.where(bad, 0.0, error)
                max_after = max(max_after, float(remaining.max()))
                squared_after += float(np.dot(remaining, remaining))
                if bad.any():
                    records = np.empty(int(np.count_nonzero(bad)), dtype=record_dtype)
                    records["row"] = np.flatnonzero(bad).astype(np.uint64) + start
                    for key in VELOCITY_FIELDS:
                        records[key] = original[key][bad]
                    writer.write(records.tobytes())
        del decoded
    corrections = None
    if corrected_count:
        corrections = {
            "format": CORRECTION_FORMAT, "path": str(correction_path),
            "row_order": "reconstructed", "count": corrected_count,
            "bytes": correction_path.stat().st_size,
        }
        manifest["artifacts"]["compressed"]["qoi_corrections"] = str(correction_path)
    else:
        correction_path.unlink()
    qoi.update({
        "state": "validated", "corrections": corrections,
        "validation": {
            "count": count, "qoi_violations_before": qoi_failures,
            "data_violations_before": data_failures,
            "corrected_particles": corrected_count,
            "max_abs_error_before": max_before, "max_abs_error": max_after,
            "rmse": math.sqrt(squared_after / count), "satisfied": max_after <= tolerance,
        },
    })
    manifest.setdefault("timing", {})["qoi_validate_wall_seconds"] = time.perf_counter() - started


def apply_qoi_corrections(manifest: Mapping[str, Any], output_h5: Path) -> None:
    """Apply packaged vectors after reconstruction, with no source-data access."""
    qoi = manifest.get("qoi")
    if qoi is None:
        return
    if qoi.get("version") != 1 or qoi.get("state") != "validated":
        raise RuntimeError("Unsupported or unvalidated QoI package.")
    corrections = qoi.get("corrections")
    if corrections is None:
        if qoi["validation"]["corrected_particles"]:
            raise RuntimeError("QoI package is missing correction metadata.")
        return
    if corrections.get("format") != CORRECTION_FORMAT or corrections.get("row_order") != "reconstructed":
        raise RuntimeError("Unsupported QoI correction format/order.")
    dtype = _record_dtype(manifest)
    total, previous_row = 0, -1
    try:
        with open(corrections["path"], "rb") as source, h5py.File(output_h5, "r+") as output:
            with zstd.ZstdDecompressor().stream_reader(source) as reader, io.BufferedReader(reader) as buffered:
                while payload := buffered.read(CHUNK_ROWS * dtype.itemsize):
                    if len(payload) % dtype.itemsize:
                        raise RuntimeError("Truncated QoI correction records.")
                    records = np.frombuffer(payload, dtype=dtype)
                    rows = records["row"]
                    if (int(rows[0]) <= previous_row or np.any(rows[1:] <= rows[:-1])
                            or int(rows[-1]) >= int(manifest["count"])):
                        raise RuntimeError("Invalid QoI correction row indices.")
                    previous_row = int(rows[-1])
                    # h5py fancy writes are expensive for large correction sets.
                    # Update bounded contiguous blocks, including for dense masks.
                    blocks = rows // CHUNK_ROWS
                    boundaries = np.r_[0, np.flatnonzero(blocks[1:] != blocks[:-1]) + 1, len(rows)]
                    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
                        start = int(blocks[lo]) * CHUNK_ROWS
                        end = min(int(manifest["count"]), start + CHUNK_ROWS)
                        offsets = (rows[lo:hi] - start).astype(np.intp)
                        for key in VELOCITY_FIELDS:
                            dataset = output[manifest["fields"][key]["h5_path"]]
                            values = dataset[start:end]
                            values[offsets] = records[key][lo:hi]
                            dataset[start:end] = values
                    total += len(records)
    except (OSError, zstd.ZstdError) as exc:
        raise RuntimeError(f"Could not read QoI corrections: {exc}") from exc
    if total != int(corrections["count"]) or total != int(qoi["validation"]["corrected_particles"]):
        raise RuntimeError("QoI correction count does not match the manifest.")


def measure_reconstructed_qoi(original: h5py.File, reconstructed: h5py.File,
                              manifest: Mapping[str, Any], order: Optional[np.ndarray]) -> Dict[str, Any]:
    """Independently measure the final HDF5 values when --metrics is requested."""
    qoi = manifest["qoi"]
    count = int(manifest["count"])
    raw_paths = manifest.get("artifacts", {}).get("preprocessed", {})
    # Normal roundtrips still have these memory-mapped source exports. The
    # fallback also permits metrics on packages with no preprocessing workspace.
    source = {
        key: (np.memmap(raw_paths[key], mode="r", dtype=manifest["fields"][key]["dtype"], shape=(count,))
              if key in raw_paths and Path(raw_paths[key]).is_file()
              else original[manifest["fields"][key]["h5_path"]][:count])
        for key in VELOCITY_FIELDS
    }
    maximum, sum_squared, violations = 0.0, 0.0, 0
    for start in range(0, count, CHUNK_ROWS):
        rows = slice(start, min(count, start + CHUNK_ROWS))
        source_rows = order[rows] if order is not None else rows
        expected = evaluate_qoi(qoi["quantity"], {key: value[source_rows] for key, value in source.items()}, qoi["mass"])
        actual = evaluate_qoi(qoi["quantity"], {
            key: reconstructed[manifest["fields"][key]["h5_path"]][rows] for key in VELOCITY_FIELDS
        }, qoi["mass"])
        with np.errstate(over="ignore", invalid="ignore"):
            error = np.abs(expected - actual)
            error = np.where(np.isfinite(error), error, math.inf)
        maximum = max(maximum, float(error.max()))
        sum_squared += float(np.dot(error, error))
        violations += int(np.count_nonzero(error > qoi["tolerance"]["absolute"]))
    return {"count": count, "max_abs_error": maximum, "rmse": math.sqrt(sum_squared / count),
            "violations": violations, "satisfied": violations == 0}


def print_qoi_summary(report: Mapping[str, Any]) -> None:
    qoi = report.get("qoi")
    if qoi is None:
        return
    print(f"QoI: {qoi['quantity']} (per particle), abs_tolerance={qoi['tolerance']['absolute']:.6g}")
    selected = qoi["compressor_velocity_abs_bounds"]
    print("  SZO abs bounds: " + ", ".join(f"{key}={selected[key]:.6g}" for key in VELOCITY_FIELDS))
    validation = qoi.get("validation")
    if validation:
        measured = qoi.get("reconstruction_validation", validation)
        correction_bytes = (qoi.get("corrections") or {}).get("bytes", 0)
        print(f"  max_abs_error={measured['max_abs_error']:.6g}, "
              f"corrected_particles={validation['corrected_particles']}/{validation['count']}, "
              f"correction_bytes={correction_bytes}, bound_ok={measured['satisfied']}")
