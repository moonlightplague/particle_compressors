"""Per-axis XnYZip bounds, package-size tuning, and compact outlier streams.

Corrections address final source-dtype values in canonical particle order.
Keeping the native decoded floats intact also preserves structure-aware orders.
Most outliers need only a small signed multiple of twice the requested bound;
source bits are stored only when rounding prevents that correction from working.
"""

import json
import math
from pathlib import Path
import struct
import tempfile
from typing import Mapping

import numpy as np
import zstandard as zstd

from src.constants import POSITION_FIELDS, VELOCITY_FIELDS
from src.models import ToolPaths
from src.position_ties import sort_position_ties
from src.raw_codecs import SZO_ADAPTIVE_1D_PROFILE, compress_szo_raw
from src.runtime import require_output_path
from src.xnyzip_codec import compress_xnyzip_triplet, run_xnyzip_decompress


OUTLIER_MAGIC = b"XNYLINF1"
OUTLIER_HEADER = struct.Struct("<8sQ")
OUTLIER_BLOCK = struct.Struct("<IBBBII")
OUTLIER_BLOCK_VALUES = 1 << 20
TUNING_VALUES = 1 << 18


def linf_axis_scales(bounds):
    positive = [float(value) for value in bounds.values() if value > 0]
    reference = min(positive) if positive else 1.0
    return {k: reference / float(bounds[k]) if bounds[k] > 0 else 1.0 for k in POSITION_FIELDS}


def estimate_linf_l2_bound(bounds, preprocess_errors, ranges, axis_scales=None) -> float:
    """TO's covering radius is b, but its maximum axis error is 2b/sqrt(5).

    Start at sqrt(5)/2 times the smallest remaining per-axis budget. A native
    lattice floor handles zero/very small budgets, repaired with exact escapes.
    """
    factors = axis_scales or dict.fromkeys(POSITION_FIELDS, 1.0)
    remaining = min(max(0.0, bounds[k] - preprocess_errors[k]) * factors[k] for k in POSITION_FIELDS)
    lattice_floor = max(ranges[k] * factors[k] for k in POSITION_FIELDS) * math.sqrt(5) / (2 * 2_097_150) * 1.001
    return max(math.sqrt(5) / 2 * remaining, lattice_floor, float(np.finfo(np.float32).tiny))


def scale_native_positions(path, count, axis_scales):
    """Normalize only the interleaved native input; keep source raws intact."""
    if all(value == 1.0 for value in axis_scales.values()):
        return
    values = np.memmap(path, mode="r+", dtype="float32", shape=(count, 3))
    factors = np.array([axis_scales[k] for k in POSITION_FIELDS], dtype=np.float64)
    for start in range(0, count, OUTLIER_BLOCK_VALUES):
        end = min(count, start + OUTLIER_BLOCK_VALUES)
        values[start:end] = values[start:end].astype(np.float64) * factors
    values.flush()


def reconstruct_source_positions(decoded, dtype, scale):
    """Match HDF5 reconstruction, including integer rounding and clipping."""
    dtype = np.dtype(dtype)
    values = decoded.astype(np.float64) * scale
    if np.issubdtype(dtype, np.integer):
        limits = np.iinfo(dtype)
        values = np.clip(np.rint(values), limits.min, limits.max)
    return values.astype(dtype)


def _errors(original, reconstructed, scale):
    # Compute in compressor units, just as roundtrip metrics do. Division before
    # subtraction matters at tight bounds on fixed-point and scaled float data.
    return np.abs(original.astype(np.float64) / scale - reconstructed.astype(np.float64) / scale)


def make_outlier_codes(decoded, original, bound, scale):
    """Return zero/signed-step/escape codes and verify the final strict bound."""
    if not math.isfinite(bound) or bound < 0 or not math.isfinite(scale) or scale <= 0:
        raise RuntimeError("Invalid XnYZip Linf bound or position scale.")
    if not np.isfinite(original).all():
        raise RuntimeError("XnYZip Linf correction requires finite source positions.")
    base = reconstruct_source_positions(decoded, original.dtype, scale)
    error = _errors(original, base, scale)
    outliers = ~np.isfinite(error) | (error > bound)
    codes = np.zeros(len(base), dtype=np.uint8)
    step = 2 * bound * scale
    if not math.isfinite(step):
        raise RuntimeError("XnYZip Linf correction step is not finite.")
    if step > 0:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            multiples = np.rint((original[outliers].astype(np.float64) - base[outliers].astype(np.float64)) / step)
        multiples = np.clip(np.nan_to_num(multiples), -127, 127).astype(np.int16)
        codes[outliers] = np.where(multiples > 0, 2 * multiples - 1, -2 * multiples).astype(np.uint8)
    corrected = apply_outlier_codes(base, codes, 255, np.empty(0, dtype=original.dtype), step)
    remaining = _errors(original, corrected, scale)
    escapes = outliers & (~np.isfinite(remaining) | (remaining > bound))
    escape_symbol = int(codes[~escapes].max(initial=0)) + 1
    codes[escapes] = escape_symbol
    exact = original[escapes].copy()
    corrected[escapes] = exact
    maximum = float(_errors(original, corrected, scale).max(initial=0))
    if not math.isfinite(maximum) or maximum > bound:
        raise RuntimeError("XnYZip outlier correction failed strict Linf validation.")
    return codes, escape_symbol, exact, {
        "outlier_count": int(outliers.sum()),
        "exact_count": int(escapes.sum()),
        "validated_max_abs_error": maximum,
    }


def apply_outlier_codes(values, codes, escape_symbol, exact, step):
    output = values.copy()
    mask = (codes != 0) & (codes != escape_symbol)
    if mask.any():
        symbols = codes[mask].astype(np.int16)
        multiples = np.where(symbols & 1, (symbols + 1) // 2, -(symbols // 2))
        corrected = values[mask].astype(np.float64) + multiples * step
        if np.issubdtype(values.dtype, np.integer):
            limits = np.iinfo(values.dtype)
            corrected = np.clip(np.rint(corrected), limits.min, limits.max)
        output[mask] = corrected.astype(values.dtype)
    escapes = codes == escape_symbol
    if int(escapes.sum()) != len(exact):
        raise RuntimeError("XnYZip outlier escape count mismatch.")
    output[escapes] = exact
    return output


def _encode_block(codes, escape_symbol, exact):
    compressor = zstd.ZstdCompressor(level=9)
    dense = compressor.compress(codes.tobytes())
    width = max(1, int(codes.max(initial=0)).bit_length())
    bits = ((codes[:, None] >> np.arange(width, dtype=np.uint8)) & 1).reshape(-1)
    packed = compressor.compress(np.packbits(bits, bitorder="little").tobytes())
    encoding, payload = (1, packed) if len(packed) < len(dense) else (0, dense)
    exact_payload = compressor.compress(exact.tobytes()) if len(exact) else b""
    return (OUTLIER_BLOCK.pack(len(codes), encoding, width, escape_symbol, len(payload), len(exact_payload))
            + payload + exact_payload)


def write_linf_outliers(path, decoded, source, order, bound, scale, force=False):
    """Write bounded-memory frames; omit the entire stream when no outlier exists."""
    path = Path(path)
    require_output_path(path, force)
    stats = {"outlier_count": 0, "exact_count": 0, "validated_max_abs_error": 0.0}
    with path.open("wb") as output:
        output.write(OUTLIER_HEADER.pack(OUTLIER_MAGIC, len(order)))
        for start in range(0, len(order), OUTLIER_BLOCK_VALUES):
            end = min(len(order), start + OUTLIER_BLOCK_VALUES)
            original = source[order[start:end]]
            codes, escape_symbol, exact, block_stats = make_outlier_codes(decoded[start:end], original, bound, scale)
            stats["outlier_count"] += block_stats["outlier_count"]
            stats["exact_count"] += block_stats["exact_count"]
            stats["validated_max_abs_error"] = max(stats["validated_max_abs_error"], block_stats["validated_max_abs_error"])
            output.write(_encode_block(codes, escape_symbol, exact))
    if stats["outlier_count"] == 0:
        path.unlink()
        return None, stats
    return {
        "container": "xnyzip_linf_outliers_v1",
        "path": str(path), "bytes": path.stat().st_size,
        "dtype": source.dtype.str, "count": len(order),
        "step_in_source_units": 2 * bound * scale,
        "abs_error_bound": bound, **stats,
    }, stats


def restore_linf_outliers(values, metadata):
    """Apply canonical-order corrections before any position-order restoration."""
    if metadata.get("container") != "xnyzip_linf_outliers_v1":
        raise RuntimeError("Unsupported XnYZip Linf outlier container.")
    if np.dtype(metadata["dtype"]) != values.dtype or int(metadata["count"]) != len(values):
        raise RuntimeError("XnYZip Linf outlier dtype or count mismatch.")
    step = float(metadata["step_in_source_units"])
    if not math.isfinite(step) or step < 0:
        raise RuntimeError("Invalid XnYZip Linf outlier step.")
    decompressor = zstd.ZstdDecompressor()
    with Path(metadata["path"]).open("rb") as stream:
        header = stream.read(OUTLIER_HEADER.size)
        if len(header) != OUTLIER_HEADER.size or OUTLIER_HEADER.unpack(header) != (OUTLIER_MAGIC, len(values)):
            raise RuntimeError("Invalid XnYZip Linf outlier header.")
        start = 0
        while start < len(values):
            header = stream.read(OUTLIER_BLOCK.size)
            if len(header) != OUTLIER_BLOCK.size:
                raise RuntimeError("Truncated XnYZip Linf outlier block.")
            count, encoding, width, escape_symbol, code_bytes, exact_bytes = OUTLIER_BLOCK.unpack(header)
            if not 0 < count <= min(OUTLIER_BLOCK_VALUES, len(values) - start) or encoding not in (0, 1) or not 1 <= width <= 8 or escape_symbol == 0:
                raise RuntimeError("Invalid XnYZip Linf outlier block.")
            size = count if encoding == 0 else (count * width + 7) // 8
            try:
                raw = decompressor.decompress(stream.read(code_bytes), max_output_size=size)
                if len(raw) != size:
                    raise RuntimeError("Invalid XnYZip Linf code length.")
                if encoding == 0:
                    codes = np.frombuffer(raw, dtype=np.uint8)
                else:
                    bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8), bitorder="little")[:count * width]
                    codes = np.sum(bits.reshape(count, width) << np.arange(width, dtype=np.uint8), axis=1, dtype=np.uint8)
                exact_count = int(np.count_nonzero(codes == escape_symbol))
                if exact_count:
                    exact_raw = decompressor.decompress(stream.read(exact_bytes), max_output_size=exact_count * values.dtype.itemsize)
                    if len(exact_raw) != exact_count * values.dtype.itemsize:
                        raise RuntimeError("Invalid XnYZip Linf escape length.")
                    exact = np.frombuffer(exact_raw, dtype=values.dtype)
                elif exact_bytes:
                    raise RuntimeError("Unexpected XnYZip Linf escape payload.")
                else:
                    exact = np.empty(0, dtype=values.dtype)
            except zstd.ZstdError as exc:
                raise RuntimeError("Corrupt XnYZip Linf outlier payload.") from exc
            values[start:start + count] = apply_outlier_codes(values[start:start + count], codes, escape_symbol, exact, step)
            start += count
        if stream.read(1):
            raise RuntimeError("Trailing XnYZip Linf outlier bytes.")
    return values


def tune_linf_l2_bound(tools: ToolPaths, raw_paths: Mapping, manifest: Mapping, workspace: Path, tie_sort: bool):
    """Score native positions + actual corrections + SZO velocities on a sample.

    Contiguous blocks spread through the input retain local particle clustering.
    This is an estimate of total bytes, not a claim of a global optimum.
    """
    count = int(manifest["count"])
    sample_count = min(count, TUNING_VALUES)
    if count <= TUNING_VALUES:
        sample = np.arange(count, dtype=np.intp)
    else:
        block = 4096
        starts = np.linspace(0, count - block, TUNING_VALUES // block, dtype=np.intp)
        sample = (starts[:, None] + np.arange(block)).reshape(-1)
    bounds = {k: float(manifest["field_error_bounds"][k]["abs"]) for k in POSITION_FIELDS}
    scale = float(manifest["position_scale"]["value"])
    estimate = float(manifest["error_bounds"]["positions_xnyzip_abs"])
    normalized = linf_axis_scales(bounds)
    scalings = [normalized]
    if any(value != 1.0 for value in normalized.values()):
        scalings.append(dict.fromkeys(POSITION_FIELDS, 1.0))
    sources = {k: np.memmap(raw_paths[f"{k}_source"], mode="r", dtype=manifest["fields"][k]["dtype"], shape=(count,))[sample]
               for k in POSITION_FIELDS}
    positions = {k: np.memmap(raw_paths[k], mode="r", dtype="float32", shape=(count,))[sample]
                 for k in POSITION_FIELDS}
    velocities = {k: np.memmap(raw_paths[k], mode="r", dtype=manifest["fields"][k]["dtype"], shape=(count,))[sample]
                  for k in VELOCITY_FIELDS}
    rows = []
    with tempfile.TemporaryDirectory(prefix="xnyzip-linf-tune-", dir=workspace) as temp:
        root = Path(temp)
        raw = root / "positions.raw"
        candidates = []
        for factors in scalings:
            envelope = max(estimate, math.hypot(*(bounds[k] * factors[k] for k in POSITION_FIELDS)))
            for quantizer, bound in dict.fromkeys([
                ("to", estimate), ("cube", estimate * math.sqrt(12 / 5)),
                ("to", max(estimate, .75 * envelope)), ("to", envelope),
                ("cube", envelope), ("to", 1.25 * envelope),
            ]):
                candidates.append((quantizer, bound, factors))
        for quantizer, bound, factors in candidates:
            try:
                np.column_stack([positions[k].astype(np.float64) * factors[k] for k in POSITION_FIELDS]).astype('float32').tofile(raw)
                archive = root / "positions.xnyzip"
                order = compress_xnyzip_triplet(tools, str(raw), str(archive), sample_count, bound, root / "order.raw", True, quantizer=quantizer)
                paths = {k: str(root / f"{k}.raw") for k in POSITION_FIELDS}
                run_xnyzip_decompress(tools, str(archive), paths, POSITION_FIELDS, sample_count, bound, root / "decoded.raw", True, quantizer=quantizer,
                                     axis_scales=factors)
                decoded = {k: np.fromfile(paths[k], dtype="float32") for k in POSITION_FIELDS}
                squared = sum(((decoded[k].astype(np.float64) - positions[k][order].astype(np.float64)) * factors[k]) ** 2 for k in POSITION_FIELDS)
                rounding = 8 * np.finfo(np.float32).eps * max(float(np.abs(a).max()) for a in positions.values())
                if not np.isfinite(squared).all() or math.sqrt(float(squared.max())) > bound + rounding:
                    continue  # Reject unsigned boundary-node corruption.
                if tie_sort:
                    sort_position_ties(order, decoded, velocities)
                correction_bytes = 0
                outlier_metadata = {}
                validation_metadata = {}
                for k in POSITION_FIELDS:
                    metadata, stats = write_linf_outliers(root / f"{k}.outliers", decoded[k], sources[k], order, bounds[k], scale, True)
                    correction_bytes += metadata["bytes"] if metadata else 0
                    validation_metadata[k] = stats
                    if metadata:
                        metadata["path"] = str(Path(manifest["artifacts"]["compressed"]["positions"]).parent / f"{k}.outliers")
                        metadata["count"] = count
                        outlier_metadata[k] = metadata
                velocity_bytes = 0
                for k in VELOCITY_FIELDS:
                    velocities[k][order].tofile(root / "velocity.raw")
                    field = compress_szo_raw(str(root / "velocity.raw"), str(velocities[k].dtype), str(root / "velocity.szo"), k, sample_count,
                                             float(manifest["field_error_bounds"][k]["abs"]), True,
                                             profile=SZO_ADAPTIVE_1D_PROFILE if tie_sort else None)
                    velocity_bytes += field["bytes"]
                position_bytes = archive.stat().st_size
                # Sidecars also add manifest entries. This matters on small
                # inputs where a tiny stream saving can cost more metadata.
                metadata_bytes = len(json.dumps({
                    "linf_outliers": outlier_metadata, "linf_validation": validation_metadata,
                    "axis_scales": factors, "quantizer": quantizer, "l2_error_bound": bound,
                    "artifacts": {f"{k}_outliers": field["path"] for k, field in outlier_metadata.items()},
                }, indent=2, sort_keys=True).encode("utf-8"))
                rows.append({"quantizer": quantizer, "l2_error_bound": bound, "axis_scales": factors,
                             "position_bytes": position_bytes, "outlier_bytes": correction_bytes,
                             "velocity_bytes": velocity_bytes, "metadata_bytes": metadata_bytes,
                             "total_bytes": position_bytes + correction_bytes + velocity_bytes + metadata_bytes})
            except RuntimeError:
                # A candidate can hit a native lattice limit. The full encoder
                # still validates the selected stream and reports codec failures.
                continue
    if not rows:
        return "to", estimate, normalized, {"sample_count": sample_count, "candidates": [], "fallback": "no_valid_candidate"}
    best = min(rows, key=lambda row: row["total_bytes"])
    return best["quantizer"], best["l2_error_bound"], best["axis_scales"], {"sample_count": sample_count, "candidates": rows, "selected": best}
