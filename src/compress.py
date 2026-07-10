import argparse
import numpy as np
import time
import math

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import src.helpers as hp


def compress_pcodec_raw_parts(
    raw_path: str,
    dtype: str,
    cmp_dir: Path,
    field_name: str,
    count: int,
    part_size: int,
    force: bool,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    standalone, ChunkConfig = hp.load_pcodec()
    part_dir = cmp_dir / f"{field_name}.pcodecparts"
    hp.prepare_output_dir(part_dir, force)
    dt = np.dtype(dtype)
    records: List[Dict[str, Any]] = []
    parts: List[Dict[str, Any]] = []
    start = 0
    for part_index, chunk in enumerate(hp.raw_chunk_reader(raw_path, dt, count, part_size)):
        part_cmp = part_dir / f"part{part_index:06d}.pco"
        hp.require_output_path(part_cmp, force)
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
        raise RuntimeError(f"pcodec part compression for {field_name} saw {start} values, expected {count}.")

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

def pysz_encoded_chunk(chunk: np.ndarray) -> Tuple[np.ndarray, int]:
    if chunk.size >= hp.PYSZ_MIN_VALUES:
        return np.ascontiguousarray(chunk), int(chunk.size)
    encoded_count = hp.PYSZ_MIN_VALUES
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
    PyszSZ, PyszConfig, PyszErrorBoundMode = hp.load_pysz()
    part_dir = cmp_dir / f"{field_name}.pyszparts"
    hp.prepare_output_dir(part_dir, force)
    dt = np.dtype(dtype)
    if dt not in (np.dtype("float32"), np.dtype("float64")):
        raise RuntimeError(f"pysz velocity compression expected float32/float64, got {dt} for {field_name}.")
    records: List[Dict[str, Any]] = []
    parts: List[Dict[str, Any]] = []
    start = 0
    for part_index, chunk in enumerate(hp.raw_chunk_reader(raw_path, dt, count, part_size)):
        part_cmp = part_dir / f"part{part_index:06d}.psz"
        hp.require_output_path(part_cmp, force)
        encoded, encoded_count = pysz_encoded_chunk(chunk)
        config = PyszConfig(encoded.shape)
        config.errorBoundMode = PyszErrorBoundMode.ABS
        config.absErrorBound = float(abs_eb)
        start_time = time.perf_counter()
        try:
            compressed, reported_ratio = PyszSZ.compress(encoded, config)
        except Exception as exc:
            raise RuntimeError(
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
        raise RuntimeError(f"pysz part compression for {field_name} saw {start} values, expected {count}.")

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

def compress(args: argparse.Namespace, 
             manifest: Dict[str, Any], 
             raw_paths: Dict[str, str],
             tools: hp.ToolPaths) -> Dict[str, Any]:
    
    work_dir = Path(args.work_dir).resolve()
    order_raw = work_dir / "preprocessed" / "order.i32.raw"
    hp.require_output_path(order_raw, args.force)
    raw_paths["order"] = str(order_raw)

    commands: Dict[str, Any] = {}
    cmp_dir = work_dir / "compressed"
    lcp_cmp = cmp_dir / "positions.lcp"
    hp.require_output_path(lcp_cmp, args.force)
    commands["lcp_compress_positions"] = hp.run_command(
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

    for logical in hp.VELOCITY_FIELDS:
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
    hp.update_compressed_size_metrics(manifest, work_dir)
    hp.write_json(work_dir / "manifest.json", manifest, force=True)
    return manifest