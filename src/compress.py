import argparse
import numpy as np
import time
import math

from pathlib import Path
from typing import Any, Dict, Tuple

import src.helpers as hp


def compress_pcodec_raw(
    raw_path: str,
    dtype: str,
    compressed_path: str,
    field_name: str,
    count: int,
    force: bool,
) -> Tuple[Dict[str, Any]]:
    standalone, ChunkConfig = hp.load_pcodec()
    dt = np.dtype(dtype)
    output = Path(compressed_path)
    hp.require_output_path(output, force)
    values = np.ascontiguousarray(hp.read_raw(raw_path, dt, count))
    payload = standalone.simple_compress(values, ChunkConfig())
    output.write_bytes(payload)
    compressed_bytes = len(payload)
    return {
        "field": field_name,
        "codec": "pcodec",
        "dtype": str(dt),
        "count": count,
        "path": str(output),
        "bytes": compressed_bytes,
    }

def pysz_encoded_values(values: np.ndarray) -> Tuple[np.ndarray, int]:
    if values.size >= hp.PYSZ_MIN_VALUES:
        return np.ascontiguousarray(values), int(values.size)
    encoded_count = hp.PYSZ_MIN_VALUES
    padded = np.empty(encoded_count, dtype=values.dtype)
    padded[: values.size] = values
    fill_value = values[-1] if values.size else np.asarray(0, dtype=values.dtype)
    padded[values.size :] = fill_value
    return padded, encoded_count

def compress_pysz_raw(
    raw_path: str,
    dtype: str,
    compressed_path: str,
    field_name: str,
    count: int,
    abs_eb: float,
    force: bool,
) -> Tuple[Dict[str, Any]]:
    PyszSZ, PyszConfig, PyszErrorBoundMode = hp.load_pysz()
    dt = np.dtype(dtype)
    if dt not in (np.dtype("float32"), np.dtype("float64")):
        raise RuntimeError(f"pysz velocity compression expected float32/float64, got {dt} for {field_name}.")
    output = Path(compressed_path)
    hp.require_output_path(output, force)
    values = hp.read_raw(raw_path, dt, count)
    encoded, encoded_count = pysz_encoded_values(values)
    config = PyszConfig(encoded.shape)
    config.errorBoundMode = PyszErrorBoundMode.ABS
    config.absErrorBound = float(abs_eb)
    try:
        compressed, _ = PyszSZ.compress(encoded, config)
    except Exception as exc:
        raise RuntimeError(
            f"pysz compression failed for {field_name} with {count} values "
            f"encoded as {encoded_count} values."
        ) from exc
    compressed = np.ascontiguousarray(compressed, dtype=np.uint8)
    compressed.tofile(output)
    compressed_bytes = int(compressed.size)
    return {
        "field": field_name,
        "codec": "pysz",
        "dtype": str(dt),
        "abs_error_bound": abs_eb,
        "count": count,
        "encoded_count": encoded_count,
        "path": str(output),
        "bytes": compressed_bytes,
    }

def compress(args: argparse.Namespace, 
             manifest: Dict[str, Any], 
             raw_paths: Dict[str, str],
             tools: hp.ToolPaths) -> Dict[str, Any]:
    
    work_dir = Path(args.work_dir).resolve()
    order_raw = work_dir / "preprocessed" / "order.i32.raw"
    hp.require_output_path(order_raw, args.force)
    raw_paths["order"] = str(order_raw)

    cmp_dir = work_dir / "compressed"
    lcp_cmp = cmp_dir / "positions.lcp"
    hp.require_output_path(lcp_cmp, args.force)
    hp.run_command(
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

    artifacts = manifest["artifacts"]["compressed"]
    manifest["compressed_fields"]["order"]= compress_pcodec_raw(
        str(order_raw),
        "int32",
        artifacts["order"],
        "order",
        manifest["count"],
        args.force,
    )

    manifest["compressed_fields"]["id"]= compress_pcodec_raw(
        raw_paths["id"],
        manifest["fields"]["id"]["dtype"],
        artifacts["id"],
        "id",
        manifest["count"],
        args.force,
    )

    for logical in hp.VELOCITY_FIELDS:
        dtype = manifest["fields"][logical]["dtype"]
        manifest["compressed_fields"][logical]= compress_pysz_raw(
            raw_paths[logical],
            dtype,
            artifacts[logical],
            logical,
            manifest["count"],
            manifest["field_error_bounds"][logical]["abs"],
            args.force,
        )

    hp.update_compressed_size_metrics(manifest, work_dir)
    hp.write_json(work_dir / "manifest.json", manifest, force=True)
    return manifest
