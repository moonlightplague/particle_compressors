"""XnYZip command construction, interleaved decoding, and order validation."""

import os
from pathlib import Path
from typing import Mapping, Tuple

import numpy as np

from src.models import ToolPaths
from src.runtime import read_raw, require_output_path, run_command


FieldTriplet = Tuple[str, str, str]

XNYZIP_QUANTIZER = "to"
XNYZIP_CURVE = "-z"
XNYZIP_STORAGE_MODE = "-rle"
# The submodule's monolithic runner also forces block mode because its direct
# encoder is not reliable for every quantized range.
XNYZIP_DIRECT_THRESHOLD = 0
XNYZIP_ORDER_DTYPE = np.dtype("uint64")


def read_xnyzip_permutation(order_path: str, count: int) -> np.ndarray:
    order = read_raw(order_path, XNYZIP_ORDER_DTYPE, count)
    if count and (int(order.min()) < 0 or int(order.max()) >= count):
        raise RuntimeError(
            "XnYZip order is not a valid index range for this particle count."
        )
    seen = np.zeros(count, dtype=np.bool_)
    seen[order] = True
    if not bool(seen.all()):
        raise RuntimeError(
            "XnYZip order is not a permutation of the particle rows."
        )
    return order


def compress_xnyzip_triplet(
    tools: ToolPaths,
    input_path: str,
    compressed_path: str,
    count: int,
    l2_error_bound: float,
    order_path: Path,
    force: bool,
) -> np.ndarray:
    bound = _positive_l2_bound(l2_error_bound)
    _validate_interleaved_size(Path(input_path), count, "input")
    compressed = Path(compressed_path)
    require_output_path(compressed, force)
    require_output_path(order_path, force)
    run_command(
        [
            str(_xnyzip_tool(tools)),
            "--compress",
            input_path,
            str(compressed),
            str(order_path),
            XNYZIP_QUANTIZER,
            str(bound),
            XNYZIP_CURVE,
            XNYZIP_STORAGE_MODE,
            str(XNYZIP_DIRECT_THRESHOLD),
        ]
    )
    if not compressed.is_file() or not compressed.stat().st_size:
        raise RuntimeError("XnYZip did not produce a compressed stream.")
    return read_xnyzip_permutation(str(order_path), count)


def run_xnyzip_decompress(
    tools: ToolPaths,
    compressed_path: str,
    output_paths: Mapping[str, str],
    fields: FieldTriplet,
    count: int,
    l2_error_bound: float,
    interleaved_path: Path,
    force: bool,
) -> None:
    bound = _positive_l2_bound(l2_error_bound)
    require_output_path(interleaved_path, force)
    for field in fields:
        require_output_path(Path(output_paths[field]), force)
    run_command(
        [
            str(_xnyzip_tool(tools)),
            "--decompress",
            compressed_path,
            str(interleaved_path),
            XNYZIP_QUANTIZER,
            str(bound),
            XNYZIP_STORAGE_MODE,
        ]
    )
    _validate_interleaved_size(interleaved_path, count, "decompressed")
    interleaved = np.memmap(
        interleaved_path,
        dtype=np.float32,
        mode="r",
        shape=(count, len(fields)),
    )
    try:
        for axis, field in enumerate(fields):
            np.ascontiguousarray(interleaved[:, axis]).tofile(
                output_paths[field]
            )
    finally:
        del interleaved
        interleaved_path.unlink(missing_ok=True)


def _xnyzip_tool(tools: ToolPaths) -> Path:
    path = getattr(tools, "xnyzip", None)
    if path is None:
        raise RuntimeError("XnYZip executable path is not configured.")
    executable = Path(path)
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise RuntimeError(
            f"XnYZip executable is missing or not executable: {executable}"
        )
    return executable


def _positive_l2_bound(value: float) -> float:
    bound = float(value)
    if bound <= 0.0:
        raise RuntimeError("XnYZip L2 error bound must be positive.")
    return bound


def _validate_interleaved_size(
    path: Path,
    count: int,
    label: str,
) -> None:
    expected = count * 3 * np.dtype(np.float32).itemsize
    actual = path.stat().st_size if path.is_file() else -1
    if actual != expected:
        raise RuntimeError(
            f"XnYZip {label} stream expected {expected} bytes for {count} "
            f"particles, got {actual}."
        )


__all__ = [
    "XNYZIP_CURVE",
    "XNYZIP_DIRECT_THRESHOLD",
    "XNYZIP_ORDER_DTYPE",
    "XNYZIP_QUANTIZER",
    "XNYZIP_STORAGE_MODE",
    "compress_xnyzip_triplet",
    "read_xnyzip_permutation",
    "run_xnyzip_decompress",
]
