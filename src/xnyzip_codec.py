"""XnYZip command construction, chunk containers, and order validation."""

import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Dict, Mapping, Tuple

import numpy as np

from src.constants import (
    XNYZIP_CHUNK_CONTAINER,
    XNYZIP_CHUNK_ENTRY,
    XNYZIP_CHUNK_HEADER,
    XNYZIP_CHUNK_MAGIC,
)
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


@dataclass(frozen=True)
class CompressionChunk:
    index: int
    start: int
    count: int


@dataclass(frozen=True)
class DecompressionChunk:
    start: int
    count: int
    archive: Path
    outputs: Dict[str, str]


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


def compress_chunked_xnyzip_triplet(
    tools: ToolPaths,
    input_path: str,
    compressed_path: str,
    count: int,
    chunk_size: int,
    l2_error_bound: float,
    order_path: Path,
    force: bool,
    workers: int = 1,
) -> Dict[str, int]:
    """Compress independent interleaved chunks into a deterministic container."""

    _validate_chunk_options(chunk_size, workers, "compression")
    bound = _positive_l2_bound(l2_error_bound)
    source_path = Path(input_path)
    _validate_interleaved_size(source_path, count, "input")
    output = Path(compressed_path)
    require_output_path(output, force)
    require_output_path(order_path, force)

    chunks = _build_chunks(count, chunk_size)
    effective_workers = min(workers, len(chunks))
    source = np.memmap(
        source_path,
        dtype=np.float32,
        mode="r",
        shape=(count, 3),
    )
    local_order = np.memmap(
        order_path,
        dtype=XNYZIP_ORDER_DTYPE,
        mode="w+",
        shape=(count,),
    )

    with tempfile.TemporaryDirectory(
        prefix="velocity_xnyzip_chunks_",
        dir=order_path.parent,
    ) as temp:
        temp_dir = Path(temp)

        def compress_chunk(
            chunk: CompressionChunk,
        ) -> Tuple[CompressionChunk, Path, Path]:
            chunk_dir = temp_dir / f"chunk_{chunk.index:08d}"
            chunk_dir.mkdir()
            chunk_input = chunk_dir / "velocities.f32.raw"
            chunk_archive = chunk_dir / "chunk.xnyzip"
            chunk_order = chunk_dir / "order.u64.raw"
            end = chunk.start + chunk.count
            np.ascontiguousarray(source[chunk.start:end]).tofile(chunk_input)
            try:
                compress_xnyzip_triplet(
                    tools,
                    str(chunk_input),
                    str(chunk_archive),
                    chunk.count,
                    bound,
                    chunk_order,
                    True,
                )
            finally:
                chunk_input.unlink(missing_ok=True)
            return chunk, chunk_archive, chunk_order

        with output.open("wb") as archive:
            archive.write(
                XNYZIP_CHUNK_HEADER.pack(
                    XNYZIP_CHUNK_MAGIC,
                    count,
                    chunk_size,
                    len(chunks),
                )
            )
            with ThreadPoolExecutor(
                max_workers=effective_workers
            ) as executor:
                results = executor.map(compress_chunk, chunks)
                for chunk, chunk_archive, chunk_order in results:
                    order = read_xnyzip_permutation(
                        str(chunk_order),
                        chunk.count,
                    )
                    start = chunk.start
                    local_order[start : start + chunk.count] = order
                    payload = chunk_archive.read_bytes()
                    archive.write(
                        XNYZIP_CHUNK_ENTRY.pack(chunk.count, len(payload))
                    )
                    archive.write(payload)
                    chunk_archive.unlink()
                    chunk_order.unlink()

    local_order.flush()
    del local_order
    del source
    return {
        "chunk_size": chunk_size,
        "chunk_count": len(chunks),
        "xnyzip_workers": effective_workers,
        "order_bits_per_particle": max(0, (chunk_size - 1).bit_length()),
    }


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


def run_chunked_xnyzip_decompress(
    tools: ToolPaths,
    compressed_path: str,
    output_paths: Mapping[str, str],
    fields: FieldTriplet,
    count: int,
    chunk_size: int,
    l2_error_bound: float,
    workers: int = 1,
) -> None:
    """Decompress a framed set of independent XnYZip velocity chunks."""

    _validate_chunk_options(chunk_size, workers, "decompression")
    bound = _positive_l2_bound(l2_error_bound)
    sinks = {
        field: np.memmap(
            output_paths[field],
            dtype=np.float32,
            mode="w+",
            shape=(count,),
        )
        for field in fields
    }

    temp_parent = Path(output_paths[fields[0]]).parent
    with tempfile.TemporaryDirectory(
        prefix="velocity_xnyzip_chunks_",
        dir=temp_parent,
    ) as temp:
        chunks = _read_chunked_archive(
            Path(compressed_path),
            Path(temp),
            fields,
            count,
            chunk_size,
        )

        def decompress_chunk(chunk: DecompressionChunk) -> None:
            interleaved = chunk.archive.parent / "velocities.f32.raw"
            end = chunk.start + chunk.count
            try:
                run_xnyzip_decompress(
                    tools,
                    str(chunk.archive),
                    chunk.outputs,
                    fields,
                    chunk.count,
                    bound,
                    interleaved,
                    True,
                )
                for field in fields:
                    sinks[field][chunk.start:end] = read_raw(
                        chunk.outputs[field],
                        np.dtype("float32"),
                        chunk.count,
                    )
            finally:
                chunk.archive.unlink(missing_ok=True)
                interleaved.unlink(missing_ok=True)
                for path in chunk.outputs.values():
                    Path(path).unlink(missing_ok=True)

        effective_workers = min(workers, len(chunks))
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            list(executor.map(decompress_chunk, chunks))

    for sink in sinks.values():
        sink.flush()


def _build_chunks(
    count: int,
    chunk_size: int,
) -> Tuple[CompressionChunk, ...]:
    return tuple(
        CompressionChunk(
            index=index,
            start=start,
            count=min(chunk_size, count - start),
        )
        for index, start in enumerate(range(0, count, chunk_size))
    )


def _read_chunked_archive(
    path: Path,
    temp_dir: Path,
    fields: FieldTriplet,
    count: int,
    chunk_size: int,
) -> Tuple[DecompressionChunk, ...]:
    chunks = []
    with path.open("rb") as archive:
        stored_count, stored_chunk_size, chunk_count = _read_chunk_header(
            archive
        )
        if stored_count != count or stored_chunk_size != chunk_size:
            raise RuntimeError(
                "Chunked XnYZip velocity archive metadata does not match "
                "the manifest."
            )

        start = 0
        for index in range(chunk_count):
            value_count, payload = _read_chunk_entry(archive, count - start)
            expected_count = min(chunk_size, count - start)
            if value_count != expected_count:
                raise RuntimeError(
                    "Chunked XnYZip velocity archive has a misaligned chunk."
                )
            chunk_dir = temp_dir / f"chunk_{index:08d}"
            chunk_dir.mkdir()
            chunk_archive = chunk_dir / "chunk.xnyzip"
            chunk_archive.write_bytes(payload)
            outputs = {
                field: str(chunk_dir / f"{field}.f32.raw")
                for field in fields
            }
            chunks.append(
                DecompressionChunk(
                    start,
                    value_count,
                    chunk_archive,
                    outputs,
                )
            )
            start += value_count

        if start != count:
            raise RuntimeError(
                "Chunked XnYZip velocity archive does not cover every "
                "particle."
            )
        if archive.read(1):
            raise RuntimeError(
                "Chunked XnYZip velocity archive contains trailing data."
            )
    return tuple(chunks)


def _read_chunk_header(archive: BinaryIO) -> Tuple[int, int, int]:
    header = archive.read(XNYZIP_CHUNK_HEADER.size)
    if len(header) != XNYZIP_CHUNK_HEADER.size:
        raise RuntimeError("Truncated chunked XnYZip velocity header.")
    magic, count, chunk_size, chunk_count = XNYZIP_CHUNK_HEADER.unpack(header)
    if magic != XNYZIP_CHUNK_MAGIC:
        raise RuntimeError("Invalid chunked XnYZip velocity archive magic.")
    return int(count), int(chunk_size), int(chunk_count)


def _read_chunk_entry(
    archive: BinaryIO,
    remaining: int,
) -> Tuple[int, bytes]:
    entry = archive.read(XNYZIP_CHUNK_ENTRY.size)
    if len(entry) != XNYZIP_CHUNK_ENTRY.size:
        raise RuntimeError("Truncated chunked XnYZip velocity entry.")
    value_count, payload_size = XNYZIP_CHUNK_ENTRY.unpack(entry)
    if not value_count or value_count > remaining:
        raise RuntimeError(
            "Chunked XnYZip velocity archive has an invalid chunk size."
        )
    payload = archive.read(payload_size)
    if len(payload) != payload_size:
        raise RuntimeError("Truncated chunked XnYZip velocity payload.")
    return int(value_count), payload


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


def _validate_chunk_options(
    chunk_size: int,
    workers: int,
    operation: str,
) -> None:
    if chunk_size <= 0:
        raise RuntimeError(
            f"Chunked XnYZip {operation} requires a positive chunk size."
        )
    if workers <= 0:
        raise RuntimeError(
            f"Chunked XnYZip {operation} requires at least one worker."
        )


__all__ = [
    "XNYZIP_CURVE",
    "XNYZIP_CHUNK_CONTAINER",
    "XNYZIP_DIRECT_THRESHOLD",
    "XNYZIP_ORDER_DTYPE",
    "XNYZIP_QUANTIZER",
    "XNYZIP_STORAGE_MODE",
    "compress_chunked_xnyzip_triplet",
    "compress_xnyzip_triplet",
    "read_xnyzip_permutation",
    "run_chunked_xnyzip_decompress",
    "run_xnyzip_decompress",
]
