"""LCP command construction, row ordering, and chunk-container handling."""

import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Callable, Dict, Mapping, Tuple

import numpy as np

from src.constants import (
    LCP_CHUNK_BATCH_VALUES,
    LCP_CHUNK_CONTAINER,
    LCP_CHUNK_ENTRY,
    LCP_CHUNK_HEADER,
    LCP_CHUNK_MAGIC,
    MAX_INT32_ORDER_VALUES,
    VELOCITY_FIELDS,
)
from src.models import ToolPaths
from src.runtime import read_raw, require_output_path, run_command


FieldTriplet = Tuple[str, str, str]


@dataclass(frozen=True)
class CompressionSegment:
    index: int
    start: int
    value_count: int
    chunk_count: int


@dataclass(frozen=True)
class DecompressionSegment:
    start: int
    value_count: int
    archive: Path
    outputs: Dict[str, str]


def read_lcp_permutation(order_path: str, count: int) -> np.ndarray:
    order = read_raw(order_path, np.dtype("int32"), count)
    _validate_global_permutation(order, count, "LCP order")
    return order.astype(np.intp, copy=False)


def reorder_raw(
    raw_path: str,
    dtype: str,
    output_path: Path,
    count: int,
    order: np.ndarray,
    force: bool,
) -> str:
    require_output_path(output_path, force)
    values = read_raw(raw_path, np.dtype(dtype), count)
    np.ascontiguousarray(values[order]).tofile(output_path)
    return str(output_path)


def compress_lcp_triplet(
    tools: ToolPaths,
    input_paths: FieldTriplet,
    compressed_path: str,
    count: int,
    abs_error_bound: float,
    order_path: Path,
    force: bool,
) -> None:
    require_output_path(Path(compressed_path), force)
    require_output_path(order_path, force)
    run_command(
        [
            str(tools.lcp),
            "-i",
            *input_paths,
            "-z",
            compressed_path,
            "-1",
            str(count),
            "-eb",
            str(abs_error_bound),
            "-ord",
            "32",
            str(order_path),
        ]
    )


def compress_lcp_triplet_batch(
    tools: ToolPaths,
    input_paths: FieldTriplet,
    compressed_path: str,
    chunks: int,
    chunk_size: int,
    abs_error_bound: float,
    order_path: Path,
    force: bool,
) -> None:
    if chunks <= 1:
        raise RuntimeError("Batched LCP compression requires at least two chunks.")
    require_output_path(Path(compressed_path), force)
    require_output_path(order_path, force)
    run_command(
        [
            str(tools.lcp),
            "-i",
            *input_paths,
            "-z",
            compressed_path,
            "-2",
            str(chunks),
            str(chunk_size),
            "-bt",
            "0",
            "-eb",
            str(abs_error_bound),
            "-ord",
            "32",
            str(order_path),
        ]
    )


def run_lcp_decompress(
    tools: ToolPaths,
    compressed_path: str,
    output_paths: Mapping[str, str],
    fields: FieldTriplet,
    count: int,
    abs_error_bound: float,
) -> None:
    run_command(
        [
            str(tools.lcp),
            "-z",
            compressed_path,
            "-o",
            *(output_paths[field] for field in fields),
            "-1",
            str(count),
            "-eb",
            str(abs_error_bound),
        ]
    )


def run_lcp_decompress_batch(
    tools: ToolPaths,
    compressed_path: str,
    output_paths: Mapping[str, str],
    fields: FieldTriplet,
    chunks: int,
    chunk_size: int,
    abs_error_bound: float,
) -> None:
    run_command(
        [
            str(tools.lcp),
            "-z",
            compressed_path,
            "-o",
            *(output_paths[field] for field in fields),
            "-2",
            str(chunks),
            str(chunk_size),
            "-bt",
            "0",
            "-eb",
            str(abs_error_bound),
        ]
    )


def velocity_order_bits(chunk_size: int) -> int:
    if chunk_size < 0:
        raise RuntimeError("Velocity chunk size must be non-negative.")
    return max(0, (chunk_size - 1).bit_length())


def compress_chunked_lcp_triplet(
    tools: ToolPaths,
    input_paths: FieldTriplet,
    compressed_path: str,
    count: int,
    chunk_size: int,
    abs_error_bound: float,
    order_path: Path,
    force: bool,
    workers: int = 1,
) -> Dict[str, int]:
    _validate_chunk_options(chunk_size, workers, "compression")
    if chunk_size > MAX_INT32_ORDER_VALUES:
        raise RuntimeError(
            "Velocity chunk size cannot exceed 2^31 with int32 local order indices."
        )

    output = Path(compressed_path)
    require_output_path(output, force)
    require_output_path(order_path, force)
    segments = _build_compression_segments(count, chunk_size)
    effective_workers = min(workers, len(segments))
    sources = tuple(
        np.memmap(path, dtype=np.float32, mode="r", shape=(count,))
        for path in input_paths
    )
    local_order = np.memmap(
        order_path,
        dtype=np.int32,
        mode="w+",
        shape=(count,),
    )

    with tempfile.TemporaryDirectory(
        prefix="velocity_lcp_chunks_",
        dir=order_path.parent,
    ) as temp:
        temp_dir = Path(temp)

        def compress_segment(
            segment: CompressionSegment,
        ) -> Tuple[CompressionSegment, Path, Path]:
            segment_dir = temp_dir / f"segment_{segment.index:08d}"
            segment_dir.mkdir()
            inputs = tuple(
                segment_dir / f"{field}.f32.raw" for field in VELOCITY_FIELDS
            )
            archive = segment_dir / "chunk.lcp"
            order = segment_dir / "order.i32.raw"
            try:
                end = segment.start + segment.value_count
                for source, path in zip(sources, inputs):
                    np.asarray(source[segment.start:end]).tofile(path)
                _compress_segment(
                    tools,
                    inputs,
                    archive,
                    order,
                    segment,
                    chunk_size,
                    abs_error_bound,
                )
            finally:
                for path in inputs:
                    path.unlink(missing_ok=True)
            return segment, archive, order

        _write_chunked_archive(
            output,
            count,
            chunk_size,
            segments,
            effective_workers,
            compress_segment,
            local_order,
        )

    local_order.flush()
    del local_order
    del sources
    return {
        "chunk_size": chunk_size,
        "chunk_count": (count + chunk_size - 1) // chunk_size,
        "lcp_segment_count": len(segments),
        "lcp_batch_values": LCP_CHUNK_BATCH_VALUES,
        "lcp_workers": effective_workers,
        "order_bits_per_particle": velocity_order_bits(chunk_size),
    }


def run_chunked_lcp_decompress(
    tools: ToolPaths,
    compressed_path: str,
    output_paths: Mapping[str, str],
    fields: FieldTriplet,
    count: int,
    chunk_size: int,
    abs_error_bound: float,
    workers: int = 1,
) -> None:
    _validate_chunk_options(chunk_size, workers, "decompression")
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
        prefix="velocity_lcp_chunks_",
        dir=temp_parent,
    ) as temp:
        segments = _read_chunked_archive(
            Path(compressed_path),
            Path(temp),
            fields,
            count,
            chunk_size,
        )

        def decompress_segment(segment: DecompressionSegment) -> None:
            _decompress_segment(
                tools,
                segment,
                sinks,
                fields,
                chunk_size,
                abs_error_bound,
            )

        effective_workers = min(workers, len(segments))
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            list(executor.map(decompress_segment, segments))

    for sink in sinks.values():
        sink.flush()


def read_lcp_order(
    path: str,
    dtype: np.dtype,
    count: int,
    label: str,
    chunk_size: int = 0,
) -> np.ndarray:
    order = np.fromfile(path, dtype=dtype, count=count)
    if order.size != count:
        raise RuntimeError(
            f"Unexpected EOF reading {path}; expected {count}, got {order.size}."
        )
    if chunk_size < 0:
        raise RuntimeError(f"{label} has an invalid negative chunk size.")
    if not chunk_size:
        _validate_global_permutation(order, count, label)
        return order.astype(np.intp, copy=False)

    expanded = np.empty(count, dtype=np.intp)
    for start in range(0, count, chunk_size):
        end = min(start + chunk_size, count)
        local = order[start:end]
        local_count = end - start
        _validate_local_permutation(local, local_count, label)
        expanded[start:end] = start + local.astype(np.intp, copy=False)
    return expanded


def _build_compression_segments(
    count: int,
    chunk_size: int,
) -> Tuple[CompressionSegment, ...]:
    chunks_per_batch = max(1, LCP_CHUNK_BATCH_VALUES // chunk_size)
    segments = []
    start = 0
    while start < count:
        remaining = count - start
        if remaining < chunk_size:
            chunks_in_segment = 1
            value_count = remaining
        else:
            full_chunks = remaining // chunk_size
            chunks_in_segment = min(chunks_per_batch, full_chunks)
            value_count = chunks_in_segment * chunk_size
        segments.append(
            CompressionSegment(
                index=len(segments),
                start=start,
                value_count=value_count,
                chunk_count=chunks_in_segment,
            )
        )
        start += value_count
    return tuple(segments)


def _compress_segment(
    tools: ToolPaths,
    inputs: Tuple[Path, Path, Path],
    archive: Path,
    order: Path,
    segment: CompressionSegment,
    chunk_size: int,
    abs_error_bound: float,
) -> None:
    input_paths = tuple(str(path) for path in inputs)
    if segment.chunk_count > 1:
        compress_lcp_triplet_batch(
            tools,
            input_paths,
            str(archive),
            segment.chunk_count,
            chunk_size,
            abs_error_bound,
            order,
            True,
        )
        return
    compress_lcp_triplet(
        tools,
        input_paths,
        str(archive),
        segment.value_count,
        abs_error_bound,
        order,
        True,
    )


def _write_chunked_archive(
    output: Path,
    count: int,
    chunk_size: int,
    segments: Tuple[CompressionSegment, ...],
    workers: int,
    compress_segment: Callable[
        [CompressionSegment],
        Tuple[CompressionSegment, Path, Path],
    ],
    local_order: np.memmap,
) -> None:
    with output.open("wb") as archive:
        archive.write(
            LCP_CHUNK_HEADER.pack(
                LCP_CHUNK_MAGIC,
                count,
                chunk_size,
                len(segments),
            )
        )
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = executor.map(compress_segment, segments)
            for segment, chunk_archive, chunk_order in results:
                order = read_raw(
                    str(chunk_order),
                    np.dtype("int32"),
                    segment.value_count,
                )
                _store_local_orders(
                    order,
                    local_order,
                    segment.start,
                    segment.value_count,
                    chunk_size,
                )
                payload = chunk_archive.read_bytes()
                archive.write(
                    LCP_CHUNK_ENTRY.pack(segment.value_count, len(payload))
                )
                archive.write(payload)
                chunk_archive.unlink()
                chunk_order.unlink()


def _store_local_orders(
    order: np.ndarray,
    destination: np.memmap,
    segment_start: int,
    segment_count: int,
    chunk_size: int,
) -> None:
    for local_start in range(0, segment_count, chunk_size):
        local_end = min(local_start + chunk_size, segment_count)
        local_values = order[local_start:local_end]
        local_count = local_end - local_start
        if local_count and (
            int(local_values.min()) < local_start
            or int(local_values.max()) >= local_end
        ):
            raise RuntimeError(
                "LCP order contains an index outside its velocity chunk."
            )
        localized = local_values - local_start
        if np.unique(localized).size != local_count:
            raise RuntimeError(
                "LCP order is not a local permutation within a velocity chunk."
            )
        output_start = segment_start + local_start
        output_end = segment_start + local_end
        destination[output_start:output_end] = localized


def _read_chunked_archive(
    path: Path,
    temp_dir: Path,
    fields: FieldTriplet,
    count: int,
    chunk_size: int,
) -> Tuple[DecompressionSegment, ...]:
    segments = []
    with path.open("rb") as archive:
        stored_count, stored_chunk_size, segment_count = _read_chunk_header(archive)
        if stored_count != count or stored_chunk_size != chunk_size:
            raise RuntimeError(
                "Chunked LCP velocity archive metadata does not match the manifest."
            )

        start = 0
        for index in range(segment_count):
            value_count, payload = _read_chunk_entry(archive, count - start)
            segment_dir = temp_dir / f"segment_{index:08d}"
            segment_dir.mkdir()
            chunk_archive = segment_dir / "chunk.lcp"
            chunk_archive.write_bytes(payload)
            outputs = {
                field: str(segment_dir / f"{field}.f32.raw")
                for field in fields
            }
            segments.append(
                DecompressionSegment(start, value_count, chunk_archive, outputs)
            )
            start += value_count

        if start != count:
            raise RuntimeError(
                "Chunked LCP velocity archive does not cover every particle."
            )
        if archive.read(1):
            raise RuntimeError(
                "Chunked LCP velocity archive contains trailing data."
            )
    return tuple(segments)


def _read_chunk_header(archive: BinaryIO) -> Tuple[int, int, int]:
    header = archive.read(LCP_CHUNK_HEADER.size)
    if len(header) != LCP_CHUNK_HEADER.size:
        raise RuntimeError("Truncated chunked LCP velocity header.")
    magic, count, chunk_size, segment_count = LCP_CHUNK_HEADER.unpack(header)
    if magic != LCP_CHUNK_MAGIC:
        raise RuntimeError("Invalid chunked LCP velocity archive magic.")
    return int(count), int(chunk_size), int(segment_count)


def _read_chunk_entry(
    archive: BinaryIO,
    remaining: int,
) -> Tuple[int, bytes]:
    entry = archive.read(LCP_CHUNK_ENTRY.size)
    if len(entry) != LCP_CHUNK_ENTRY.size:
        raise RuntimeError("Truncated chunked LCP velocity entry.")
    value_count, payload_size = LCP_CHUNK_ENTRY.unpack(entry)
    if not value_count or value_count > remaining:
        raise RuntimeError(
            "Chunked LCP velocity archive has an invalid segment size."
        )
    payload = archive.read(payload_size)
    if len(payload) != payload_size:
        raise RuntimeError("Truncated chunked LCP velocity payload.")
    return int(value_count), payload


def _decompress_segment(
    tools: ToolPaths,
    segment: DecompressionSegment,
    sinks: Mapping[str, np.memmap],
    fields: FieldTriplet,
    chunk_size: int,
    abs_error_bound: float,
) -> None:
    end = segment.start + segment.value_count
    try:
        if segment.value_count > chunk_size:
            if segment.value_count % chunk_size:
                raise RuntimeError(
                    "Chunked LCP velocity batch is not aligned to the "
                    "configured chunk size."
                )
            run_lcp_decompress_batch(
                tools,
                str(segment.archive),
                segment.outputs,
                fields,
                segment.value_count // chunk_size,
                chunk_size,
                abs_error_bound,
            )
        else:
            run_lcp_decompress(
                tools,
                str(segment.archive),
                segment.outputs,
                fields,
                segment.value_count,
                abs_error_bound,
            )
        for field in fields:
            sinks[field][segment.start:end] = read_raw(
                segment.outputs[field],
                np.dtype("float32"),
                segment.value_count,
            )
    finally:
        segment.archive.unlink(missing_ok=True)
        for path in segment.outputs.values():
            Path(path).unlink(missing_ok=True)


def _validate_chunk_options(
    chunk_size: int,
    workers: int,
    operation: str,
) -> None:
    if chunk_size <= 0:
        raise RuntimeError(
            f"Chunked LCP {operation} requires a positive chunk size."
        )
    if workers <= 0:
        raise RuntimeError(
            f"Chunked LCP {operation} requires at least one worker."
        )


def _validate_global_permutation(
    order: np.ndarray,
    count: int,
    label: str,
) -> None:
    if count and (int(order.min()) < 0 or int(order.max()) >= count):
        raise RuntimeError(
            f"{label} is not a valid index range for this particle count."
        )
    if np.unique(order).size != count:
        raise RuntimeError(f"{label} is not a permutation of the particle rows.")


def _validate_local_permutation(
    order: np.ndarray,
    count: int,
    label: str,
) -> None:
    if count and (int(order.min()) < 0 or int(order.max()) >= count):
        raise RuntimeError(f"{label} contains an index outside its velocity chunk.")
    if np.unique(order).size != count:
        raise RuntimeError(
            f"{label} contains a chunk that is not a local permutation."
        )


__all__ = [
    "LCP_CHUNK_CONTAINER",
    "compress_chunked_lcp_triplet",
    "compress_lcp_triplet",
    "compress_lcp_triplet_batch",
    "read_lcp_order",
    "read_lcp_permutation",
    "reorder_raw",
    "run_chunked_lcp_decompress",
    "run_lcp_decompress",
    "run_lcp_decompress_batch",
    "velocity_order_bits",
]
