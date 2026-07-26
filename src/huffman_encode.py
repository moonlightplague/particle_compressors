"""Canonical Huffman coding for LCP's uint32/uint64 block-ID sidecar."""

import heapq
import struct
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from src.runtime import require_output_path


HUFFMAN_MAGIC = b"LCPHUF3\0"
HUFFMAN_HEADER = struct.Struct("<8sQQIB3x")
HUFFMAN_ENTRY = struct.Struct("<QB")
HUFFMAN_V2_MAGIC = b"LCPHUF2\0"
HUFFMAN_V2_HEADER = struct.Struct("<8sQQI")
HUFFMAN_V2_ENTRY = struct.Struct("<IB")
BLOCK_ID_DTYPE = np.dtype("uint32")
BLOCK_ID_DTYPES = {
    32: BLOCK_ID_DTYPE,
    64: np.dtype("uint64"),
}


def huffman_encode_file(
    input_path: Path,
    output_path: Path,
    force: bool,
    expected_count: Optional[int] = None,
) -> Dict[str, Any]:
    """Encode a raw uint32/uint64 block-ID file into a Huffman stream."""

    if input_path.resolve() == output_path.resolve():
        raise RuntimeError("Huffman input and output paths must be different.")

    data_type = _block_id_dtype(input_path, expected_count)
    values = np.fromfile(input_path, dtype=data_type)
    if expected_count is not None and values.size != expected_count:
        raise RuntimeError(
            "LCP block-ID sidecar contains "
            f"{values.size} values, expected {expected_count}."
        )

    encoded_values = _delta_encode(values)
    symbols, frequencies = np.unique(encoded_values, return_counts=True)
    lengths = _huffman_code_lengths(symbols, frequencies)
    codes = _canonical_codes(lengths)
    bit_count = sum(
        int(frequency) * lengths[int(symbol)]
        for symbol, frequency in zip(symbols, frequencies)
    )
    payload = _encode_values(encoded_values, codes, bit_count)

    require_output_path(output_path, force)
    with output_path.open("wb") as output:
        output.write(
            HUFFMAN_HEADER.pack(
                HUFFMAN_MAGIC,
                int(values.size),
                bit_count,
                len(lengths),
                data_type.itemsize * 8,
            )
        )
        for symbol in sorted(lengths):
            output.write(HUFFMAN_ENTRY.pack(symbol, lengths[symbol]))
        output.write(payload)

    return {
        "codec": "canonical_huffman",
        "symbol_dtype": str(data_type),
        "value_count": int(values.size),
        "unique_symbol_count": len(lengths),
        "encoded_bit_count": bit_count,
        "encoded_bytes": output_path.stat().st_size,
        "transform": f"{data_type.name}_delta_modulo",
        "format": HUFFMAN_MAGIC.rstrip(b"\0").decode("ascii"),
    }


def huffman_decode_file(
    input_path: Path,
    output_path: Path,
    force: bool,
    expected_count: Optional[int] = None,
) -> Dict[str, Any]:
    """Decode a canonical Huffman stream to LCP's raw block-ID words."""

    encoded = input_path.read_bytes()
    if len(encoded) < 8:
        raise RuntimeError("Truncated LCP block-ID Huffman header.")
    magic = encoded[:8]
    if magic == HUFFMAN_MAGIC:
        if len(encoded) < HUFFMAN_HEADER.size:
            raise RuntimeError("Truncated LCP block-ID Huffman header.")
        (
            _,
            value_count,
            bit_count,
            symbol_count,
            symbol_bits,
        ) = HUFFMAN_HEADER.unpack_from(encoded)
        try:
            data_type = BLOCK_ID_DTYPES[symbol_bits]
        except KeyError as exc:
            raise RuntimeError(
                "LCP block-ID Huffman symbol width must be 32 or 64 bits."
            ) from exc
        header = HUFFMAN_HEADER
        entry = HUFFMAN_ENTRY
    elif magic == HUFFMAN_V2_MAGIC:
        if len(encoded) < HUFFMAN_V2_HEADER.size:
            raise RuntimeError("Truncated LCP block-ID Huffman header.")
        _, value_count, bit_count, symbol_count = (
            HUFFMAN_V2_HEADER.unpack_from(encoded)
        )
        data_type = BLOCK_ID_DTYPE
        header = HUFFMAN_V2_HEADER
        entry = HUFFMAN_V2_ENTRY
    else:
        raise RuntimeError("Invalid LCP block-ID Huffman magic.")
    if expected_count is not None and value_count != expected_count:
        raise RuntimeError(
            "LCP block-ID Huffman stream contains "
            f"{value_count} values, expected {expected_count}."
        )
    if symbol_count > value_count:
        raise RuntimeError(
            "LCP block-ID Huffman stream has more symbols than values."
        )

    table_bytes = symbol_count * entry.size
    payload_offset = header.size + table_bytes
    expected_payload_bytes = (bit_count + 7) // 8
    if payload_offset > len(encoded):
        raise RuntimeError("Truncated LCP block-ID Huffman code table.")
    if len(encoded) - payload_offset != expected_payload_bytes:
        raise RuntimeError(
            "LCP block-ID Huffman payload size does not match its header."
        )

    lengths: Dict[int, int] = {}
    offset = header.size
    for _ in range(symbol_count):
        symbol, length = entry.unpack_from(encoded, offset)
        offset += entry.size
        if symbol > np.iinfo(data_type).max:
            raise RuntimeError(
                "LCP block-ID Huffman symbol exceeds its stored width."
            )
        if not length:
            raise RuntimeError("LCP block-ID Huffman code has zero length.")
        if symbol in lengths:
            raise RuntimeError("Duplicate symbol in LCP Huffman code table.")
        lengths[symbol] = length

    if bool(value_count) != bool(symbol_count):
        raise RuntimeError(
            "LCP block-ID Huffman stream has inconsistent empty metadata."
        )
    codes = _canonical_codes(lengths)
    encoded_values = _decode_values(
        encoded[payload_offset:],
        bit_count,
        value_count,
        codes,
        data_type,
    )
    values = _delta_decode(encoded_values)

    require_output_path(output_path, force)
    values.tofile(output_path)
    return {
        "codec": "canonical_huffman",
        "symbol_dtype": str(data_type),
        "value_count": int(value_count),
        "unique_symbol_count": int(symbol_count),
        "encoded_bit_count": int(bit_count),
        "encoded_bytes": len(encoded),
        "decoded_bytes": int(values.nbytes),
        "transform": f"{data_type.name}_delta_modulo",
        "format": magic.rstrip(b"\0").decode("ascii"),
    }


def _block_id_dtype(
    input_path: Path,
    expected_count: Optional[int],
) -> np.dtype:
    file_size = input_path.stat().st_size
    if expected_count is not None:
        if expected_count < 0:
            raise RuntimeError("LCP block-ID count must be non-negative.")
        matching = [
            data_type
            for data_type in BLOCK_ID_DTYPES.values()
            if file_size == expected_count * data_type.itemsize
        ]
        if matching:
            return matching[0]
        expected_sizes = (
            expected_count * BLOCK_ID_DTYPES[32].itemsize,
            expected_count * BLOCK_ID_DTYPES[64].itemsize,
        )
        raise RuntimeError(
            f"LCP block-ID sidecar has {file_size} bytes; expected "
            f"{expected_sizes[0]} for uint32 or {expected_sizes[1]} "
            f"for uint64."
        )
    if file_size % BLOCK_ID_DTYPE.itemsize:
        raise RuntimeError(
            f"LCP block-ID sidecar has a partial uint32 value: {input_path}."
        )
    return BLOCK_ID_DTYPE


def _huffman_code_lengths(
    symbols: np.ndarray,
    frequencies: np.ndarray,
) -> Dict[int, int]:
    if symbols.size == 0:
        return {}
    if symbols.size == 1:
        return {int(symbols[0]): 1}

    heap = []
    serial = 0
    for symbol, frequency in zip(symbols, frequencies):
        value = int(symbol)
        heap.append((int(frequency), value, serial, value))
        serial += 1
    heapq.heapify(heap)

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        node = (left[3], right[3])
        heapq.heappush(
            heap,
            (
                left[0] + right[0],
                min(left[1], right[1]),
                serial,
                node,
            ),
        )
        serial += 1

    lengths: Dict[int, int] = {}
    stack = [(heap[0][3], 0)]
    while stack:
        node, depth = stack.pop()
        if isinstance(node, int):
            if depth > 255:
                raise RuntimeError("LCP block-ID Huffman code exceeds 255 bits.")
            lengths[node] = max(1, depth)
            continue
        left, right = node
        stack.append((right, depth + 1))
        stack.append((left, depth + 1))
    return lengths


def _delta_encode(values: np.ndarray) -> np.ndarray:
    encoded = np.empty_like(values)
    if values.size:
        encoded[0] = values[0]
        np.subtract(
            values[1:],
            values[:-1],
            out=encoded[1:],
            dtype=values.dtype,
        )
    return encoded


def _delta_decode(values: np.ndarray) -> np.ndarray:
    return np.add.accumulate(values, dtype=values.dtype)


def _canonical_codes(
    lengths: Mapping[int, int],
) -> Dict[int, Tuple[int, int]]:
    ordered = sorted(
        ((int(length), int(symbol)) for symbol, length in lengths.items())
    )
    codes: Dict[int, Tuple[int, int]] = {}
    code = 0
    previous_length = 0
    for length, symbol in ordered:
        if length <= 0 or length > 255:
            raise RuntimeError("Invalid LCP block-ID Huffman code length.")
        code <<= length - previous_length
        if code >= 1 << length:
            raise RuntimeError("Oversubscribed LCP block-ID Huffman code table.")
        codes[symbol] = (code, length)
        code += 1
        previous_length = length
    return codes


def _encode_values(
    values: np.ndarray,
    codes: Mapping[int, Tuple[int, int]],
    bit_count: int,
) -> bytes:
    payload = bytearray()
    buffer = 0
    buffered_bits = 0
    for value in values:
        code, length = codes[int(value)]
        buffer = (buffer << length) | code
        buffered_bits += length
        while buffered_bits >= 8:
            shift = buffered_bits - 8
            payload.append((buffer >> shift) & 0xFF)
            buffer &= (1 << shift) - 1
            buffered_bits = shift
    if buffered_bits:
        payload.append((buffer << (8 - buffered_bits)) & 0xFF)
    if len(payload) != (bit_count + 7) // 8:
        raise RuntimeError("Internal LCP block-ID Huffman size mismatch.")
    return bytes(payload)


def _decode_values(
    payload: bytes,
    bit_count: int,
    value_count: int,
    codes: Mapping[int, Tuple[int, int]],
    data_type: np.dtype,
) -> np.ndarray:
    decode_table = {
        (length, code): symbol
        for symbol, (code, length) in codes.items()
    }
    values = np.empty(value_count, dtype=data_type)
    output_index = 0
    code = 0
    length = 0

    for bit_index in range(bit_count):
        byte = payload[bit_index // 8]
        bit = (byte >> (7 - bit_index % 8)) & 1
        code = (code << 1) | bit
        length += 1
        symbol = decode_table.get((length, code))
        if symbol is None:
            continue
        if output_index >= value_count:
            raise RuntimeError(
                "LCP block-ID Huffman payload decodes too many values."
            )
        values[output_index] = symbol
        output_index += 1
        code = 0
        length = 0

    if length:
        raise RuntimeError("Truncated LCP block-ID Huffman code.")
    if output_index != value_count:
        raise RuntimeError(
            "LCP block-ID Huffman payload decoded "
            f"{output_index} values, expected {value_count}."
        )
    if bit_count % 8 and payload:
        padding_mask = (1 << (8 - bit_count % 8)) - 1
        if payload[-1] & padding_mask:
            raise RuntimeError("Nonzero padding in LCP block-ID Huffman payload.")
    return values


__all__ = [
    "BLOCK_ID_DTYPE",
    "HUFFMAN_MAGIC",
    "huffman_decode_file",
    "huffman_encode_file",
]
