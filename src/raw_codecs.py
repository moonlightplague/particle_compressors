"""Adapters for fieldwise pcodec, SZ3, and SZo streams."""

from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import numpy as np

from src.constants import MIN_CODEC_VALUES
from src.runtime import (
    load_pcodec,
    load_pysz,
    load_pyszo,
    read_raw,
    require_output_path,
)


FLOAT_DTYPES = (np.dtype("float32"), np.dtype("float64"))


def compress_pcodec_raw(
    raw_path: str,
    dtype: str,
    compressed_path: str,
    field_name: str,
    count: int,
    force: bool,
) -> Dict[str, Any]:
    standalone, chunk_config_type = load_pcodec()
    data_type = np.dtype(dtype)
    output = Path(compressed_path)
    require_output_path(output, force)

    values = np.ascontiguousarray(read_raw(raw_path, data_type, count))
    payload = standalone.simple_compress(values, chunk_config_type())
    output.write_bytes(payload)
    return _field_metadata(
        field_name,
        "pcodec",
        data_type,
        count,
        output,
        len(payload),
    )


def compress_integer_raw(
    codec: str,
    raw_path: str,
    dtype: str,
    compressed_path: str,
    field_name: str,
    count: int,
    force: bool,
) -> Dict[str, Any]:
    if codec != "pcodec":
        raise RuntimeError(f"Unsupported lossless compressor: {codec}.")
    return compress_pcodec_raw(
        raw_path,
        dtype,
        compressed_path,
        field_name,
        count,
        force,
    )


def compress_szo_raw(
    raw_path: str,
    dtype: str,
    compressed_path: str,
    field_name: str,
    count: int,
    abs_error_bound: float,
    force: bool,
) -> Dict[str, Any]:
    data_type = _require_float_dtype(dtype, field_name, "SZO compression")
    output = Path(compressed_path)
    require_output_path(output, force)
    szo, config_type, error_bound_mode, algorithm = load_pyszo()

    values = read_raw(raw_path, data_type, count)
    encoded, encoded_count = pad_codec_input(values)
    config = config_type((encoded_count,))
    config.errorBoundMode = error_bound_mode.ABS
    config.absErrorBound = float(abs_error_bound)
    try:
        compressed, _ = szo.compress(encoded, config, copy=True)
    except Exception as exc:
        raise RuntimeError(f"SZO compression failed for {field_name}.") from exc

    payload = np.ascontiguousarray(compressed, dtype=np.uint8)
    payload.tofile(output)
    metadata = _field_metadata(
        field_name,
        "szo",
        data_type,
        count,
        output,
        int(payload.size),
    )
    metadata.update(
        {
            "abs_error_bound": float(abs_error_bound),
            "encoded_count": encoded_count,
        }
    )
    return metadata


def compress_pysz_raw(
    raw_path: str,
    dtype: str,
    compressed_path: str,
    field_name: str,
    count: int,
    abs_error_bound: float,
    force: bool,
) -> Dict[str, Any]:
    data_type = _require_float_dtype(dtype, field_name, "pysz compression")
    output = Path(compressed_path)
    require_output_path(output, force)
    pysz, config_type, error_bound_mode = load_pysz()

    values = read_raw(raw_path, data_type, count)
    encoded, encoded_count = pad_codec_input(values)
    config = config_type(encoded.shape)
    config.errorBoundMode = error_bound_mode.ABS
    config.absErrorBound = float(abs_error_bound)
    try:
        compressed, _ = pysz.compress(encoded, config)
    except Exception as exc:
        raise RuntimeError(
            f"pysz compression failed for {field_name} with {count} values "
            f"encoded as {encoded_count} values."
        ) from exc

    payload = np.ascontiguousarray(compressed, dtype=np.uint8)
    payload.tofile(output)
    metadata = _field_metadata(
        field_name,
        "pysz",
        data_type,
        count,
        output,
        int(payload.size),
    )
    metadata.update(
        {
            "abs_error_bound": abs_error_bound,
            "encoded_count": encoded_count,
        }
    )
    return metadata


def compress_lossy_raw(
    codec: str,
    raw_path: str,
    dtype: str,
    compressed_path: str,
    field_name: str,
    count: int,
    abs_error_bound: float,
    force: bool,
) -> Dict[str, Any]:
    compressors = {
        "sz3": compress_pysz_raw,
        "szo": compress_szo_raw,
    }
    try:
        compressor = compressors[codec]
    except KeyError as exc:
        raise RuntimeError(f"Unsupported lossy compressor: {codec}.") from exc
    return compressor(
        raw_path,
        dtype,
        compressed_path,
        field_name,
        count,
        abs_error_bound,
        force,
    )


def decompress_pcodec_raw(
    field: Mapping[str, Any],
    out_path: str,
    force: bool,
) -> None:
    standalone, _ = load_pcodec()
    data_type = np.dtype(field["dtype"])
    count = int(field["count"])
    output = Path(out_path)
    require_output_path(output, force)

    payload = Path(field["path"]).read_bytes()
    data = standalone.simple_decompress(payload)
    if data is None:
        raise RuntimeError(
            f"pcodec decompression for {field['field']} returned no data."
        )
    values = np.asarray(data)
    _validate_decoded_field(values, data_type, count, field["field"], "pcodec")
    values.tofile(output)


def decompress_integer_raw(
    field: Mapping[str, Any],
    out_path: str,
    force: bool,
) -> None:
    decompress_pcodec_raw(field, out_path, force)


def decompress_szo_raw(
    field: Mapping[str, Any],
    out_path: str,
    force: bool,
) -> None:
    data_type = _require_float_dtype(
        field["dtype"],
        str(field["field"]),
        "SZO decompression",
    )
    szo, _, _, _ = load_pyszo()
    _decompress_padded_float_field(
        field,
        out_path,
        force,
        data_type,
        "SZO",
        lambda payload, shape: szo.decompress(payload, data_type, shape)[0],
    )


def decompress_pysz_raw(
    field: Mapping[str, Any],
    out_path: str,
    force: bool,
) -> None:
    data_type = _require_float_dtype(
        field["dtype"],
        str(field["field"]),
        "pysz decompression",
    )
    pysz, _, _ = load_pysz()
    _decompress_padded_float_field(
        field,
        out_path,
        force,
        data_type,
        "pysz",
        lambda payload, shape: pysz.decompress(payload, data_type, shape)[0],
    )


def decompress_lossy_raw(
    field: Mapping[str, Any],
    out_path: str,
    force: bool,
) -> None:
    decompressors = {
        "pysz": decompress_pysz_raw,
        "szo": decompress_szo_raw,
    }
    codec = str(field.get("codec"))
    try:
        decompressor = decompressors[codec]
    except KeyError as exc:
        label = field.get("field", "field")
        raise RuntimeError(
            f"Unsupported lossy codec for {label}: {codec}."
        ) from exc
    decompressor(field, out_path, force)


def pad_codec_input(
    values: np.ndarray,
    minimum_count: int = MIN_CODEC_VALUES,
) -> Tuple[np.ndarray, int]:
    if values.size >= minimum_count:
        return np.ascontiguousarray(values), int(values.size)

    padded = np.empty(minimum_count, dtype=values.dtype)
    padded[: values.size] = values
    fill_value = values[-1] if values.size else np.asarray(0, dtype=values.dtype)
    padded[values.size :] = fill_value
    return padded, minimum_count


def _field_metadata(
    field_name: str,
    codec: str,
    dtype: np.dtype,
    count: int,
    path: Path,
    compressed_bytes: int,
) -> Dict[str, Any]:
    return {
        "field": field_name,
        "codec": codec,
        "dtype": str(dtype),
        "count": count,
        "path": str(path),
        "bytes": compressed_bytes,
    }


def _require_float_dtype(dtype: Any, field_name: str, operation: str) -> np.dtype:
    data_type = np.dtype(dtype)
    if data_type not in FLOAT_DTYPES:
        raise RuntimeError(
            f"{operation} expected float32/float64, got {data_type} "
            f"for {field_name}."
        )
    return data_type


def _validate_decoded_field(
    values: np.ndarray,
    expected_dtype: np.dtype,
    expected_count: int,
    field_name: str,
    codec: str,
) -> None:
    if values.size != expected_count:
        raise RuntimeError(
            f"{codec} decompression for {field_name} returned {values.size} "
            f"values, expected {expected_count}."
        )
    if np.dtype(values.dtype) != expected_dtype:
        raise RuntimeError(
            f"{codec} decompression for {field_name} returned dtype "
            f"{values.dtype}, expected {expected_dtype}."
        )


def _decompress_padded_float_field(
    field: Mapping[str, Any],
    out_path: str,
    force: bool,
    dtype: np.dtype,
    codec_label: str,
    decompress: Any,
) -> None:
    count = int(field["count"])
    encoded_count = int(field.get("encoded_count", count))
    payload = np.fromfile(field["path"], dtype=np.uint8)
    try:
        values = decompress(payload, (encoded_count,))
    except Exception as exc:
        raise RuntimeError(
            f"{codec_label} decompression failed for {field['field']}."
        ) from exc

    decoded = np.asarray(values, dtype=dtype).reshape(-1)
    if decoded.size < count:
        raise RuntimeError(
            f"{codec_label} decompression for {field['field']} returned "
            f"{decoded.size} values, expected at least {count}."
        )
    output = Path(out_path)
    require_output_path(output, force)
    decoded[:count].tofile(output)
