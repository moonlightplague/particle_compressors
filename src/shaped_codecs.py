"""N-dimensional lossy-codec adapters used by lattice packages."""

from __future__ import annotations

import itertools
import math
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Tuple

import numpy as np

from src.raw_codecs import require_float_dtype
from src.runtime import (
    load_pysz,
    load_pyszo,
    read_raw,
    require_output_path,
)


def compress_shaped_lossy_raw(
    codec: str,
    raw_path: str,
    dtype: str,
    compressed_path: str,
    field_name: str,
    count: int,
    abs_error_bound: float,
    force: bool,
    encoded_shape: Tuple[int, int, int],
    axis_search: bool,
) -> Dict[str, Any]:
    if codec == "szo":
        return _compress_shaped_szo(
            raw_path,
            dtype,
            compressed_path,
            field_name,
            count,
            abs_error_bound,
            force,
            encoded_shape,
            axis_search,
        )
    if codec == "sz3":
        return _compress_shaped_sz3(
            raw_path,
            dtype,
            compressed_path,
            field_name,
            count,
            abs_error_bound,
            force,
            encoded_shape,
            axis_search,
        )
    raise RuntimeError(f"Unsupported shaped lossy compressor: {codec}.")


def decompress_shaped_lossy_raw(
    field: Mapping[str, Any],
    out_path: str,
    force: bool,
) -> None:
    codec = str(field.get("codec"))
    if codec == "szo":
        data_type = require_float_dtype(
            field["dtype"],
            str(field["field"]),
            "SZO decompression",
        )
        szo, _, _, _ = load_pyszo()
        _decompress_shaped(
            field,
            out_path,
            force,
            data_type,
            "SZO",
            lambda payload, shape: szo.decompress(
                payload,
                data_type,
                shape,
            )[0],
        )
        return
    if codec == "pysz":
        data_type = require_float_dtype(
            field["dtype"],
            str(field["field"]),
            "pysz decompression",
        )
        pysz, _, _ = load_pysz()
        _decompress_shaped(
            field,
            out_path,
            force,
            data_type,
            "pysz",
            lambda payload, shape: pysz.decompress(
                payload,
                data_type,
                shape,
            )[0],
        )
        return
    raise RuntimeError(
        f"Unsupported shaped lossy codec for {field.get('field')}: {codec}."
    )


def _compress_shaped_szo(
    raw_path: str,
    dtype: str,
    compressed_path: str,
    field_name: str,
    count: int,
    abs_error_bound: float,
    force: bool,
    encoded_shape: Tuple[int, int, int],
    axis_search: bool,
) -> Dict[str, Any]:
    data_type = require_float_dtype(
        dtype,
        field_name,
        "SZO compression",
    )
    output = Path(compressed_path)
    require_output_path(output, force)
    szo, config_type, error_bound_mode, algorithms = load_pyszo()
    values = _read_shaped(raw_path, data_type, encoded_shape)
    best_payload = None
    best_permutation = tuple(range(values.ndim))
    best_flips = (False,) * values.ndim
    best_shape = values.shape
    for permutation, candidate in _axis_candidates(values, axis_search):
        config = config_type(candidate.shape)
        config.errorBoundMode = error_bound_mode.ABS
        config.absErrorBound = float(abs_error_bound)
        if hasattr(algorithms, "INTERP_LORENZO"):
            config.cmprAlgo = algorithms.INTERP_LORENZO
        elif hasattr(algorithms, "LORENZO_REG"):
            config.cmprAlgo = algorithms.LORENZO_REG
        try:
            compressed, _ = szo.compress(candidate, config, copy=True)
        except Exception as exc:
            raise RuntimeError(
                f"SZO compression failed for shaped field {field_name}."
            ) from exc
        payload = np.ascontiguousarray(compressed, dtype=np.uint8)
        if best_payload is None or payload.size < best_payload.size:
            best_payload = payload
            best_permutation = permutation
            best_shape = candidate.shape
    if axis_search:
        for flips, candidate in _flip_candidates(values, best_permutation):
            config = config_type(candidate.shape)
            config.errorBoundMode = error_bound_mode.ABS
            config.absErrorBound = float(abs_error_bound)
            if hasattr(algorithms, "INTERP_LORENZO"):
                config.cmprAlgo = algorithms.INTERP_LORENZO
            elif hasattr(algorithms, "LORENZO_REG"):
                config.cmprAlgo = algorithms.LORENZO_REG
            try:
                compressed, _ = szo.compress(candidate, config, copy=True)
            except Exception as exc:
                raise RuntimeError(
                    f"SZO compression failed for shaped field {field_name}."
                ) from exc
            payload = np.ascontiguousarray(compressed, dtype=np.uint8)
            if payload.size < best_payload.size:
                best_payload = payload
                best_flips = flips
                best_shape = candidate.shape
    assert best_payload is not None
    best_payload.tofile(output)
    return _shaped_metadata(
        field_name,
        "szo",
        data_type,
        count,
        output,
        int(best_payload.size),
        values.shape,
        best_shape,
        best_permutation,
        best_flips,
        abs_error_bound,
        axis_search,
    )


def _compress_shaped_sz3(
    raw_path: str,
    dtype: str,
    compressed_path: str,
    field_name: str,
    count: int,
    abs_error_bound: float,
    force: bool,
    encoded_shape: Tuple[int, int, int],
    axis_search: bool,
) -> Dict[str, Any]:
    data_type = require_float_dtype(
        dtype,
        field_name,
        "pysz compression",
    )
    output = Path(compressed_path)
    require_output_path(output, force)
    pysz, config_type, error_bound_mode = load_pysz()
    values = _read_shaped(raw_path, data_type, encoded_shape)
    best_payload = None
    best_permutation = tuple(range(values.ndim))
    best_flips = (False,) * values.ndim
    best_shape = values.shape
    for permutation, candidate in _axis_candidates(values, axis_search):
        config = config_type(candidate.shape)
        config.errorBoundMode = error_bound_mode.ABS
        config.absErrorBound = float(abs_error_bound)
        try:
            compressed, _ = pysz.compress(candidate, config)
        except Exception as exc:
            raise RuntimeError(
                f"pysz compression failed for shaped field {field_name}."
            ) from exc
        payload = np.ascontiguousarray(compressed, dtype=np.uint8)
        if best_payload is None or payload.size < best_payload.size:
            best_payload = payload
            best_permutation = permutation
            best_shape = candidate.shape
    if axis_search:
        for flips, candidate in _flip_candidates(values, best_permutation):
            config = config_type(candidate.shape)
            config.errorBoundMode = error_bound_mode.ABS
            config.absErrorBound = float(abs_error_bound)
            try:
                compressed, _ = pysz.compress(candidate, config)
            except Exception as exc:
                raise RuntimeError(
                    f"pysz compression failed for shaped field {field_name}."
                ) from exc
            payload = np.ascontiguousarray(compressed, dtype=np.uint8)
            if payload.size < best_payload.size:
                best_payload = payload
                best_flips = flips
                best_shape = candidate.shape
    assert best_payload is not None
    best_payload.tofile(output)
    return _shaped_metadata(
        field_name,
        "pysz",
        data_type,
        count,
        output,
        int(best_payload.size),
        values.shape,
        best_shape,
        best_permutation,
        best_flips,
        abs_error_bound,
        axis_search,
    )


def _read_shaped(
    raw_path: str,
    dtype: np.dtype,
    shape: Tuple[int, int, int],
) -> np.ndarray:
    resolved_shape = tuple(int(value) for value in shape)
    if len(resolved_shape) != 3 or any(value <= 0 for value in resolved_shape):
        raise RuntimeError(f"Invalid dense codec shape: {shape!r}.")
    count = math.prod(resolved_shape)
    return read_raw(raw_path, dtype, count).reshape(resolved_shape)


def _axis_candidates(
    values: np.ndarray,
    axis_search: bool,
) -> Iterator[Tuple[Tuple[int, ...], np.ndarray]]:
    identity = tuple(range(values.ndim))
    permutations = (
        itertools.permutations(range(values.ndim))
        if axis_search
        else (identity,)
    )
    for permutation_values in permutations:
        permutation = tuple(int(value) for value in permutation_values)
        yield permutation, np.ascontiguousarray(
            np.transpose(values, permutation)
        )


def _flip_candidates(
    values: np.ndarray,
    permutation: Tuple[int, ...],
) -> Iterator[Tuple[Tuple[bool, ...], np.ndarray]]:
    permuted = np.transpose(values, permutation)
    for flips in itertools.product((False, True), repeat=values.ndim):
        if not any(flips):
            continue
        slices = tuple(
            slice(None, None, -1) if flipped else slice(None)
            for flipped in flips
        )
        yield flips, np.ascontiguousarray(permuted[slices])


def _shaped_metadata(
    field_name: str,
    codec: str,
    dtype: np.dtype,
    count: int,
    output: Path,
    compressed_bytes: int,
    base_shape: Tuple[int, ...],
    encoded_shape: Tuple[int, ...],
    permutation: Tuple[int, ...],
    flips: Tuple[bool, ...],
    abs_error_bound: float,
    axis_search: bool,
) -> Dict[str, Any]:
    return {
        "field": field_name,
        "codec": codec,
        "dtype": str(dtype),
        "count": count,
        "path": str(output),
        "bytes": compressed_bytes,
        "abs_error_bound": float(abs_error_bound),
        "encoded_count": math.prod(base_shape),
        "base_shape": list(base_shape),
        "encoded_shape": list(encoded_shape),
        "axis_permutation": list(permutation),
        "axis_flips": list(flips),
        "axis_search": bool(axis_search),
    }


def _decompress_shaped(
    field: Mapping[str, Any],
    out_path: str,
    force: bool,
    dtype: np.dtype,
    codec_label: str,
    decompress: Any,
) -> None:
    encoded_shape = tuple(int(value) for value in field["encoded_shape"])
    base_shape = tuple(int(value) for value in field["base_shape"])
    encoded_count = int(field["encoded_count"])
    if math.prod(encoded_shape) != encoded_count or math.prod(
        base_shape
    ) != encoded_count:
        raise RuntimeError(
            f"{codec_label} shaped metadata for {field['field']} is inconsistent."
        )
    payload = np.fromfile(field["path"], dtype=np.uint8)
    try:
        values = decompress(payload, encoded_shape)
    except Exception as exc:
        raise RuntimeError(
            f"{codec_label} decompression failed for {field['field']}."
        ) from exc
    decoded = np.asarray(values, dtype=dtype)
    if decoded.size != encoded_count:
        raise RuntimeError(
            f"{codec_label} decompression for {field['field']} returned "
            f"{decoded.size} values, expected {encoded_count}."
        )
    permutation = tuple(int(value) for value in field["axis_permutation"])
    if sorted(permutation) != list(range(len(encoded_shape))):
        raise RuntimeError(
            f"Invalid shaped axis permutation for {field['field']}: "
            f"{permutation}."
        )
    flips = tuple(
        bool(value)
        for value in field.get(
            "axis_flips",
            [False] * len(encoded_shape),
        )
    )
    if len(flips) != len(encoded_shape):
        raise RuntimeError(
            f"Invalid shaped axis flips for {field['field']}: {flips}."
        )
    oriented = decoded.reshape(encoded_shape)
    if any(flips):
        oriented = np.flip(
            oriented,
            axis=tuple(index for index, flipped in enumerate(flips) if flipped),
        )
    restored = np.transpose(
        oriented,
        np.argsort(permutation),
    )
    if restored.shape != base_shape:
        raise RuntimeError(
            f"{codec_label} restored shape for {field['field']} is "
            f"{restored.shape}, expected {base_shape}."
        )
    output = Path(out_path)
    require_output_path(output, force)
    np.ascontiguousarray(restored).tofile(output)
