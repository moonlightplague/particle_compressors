import argparse
import numpy as np
import tempfile

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
) -> Dict[str, Any]:
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


def compress_szo_raw(
    raw_path: str,
    dtype: str,
    compressed_path: str,
    field_name: str,
    count: int,
    abs_error_bound: float,
    force: bool,
) -> Dict[str, Any]:
    if not 0.0 <= abs_error_bound < 1.0:
        raise RuntimeError("SZO integer absolute error bound must be at least 0 and less than 1.")
    output = Path(compressed_path)
    hp.require_output_path(output, force)
    SZo, SZoConfig, SZoErrorBoundMode, SZoAlgorithm = hp.load_pyszo()
    source_dtype = np.dtype(dtype)
    source = np.ascontiguousarray(hp.read_raw(raw_path, source_dtype, count))
    encoded, transform = hp.encode_integers_for_szo(source)
    encoded_count = max(count, hp.SZO_MIN_VALUES)
    if encoded_count != count:
        padded = np.empty(encoded_count, dtype=encoded.dtype)
        padded[:count] = encoded
        padded[count:] = encoded[-1] if count else 0
        encoded = padded

    config = SZoConfig((encoded_count,))
    config.errorBoundMode = SZoErrorBoundMode.ABS
    config.absErrorBound = float(abs_error_bound)
    # Interpolation's RLE/FSE encoder crashes on large, high-entropy permutations.
    config.cmprAlgo = SZoAlgorithm.LORENZO_REG
    try:
        compressed, _ = SZo.compress(encoded, config, copy=True)
    except Exception as exc:
        raise RuntimeError(f"SZO compression failed for {field_name}.") from exc

    compressed = np.ascontiguousarray(compressed, dtype=np.uint8)
    compressed.tofile(output)
    return {
        "field": field_name,
        "codec": "szo",
        "dtype": str(source_dtype),
        "encoded_dtype": str(encoded.dtype),
        "integer_transform": transform,
        "abs_error_bound": float(abs_error_bound),
        "algorithm": "lorenzo_reg",
        "count": count,
        "encoded_count": encoded_count,
        "sha256": hp.integer_stream_sha256(source),
        "path": str(output),
        "bytes": int(compressed.size),
    }


def compress_integer_raw(
    codec: str,
    raw_path: str,
    dtype: str,
    compressed_path: str,
    field_name: str,
    count: int,
    szo_abs_error_bound: float,
    force: bool,
) -> Dict[str, Any]:
    if codec == "szo":
        return compress_szo_raw(
            raw_path,
            dtype,
            compressed_path,
            field_name,
            count,
            szo_abs_error_bound,
            force,
        )
    return compress_pcodec_raw(raw_path, dtype, compressed_path, field_name, count, force)

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
) -> Dict[str, Any]:
    PyszSZ, PyszConfig, PyszErrorBoundMode = hp.load_pysz()
    dt = np.dtype(dtype)
    if dt not in (np.dtype("float32"), np.dtype("float64")):
        raise RuntimeError(f"pysz compression expected float32/float64, got {dt} for {field_name}.")
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


def read_lcp_permutation(order_path: str, count: int) -> np.ndarray:
    order = hp.read_raw(order_path, np.dtype("int32"), count)
    if count and (int(order.min()) < 0 or int(order.max()) >= count):
        raise RuntimeError("LCP order is not a valid index range for this particle count.")
    if np.unique(order).size != count:
        raise RuntimeError("LCP order is not a permutation of the particle rows.")
    return order.astype(np.intp, copy=False)


def reorder_raw(
    raw_path: str,
    dtype: str,
    output_path: Path,
    count: int,
    order: np.ndarray,
    force: bool,
) -> str:
    hp.require_output_path(output_path, force)
    values = hp.read_raw(raw_path, np.dtype(dtype), count)
    np.ascontiguousarray(values[order]).tofile(output_path)
    return str(output_path)


def compress_lcp_triplet(
    tools: hp.ToolPaths,
    input_paths: Tuple[str, str, str],
    compressed_path: str,
    count: int,
    abs_error_bound: float,
    order_path: Path,
    force: bool,
) -> None:
    hp.require_output_path(Path(compressed_path), force)
    hp.require_output_path(order_path, force)
    hp.run_command(
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
    tools: hp.ToolPaths,
    input_paths: Tuple[str, str, str],
    compressed_path: str,
    chunks: int,
    chunk_size: int,
    abs_error_bound: float,
    order_path: Path,
    force: bool,
) -> None:
    if chunks <= 1:
        raise RuntimeError("Batched LCP compression requires at least two chunks.")
    hp.require_output_path(Path(compressed_path), force)
    hp.require_output_path(order_path, force)
    hp.run_command(
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


def velocity_order_bits(chunk_size: int) -> int:
    if chunk_size < 0:
        raise RuntimeError("Velocity chunk size must be non-negative.")
    return max(0, (chunk_size - 1).bit_length())


def compress_chunked_lcp_triplet(
    tools: hp.ToolPaths,
    input_paths: Tuple[str, str, str],
    compressed_path: str,
    count: int,
    chunk_size: int,
    abs_error_bound: float,
    order_path: Path,
    force: bool,
) -> Dict[str, int]:
    if chunk_size <= 0:
        raise RuntimeError("Chunked LCP compression requires a positive chunk size.")
    if chunk_size > 2**31:
        raise RuntimeError("Velocity chunk size cannot exceed 2^31 with int32 local order indices.")

    output = Path(compressed_path)
    hp.require_output_path(output, force)
    hp.require_output_path(order_path, force)
    chunk_count = (count + chunk_size - 1) // chunk_size
    chunks_per_batch = max(1, hp.LCP_CHUNK_BATCH_VALUES // chunk_size)
    full_chunks = count // chunk_size
    remainder = count % chunk_size
    segment_count = (
        (full_chunks + chunks_per_batch - 1) // chunks_per_batch
        + (1 if remainder else 0)
    )
    sources = tuple(
        np.memmap(path, dtype=np.float32, mode="r", shape=(count,))
        for path in input_paths
    )
    local_order = np.memmap(order_path, dtype=np.int32, mode="w+", shape=(count,))

    with tempfile.TemporaryDirectory(prefix="velocity_lcp_chunks_", dir=order_path.parent) as temp:
        temp_dir = Path(temp)
        chunk_inputs = tuple(temp_dir / f"{field}.f32.raw" for field in hp.VELOCITY_FIELDS)
        chunk_archive = temp_dir / "chunk.lcp"
        chunk_order = temp_dir / "order.i32.raw"

        with output.open("wb") as archive:
            archive.write(
                hp.LCP_CHUNK_HEADER.pack(
                    hp.LCP_CHUNK_MAGIC,
                    count,
                    chunk_size,
                    segment_count,
                )
            )
            start = 0
            while start < count:
                remaining = count - start
                if remaining < chunk_size:
                    chunks_in_segment = 1
                    segment_values = remaining
                else:
                    available_full_chunks = remaining // chunk_size
                    chunks_in_segment = min(chunks_per_batch, available_full_chunks)
                    segment_values = chunks_in_segment * chunk_size
                end = start + segment_values
                for source, path in zip(sources, chunk_inputs):
                    np.asarray(source[start:end]).tofile(path)

                if chunks_in_segment > 1:
                    compress_lcp_triplet_batch(
                        tools,
                        tuple(str(path) for path in chunk_inputs),
                        str(chunk_archive),
                        chunks_in_segment,
                        chunk_size,
                        abs_error_bound,
                        chunk_order,
                        True,
                    )
                else:
                    compress_lcp_triplet(
                        tools,
                        tuple(str(path) for path in chunk_inputs),
                        str(chunk_archive),
                        segment_values,
                        abs_error_bound,
                        chunk_order,
                        True,
                    )

                order = hp.read_raw(str(chunk_order), np.dtype("int32"), segment_values)
                for local_start in range(0, segment_values, chunk_size):
                    local_end = min(local_start + chunk_size, segment_values)
                    local_count = local_end - local_start
                    chunk_order_values = order[local_start:local_end]
                    if local_count and (
                        int(chunk_order_values.min()) < local_start
                        or int(chunk_order_values.max()) >= local_end
                    ):
                        raise RuntimeError("LCP order contains an index outside its velocity chunk.")
                    localized = chunk_order_values - local_start
                    if np.unique(localized).size != local_count:
                        raise RuntimeError("LCP order is not a local permutation within a velocity chunk.")
                    output_start = start + local_start
                    output_end = start + local_end
                    local_order[output_start:output_end] = localized

                payload = chunk_archive.read_bytes()
                archive.write(hp.LCP_CHUNK_ENTRY.pack(segment_values, len(payload)))
                archive.write(payload)
                start = end

    local_order.flush()
    del local_order
    del sources
    return {
        "chunk_size": chunk_size,
        "chunk_count": chunk_count,
        "lcp_segment_count": segment_count,
        "lcp_batch_values": hp.LCP_CHUNK_BATCH_VALUES,
        "order_bits_per_particle": velocity_order_bits(chunk_size),
    }


def compress(args: argparse.Namespace,
             manifest: Dict[str, Any],
             raw_paths: Dict[str, str],
             tools: hp.ToolPaths) -> Dict[str, Any]:
    work_dir = Path(args.work_dir).resolve()
    pre_dir = work_dir / "preprocessed"
    artifacts = manifest["artifacts"]["compressed"]
    count = int(manifest["count"])
    position_compressor = getattr(args, "pos_compressor", "lcp")
    velocity_compressor = args.vel_compressor
    velocity_chunk_size = int(getattr(args, "vel_chunk_size", 0))
    if velocity_chunk_size < 0:
        raise RuntimeError("--vel-chunk-size must be non-negative.")
    if velocity_chunk_size > 2**31:
        raise RuntimeError("--vel-chunk-size cannot exceed 2^31 when using int32 order indices.")
    if velocity_chunk_size and not (
        position_compressor == "lcp" and velocity_compressor == "lcp"
    ):
        raise RuntimeError(
            "--vel-chunk-size is only supported when both position and velocity compressors are lcp."
        )
    canonical_order = None
    canonical_mapping = "original_row"
    canonical_field = None
    canonical_order_artifact = None

    if position_compressor == "lcp":
        position_order_raw = pre_dir / "order.i32.raw"
        raw_paths["position_order"] = str(position_order_raw)
        compress_lcp_triplet(
            tools,
            tuple(raw_paths[logical] for logical in hp.POSITION_FIELDS),
            artifacts["positions"],
            count,
            float(manifest["error_bounds"]["positions_lcp_abs"]),
            position_order_raw,
            args.force,
        )
        canonical_order = read_lcp_permutation(str(position_order_raw), count)
        canonical_mapping = "lcp_position_sorted"
        canonical_field = "positions"
        canonical_order_artifact = "position_order"
        position_lcp = Path(artifacts["positions"])
        manifest["compressed_fields"]["positions"] = {
            "field": "positions",
            "codec": "lcp",
            "dtype": "float32",
            "source_dtypes": {
                logical: manifest["fields"][logical]["dtype"]
                for logical in hp.POSITION_FIELDS
            },
            "count": count,
            "abs_error_bound": manifest["error_bounds"]["positions_lcp_abs"],
            "path": str(position_lcp),
            "bytes": position_lcp.stat().st_size,
        }
    elif velocity_compressor == "lcp":
        velocity_order_raw = pre_dir / "velocity_order.i32.raw"
        raw_paths["velocity_order"] = str(velocity_order_raw)
        compress_lcp_triplet(
            tools,
            tuple(raw_paths[f"{logical}_lcp"] for logical in hp.VELOCITY_FIELDS),
            artifacts["velocities"],
            count,
            float(manifest["error_bounds"]["velocities_lcp_abs"]),
            velocity_order_raw,
            args.force,
        )
        canonical_order = read_lcp_permutation(str(velocity_order_raw), count)
        canonical_mapping = "lcp_velocity_sorted"
        canonical_field = "velocities"
        canonical_order_artifact = "velocity_order"

    id_dtype = manifest["fields"]["id"]["dtype"]
    ordered_id_path = raw_paths["id"]
    if canonical_order is not None:
        ordered_id_path = reorder_raw(
            raw_paths["id"],
            id_dtype,
            pre_dir / f"id.{canonical_mapping}.{np.dtype(id_dtype).name}.raw",
            count,
            canonical_order,
            args.force,
        )
        raw_paths["id_canonical_ordered"] = ordered_id_path

    manifest["ordering"] = {
        "reconstructed_rows": {
            "mapping": canonical_mapping,
            "original_row_order_restored": canonical_order is None,
            "canonical_lcp_field": canonical_field,
            "lcp_permutation_stored": False,
            "temporary_permutation_artifact": canonical_order_artifact,
        },
        "id": {"mapping": canonical_mapping},
    }
    if canonical_field == "positions":
        manifest["ordering"]["reconstructed_rows"]["position_permutation_stored"] = False
        manifest["ordering"]["id"]["replaces_lcp_position_order"] = True
    elif canonical_field == "velocities":
        manifest["ordering"]["reconstructed_rows"]["velocity_permutation_stored"] = False
        manifest["ordering"]["id"]["replaces_lcp_velocity_order"] = True
    manifest["format_version"] = 3

    manifest["compressed_fields"]["id"] = compress_integer_raw(
        args.lossless,
        ordered_id_path,
        id_dtype,
        artifacts["id"],
        "id",
        count,
        args.szo_abs_eb,
        args.force,
    )

    if position_compressor == "lcp":
        manifest["ordering"]["positions"] = {"mapping": canonical_mapping}
    else:
        for logical in hp.POSITION_FIELDS:
            position_path = raw_paths[logical]
            if canonical_order is not None:
                position_path = reorder_raw(
                    position_path,
                    "float32",
                    pre_dir / f"{logical}.{canonical_mapping}.float32.raw",
                    count,
                    canonical_order,
                    args.force,
                )
                raw_paths[f"{logical}_canonical_ordered"] = position_path
            manifest["compressed_fields"][logical] = compress_pysz_raw(
                position_path,
                "float32",
                artifacts[logical],
                logical,
                count,
                float(manifest["field_error_bounds"][logical]["compressor_abs"]),
                args.force,
            )
        manifest["ordering"]["positions"] = {"mapping": canonical_mapping}

    if velocity_compressor == "lcp":
        if position_compressor == "lcp":
            ordered_velocity_paths: Dict[str, str] = {}
            for logical in hp.VELOCITY_FIELDS:
                ordered_path = reorder_raw(
                    raw_paths[f"{logical}_lcp"],
                    "float32",
                    pre_dir / f"{logical}.{canonical_mapping}.float32.raw",
                    count,
                    canonical_order,
                    args.force,
                )
                ordered_velocity_paths[logical] = ordered_path
                raw_paths[f"{logical}_canonical_ordered"] = ordered_path

            velocity_order_raw = pre_dir / "velocity_order.i32.raw"
            raw_paths["velocity_order"] = str(velocity_order_raw)
            velocity_chunking = None
            if velocity_chunk_size:
                velocity_chunking = compress_chunked_lcp_triplet(
                    tools,
                    tuple(ordered_velocity_paths[logical] for logical in hp.VELOCITY_FIELDS),
                    artifacts["velocities"],
                    count,
                    velocity_chunk_size,
                    float(manifest["error_bounds"]["velocities_lcp_abs"]),
                    velocity_order_raw,
                    args.force,
                )
            else:
                compress_lcp_triplet(
                    tools,
                    tuple(ordered_velocity_paths[logical] for logical in hp.VELOCITY_FIELDS),
                    artifacts["velocities"],
                    count,
                    float(manifest["error_bounds"]["velocities_lcp_abs"]),
                    velocity_order_raw,
                    args.force,
                )
            velocity_order_field = compress_integer_raw(
                args.lossless,
                str(velocity_order_raw),
                "int32",
                artifacts["velocity_order"],
                "velocity_order",
                count,
                args.szo_abs_eb,
                args.force,
            )
            if velocity_chunking is not None:
                velocity_order_field.update(
                    {
                        **velocity_chunking,
                        "index_scope": "chunk_local",
                        "uncompressed_storage_dtype": "int32",
                        "compressed_bits_per_particle": (
                            8.0 * float(velocity_order_field["bytes"]) / count
                            if count
                            else 0.0
                        ),
                    }
                )
            else:
                velocity_order_field.update(
                    {
                        "chunk_size": 0,
                        "chunk_count": 1,
                        "index_scope": "global",
                        "order_bits_per_particle": velocity_order_bits(count),
                        "uncompressed_storage_dtype": "int32",
                        "compressed_bits_per_particle": (
                            8.0 * float(velocity_order_field["bytes"]) / count
                            if count
                            else 0.0
                        ),
                    }
                )
            manifest["compressed_fields"]["velocity_order"] = velocity_order_field

        velocity_lcp = Path(artifacts["velocities"])
        manifest["compressed_fields"]["velocities"] = {
            "field": "velocities",
            "codec": "lcp",
            "dtype": "float32",
            "source_dtypes": {
                logical: manifest["fields"][logical]["dtype"]
                for logical in hp.VELOCITY_FIELDS
            },
            "count": count,
            "abs_error_bound": manifest["error_bounds"]["velocities_lcp_abs"],
            "chunk_size": velocity_chunk_size if position_compressor == "lcp" else 0,
            "chunk_count": (
                (count + velocity_chunk_size - 1) // velocity_chunk_size
                if position_compressor == "lcp" and velocity_chunk_size
                else 1
            ),
            "container": (
                hp.LCP_CHUNK_CONTAINER
                if position_compressor == "lcp" and velocity_chunk_size
                else "native_lcp"
            ),
            "path": str(velocity_lcp),
            "bytes": velocity_lcp.stat().st_size,
        }
        if position_compressor == "lcp":
            manifest["ordering"]["velocities"] = {
                "mapping": "lcp_velocity_sorted_index_to_lcp_position_sorted_row",
                "field": "velocity_order",
                "index_scope": "chunk_local" if velocity_chunk_size else "global",
                "chunk_size": velocity_chunk_size,
            }
        else:
            manifest["ordering"]["velocities"] = {"mapping": canonical_mapping}
    else:
        for logical in hp.VELOCITY_FIELDS:
            dtype = manifest["fields"][logical]["dtype"]
            velocity_path = raw_paths[logical]
            if canonical_order is not None:
                velocity_path = reorder_raw(
                    velocity_path,
                    dtype,
                    pre_dir / f"{logical}.{canonical_mapping}.{np.dtype(dtype).name}.raw",
                    count,
                    canonical_order,
                    args.force,
                )
                raw_paths[f"{logical}_canonical_ordered"] = velocity_path
            manifest["compressed_fields"][logical] = compress_pysz_raw(
                velocity_path,
                dtype,
                artifacts[logical],
                logical,
                count,
                manifest["field_error_bounds"][logical]["abs"],
                args.force,
            )
        manifest["ordering"]["velocities"] = {"mapping": canonical_mapping}

    manifest["velocity_chunking"] = {
        "enabled": bool(velocity_chunk_size),
        "chunk_size": velocity_chunk_size,
        "chunk_count": (
            (count + velocity_chunk_size - 1) // velocity_chunk_size
            if velocity_chunk_size
            else 1
        ),
    }
    manifest["format_version"] = 4 if velocity_chunk_size else 3
    hp.update_compressed_size_metrics(manifest, work_dir)
    hp.write_json(work_dir / "manifest.json", manifest, force=True)
    return manifest
