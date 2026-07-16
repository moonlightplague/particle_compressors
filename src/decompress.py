import argparse
import time
import tempfile
import numpy as np
import h5py

from concurrent.futures import ThreadPoolExecutor

from pathlib import Path
from typing import Any, Dict, Mapping

import src.helpers as hp


def decompress_pcodec_raw(
    field: Mapping[str, Any],
    out_path: str,
    force: bool,
):
    standalone, _ = hp.load_pcodec()
    dt = np.dtype(field["dtype"])
    count = int(field["count"])
    out = Path(out_path)
    hp.require_output_path(out, force)
    payload = Path(field["path"]).read_bytes()
    data = standalone.simple_decompress(payload)
    if data is None:
        raise RuntimeError(f"pcodec decompression for {field['field']} returned no data.")
    data = np.asarray(data)
    if data.size != count:
        raise RuntimeError(
            f"pcodec decompression for {field['field']} returned {data.size} values, expected {count}."
        )
    if np.dtype(data.dtype) != dt:
        raise RuntimeError(
            f"pcodec decompression for {field['field']} returned dtype {data.dtype}, expected {dt}."
        )
    data.tofile(out)


def decompress_szo_raw(
    field: Mapping[str, Any],
    out_path: str,
    force: bool,
) -> None:
    SZo, _, _, _ = hp.load_pyszo()
    source_dtype = np.dtype(field["dtype"])
    encoded_dtype = np.dtype(field["encoded_dtype"])
    count = int(field["count"])
    encoded_count = int(field.get("encoded_count", count))
    compressed = np.fromfile(field["path"], dtype=np.uint8)
    try:
        encoded, _ = SZo.decompress(compressed, encoded_dtype, (encoded_count,))
    except Exception as exc:
        raise RuntimeError(f"SZO decompression failed for {field['field']}.") from exc
    encoded = np.asarray(encoded, dtype=encoded_dtype).reshape(-1)
    if encoded.size < count:
        raise RuntimeError(
            f"SZO decompression for {field['field']} returned {encoded.size} values, "
            f"expected at least {count}."
        )
    data = hp.decode_integers_from_szo(
        encoded[:count],
        source_dtype,
        str(field["integer_transform"]),
    )
    expected_sha256 = field.get("sha256")
    actual_sha256 = hp.integer_stream_sha256(data)
    if expected_sha256 and actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"SZO integrity check failed for {field['field']}: integer data was not reconstructed exactly."
        )
    out = Path(out_path)
    hp.require_output_path(out, force)
    data.tofile(out)


def decompress_integer_raw(
    field: Mapping[str, Any],
    out_path: str,
    force: bool,
) -> None:
    if field.get("codec") == "szo":
        decompress_szo_raw(field, out_path, force)
    else:
        decompress_pcodec_raw(field, out_path, force)


def decompress_pysz_raw(
    field: Mapping[str, Any],
    out_path: str,
    force: bool,
):
    PyszSZ, _, _ = hp.load_pysz()
    dt = np.dtype(field["dtype"])
    count = int(field["count"])
    encoded_count = int(field.get("encoded_count", count))
    out = Path(out_path)
    hp.require_output_path(out, force)
    compressed = np.fromfile(field["path"], dtype=np.uint8)
    try:
        data, _ = PyszSZ.decompress(compressed, dt, (encoded_count,))
    except Exception as exc:
        raise RuntimeError(f"pysz decompression failed for {field['field']}.") from exc
    data = np.asarray(data, dtype=dt)
    if data.size < count:
        raise RuntimeError(
            f"pysz decompression for {field['field']} returned {data.size} values, expected at least {count}."
        )
    data[:count].tofile(out)


def restore_attr(payload: Mapping[str, Any]) -> Any:
    dtype = np.dtype(payload["dtype"])
    value = payload["value"]
    shape = tuple(payload.get("shape", []))
    if dtype.kind == "S" and isinstance(value, str):
        value = value.encode("utf-8", errors="surrogateescape")
    arr = np.asarray(value, dtype=dtype)
    if shape:
        return arr.reshape(shape)
    return arr[()]

def apply_attrs(obj: Any, attrs: Mapping[str, Mapping[str, Any]]) -> None:
    for name, payload in attrs.items():
        obj.attrs[name] = restore_attr(payload)

def create_dataset(h5: h5py.File, path: str, dtype: np.dtype, count: int) -> h5py.Dataset:
    parent_path, _, name = path.rpartition("/")
    group = h5
    if parent_path:
        group = h5.require_group(parent_path)
    return group.create_dataset(name, shape=(count,), dtype=dtype)


def velocity_compressor_from_manifest(manifest: Mapping[str, Any]) -> str:
    configured = manifest.get("compressors", {}).get("velocities")
    if configured:
        return str(configured)
    if manifest.get("compressed_fields", {}).get("velocities", {}).get("codec") == "lcp":
        return "lcp"
    return "sz3"


def position_compressor_from_manifest(manifest: Mapping[str, Any]) -> str:
    if manifest.get("compressed_fields", {}).get("positions", {}).get("codec") == "lcp":
        return "lcp"
    if all(
        manifest.get("compressed_fields", {}).get(logical, {}).get("codec") == "pysz"
        for logical in hp.POSITION_FIELDS
    ):
        return "sz3"
    if "positions" in manifest.get("artifacts", {}).get("compressed", {}):
        return "lcp"
    configured = manifest.get("compressors", {}).get("positions")
    if configured:
        return str(configured)
    return "lcp"


def read_lcp_order(
    path: str,
    dtype: np.dtype,
    count: int,
    label: str,
    chunk_size: int = 0,
) -> np.ndarray:
    order = np.fromfile(path, dtype=dtype, count=count)
    if order.size != count:
        raise RuntimeError(f"Unexpected EOF reading {path}; expected {count}, got {order.size}.")
    if chunk_size < 0:
        raise RuntimeError(f"{label} has an invalid negative chunk size.")
    if not chunk_size:
        if count and (int(order.min()) < 0 or int(order.max()) >= count):
            raise RuntimeError(f"{label} is not a valid index range for this particle count.")
        if np.unique(order).size != count:
            raise RuntimeError(f"{label} is not a permutation of the particle rows.")
        return order.astype(np.intp, copy=False)

    expanded = np.empty(count, dtype=np.intp)
    for start in range(0, count, chunk_size):
        end = min(start + chunk_size, count)
        local = order[start:end]
        local_count = end - start
        if local_count and (int(local.min()) < 0 or int(local.max()) >= local_count):
            raise RuntimeError(f"{label} contains an index outside its velocity chunk.")
        if np.unique(local).size != local_count:
            raise RuntimeError(f"{label} contains a chunk that is not a local permutation.")
        expanded[start:end] = start + local.astype(np.intp, copy=False)
    return expanded


def run_lcp_decompress(
    tools: hp.ToolPaths,
    compressed_path: str,
    output_paths: Mapping[str, str],
    fields: tuple[str, str, str],
    count: int,
    abs_error_bound: float,
) -> None:
    hp.run_command(
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
    tools: hp.ToolPaths,
    compressed_path: str,
    output_paths: Mapping[str, str],
    fields: tuple[str, str, str],
    chunks: int,
    chunk_size: int,
    abs_error_bound: float,
) -> None:
    hp.run_command(
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


def run_chunked_lcp_decompress(
    tools: hp.ToolPaths,
    compressed_path: str,
    output_paths: Mapping[str, str],
    fields: tuple[str, str, str],
    count: int,
    chunk_size: int,
    abs_error_bound: float,
    workers: int = 1,
) -> None:
    if chunk_size <= 0:
        raise RuntimeError("Chunked LCP decompression requires a positive chunk size.")
    if workers <= 0:
        raise RuntimeError("Chunked LCP decompression requires at least one worker.")
    sinks = {
        field: np.memmap(output_paths[field], dtype=np.float32, mode="w+", shape=(count,))
        for field in fields
    }

    temp_parent = Path(output_paths[fields[0]]).parent
    with tempfile.TemporaryDirectory(prefix="velocity_lcp_chunks_", dir=temp_parent) as temp:
        temp_dir = Path(temp)
        segments = []
        with Path(compressed_path).open("rb") as archive:
            header = archive.read(hp.LCP_CHUNK_HEADER.size)
            if len(header) != hp.LCP_CHUNK_HEADER.size:
                raise RuntimeError("Truncated chunked LCP velocity header.")
            magic, stored_count, stored_chunk_size, stored_segment_count = (
                hp.LCP_CHUNK_HEADER.unpack(header)
            )
            if magic != hp.LCP_CHUNK_MAGIC:
                raise RuntimeError("Invalid chunked LCP velocity archive magic.")
            if stored_count != count or stored_chunk_size != chunk_size:
                raise RuntimeError(
                    "Chunked LCP velocity archive metadata does not match the manifest."
                )
            start = 0
            for segment_index in range(stored_segment_count):
                entry = archive.read(hp.LCP_CHUNK_ENTRY.size)
                if len(entry) != hp.LCP_CHUNK_ENTRY.size:
                    raise RuntimeError("Truncated chunked LCP velocity entry.")
                segment_values, payload_size = hp.LCP_CHUNK_ENTRY.unpack(entry)
                if not segment_values or segment_values > count - start:
                    raise RuntimeError("Chunked LCP velocity archive has an invalid segment size.")
                end = start + segment_values
                payload = archive.read(payload_size)
                if len(payload) != payload_size:
                    raise RuntimeError("Truncated chunked LCP velocity payload.")
                segment_dir = temp_dir / f"segment_{segment_index:08d}"
                segment_dir.mkdir()
                chunk_archive = segment_dir / "chunk.lcp"
                chunk_archive.write_bytes(payload)
                chunk_outputs = {
                    field: str(segment_dir / f"{field}.f32.raw")
                    for field in fields
                }
                segments.append(
                    (start, segment_values, chunk_archive, chunk_outputs)
                )
                start = end

            if start != count:
                raise RuntimeError("Chunked LCP velocity archive does not cover every particle.")

            if archive.read(1):
                raise RuntimeError("Chunked LCP velocity archive contains trailing data.")

        def decompress_segment(segment) -> None:
            start, segment_values, chunk_archive, chunk_outputs = segment
            end = start + segment_values
            try:
                if segment_values > chunk_size:
                    if segment_values % chunk_size:
                        raise RuntimeError(
                            "Chunked LCP velocity batch is not aligned to the configured chunk size."
                        )
                    run_lcp_decompress_batch(
                        tools,
                        str(chunk_archive),
                        chunk_outputs,
                        fields,
                        segment_values // chunk_size,
                        chunk_size,
                        abs_error_bound,
                    )
                else:
                    run_lcp_decompress(
                        tools,
                        str(chunk_archive),
                        chunk_outputs,
                        fields,
                        segment_values,
                        abs_error_bound,
                    )
                for field in fields:
                    sinks[field][start:end] = hp.read_raw(
                        chunk_outputs[field], np.dtype("float32"), segment_values
                    )
            finally:
                chunk_archive.unlink(missing_ok=True)
                for path in chunk_outputs.values():
                    Path(path).unlink(missing_ok=True)

        effective_workers = min(workers, len(segments))
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            list(executor.map(decompress_segment, segments))

    for sink in sinks.values():
        sink.flush()


def recombine_h5(manifest: Mapping[str, Any], dec_paths: Mapping[str, str], output_h5: Path) -> None:
    count = int(manifest["count"])
    scale = float(manifest["position_scale"]["value"])
    order_index = None
    if "order" in dec_paths:
        order_index = read_lcp_order(
            dec_paths["order"],
            hp.order_dtype_from_manifest(manifest),
            count,
            "LCP position order sidecar",
        )
    velocity_compressor = velocity_compressor_from_manifest(manifest)
    velocity_order_index = None
    if velocity_compressor == "lcp" and "velocity_order" in dec_paths:
        velocity_order_field = manifest["compressed_fields"]["velocity_order"]
        velocity_order_index = read_lcp_order(
            dec_paths["velocity_order"],
            np.dtype(velocity_order_field["dtype"]),
            count,
            "LCP velocity order sidecar",
            int(velocity_order_field.get("chunk_size", 0)),
        )

    with h5py.File(output_h5, "w") as out:
        apply_attrs(out, manifest.get("root_attrs", {}))

        for logical in hp.LOGICAL_ORDER:
            field = manifest["fields"][logical]
            target_dtype = np.dtype(field["dtype"])
            dset = create_dataset(out, field["h5_path"], target_dtype, count)
            apply_attrs(dset, field.get("attrs", {}))

            if logical == "id":
                dset[:] = hp.read_raw(dec_paths[logical], target_dtype, count)
            elif logical in hp.POSITION_FIELDS:
                info = np.iinfo(target_dtype) if np.issubdtype(target_dtype, np.integer) else None
                decoded = np.fromfile(dec_paths[logical], dtype=np.float32, count=count)
                if decoded.size != count:
                    raise RuntimeError(
                        f"Unexpected EOF reading {dec_paths[logical]}; expected {count}, got {decoded.size}."
                    )
                values64 = decoded.astype(np.float64) * scale
                if info is not None:
                    values64 = np.rint(values64)
                    values64 = np.clip(values64, info.min, info.max)
                converted = values64.astype(target_dtype)
                if order_index is None:
                    dset[:] = converted
                else:
                    restored = np.empty(count, dtype=target_dtype)
                    restored[order_index] = converted
                    dset[:] = restored
            elif logical in hp.VELOCITY_FIELDS and velocity_order_index is not None:
                decoded = np.fromfile(dec_paths[logical], dtype=np.float32, count=count)
                if decoded.size != count:
                    raise RuntimeError(
                        f"Unexpected EOF reading {dec_paths[logical]}; expected {count}, got {decoded.size}."
                    )
                converted = decoded.astype(target_dtype)
                restored = np.empty(count, dtype=target_dtype)
                restored[velocity_order_index] = converted
                dset[:] = restored
            else:
                dset[:] = hp.read_raw(dec_paths[logical], target_dtype, count)


def decompress(args: argparse.Namespace, 
                       tools: hp.ToolPaths) -> Dict[str, Any]:
    work_dir = Path(args.work_dir).resolve()
    manifest_path = work_dir / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"Missing manifest: {manifest_path}")
    manifest = hp.read_json(manifest_path)

    count = int(manifest["count"])
    fields = manifest.get("compressed_fields")
    if not fields:
        raise RuntimeError("Manifest does not contain compressed_fields for the Python compressor pipeline.")
    has_position_order = "order" in fields
    has_velocity_order = "velocity_order" in fields
    position_compressor = position_compressor_from_manifest(manifest)
    velocity_compressor = velocity_compressor_from_manifest(manifest)
    dec_dir = work_dir / "decompressed"
    dec_dir.mkdir(parents=True, exist_ok=True)
    output_h5 = work_dir / "reconstructed.h5"
    hp.require_output_path(output_h5, args.force)

    t0 = time.perf_counter()
    dec_paths = {
        "x": str(dec_dir / "x.f32.raw"),
        "y": str(dec_dir / "y.f32.raw"),
        "z": str(dec_dir / "z.f32.raw"),
        "id": str(dec_dir / f"id.{np.dtype(manifest['fields']['id']['dtype']).name}.raw"),
        "vx": str(dec_dir / f"vx.{manifest['fields']['vx']['dtype']}.raw"),
        "vy": str(dec_dir / f"vy.{manifest['fields']['vy']['dtype']}.raw"),
        "vz": str(dec_dir / f"vz.{manifest['fields']['vz']['dtype']}.raw"),
    }
    if has_position_order:
        order_dtype = hp.order_dtype_from_manifest(manifest)
        dec_paths["order"] = str(dec_dir / f"order.{order_dtype.name}.raw")
    if velocity_compressor == "lcp":
        dec_paths["vx"] = str(dec_dir / "vx.f32.raw")
        dec_paths["vy"] = str(dec_dir / "vy.f32.raw")
        dec_paths["vz"] = str(dec_dir / "vz.f32.raw")
    if has_velocity_order:
        velocity_order_dtype = np.dtype(manifest["compressed_fields"]["velocity_order"]["dtype"])
        dec_paths["velocity_order"] = str(
            dec_dir / f"velocity_order.{velocity_order_dtype.name}.raw"
        )
    for path in dec_paths.values():
        hp.require_output_path(Path(path), args.force)

    artifacts = manifest["artifacts"]["compressed"]
    if position_compressor == "lcp":
        run_lcp_decompress(
            tools,
            artifacts["positions"],
            dec_paths,
            hp.POSITION_FIELDS,
            count,
            float(manifest["error_bounds"]["positions_lcp_abs"]),
        )
    else:
        for logical in hp.POSITION_FIELDS:
            decompress_pysz_raw(
                fields[logical], dec_paths[logical], args.force
            )

    if has_position_order:
        decompress_integer_raw(
            fields["order"], dec_paths["order"], args.force
        )
    decompress_integer_raw(
        fields["id"], dec_paths["id"], args.force
    )
    if velocity_compressor == "lcp":
        velocity_chunk_size = int(fields["velocities"].get("chunk_size", 0))
        if velocity_chunk_size:
            velocity_lcp_start = time.perf_counter()
            run_chunked_lcp_decompress(
                tools,
                artifacts["velocities"],
                dec_paths,
                hp.VELOCITY_FIELDS,
                count,
                velocity_chunk_size,
                float(manifest["error_bounds"]["velocities_lcp_abs"]),
                hp.resolve_lcp_chunk_workers(
                    int(getattr(args, "vel_chunk_workers", 0))
                ),
            )
            manifest.setdefault("timing", {})["velocity_lcp_decompress_wall_seconds"] = (
                time.perf_counter() - velocity_lcp_start
            )
        else:
            run_lcp_decompress(
                tools,
                artifacts["velocities"],
                dec_paths,
                hp.VELOCITY_FIELDS,
                count,
                float(manifest["error_bounds"]["velocities_lcp_abs"]),
            )
        if has_velocity_order:
            decompress_integer_raw(
                fields["velocity_order"], dec_paths["velocity_order"], args.force
            )
    else:
        for logical in hp.VELOCITY_FIELDS:
            decompress_pysz_raw(
                fields[logical], dec_paths[logical], args.force
            )

    recombine_start = time.perf_counter()
    recombine_h5(manifest, dec_paths, output_h5)
    recombine_seconds = time.perf_counter() - recombine_start

    manifest.setdefault("timing", {})["decompress_and_recombine_wall_seconds"] = time.perf_counter() - t0
    manifest["timing"]["recombine_h5_wall_seconds"] = recombine_seconds
    manifest["artifacts"]["decompressed"] = dec_paths
    manifest["artifacts"]["reconstructed_h5"] = str(output_h5)
    manifest.setdefault("sizes", {})["reconstructed_h5_file_bytes"] = output_h5.stat().st_size
    hp.update_compressed_size_metrics(manifest, work_dir)
    hp.write_json(manifest_path, manifest, force=True)
    return manifest
