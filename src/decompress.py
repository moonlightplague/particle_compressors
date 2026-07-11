import argparse
import time
import numpy as np
import h5py

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

def recombine_h5(manifest: Mapping[str, Any], dec_paths: Mapping[str, str], output_h5: Path) -> None:
    count = int(manifest["count"])
    scale = float(manifest["position_scale"]["value"])
    order_dtype = hp.order_dtype_from_manifest(manifest)
    order = np.fromfile(dec_paths["order"], dtype=order_dtype, count=count)
    if order.size != count:
        raise RuntimeError(f"Unexpected EOF reading {dec_paths['order']}; expected {count}, got {order.size}.")
    if count and (int(order.min()) < 0 or int(order.max()) >= count):
        raise RuntimeError("LCP order sidecar is not a valid index range for this particle count.")
    order_index = order.astype(np.intp, copy=False)

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
                restored = np.empty(count, dtype=target_dtype)
                restored[order_index] = converted
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
    dec_dir = work_dir / "decompressed"
    dec_dir.mkdir(parents=True, exist_ok=True)
    output_h5 = work_dir / "reconstructed.h5"
    hp.require_output_path(output_h5, args.force)

    t0 = time.perf_counter()
    order_dtype = hp.order_dtype_from_manifest(manifest)
    dec_paths = {
        "x": str(dec_dir / "x.f32.raw"),
        "y": str(dec_dir / "y.f32.raw"),
        "z": str(dec_dir / "z.f32.raw"),
        "order": str(dec_dir / f"order.{order_dtype.name}.raw"),
        "id": str(dec_dir / f"id.{np.dtype(manifest['fields']['id']['dtype']).name}.raw"),
        "vx": str(dec_dir / f"vx.{manifest['fields']['vx']['dtype']}.raw"),
        "vy": str(dec_dir / f"vy.{manifest['fields']['vy']['dtype']}.raw"),
        "vz": str(dec_dir / f"vz.{manifest['fields']['vz']['dtype']}.raw"),
    }
    for path in dec_paths.values():
        hp.require_output_path(Path(path), args.force)

    artifacts = manifest["artifacts"]["compressed"]
    hp.run_command(
        [
            str(tools.lcp),
            "-z",
            artifacts["positions"],
            "-o",
            dec_paths["x"],
            dec_paths["y"],
            dec_paths["z"],
            "-1",
            str(count),
            "-eb",
            str(manifest["error_bounds"]["positions_lcp_abs"]),
        ]
    )

    fields = manifest.get("compressed_fields")
    if not fields:
        raise RuntimeError("Manifest does not contain compressed_fields for the Python compressor pipeline.")
    decompress_pcodec_raw(
        fields["order"], dec_paths["order"], args.force
    )
    decompress_pcodec_raw(
        fields["id"], dec_paths["id"], args.force
    )
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
