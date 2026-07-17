"""Manifest queries and compressed-size bookkeeping."""

import math
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np

from src.constants import POSITION_FIELDS, VELOCITY_FIELDS
from src.runtime import json_size_bytes


def compressed_sizes(work_dir: Path) -> Dict[str, int]:
    sizes: Dict[str, int] = {}
    manifest_path = work_dir / "manifest.json"
    compressed_dir = work_dir / "compressed"

    if manifest_path.exists():
        sizes["manifest.json"] = manifest_path.stat().st_size
    if compressed_dir.exists():
        for path in sorted(compressed_dir.rglob("*")):
            if path.is_file():
                sizes[str(path.relative_to(work_dir))] = path.stat().st_size
    return sizes


def update_compressed_size_metrics(
    manifest: Dict[str, Any],
    work_dir: Path,
) -> None:
    components = compressed_sizes(work_dir)
    for _ in range(10):
        sizes = manifest.setdefault("sizes", {})
        compressed_total = int(sum(components.values()))
        selected_total = int(sizes["selected_original_payload_bytes"])

        sizes["compressed_components_bytes"] = dict(components)
        sizes["compressed_total_bytes"] = compressed_total
        sizes["payload_compression_ratio"] = (
            selected_total / compressed_total if compressed_total else math.inf
        )
        if "input_h5_file_bytes" in manifest:
            sizes["h5_file_to_compressed_ratio"] = (
                int(manifest["input_h5_file_bytes"]) / compressed_total
                if compressed_total
                else math.inf
            )

        rendered_manifest_bytes = json_size_bytes(manifest)
        if components.get("manifest.json") == rendered_manifest_bytes:
            break
        components["manifest.json"] = rendered_manifest_bytes


def order_dtype_from_manifest(manifest: Mapping[str, Any]) -> np.dtype:
    field = manifest.get("compressed_fields", {}).get("order", {})
    dtype = np.dtype(field.get("dtype", manifest.get("order_dtype", "int64")))
    if dtype not in (np.dtype("int32"), np.dtype("int64")):
        raise RuntimeError(f"Unsupported LCP order dtype in manifest: {dtype}.")
    return dtype


def velocity_compressor_from_manifest(manifest: Mapping[str, Any]) -> str:
    configured = manifest.get("compressors", {}).get("velocities")
    if configured:
        return str(configured)

    compressed_fields = manifest.get("compressed_fields", {})
    if compressed_fields.get("velocities", {}).get("codec") == "lcp":
        return "lcp"
    if _all_fields_use_codec(compressed_fields, VELOCITY_FIELDS, "szo"):
        return "szo"
    return "sz3"


def position_compressor_from_manifest(manifest: Mapping[str, Any]) -> str:
    compressed_fields = manifest.get("compressed_fields", {})
    if compressed_fields.get("positions", {}).get("codec") == "lcp":
        return "lcp"
    if _all_fields_use_codec(compressed_fields, POSITION_FIELDS, "pysz"):
        return "sz3"
    if _all_fields_use_codec(compressed_fields, POSITION_FIELDS, "szo"):
        return "szo"
    if "positions" in manifest.get("artifacts", {}).get("compressed", {}):
        return "lcp"

    configured = manifest.get("compressors", {}).get("positions")
    return str(configured) if configured else "lcp"


def _all_fields_use_codec(
    compressed_fields: Mapping[str, Any],
    fields: tuple[str, str, str],
    codec: str,
) -> bool:
    return all(
        compressed_fields.get(logical, {}).get("codec") == codec
        for logical in fields
    )

