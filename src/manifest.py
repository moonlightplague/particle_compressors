"""Manifest queries and compressed-size bookkeeping."""

import math
from pathlib import Path
from typing import Any, Dict, Mapping

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


def lossy_compressor_from_manifest(manifest: Mapping[str, Any]) -> str:
    configured = manifest.get("compressors", {}).get("lossy")
    compressed_fields = manifest.get("compressed_fields", {})
    lossy_fields = (*POSITION_FIELDS, *VELOCITY_FIELDS)
    codecs = {
        compressed_fields.get(field, {}).get("codec")
        for field in lossy_fields
    }
    if configured in ("szo", "sz3", "sperr"):
        if compressed_fields:
            expected_codec = {
                "szo": "szo",
                "sz3": "pysz",
                "sperr": "sperr",
            }[configured]
            if codecs != {expected_codec}:
                raise RuntimeError(
                    "Manifest lossy field metadata does not match its "
                    "configured compressor."
                )
        return str(configured)

    if codecs == {"szo"}:
        return "szo"
    if codecs == {"pysz"}:
        return "sz3"
    if codecs == {"sperr"}:
        return "sperr"
    raise RuntimeError(
        "Manifest does not select one supported lossy compressor for all "
        "position and velocity fields."
    )
