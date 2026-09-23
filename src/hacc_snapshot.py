"""Expose HACC's six little-endian float32 files without copying the data."""

from pathlib import Path
from typing import Any, Dict

import h5py

HACC_FORMAT = "hacc_f32_fields_v1"
HACC_FIELDS = {
    "x": "xx", "y": "yy", "z": "zz",
    "vx": "vx", "vy": "vy", "vz": "vz",
}


def is_hacc_directory(path: Path) -> bool:
    # Recognize incomplete inputs too, so validation reports missing fields.
    return path.is_dir() and any(
        (path / f"{name}.f32").exists() for name in HACC_FIELDS.values()
    )


def hacc_source_metadata(path: Path) -> Dict[str, Any]:
    fields = {}
    for logical, name in HACC_FIELDS.items():
        file = path / f"{name}.f32"
        if not file.is_file():
            raise RuntimeError(f"HACC input is missing field file: {file}")
        size = file.stat().st_size
        if size == 0 or size % 4:
            raise RuntimeError(
                f"HACC field {file} must contain a non-empty sequence "
                "of float32 values."
            )
        fields[logical] = {
            "file": str(file.resolve()),
            "dtype": "<f4",
            "shape": [size // 4],
            "byte_count": size,
        }
    if len({field["shape"][0] for field in fields.values()}) != 1:
        raise RuntimeError("HACC fields do not have the same length.")
    return {
        "format": HACC_FORMAT,
        "byte_order": "little_endian",
        "bytes_per_particle": 24,
        "fields": fields,
    }


def write_hacc_adapter(path: Path, directory: Path) -> Path:
    metadata = hacc_source_metadata(path)
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "hacc.h5"
    partial = output.with_suffix(".h5.partial")
    try:
        with h5py.File(partial, "w") as h5:
            for logical, field in metadata["fields"].items():
                h5.create_dataset(
                    HACC_FIELDS[logical],
                    shape=tuple(field["shape"]),
                    dtype="<f4",
                    external=[(field["file"], 0, field["byte_count"])],
                )
        partial.replace(output)
    finally:
        partial.unlink(missing_ok=True)
    return output
