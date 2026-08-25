"""HDF5 schema discovery, attribute transport, and reconstruction."""

from pathlib import Path
from typing import Any, Dict, Mapping

import h5py
import numpy as np

from src.constants import FIELD_ALIASES, LOGICAL_ORDER, POSITION_FIELDS
from src.runtime import read_raw


def resolve_fields(h5: h5py.File) -> Dict[str, str]:
    """Map logical particle fields to datasets using basename aliases."""

    available: Dict[str, str] = {}

    def visit(name: str, obj: Any) -> None:
        if isinstance(obj, h5py.Dataset):
            available[name.rsplit("/", 1)[-1].lower()] = name

    h5.visititems(visit)
    resolved = {}
    for logical, aliases in FIELD_ALIASES.items():
        matched = next(
            (
                available[alias.lower()]
                for alias in aliases
                if alias.lower() in available
            ),
            None,
        )
        if matched is None:
            raise RuntimeError(
                f"Could not find dataset for logical field {logical!r}; "
                f"tried aliases {aliases}."
            )
        resolved[logical] = matched
    return resolved


def serialize_attribute(value: Any) -> Dict[str, Any]:
    array = np.asarray(value)
    payload: Any = array.item() if array.shape == () else array.tolist()
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8", errors="surrogateescape")
    return {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "value": payload,
    }


def collect_attributes(obj: Any) -> Dict[str, Dict[str, Any]]:
    return {
        name: serialize_attribute(obj.attrs[name])
        for name in obj.attrs.keys()
    }


def restore_attribute(payload: Mapping[str, Any]) -> Any:
    dtype = np.dtype(payload["dtype"])
    value = payload["value"]
    shape = tuple(payload.get("shape", []))
    if dtype.kind == "S" and isinstance(value, str):
        value = value.encode("utf-8", errors="surrogateescape")
    array = np.asarray(value, dtype=dtype)
    return array.reshape(shape) if shape else array[()]


def apply_attributes(
    obj: Any,
    attributes: Mapping[str, Mapping[str, Any]],
) -> None:
    for name, payload in attributes.items():
        obj.attrs[name] = restore_attribute(payload)


def create_dataset(
    h5: h5py.File,
    path: str,
    dtype: np.dtype,
    count: int,
) -> h5py.Dataset:
    parent_path, _, name = path.rpartition("/")
    group = h5.require_group(parent_path) if parent_path else h5
    return group.create_dataset(name, shape=(count,), dtype=dtype)


class HDF5Recombiner:
    """Rebuild the source HDF5 schema from decompressed raw fields."""

    def __init__(
        self,
        manifest: Mapping[str, Any],
        decompressed_paths: Mapping[str, str],
        output_h5: Path,
    ) -> None:
        self.manifest = manifest
        self.paths = decompressed_paths
        self.output_h5 = output_h5
        self.count = int(manifest["count"])
        self.position_scale = float(manifest["position_scale"]["value"])

    def run(self) -> None:
        with h5py.File(self.output_h5, "w") as output:
            apply_attributes(output, self.manifest.get("root_attrs", {}))
            for logical in LOGICAL_ORDER:
                self._write_field(output, logical)

    def _write_field(self, output: h5py.File, logical: str) -> None:
        field = self.manifest["fields"][logical]
        target_dtype = np.dtype(field["dtype"])
        dataset = create_dataset(
            output,
            field["h5_path"],
            target_dtype,
            self.count,
        )
        apply_attributes(dataset, field.get("attrs", {}))

        if logical in POSITION_FIELDS:
            dataset[:] = self._reconstructed_position(logical, target_dtype)
        else:
            dataset[:] = read_raw(
                self.paths[logical],
                target_dtype,
                self.count,
            )

    def _reconstructed_position(
        self,
        logical: str,
        target_dtype: np.dtype,
    ) -> np.ndarray:
        decoded = read_raw(
            self.paths[logical],
            np.dtype("float32"),
            self.count,
        )
        values = decoded.astype(np.float64) * self.position_scale
        if np.issubdtype(target_dtype, np.integer):
            limits = np.iinfo(target_dtype)
            values = np.clip(np.rint(values), limits.min, limits.max)
        return values.astype(target_dtype)


def recombine_h5(
    manifest: Mapping[str, Any],
    decompressed_paths: Mapping[str, str],
    output_h5: Path,
) -> None:
    HDF5Recombiner(manifest, decompressed_paths, output_h5).run()
