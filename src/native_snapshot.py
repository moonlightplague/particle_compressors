"""Read field-major ``cfg_*``/``dat_*`` particle snapshot files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import h5py
import numpy as np


CONFIG_VALUE_COUNT = 43
BYTES_PER_PARTICLE = 32
NATIVE_FORMAT = "cfg_dat_field_major_v1"
FIELD_LAYOUT: Tuple[Tuple[str, str], ...] = (
    ("posx", "<i4"),
    ("posy", "<i4"),
    ("posz", "<i4"),
    ("velx", "<f4"),
    ("vely", "<f4"),
    ("velz", "<f4"),
    ("id", "<u8"),
)


@dataclass(frozen=True)
class NativeSnapshotHeader:
    """Header values needed to expose one native particle partition."""

    config_path: Path
    data_path: Path
    rank: int
    proc_size: int
    npart_total: int
    npart: int
    nsidemesh: int
    bitwidth: int

    @property
    def expected_data_bytes(self) -> int:
        return self.npart * BYTES_PER_PARTICLE

    def source_metadata(self) -> Dict[str, Any]:
        offset = 0
        fields: Dict[str, Dict[str, Any]] = {}
        for name, dtype_text in FIELD_LAYOUT:
            dtype = np.dtype(dtype_text)
            byte_count = self.npart * dtype.itemsize
            fields[name] = {
                "byte_offset": offset,
                "byte_count": byte_count,
                "dtype": dtype.name,
                "shape": [self.npart],
            }
            offset += byte_count
        return {
            "format": NATIVE_FORMAT,
            "config_file": str(self.config_path),
            "data_file": str(self.data_path),
            "data_file_bytes": self.data_path.stat().st_size,
            "storage_order": "field_major",
            "byte_order": "little_endian",
            "bytes_per_particle": BYTES_PER_PARTICLE,
            "fields": fields,
        }


@dataclass(frozen=True)
class AdaptedParticleInput:
    """An original input and the HDF5 path used by the existing pipeline."""

    original_path: Path
    h5_path: Path
    native_header: Optional[NativeSnapshotHeader] = None


def native_config_path(data_path: Path) -> Path:
    """Return the configuration filename paired with a ``dat_*`` file."""

    if not data_path.name.startswith("dat_"):
        raise RuntimeError(
            f"Native particle data filename must start with 'dat_': {data_path}"
        )
    return data_path.with_name(f"cfg_{data_path.name[4:]}")


def is_native_data_path(path: Path) -> bool:
    """Return whether a path has the native data naming and config pair."""

    return (
        path.is_file()
        and path.name.startswith("dat_")
        and path.suffix != ".h5"
        and native_config_path(path).is_file()
    )


def read_native_header(data_path: Path) -> NativeSnapshotHeader:
    """Parse and validate the fixed 43-value native configuration file."""

    data_path = data_path.resolve()
    if not data_path.is_file():
        raise RuntimeError(
            f"Native particle data file does not exist: {data_path}"
        )
    config_path = native_config_path(data_path)
    if not config_path.is_file():
        raise RuntimeError(
            f"Native particle data file {data_path} has no matching "
            f"configuration file {config_path}."
        )

    parsed_integers = _read_config_integers(config_path)

    header = NativeSnapshotHeader(
        config_path=config_path,
        data_path=data_path,
        rank=parsed_integers[0],
        proc_size=parsed_integers[1],
        npart_total=parsed_integers[2],
        npart=parsed_integers[9],
        nsidemesh=parsed_integers[41],
        bitwidth=parsed_integers[42],
    )
    _validate_header(header)
    return header


def discover_native_data_files(directory: Path) -> List[Path]:
    """Discover and validate non-empty native partitions in a directory."""

    for config_path in directory.iterdir():
        if (
            not config_path.is_file()
            or not config_path.name.startswith("cfg_")
        ):
            continue
        suffix = config_path.name[4:]
        data_path = config_path.with_name(f"dat_{suffix}")
        converted_h5 = config_path.with_name(f"dat_{suffix}.h5")
        if data_path.is_file() or converted_h5.is_file():
            continue
        parsed_integers = _read_config_integers(config_path.resolve())
        if parsed_integers[9] != 0:
            raise RuntimeError(
                f"Native particle configuration {config_path.resolve()} "
                f"declares {parsed_integers[9]} particles but its data file "
                f"{data_path.resolve()} is missing."
            )

    candidates = sorted(
        (
            path.resolve()
            for path in directory.iterdir()
            if path.is_file()
            and path.name.startswith("dat_")
            and path.suffix != ".h5"
        ),
        key=_natural_name_key,
    )
    return [read_native_header(path).data_path for path in candidates]


def adapt_particle_input(
    input_path: Path,
    adapter_directory: Path,
) -> AdaptedParticleInput:
    """Expose native blocks as HDF5 external datasets, or pass HDF5 through."""

    input_path = input_path.resolve()
    if not input_path.is_file():
        raise RuntimeError(f"Particle input file does not exist: {input_path}")
    if h5py.is_hdf5(input_path):
        return AdaptedParticleInput(input_path, input_path)
    if not input_path.name.startswith("dat_") or input_path.suffix == ".h5":
        raise RuntimeError(
            f"Unsupported particle input file {input_path}; expected HDF5 or "
            "a native dat_* file with a matching cfg_* file."
        )

    header = read_native_header(input_path)
    adapter_path = adapter_directory.resolve() / f"{input_path.name}.h5"
    _write_hdf5_adapter(header, adapter_path)
    return AdaptedParticleInput(input_path, adapter_path, header)


def adapt_particle_inputs(
    input_paths: Iterable[Path],
    adapter_directory: Path,
) -> List[AdaptedParticleInput]:
    """Adapt a sequence while preserving its deterministic input order."""

    return [
        adapt_particle_input(path, adapter_directory)
        for path in input_paths
    ]


def _write_hdf5_adapter(
    header: NativeSnapshotHeader,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_name(f".{output_path.name}.partial")
    if partial_path.exists():
        partial_path.unlink()

    try:
        with h5py.File(partial_path, "w") as output:
            output.attrs["bitwidth"] = np.int32(header.bitwidth)
            output.attrs["npart"] = np.int32(header.npart)
            output.attrs["npart_total"] = np.uint64(header.npart_total)
            output.attrs["nsidemesh"] = np.int32(header.nsidemesh)
            output.attrs["proc_size"] = np.int32(header.proc_size)
            output.attrs["rank"] = np.int32(header.rank)

            offset = 0
            for name, dtype_text in FIELD_LAYOUT:
                dtype = np.dtype(dtype_text)
                byte_count = header.npart * dtype.itemsize
                output.create_dataset(
                    name,
                    shape=(header.npart,),
                    dtype=dtype,
                    external=[
                        (str(header.data_path), offset, byte_count),
                    ],
                )
                offset += byte_count
        partial_path.replace(output_path)
    except Exception:
        if partial_path.exists():
            partial_path.unlink()
        raise


def _validate_header(header: NativeSnapshotHeader) -> None:
    if header.rank < 0 or header.rank >= header.proc_size:
        raise RuntimeError(
            f"Native particle configuration {header.config_path} has rank "
            f"{header.rank} outside [0, {header.proc_size})."
        )
    if header.proc_size <= 0:
        raise RuntimeError(
            f"Native particle configuration {header.config_path} has a "
            "non-positive process count."
        )
    if header.npart <= 0:
        raise RuntimeError(
            f"Native particle data file {header.data_path} must contain at "
            "least one particle."
        )
    if header.npart_total < header.npart:
        raise RuntimeError(
            f"Native particle configuration {header.config_path} has local "
            "particle count larger than its total."
        )
    if header.nsidemesh <= 0 or header.bitwidth <= 0:
        raise RuntimeError(
            f"Native particle configuration {header.config_path} has an "
            "invalid mesh size or position bit width."
        )
    actual_bytes = header.data_path.stat().st_size
    if actual_bytes != header.expected_data_bytes:
        raise RuntimeError(
            f"Native particle data file {header.data_path} has "
            f"{actual_bytes} bytes; expected {header.expected_data_bytes} "
            f"({header.npart} particles x {BYTES_PER_PARTICLE} bytes)."
        )


def _read_config_integers(config_path: Path) -> Dict[int, int]:
    try:
        values = config_path.read_text(encoding="ascii").split()
    except (OSError, UnicodeError) as exc:
        raise RuntimeError(
            "Could not read native particle configuration "
            f"{config_path}: {exc}"
        ) from exc
    if len(values) != CONFIG_VALUE_COUNT:
        raise RuntimeError(
            f"Native particle configuration {config_path} must contain "
            f"{CONFIG_VALUE_COUNT} values, got {len(values)}."
        )

    integer_lines = (0, 1, 2, *range(9, CONFIG_VALUE_COUNT))
    parsed_integers: Dict[int, int] = {}
    try:
        for index in integer_lines:
            parsed_integers[index] = int(values[index])
        for index in range(3, 9):
            float(values[index])
    except ValueError as exc:
        raise RuntimeError(
            f"Native particle configuration {config_path} contains an "
            "invalid numeric value."
        ) from exc
    return parsed_integers


def _natural_name_key(path: Path) -> Tuple[Any, ...]:
    parts = path.name.split(".")
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part)
        for part in parts
    )


__all__ = [
    "AdaptedParticleInput",
    "BYTES_PER_PARTICLE",
    "FIELD_LAYOUT",
    "NATIVE_FORMAT",
    "NativeSnapshotHeader",
    "adapt_particle_input",
    "adapt_particle_inputs",
    "discover_native_data_files",
    "is_native_data_path",
    "native_config_path",
    "read_native_header",
]
