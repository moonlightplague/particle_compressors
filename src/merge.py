"""Streaming validation and concatenation of particle HDF5 chunks."""

from __future__ import annotations

import math
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import h5py
import numpy as np

from src.constants import LOGICAL_ORDER
from src.hdf5_io import (
    apply_attributes,
    collect_attributes,
    create_dataset,
    resolve_fields,
    restore_attribute,
    serialize_attribute,
)
from src.runtime import require_output_path


COPY_CHUNK_ROWS = 1_048_576
MAX_BITMAP_BYTES = 512 * 1024 * 1024
MAX_IN_MEMORY_ID_BYTES = 256 * 1024 * 1024
PARTITION_TARGET_BYTES = 32 * 1024 * 1024
MUTABLE_ROOT_ATTRIBUTES = ("npart", "npart_total", "proc_size", "rank")


@dataclass(frozen=True)
class SourceFile:
    path: Path
    fields: Dict[str, str]
    count: int


@dataclass(frozen=True)
class MergePlan:
    sources: Tuple[SourceFile, ...]
    output_fields: Dict[str, str]
    field_dtypes: Dict[str, str]
    field_attributes: Dict[str, Dict[str, Dict[str, Any]]]
    root_attributes: Dict[str, Dict[str, Any]]
    count: int
    id_minimum: int
    id_maximum: int
    input_file_bytes: int


@dataclass(frozen=True)
class MergeResult:
    output_h5: Path
    metadata: Dict[str, Any]


class _IdOverlapChecker:
    """Incrementally collect IDs and reject any repeated value."""

    algorithm = "unknown"

    def add(self, values: np.ndarray, source: Path) -> None:
        raise NotImplementedError

    def finish(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass


class _BitmapIdOverlapChecker(_IdOverlapChecker):
    algorithm = "dense_bitmap"

    def __init__(self, minimum: int, maximum: int) -> None:
        self.minimum = minimum
        span = maximum - minimum + 1
        self.bitmap = np.zeros((span + 7) // 8, dtype=np.uint8)

    def add(self, values: np.ndarray, source: Path) -> None:
        if values.dtype.kind == "u":
            offsets = values.astype(np.uint64, copy=False) - np.uint64(
                self.minimum
            )
        else:
            offsets = values.astype(np.int64, copy=False) - self.minimum
        unique, counts = np.unique(offsets, return_counts=True)
        duplicate = np.flatnonzero(counts > 1)
        if duplicate.size:
            self._raise_duplicate(int(unique[int(duplicate[0])]), source)

        byte_indices = (unique // 8).astype(np.intp, copy=False)
        masks = np.left_shift(
            np.uint8(1),
            (unique % 8).astype(np.uint8, copy=False),
        )
        repeated = np.flatnonzero(self.bitmap[byte_indices] & masks)
        if repeated.size:
            self._raise_duplicate(int(unique[int(repeated[0])]), source)
        np.bitwise_or.at(self.bitmap, byte_indices, masks)

    def finish(self) -> None:
        pass

    def _raise_duplicate(self, offset: int, source: Path) -> None:
        raise RuntimeError(
            "Input HDF5 files contain overlapping or duplicate particle ID "
            f"{self.minimum + offset} (detected while reading {source})."
        )


class _InMemoryIdOverlapChecker(_IdOverlapChecker):
    algorithm = "in_memory_sort"

    def __init__(self, dtype: np.dtype, count: int) -> None:
        self.values = np.empty(count, dtype=dtype)
        self.offset = 0

    def add(self, values: np.ndarray, source: Path) -> None:
        end = self.offset + values.size
        self.values[self.offset:end] = values
        self.offset = end

    def finish(self) -> None:
        if self.offset != self.values.size:
            raise RuntimeError(
                "Internal ID overlap check did not receive the expected "
                f"number of IDs ({self.offset}/{self.values.size})."
            )
        self.values.sort(kind="quicksort")
        duplicate = np.flatnonzero(self.values[1:] == self.values[:-1])
        if duplicate.size:
            value = self.values[int(duplicate[0])].item()
            raise RuntimeError(
                "Input HDF5 files contain overlapping or duplicate particle "
                f"ID {value}."
            )


class _PartitionedIdOverlapChecker(_IdOverlapChecker):
    algorithm = "disk_partitioned_sort"

    def __init__(self, dtype: np.dtype, count: int, parent: Path) -> None:
        self.dtype = dtype
        total_bytes = count * dtype.itemsize
        requested = max(2, math.ceil(total_bytes / PARTITION_TARGET_BYTES))
        self.partition_count = 1 << (requested - 1).bit_length()
        self.temporary = tempfile.TemporaryDirectory(
            prefix="id-overlap-",
            dir=parent,
        )
        self.root = Path(self.temporary.name)
        self.paths = [
            self.root / f"partition-{index:05d}.raw"
            for index in range(self.partition_count)
        ]

    def add(self, values: np.ndarray, source: Path) -> None:
        values = np.asarray(values, dtype=self.dtype)
        unique = np.unique(values)
        if unique.size != values.size:
            sorted_values = np.sort(values)
            duplicate = np.flatnonzero(
                sorted_values[1:] == sorted_values[:-1]
            )
            value = sorted_values[int(duplicate[0])].item()
            raise RuntimeError(
                "Input HDF5 files contain overlapping or duplicate particle "
                f"ID {value} (detected while reading {source})."
            )

        buckets = self._buckets(values)
        order = np.argsort(buckets, kind="stable")
        sorted_buckets = buckets[order]
        sorted_values = values[order]
        boundaries = np.flatnonzero(
            sorted_buckets[1:] != sorted_buckets[:-1]
        ) + 1
        starts = np.concatenate((np.array([0]), boundaries))
        ends = np.concatenate((boundaries, np.array([values.size])))
        for start, end in zip(starts, ends):
            bucket = int(sorted_buckets[int(start)])
            with self.paths[bucket].open("ab") as stream:
                sorted_values[int(start):int(end)].tofile(stream)

    def finish(self) -> None:
        for path in self.paths:
            if not path.exists():
                continue
            values = np.fromfile(path, dtype=self.dtype)
            values.sort(kind="quicksort")
            duplicate = np.flatnonzero(values[1:] == values[:-1])
            if duplicate.size:
                value = values[int(duplicate[0])].item()
                raise RuntimeError(
                    "Input HDF5 files contain overlapping or duplicate "
                    f"particle ID {value}."
                )

    def close(self) -> None:
        self.temporary.cleanup()

    def _buckets(self, values: np.ndarray) -> np.ndarray:
        hashed = values.astype(np.uint64, copy=True)
        hashed ^= hashed >> np.uint64(30)
        hashed *= np.uint64(0xBF58476D1CE4E5B9)
        hashed ^= hashed >> np.uint64(27)
        hashed *= np.uint64(0x94D049BB133111EB)
        hashed ^= hashed >> np.uint64(31)
        return (hashed & np.uint64(self.partition_count - 1)).astype(
            np.intp,
            copy=False,
        )


def merge_h5_files(
    input_files: Sequence[Path],
    output_h5: Path,
    force: bool,
    input_directory: Optional[Path] = None,
) -> MergeResult:
    """Validate and concatenate compatible particle files into one HDF5."""

    started = time.perf_counter()
    plan = _build_merge_plan(input_files)
    output_h5 = output_h5.resolve()
    partial_h5 = output_h5.with_name(f".{output_h5.name}.partial")
    require_output_path(output_h5, force)
    require_output_path(partial_h5, force)
    checker = _make_overlap_checker(plan, output_h5.parent)
    try:
        _write_merged_h5(plan, partial_h5, checker)
        checker.finish()
        partial_h5.replace(output_h5)
    except Exception:
        if partial_h5.exists():
            partial_h5.unlink()
        raise
    finally:
        checker.close()

    elapsed = time.perf_counter() - started
    directory = (
        input_directory.resolve()
        if input_directory is not None
        else plan.sources[0].path.parent
    )
    metadata = {
        "enabled": True,
        "input_directory": str(directory),
        "input_files": [str(source.path) for source in plan.sources],
        "source_file_count": len(plan.sources),
        "source_particle_counts": {
            source.path.name: source.count for source in plan.sources
        },
        "source_h5_file_bytes_total": plan.input_file_bytes,
        "merged_h5": str(output_h5),
        "merged_h5_file_bytes": output_h5.stat().st_size,
        "merged_particle_count": plan.count,
        "normalized_root_attributes": list(MUTABLE_ROOT_ATTRIBUTES),
        "id_overlap_check": {
            "status": "passed",
            "algorithm": checker.algorithm,
            "minimum": plan.id_minimum,
            "maximum": plan.id_maximum,
            "count": plan.count,
        },
        "wall_seconds": elapsed,
    }
    return MergeResult(output_h5=output_h5, metadata=metadata)


def _build_merge_plan(input_files: Sequence[Path]) -> MergePlan:
    paths = tuple(Path(path).resolve() for path in input_files)
    if not paths:
        raise RuntimeError("Cannot merge an empty HDF5 file list.")

    sources: List[SourceFile] = []
    output_fields: Optional[Dict[str, str]] = None
    field_dtypes: Optional[Dict[str, str]] = None
    field_attributes: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None
    root_attributes: Optional[Dict[str, Dict[str, Any]]] = None
    total_count = 0
    input_file_bytes = 0
    id_minimum: Optional[int] = None
    id_maximum: Optional[int] = None

    for path in paths:
        if not path.is_file():
            raise RuntimeError(f"Input HDF5 file does not exist: {path}")
        try:
            with h5py.File(path, "r") as source:
                fields = resolve_fields(source)
                count = _validate_field_lengths(source, fields, path)
                dtypes = {
                    logical: str(source[fields[logical]].dtype)
                    for logical in LOGICAL_ORDER
                }
                attributes = {
                    logical: collect_attributes(source[fields[logical]])
                    for logical in LOGICAL_ORDER
                }
                current_root_attributes = collect_attributes(source)
                if output_fields is None:
                    output_fields = dict(fields)
                    field_dtypes = dtypes
                    field_attributes = attributes
                    root_attributes = current_root_attributes
                else:
                    assert field_dtypes is not None
                    assert field_attributes is not None
                    assert root_attributes is not None
                    _validate_schema(
                        path,
                        dtypes,
                        attributes,
                        current_root_attributes,
                        field_dtypes,
                        field_attributes,
                        root_attributes,
                    )
                minimum, maximum = _id_range(
                    source[fields["id"]],
                    path,
                )
        except OSError as exc:
            raise RuntimeError(f"Could not read HDF5 file {path}: {exc}") from exc

        sources.append(SourceFile(path, fields, count))
        total_count += count
        input_file_bytes += path.stat().st_size
        id_minimum = minimum if id_minimum is None else min(id_minimum, minimum)
        id_maximum = maximum if id_maximum is None else max(id_maximum, maximum)

    assert output_fields is not None
    assert field_dtypes is not None
    assert field_attributes is not None
    assert root_attributes is not None
    assert id_minimum is not None and id_maximum is not None
    id_dtype = np.dtype(field_dtypes["id"])
    if not np.issubdtype(id_dtype, np.integer):
        raise RuntimeError(f"Particle ID dtype must be integer, got {id_dtype}.")
    return MergePlan(
        sources=tuple(sources),
        output_fields=output_fields,
        field_dtypes=field_dtypes,
        field_attributes=field_attributes,
        root_attributes=_merged_root_attributes(root_attributes, total_count),
        count=total_count,
        id_minimum=id_minimum,
        id_maximum=id_maximum,
        input_file_bytes=input_file_bytes,
    )


def _validate_field_lengths(
    source: h5py.File,
    fields: Mapping[str, str],
    path: Path,
) -> int:
    shapes = {
        logical: tuple(source[h5_path].shape)
        for logical, h5_path in fields.items()
    }
    if any(len(shape) != 1 for shape in shapes.values()):
        raise RuntimeError(
            f"Particle fields in {path} must be one-dimensional: {shapes}"
        )
    lengths = {logical: shape[0] for logical, shape in shapes.items()}
    if len(set(lengths.values())) != 1:
        raise RuntimeError(
            f"Particle fields in {path} do not have the same length: "
            f"{lengths}"
        )
    count = int(next(iter(lengths.values())))
    if count == 0:
        raise RuntimeError(f"Particle fields in {path} are empty.")
    return count


def _validate_schema(
    path: Path,
    dtypes: Mapping[str, str],
    field_attributes: Mapping[str, Mapping[str, Mapping[str, Any]]],
    root_attributes: Mapping[str, Mapping[str, Any]],
    expected_dtypes: Mapping[str, str],
    expected_field_attributes: Mapping[str, Mapping[str, Mapping[str, Any]]],
    expected_root_attributes: Mapping[str, Mapping[str, Any]],
) -> None:
    for logical in LOGICAL_ORDER:
        if np.dtype(dtypes[logical]) != np.dtype(expected_dtypes[logical]):
            raise RuntimeError(
                f"Cannot merge {path}: field {logical!r} has dtype "
                f"{dtypes[logical]}, expected {expected_dtypes[logical]}."
            )
        if not _attributes_equal(
            field_attributes[logical],
            expected_field_attributes[logical],
        ):
            raise RuntimeError(
                f"Cannot merge {path}: field {logical!r} attributes differ."
            )

    ignored = set(MUTABLE_ROOT_ATTRIBUTES)
    actual_common = {
        name: value for name, value in root_attributes.items()
        if name not in ignored
    }
    expected_common = {
        name: value for name, value in expected_root_attributes.items()
        if name not in ignored
    }
    if not _attributes_equal(actual_common, expected_common):
        raise RuntimeError(
            f"Cannot merge {path}: common root attributes differ."
        )


def _attributes_equal(
    actual: Mapping[str, Mapping[str, Any]],
    expected: Mapping[str, Mapping[str, Any]],
) -> bool:
    if set(actual) != set(expected):
        return False
    for name in actual:
        left = actual[name]
        right = expected[name]
        if left.get("dtype") != right.get("dtype"):
            return False
        if tuple(left.get("shape", [])) != tuple(right.get("shape", [])):
            return False
        left_value = np.asarray(restore_attribute(left))
        right_value = np.asarray(restore_attribute(right))
        if left_value.dtype.kind in "fc":
            equal = np.array_equal(left_value, right_value, equal_nan=True)
        else:
            equal = np.array_equal(left_value, right_value)
        if not equal:
            return False
    return True


def _id_range(dataset: h5py.Dataset, path: Path) -> Tuple[int, int]:
    dtype = np.dtype(dataset.dtype)
    if not np.issubdtype(dtype, np.integer):
        raise RuntimeError(
            f"Particle ID dtype in {path} must be integer, got {dtype}."
        )
    minimum: Optional[int] = None
    maximum: Optional[int] = None
    for start in range(0, int(dataset.shape[0]), COPY_CHUNK_ROWS):
        values = np.asarray(dataset[start:start + COPY_CHUNK_ROWS])
        chunk_minimum = int(values.min())
        chunk_maximum = int(values.max())
        minimum = (
            chunk_minimum if minimum is None else min(minimum, chunk_minimum)
        )
        maximum = (
            chunk_maximum if maximum is None else max(maximum, chunk_maximum)
        )
    assert minimum is not None and maximum is not None
    return minimum, maximum


def _merged_root_attributes(
    root_attributes: Mapping[str, Mapping[str, Any]],
    count: int,
) -> Dict[str, Dict[str, Any]]:
    result = {name: dict(payload) for name, payload in root_attributes.items()}
    replacements = {
        "npart": count,
        "npart_total": count,
        "proc_size": 1,
        "rank": 0,
    }
    for name, value in replacements.items():
        if name not in result:
            continue
        payload = result[name]
        if tuple(payload.get("shape", [])):
            raise RuntimeError(
                f"Root attribute {name!r} must be scalar to merge files."
            )
        dtype = np.dtype(payload["dtype"])
        try:
            converted = np.asarray(value, dtype=dtype)[()]
        except (OverflowError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Merged value {value} does not fit root attribute "
                f"{name!r} dtype {dtype}."
            ) from exc
        if np.issubdtype(dtype, np.integer) and int(converted) != value:
            raise RuntimeError(
                f"Merged value {value} does not fit root attribute "
                f"{name!r} dtype {dtype}."
            )
        result[name] = serialize_attribute(converted)
    return result


def _make_overlap_checker(
    plan: MergePlan,
    temporary_parent: Path,
) -> _IdOverlapChecker:
    span = plan.id_maximum - plan.id_minimum + 1
    bitmap_bytes = (span + 7) // 8
    if bitmap_bytes <= MAX_BITMAP_BYTES:
        return _BitmapIdOverlapChecker(plan.id_minimum, plan.id_maximum)
    dtype = np.dtype(plan.field_dtypes["id"])
    if plan.count * dtype.itemsize <= MAX_IN_MEMORY_ID_BYTES:
        return _InMemoryIdOverlapChecker(dtype, plan.count)
    return _PartitionedIdOverlapChecker(dtype, plan.count, temporary_parent)


def _write_merged_h5(
    plan: MergePlan,
    output_h5: Path,
    checker: _IdOverlapChecker,
) -> None:
    with h5py.File(output_h5, "w") as output:
        apply_attributes(output, plan.root_attributes)
        output_datasets = {}
        for logical in LOGICAL_ORDER:
            dataset = create_dataset(
                output,
                plan.output_fields[logical],
                np.dtype(plan.field_dtypes[logical]),
                plan.count,
            )
            apply_attributes(dataset, plan.field_attributes[logical])
            output_datasets[logical] = dataset

        destination_start = 0
        for source_file in plan.sources:
            destination_end = destination_start + source_file.count
            with h5py.File(source_file.path, "r") as source:
                for logical in LOGICAL_ORDER:
                    source_dataset = source[source_file.fields[logical]]
                    destination = output_datasets[logical]
                    for local_start in range(
                        0,
                        source_file.count,
                        COPY_CHUNK_ROWS,
                    ):
                        local_end = min(
                            source_file.count,
                            local_start + COPY_CHUNK_ROWS,
                        )
                        values = np.asarray(
                            source_dataset[local_start:local_end]
                        )
                        if logical == "id":
                            checker.add(values, source_file.path)
                        output_start = destination_start + local_start
                        destination[
                            output_start:output_start + values.size
                        ] = values
            destination_start = destination_end


__all__ = ["MergeResult", "merge_h5_files"]
