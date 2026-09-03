"""Preprocess HDF5 particle fields and initialize a package manifest."""

import argparse
import importlib.metadata
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import h5py
import numpy as np

from src.constants import LOGICAL_ORDER, POSITION_FIELDS, VELOCITY_FIELDS
from src.error_bounds import ResolvedErrorBounds, resolve_error_bounds
from src.field_export import (
    export_float_field,
    export_id_for_pcodec,
    export_positions,
    get_selected_count,
    resolve_position_scale,
)
from src.hdf5_io import collect_attributes, resolve_fields, serialize_attribute
from src.models import PositionScale
from src.runtime import write_json


@dataclass(frozen=True)
class PreprocessWorkspace:
    root: Path
    raw: Path
    compressed: Path

    @classmethod
    def prepare(
        cls,
        work_dir: str,
        force: bool,
    ) -> "PreprocessWorkspace":
        root = Path(work_dir).resolve()
        raw = root / "preprocessed"
        compressed = root / "compressed"
        root.mkdir(parents=True, exist_ok=True)
        if force and compressed.exists():
            shutil.rmtree(compressed)
        raw.mkdir(parents=True, exist_ok=True)
        compressed.mkdir(parents=True, exist_ok=True)
        return cls(root, raw, compressed)


class PreprocessingPipeline:
    """Export compressor-ready raw fields and create their manifest."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        _validate_preprocess_args(args)
        self.input_h5 = Path(args.input_h5).resolve()
        if not self.input_h5.is_file():
            raise RuntimeError(
                f"Input HDF5 file does not exist: {self.input_h5}"
            )
        self.workspace = PreprocessWorkspace.prepare(
            args.work_dir,
            bool(args.force),
        )
        self.raw_paths: Dict[str, str] = {}
        self.statistics: Dict[str, Any] = {}

    def run(self) -> Tuple[Dict[str, Any], Dict[str, str]]:
        started = time.perf_counter()
        with h5py.File(self.input_h5, "r") as source:
            fields = resolve_fields(source)
            count = get_selected_count(source, fields, self.args.limit)
            position_scale = resolve_position_scale(
                source,
                self.args.position_scale,
                self.args.position_scale_attr,
                self.args.position_scale_value,
                np.dtype(source[fields["x"]].dtype),
            )
            self._export_fields(source, fields, count, position_scale)
            bounds = resolve_error_bounds(
                self.args,
                source,
                fields,
                position_scale,
                self.statistics,
            )
            manifest = _make_manifest(
                self.input_h5,
                source,
                fields,
                count,
                self.args.limit,
                position_scale,
                bounds,
            )
            selected_payload_bytes = _selected_payload_bytes(
                source,
                fields,
                count,
            )

        self._complete_manifest(
            manifest,
            selected_payload_bytes,
            started,
        )
        return manifest, self.raw_paths

    def _export_fields(
        self,
        source: h5py.File,
        fields: Mapping[str, str],
        count: int,
        position_scale: PositionScale,
    ) -> None:
        position_paths, position_stats = export_positions(
            source,
            fields,
            self.workspace.raw,
            count,
            position_scale,
            self.args.force,
        )
        self.raw_paths.update(position_paths)
        self.statistics["positions"] = position_stats
        self._export_id(source, fields, count)
        self._export_velocities(source, fields, count)

    def _export_id(
        self,
        source: h5py.File,
        fields: Mapping[str, str],
        count: int,
    ) -> None:
        dtype = np.dtype(source[fields["id"]].dtype)
        path, statistics = export_id_for_pcodec(
            source,
            fields["id"],
            self.workspace.raw / f"id.{dtype.name}.raw",
            count,
            self.args.force,
        )
        self.raw_paths["id"] = path
        self.statistics["id"] = statistics

    def _export_velocities(
        self,
        source: h5py.File,
        fields: Mapping[str, str],
        count: int,
    ) -> None:
        velocity_stats = {}
        for logical in VELOCITY_FIELDS:
            dtype = np.dtype(source[fields[logical]].dtype)
            raw_path, stats = export_float_field(
                source,
                fields[logical],
                self.workspace.raw / f"{logical}.{dtype.name}.raw",
                count,
                self.args.force,
            )
            self.raw_paths[logical] = raw_path
            velocity_stats[logical] = stats
        self.statistics["velocities"] = velocity_stats

    def _complete_manifest(
        self,
        manifest: Dict[str, Any],
        selected_payload_bytes: int,
        started: float,
    ) -> None:
        manifest["compressors"] = {
            "lossy": self.args.lossy_compressor,
            "lossless": self.args.lossless,
        }
        manifest["artifacts"] = {
            "preprocessed": self.raw_paths,
            "compressed": build_compressed_artifacts(
                self.workspace.compressed,
                self.args.lossy_compressor,
            ),
        }
        manifest["compressed_fields"] = {}
        manifest["preprocess"] = self.statistics
        manifest["sizes"] = {
            "selected_original_payload_bytes": selected_payload_bytes
        }
        manifest.setdefault("timing", {})["preprocess_wall_seconds"] = (
            time.perf_counter() - started
        )
        write_json(
            self.workspace.root / "manifest.json",
            manifest,
            force=True,
        )


def _make_manifest(
    input_h5: Path,
    h5: h5py.File,
    fields: Mapping[str, str],
    count: int,
    limit: Optional[int],
    position_scale: PositionScale,
    bounds: ResolvedErrorBounds,
) -> Dict[str, Any]:
    datasets = {
        logical: {
            "h5_path": h5_path,
            "dtype": str(h5[h5_path].dtype),
            "shape": list(h5[h5_path].shape),
            "selected_shape": [count],
            "attrs": collect_attributes(h5[h5_path]),
        }
        for logical, h5_path in fields.items()
    }
    root_attributes = collect_attributes(h5)
    if limit is not None and "npart" in root_attributes:
        root_attributes["npart"] = serialize_attribute(
            np.asarray(
                count,
                dtype=np.asarray(h5.attrs["npart"]).dtype,
            )
        )
    return {
        "format_version": 2,
        "input_h5": str(input_h5),
        "input_h5_file_bytes": input_h5.stat().st_size,
        "count": count,
        "limit": limit,
        "fields": datasets,
        "root_attrs": root_attributes,
        "position_scale": {
            "mode": position_scale.mode,
            "value": position_scale.value,
            "attr": position_scale.attr,
        },
        "field_error_bounds": bounds.fields,
        "tools": {
            "pcodec": package_version("pcodec"),
            "pysz": package_version("pysz"),
            "pyszo": package_version("pyszo"),
            "qoz": package_version("qoz-compressor"),
            "sperr": package_version("sperr"),
            "tthresh": package_version("tthresh"),
        },
    }


def build_compressed_artifacts(
    compressed_dir: Path,
    lossy_compressor: str,
) -> Dict[str, str]:
    extension = _lossy_extension(lossy_compressor)
    return {
        "id": str(compressed_dir / "id.pco"),
        **{
            field: str(compressed_dir / f"{field}.{extension}")
            for field in (*POSITION_FIELDS, *VELOCITY_FIELDS)
        },
    }


def package_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def preprocess(
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    return PreprocessingPipeline(args).run()


def _validate_preprocess_args(args: argparse.Namespace) -> None:
    if args.lossy_compressor not in (
        "szo",
        "sz3",
        "sperr",
        "qoz",
        "tthresh",
    ):
        raise RuntimeError(
            "--lossy-compressor must be one of: szo, sz3, sperr, qoz, "
            "tthresh."
        )


def _selected_payload_bytes(
    h5: h5py.File,
    fields: Mapping[str, str],
    count: int,
) -> int:
    return sum(
        int(np.dtype(h5[fields[field]].dtype).itemsize * count)
        for field in LOGICAL_ORDER
    )


def _lossy_extension(codec: str) -> str:
    try:
        return {
            "sz3": "psz",
            "szo": "szo",
            "sperr": "sperr",
            "qoz": "qoz",
            "tthresh": "tthresh",
        }[codec]
    except KeyError as exc:
        raise RuntimeError(f"Unsupported lossy compressor: {codec}.") from exc
