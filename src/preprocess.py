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

from src.cli import validate_compressor_combination
from src.constants import (
    LOGICAL_ORDER,
    MAX_INT32_ORDER_VALUES,
    POSITION_FIELDS,
    VELOCITY_FIELDS,
)
from src.error_bounds import (
    ResolvedErrorBounds,
    resolve_error_bounds,
    select_relative_or_absolute,
    serialize_error_bound_selection,
    validate_error_bound,
)
from src.field_export import (
    export_float32_for_lcp,
    export_float_field,
    export_id_for_pcodec,
    export_positions_for_lcp,
    export_positions_for_xnyzip,
    finalize_numeric_stats,
    get_selected_count,
    resolve_position_scale,
    update_numeric_stats,
    update_numeric_stats_int,
)
from src.hdf5_io import (
    as_jsonable_attr,
    collect_attrs,
    resolve_fields,
)
from src.models import ErrorBoundSelection, PositionScale, ToolPaths
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
        self.tools = ToolPaths(
            lcp=Path(args.lcp),
            xnyzip=(
                Path(args.xnyzip)
                if getattr(args, "xnyzip", None) is not None
                else None
            ),
        )
        self.workspace = PreprocessWorkspace.prepare(
            args.work_dir,
            bool(args.force),
        )
        self.raw_paths: Dict[str, str] = {}
        self.statistics: Dict[str, Any] = {}

    def run(self) -> Tuple[Dict[str, Any], Dict[str, str], ToolPaths]:
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
                self.tools,
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
        return manifest, self.raw_paths, self.tools

    def _export_fields(
        self,
        source: h5py.File,
        fields: Mapping[str, str],
        count: int,
        position_scale: PositionScale,
    ) -> None:
        xnyzip_position_path = (
            self.workspace.raw / "positions.xnyzip.f32.raw"
            if self.args.pos_compressor == "xnyzip"
            else None
        )
        position_paths, position_stats = export_positions_for_lcp(
            source,
            fields,
            self.workspace.raw,
            count,
            position_scale,
            self.args.force,
            xnyzip_output=xnyzip_position_path,
        )
        self.raw_paths.update(position_paths)
        self.statistics["positions"] = position_stats
        if xnyzip_position_path is not None:
            self.raw_paths["positions_xnyzip"] = str(
                xnyzip_position_path
            )
            self.statistics["positions_xnyzip"] = {
                "dtype": "float32",
                "shape": [count, 3],
                "layout": "xyz_interleaved",
                "fields": list(POSITION_FIELDS),
                "file_bytes": (
                    count * 3 * np.dtype(np.float32).itemsize
                ),
                "interleaved_during_position_export": True,
            }
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
            stats["preprocess_cast_max_abs"] = 0.0
            if self.args.vel_compressor in ("lcp", "xnyzip"):
                vector_codec = self.args.vel_compressor
                vector_path, cast_error = export_float32_for_lcp(
                    raw_path,
                    dtype,
                    self.workspace.raw
                    / f"{logical}.{vector_codec}.f32.raw",
                    count,
                    self.args.force,
                )
                self.raw_paths[f"{logical}_{vector_codec}"] = vector_path
                stats["preprocess_cast_max_abs"] = cast_error
            velocity_stats[logical] = stats
        self.statistics["velocities"] = velocity_stats

    def _complete_manifest(
        self,
        manifest: Dict[str, Any],
        selected_payload_bytes: int,
        started: float,
    ) -> None:
        chunk_size = int(getattr(self.args, "vel_chunk_size", 0))
        manifest["compressors"] = {
            "positions": self.args.pos_compressor,
            "velocities": self.args.vel_compressor,
            "lossless": self.args.lossless,
        }
        manifest["velocity_chunking"] = {
            "chunk_size": chunk_size,
            "enabled": bool(chunk_size),
            "configured_workers": int(
                getattr(self.args, "vel_chunk_workers", 0)
            ),
        }
        if self.args.vel_compressor == "lcp":
            manifest["error_bounds"]["velocities_lcp_abs"] = float(
                manifest["field_error_bounds"]["vx"]["compressor_abs"]
            )
        if self.args.vel_compressor == "xnyzip":
            manifest["error_bounds"]["velocities_xnyzip_abs"] = float(
                manifest["field_error_bounds"]["velocities_xnyzip"][
                    "compressor_abs"
                ]
            )
        else:
            manifest["error_bounds"].pop(
                "velocities_xnyzip_abs",
                None,
            )
            manifest["field_error_bounds"].pop(
                "velocities_xnyzip",
                None,
            )
        if (
            self.args.pos_compressor == "xnyzip"
            or self.args.vel_compressor == "xnyzip"
        ):
            manifest["tools"]["xnyzip"] = str(self.tools.xnyzip)
        manifest["artifacts"] = {
            "preprocessed": self.raw_paths,
            "compressed": build_compressed_artifacts(
                self.workspace.compressed,
                self.args.pos_compressor,
                self.args.vel_compressor,
            ),
        }
        manifest["order_dtype"] = (
            "uint64"
            if self.args.pos_compressor == "xnyzip"
            else "int32"
        )
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
    tools: ToolPaths,
) -> Dict[str, Any]:
    datasets = {
        logical: {
            "h5_path": h5_path,
            "dtype": str(h5[h5_path].dtype),
            "shape": list(h5[h5_path].shape),
            "selected_shape": [count],
            "attrs": collect_attrs(h5[h5_path]),
        }
        for logical, h5_path in fields.items()
    }
    root_attributes = collect_attrs(h5)
    if limit is not None and "npart" in root_attributes:
        root_attributes["npart"] = as_jsonable_attr(
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
        "error_bounds": {
            "positions_lcp_abs": bounds.position_lcp_abs,
            "positions_xnyzip_abs": bounds.position_vector_abs,
            "velocities_sz3_abs": bounds.velocity_abs,
            "id_sz3_abs": bounds.id_abs,
        },
        "field_error_bounds": bounds.fields,
        "tools": {
            "lcp": str(tools.lcp),
            "pcodec": package_version("pcodec"),
            "pysz": package_version("pysz"),
            "pyszo": package_version("pyszo"),
        },
    }


def make_manifest(
    input_h5: Path,
    h5: h5py.File,
    fields: Mapping[str, str],
    count: int,
    limit: Optional[int],
    position_scale: PositionScale,
    position_abs: float,
    position_vector_abs: float,
    velocity_abs: float,
    id_abs: float,
    field_error_bounds: Mapping[str, Mapping[str, Any]],
    tools: ToolPaths,
) -> Dict[str, Any]:
    """Compatibility wrapper for the original manifest-builder signature."""

    bounds = ResolvedErrorBounds(
        position=ErrorBoundSelection(
            "manifest",
            {},
            compressor_abs=position_abs,
        ),
        velocity=ErrorBoundSelection(
            "manifest",
            {},
            compressor_abs=velocity_abs,
        ),
        position_lcp_abs=position_abs,
        position_vector_abs=position_vector_abs,
        velocity_abs=velocity_abs,
        velocity_vector_abs=velocity_abs,
        id_abs=id_abs,
        fields={
            name: dict(payload)
            for name, payload in field_error_bounds.items()
        },
    )
    return _make_manifest(
        input_h5,
        h5,
        fields,
        count,
        limit,
        position_scale,
        bounds,
        tools,
    )


def build_compressed_artifacts(
    compressed_dir: Path,
    position_codec: str,
    velocity_codec: str,
) -> Dict[str, str]:
    validate_compressor_combination(position_codec, velocity_codec)
    artifacts = {"id": str(compressed_dir / "id.pco")}
    if position_codec in ("lcp", "xnyzip"):
        artifacts["positions"] = str(
            compressed_dir
            / (
                "positions.lcp"
                if position_codec == "lcp"
                else "positions.xnyzip"
            )
        )
    else:
        extension = _lossy_extension(position_codec)
        artifacts.update(
            {
                field: str(compressed_dir / f"{field}.{extension}")
                for field in POSITION_FIELDS
            }
        )
    if velocity_codec in ("lcp", "xnyzip"):
        artifacts["velocities"] = str(
            compressed_dir
            / (
                "velocities.lcp"
                if velocity_codec == "lcp"
                else "velocities.xnyzip"
            )
        )
        artifacts["velocity_order"] = str(
            compressed_dir / "velocity_order.pco"
        )
    else:
        extension = _lossy_extension(velocity_codec)
        artifacts.update(
            {
                field: str(compressed_dir / f"{field}.{extension}")
                for field in VELOCITY_FIELDS
            }
        )
    return artifacts


def package_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def preprocess(
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], Dict[str, str], ToolPaths]:
    return PreprocessingPipeline(args).run()


def _validate_preprocess_args(args: argparse.Namespace) -> None:
    validate_compressor_combination(
        args.pos_compressor,
        args.vel_compressor,
    )
    chunk_size = int(getattr(args, "vel_chunk_size", 0))
    workers = int(getattr(args, "vel_chunk_workers", 0))
    if chunk_size < 0:
        raise RuntimeError("--vel-chunk-size must be non-negative.")
    if (
        args.pos_compressor == "lcp"
        and args.vel_compressor == "lcp"
        and chunk_size > MAX_INT32_ORDER_VALUES
    ):
        raise RuntimeError(
            "--vel-chunk-size cannot exceed 2^31 when using int32 order indices."
        )
    if workers < 0:
        raise RuntimeError("--vel-chunk-workers must be non-negative.")
    chunked_pair = (
        args.pos_compressor == args.vel_compressor
        and args.pos_compressor in ("lcp", "xnyzip")
    )
    if chunk_size and not chunked_pair:
        raise RuntimeError(
            "--vel-chunk-size is only supported when both --pos-compressor "
            "and --vel-compressor are lcp or both are xnyzip."
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
        return {"sz3": "psz", "szo": "szo"}[codec]
    except KeyError as exc:
        raise RuntimeError(f"Unsupported lossy compressor: {codec}.") from exc


# Backwards-compatible names retained for existing callers.
export_float_for_pysz = export_float_field


__all__ = [
    "ErrorBoundSelection",
    "PositionScale",
    "PreprocessingPipeline",
    "PreprocessWorkspace",
    "ResolvedErrorBounds",
    "build_compressed_artifacts",
    "export_float32_for_lcp",
    "export_float_field",
    "export_float_for_pysz",
    "export_id_for_pcodec",
    "export_positions_for_lcp",
    "export_positions_for_xnyzip",
    "finalize_numeric_stats",
    "get_selected_count",
    "make_manifest",
    "package_version",
    "preprocess",
    "resolve_error_bounds",
    "resolve_fields",
    "resolve_position_scale",
    "select_relative_or_absolute",
    "serialize_error_bound_selection",
    "update_numeric_stats",
    "update_numeric_stats_int",
    "validate_error_bound",
]
