"""Decompression-stage orchestration and HDF5 reconstruction."""

import argparse
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

from src.constants import POSITION_FIELDS, VELOCITY_FIELDS
from src.hdf5_io import recombine_h5
from src.lattice_layout import (
    LATTICE_LAYOUT_NAME,
    DenseLatticeLayout,
    lattice_layout_from_metadata,
)
from src.manifest import (
    lossy_compressor_from_manifest,
    update_compressed_size_metrics,
)
from src.raw_codecs import decompress_integer_raw, decompress_lossy_raw
from src.runtime import (
    read_json,
    read_raw,
    require_output_path,
    write_json,
)
from src.shaped_codecs import decompress_shaped_lossy_raw


class DecompressionPipeline:
    """Decompress all package components and rebuild the source HDF5 file."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.work_dir = Path(args.work_dir).resolve()
        self.manifest_path = self.work_dir / "manifest.json"
        if not self.manifest_path.is_file():
            raise RuntimeError(f"Missing manifest: {self.manifest_path}")

        self.manifest = read_json(self.manifest_path)
        self.fields = self.manifest.get("compressed_fields")
        if not self.fields:
            raise RuntimeError(
                "Manifest does not contain compressed_fields for the "
                "Python compressor pipeline."
            )
        lossy_compressor_from_manifest(self.manifest)
        self.count = int(self.manifest["count"])
        self.decompressed_dir = self.work_dir / "decompressed"
        self.decompressed_dir.mkdir(parents=True, exist_ok=True)
        self.output_h5 = self.work_dir / "reconstructed.h5"
        require_output_path(self.output_h5, args.force)
        self.output_paths = self._build_output_paths()
        self._decoded_lattice: DenseLatticeLayout | None = None
        for path in self.output_paths.values():
            require_output_path(Path(path), args.force)

    def run(self) -> Dict[str, Any]:
        started = time.perf_counter()
        decompress_integer_raw(
            self.fields["id"],
            self.output_paths["id"],
            self.args.force,
        )
        for logical in (*POSITION_FIELDS, *VELOCITY_FIELDS):
            self._decompress_field(logical)

        recombine_started = time.perf_counter()
        recombine_h5(self.manifest, self.output_paths, self.output_h5)
        recombine_seconds = time.perf_counter() - recombine_started
        self._finalize(started, recombine_seconds)
        return self.manifest

    def _build_output_paths(self) -> Dict[str, str]:
        field_metadata = self.manifest["fields"]
        return {
            "x": str(self.decompressed_dir / "x.f32.raw"),
            "y": str(self.decompressed_dir / "y.f32.raw"),
            "z": str(self.decompressed_dir / "z.f32.raw"),
            "id": str(
                self.decompressed_dir
                / f"id.{np.dtype(field_metadata['id']['dtype']).name}.raw"
            ),
            **{
                logical: str(
                    self.decompressed_dir
                    / f"{logical}.{field_metadata[logical]['dtype']}.raw"
                )
                for logical in VELOCITY_FIELDS
            },
        }

    def _decompress_field(self, logical: str) -> None:
        field = self.fields[logical]
        if field.get("spatial_layout") != LATTICE_LAYOUT_NAME:
            decompress_lossy_raw(
                field,
                self.output_paths[logical],
                self.args.force,
            )
            return

        dtype = np.dtype(field["dtype"])
        dense_path = (
            self.decompressed_dir
            / f"{logical}.lattice-encoded.{dtype.name}.raw"
        )
        decompress_shaped_lossy_raw(
            field,
            str(dense_path),
            self.args.force,
        )
        layout = self._lattice_for_decode()
        dense_values = read_raw(
            str(dense_path),
            dtype,
            layout.dense_count,
        )
        wrap_offsets = None
        if "lattice_wrap_field" in field:
            wrap_field = field["lattice_wrap_field"]
            wrap_dtype = np.dtype(wrap_field["dtype"])
            wrap_path = (
                self.decompressed_dir
                / f"{logical}.lattice-wrap.{wrap_dtype.name}.raw"
            )
            decompress_integer_raw(
                wrap_field,
                str(wrap_path),
                self.args.force,
            )
            wrap_offsets = read_raw(
                str(wrap_path),
                wrap_dtype,
                self.count,
            )
        decoded = layout.decode_field(
            dense_values,
            logical,
            str(field.get("lattice_transform", "identity")),
            dtype,
            wrap_offsets,
        )
        output = Path(self.output_paths[logical])
        require_output_path(output, self.args.force)
        decoded.tofile(output)

    def _lattice_for_decode(self) -> DenseLatticeLayout:
        if self._decoded_lattice is not None:
            return self._decoded_lattice
        metadata = self.manifest.get("lattice_layout", {})
        id_dtype = np.dtype(self.manifest["fields"]["id"]["dtype"])
        sorted_ids = read_raw(
            self.output_paths["id"],
            id_dtype,
            self.count,
        )
        self._decoded_lattice = lattice_layout_from_metadata(
            sorted_ids,
            metadata,
        )
        return self._decoded_lattice

    def _finalize(self, started: float, recombine_seconds: float) -> None:
        timing = self.manifest.setdefault("timing", {})
        timing["decompress_and_recombine_wall_seconds"] = (
            time.perf_counter() - started
        )
        timing["recombine_h5_wall_seconds"] = recombine_seconds
        self.manifest["artifacts"]["decompressed"] = self.output_paths
        self.manifest["artifacts"]["reconstructed_h5"] = str(self.output_h5)
        self.manifest.setdefault("sizes", {})[
            "reconstructed_h5_file_bytes"
        ] = self.output_h5.stat().st_size
        update_compressed_size_metrics(self.manifest, self.work_dir)
        write_json(self.manifest_path, self.manifest, force=True)


def decompress(args: argparse.Namespace) -> Dict[str, Any]:
    return DecompressionPipeline(args).run()
