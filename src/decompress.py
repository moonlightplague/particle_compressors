"""Decompression-stage orchestration and HDF5 reconstruction."""

import argparse
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

from src.constants import POSITION_FIELDS, VELOCITY_FIELDS
from src.hdf5_io import (
    HDF5Recombiner,
    apply_attrs,
    create_dataset,
    recombine_h5,
    restore_attr,
)
from src.lcp_codec import (
    read_lcp_order,
    run_chunked_lcp_decompress,
    run_lcp_decompress,
    run_lcp_decompress_batch,
)
from src.manifest import (
    order_dtype_from_manifest,
    position_compressor_from_manifest,
    update_compressed_size_metrics,
    velocity_compressor_from_manifest,
)
from src.models import ToolPaths
from src.raw_codecs import (
    decompress_integer_raw,
    decompress_lossy_raw,
    decompress_pcodec_raw,
    decompress_pysz_raw,
    decompress_szo_raw,
)
from src.runtime import (
    read_json,
    require_output_path,
    resolve_lcp_chunk_workers,
    write_json,
)


class DecompressionPipeline:
    """Decompress all package components and rebuild the source HDF5 file."""

    def __init__(self, args: argparse.Namespace, tools: ToolPaths) -> None:
        self.args = args
        self.tools = tools
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
        self.count = int(self.manifest["count"])
        self.position_codec = position_compressor_from_manifest(self.manifest)
        self.velocity_codec = velocity_compressor_from_manifest(self.manifest)
        self.compressed_artifacts = self.manifest["artifacts"]["compressed"]
        self.decompressed_dir = self.work_dir / "decompressed"
        self.decompressed_dir.mkdir(parents=True, exist_ok=True)
        self.output_h5 = self.work_dir / "reconstructed.h5"
        require_output_path(self.output_h5, args.force)
        self.output_paths = self._build_output_paths()
        for path in self.output_paths.values():
            require_output_path(Path(path), args.force)

    def run(self) -> Dict[str, Any]:
        started = time.perf_counter()
        self._decompress_positions()
        self._decompress_integer_fields()
        self._decompress_velocities()

        recombine_started = time.perf_counter()
        recombine_h5(self.manifest, self.output_paths, self.output_h5)
        recombine_seconds = time.perf_counter() - recombine_started
        self._finalize(started, recombine_seconds)
        return self.manifest

    def _build_output_paths(self) -> Dict[str, str]:
        field_metadata = self.manifest["fields"]
        paths = {
            "x": str(self.decompressed_dir / "x.f32.raw"),
            "y": str(self.decompressed_dir / "y.f32.raw"),
            "z": str(self.decompressed_dir / "z.f32.raw"),
            "id": str(
                self.decompressed_dir
                / f"id.{np.dtype(field_metadata['id']['dtype']).name}.raw"
            ),
            "vx": str(
                self.decompressed_dir
                / f"vx.{field_metadata['vx']['dtype']}.raw"
            ),
            "vy": str(
                self.decompressed_dir
                / f"vy.{field_metadata['vy']['dtype']}.raw"
            ),
            "vz": str(
                self.decompressed_dir
                / f"vz.{field_metadata['vz']['dtype']}.raw"
            ),
        }
        if "order" in self.fields:
            order_dtype = order_dtype_from_manifest(self.manifest)
            paths["order"] = str(
                self.decompressed_dir / f"order.{order_dtype.name}.raw"
            )
        if self.velocity_codec == "lcp":
            for logical in VELOCITY_FIELDS:
                paths[logical] = str(
                    self.decompressed_dir / f"{logical}.f32.raw"
                )
        if "velocity_order" in self.fields:
            order_dtype = np.dtype(self.fields["velocity_order"]["dtype"])
            paths["velocity_order"] = str(
                self.decompressed_dir
                / f"velocity_order.{order_dtype.name}.raw"
            )
        return paths

    def _decompress_positions(self) -> None:
        if self.position_codec == "lcp":
            run_lcp_decompress(
                self.tools,
                self.compressed_artifacts["positions"],
                self.output_paths,
                POSITION_FIELDS,
                self.count,
                float(
                    self.manifest["error_bounds"]["positions_lcp_abs"]
                ),
            )
            return
        for logical in POSITION_FIELDS:
            decompress_lossy_raw(
                self.fields[logical],
                self.output_paths[logical],
                self.args.force,
            )

    def _decompress_integer_fields(self) -> None:
        if "order" in self.fields:
            decompress_integer_raw(
                self.fields["order"],
                self.output_paths["order"],
                self.args.force,
            )
        decompress_integer_raw(
            self.fields["id"],
            self.output_paths["id"],
            self.args.force,
        )

    def _decompress_velocities(self) -> None:
        if self.velocity_codec != "lcp":
            for logical in VELOCITY_FIELDS:
                decompress_lossy_raw(
                    self.fields[logical],
                    self.output_paths[logical],
                    self.args.force,
                )
            return

        velocity_field = self.fields["velocities"]
        chunk_size = int(velocity_field.get("chunk_size", 0))
        if chunk_size:
            started = time.perf_counter()
            run_chunked_lcp_decompress(
                self.tools,
                self.compressed_artifacts["velocities"],
                self.output_paths,
                VELOCITY_FIELDS,
                self.count,
                chunk_size,
                float(
                    self.manifest["error_bounds"]["velocities_lcp_abs"]
                ),
                resolve_lcp_chunk_workers(
                    int(getattr(self.args, "vel_chunk_workers", 0))
                ),
            )
            self.manifest.setdefault("timing", {})[
                "velocity_lcp_decompress_wall_seconds"
            ] = time.perf_counter() - started
        else:
            run_lcp_decompress(
                self.tools,
                self.compressed_artifacts["velocities"],
                self.output_paths,
                VELOCITY_FIELDS,
                self.count,
                float(
                    self.manifest["error_bounds"]["velocities_lcp_abs"]
                ),
            )

        if "velocity_order" in self.fields:
            decompress_integer_raw(
                self.fields["velocity_order"],
                self.output_paths["velocity_order"],
                self.args.force,
            )

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


def decompress(
    args: argparse.Namespace,
    tools: ToolPaths,
) -> Dict[str, Any]:
    return DecompressionPipeline(args, tools).run()


__all__ = [
    "DecompressionPipeline",
    "HDF5Recombiner",
    "apply_attrs",
    "create_dataset",
    "decompress",
    "decompress_integer_raw",
    "decompress_lossy_raw",
    "decompress_pcodec_raw",
    "decompress_pysz_raw",
    "decompress_szo_raw",
    "position_compressor_from_manifest",
    "read_lcp_order",
    "recombine_h5",
    "restore_attr",
    "run_chunked_lcp_decompress",
    "run_lcp_decompress",
    "run_lcp_decompress_batch",
    "velocity_compressor_from_manifest",
]
