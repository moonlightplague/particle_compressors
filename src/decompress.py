"""Decompression-stage orchestration and HDF5 reconstruction."""

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

from src.constants import POSITION_FIELDS, VELOCITY_FIELDS
from src.hdf5_io import (
    HDF5Recombiner,
    apply_attrs,
    create_dataset,
    recombine_h5,
    restore_attr,
)
from src.lattice_layout import (
    LATTICE_LAYOUT_NAME,
    DenseLatticeLayout,
    lattice_layout_from_metadata,
)
from src.huffman_encode import huffman_decode_file
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
from src.shaped_codecs import decompress_shaped_lossy_raw
from src.structured_layout import (
    HYBRID_VELOCITY_LAYOUT, StructuredParticleLayout, hybrid_velocity_order,
    validate_structured_package,
)
from src.runtime import (
    read_json,
    read_raw,
    require_output_path,
    resolve_field_workers,
    resolve_velocity_chunk_workers,
    write_json,
)
from src.xnyzip_codec import (
    run_chunked_xnyzip_decompress,
    run_xnyzip_decompress,
)


@dataclass(frozen=True)
class FieldDecompressionJob:
    """Pickle-friendly description of one independent field decode."""

    field: Dict[str, Any]
    output_path: str
    decompressed_dir: str
    force: bool
    count: int
    lattice_metadata: Optional[Dict[str, Any]] = None
    id_path: Optional[str] = None
    id_dtype: Optional[str] = None


def _decompress_field_job(job: FieldDecompressionJob) -> None:
    field = job.field
    if field.get("spatial_layout") != LATTICE_LAYOUT_NAME:
        decompress_lossy_raw(field, job.output_path, job.force)
        return

    logical = str(field["field"])
    dtype = np.dtype(field["dtype"])
    dense_path = (
        Path(job.decompressed_dir)
        / f"{logical}.lattice-encoded.{dtype.name}.raw"
    )
    decompress_shaped_lossy_raw(field, str(dense_path), job.force)
    if job.lattice_metadata is None:
        raise RuntimeError("Parallel lattice decode is missing layout metadata.")
    implicit_full = bool(
        job.lattice_metadata.get("implicit_full_lattice", False)
    )
    if implicit_full:
        sorted_ids = None
    else:
        if job.id_path is None or job.id_dtype is None:
            raise RuntimeError(
                "Parallel lattice decode is missing its ID field."
            )
        sorted_ids = read_raw(job.id_path, np.dtype(job.id_dtype), job.count)
    layout = lattice_layout_from_metadata(sorted_ids, job.lattice_metadata)
    dense_values = read_raw(str(dense_path), dtype, layout.dense_count)
    wrap_offsets = None
    if "lattice_wrap_field" in field:
        wrap_field = field["lattice_wrap_field"]
        wrap_dtype = np.dtype(wrap_field["dtype"])
        wrap_path = (
            Path(job.decompressed_dir)
            / f"{logical}.lattice-wrap.{wrap_dtype.name}.raw"
        )
        decompress_integer_raw(wrap_field, str(wrap_path), job.force)
        wrap_offsets = read_raw(str(wrap_path), wrap_dtype, job.count)
    decoded = layout.decode_field(
        dense_values,
        logical,
        str(field.get("lattice_transform", "identity")),
        dtype,
        wrap_offsets,
    )
    output = Path(job.output_path)
    require_output_path(output, job.force)
    decoded.tofile(output)


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
        validate_structured_package(self.manifest)
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
        self._decoded_lattice: DenseLatticeLayout | None = None
        self.field_workers = resolve_field_workers(
            int(getattr(args, "field_workers", 1)),
        )
        self.lossy_field_workers_used = 0
        self.lossy_fields_seconds = 0.0
        for path in self.output_paths.values():
            require_output_path(Path(path), args.force)

    def run(self) -> Dict[str, Any]:
        started = time.perf_counter()
        self._decompress_integer_fields()
        if (
            self.position_codec not in ("lcp", "xnyzip")
            and self.velocity_codec not in ("lcp", "xnyzip")
        ):
            self._decompress_fieldwise_fields(
                (*POSITION_FIELDS, *VELOCITY_FIELDS)
            )
        else:
            self._decompress_positions()
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
        if self.velocity_codec in ("lcp", "xnyzip"):
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
        if "velocity_block_ids" in self.fields:
            block_id_field = self.fields["velocity_block_ids"]
            encoded_dtype = np.dtype(block_id_field["dtype"])
            decoded_dtype = np.dtype(block_id_field["decoded_dtype"])
            paths["velocity_block_ids_huffman"] = str(
                self.decompressed_dir
                / f"velocity_block_ids.huffman.{encoded_dtype.name}.raw"
            )
            paths["velocity_block_ids"] = str(
                self.decompressed_dir
                / f"velocity_block_ids.{decoded_dtype.name}.raw"
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
        if self.position_codec == "xnyzip":
            run_xnyzip_decompress(
                self.tools,
                self.compressed_artifacts["positions"],
                self.output_paths,
                POSITION_FIELDS,
                self.count,
                float(
                    self.manifest["error_bounds"][
                        "positions_xnyzip_abs"
                    ]
                ),
                self.decompressed_dir / "positions.xnyzip.f32.raw",
                self.args.force,
                **({"quantizer": self.fields["positions"]["quantizer"]}
                   if "quantizer" in self.fields["positions"] else {}),
            )
            return
        for logical in POSITION_FIELDS:
            self._decompress_fieldwise(logical)

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
        if self.velocity_codec == "xnyzip":
            if "velocity_order" not in self.fields:
                raise RuntimeError(
                    "XnYZip velocity package is missing its velocity_order "
                    "sidecar metadata."
                )
            started = time.perf_counter()
            velocity_field = self.fields["velocities"]
            chunk_size = int(velocity_field.get("chunk_size", 0))
            if chunk_size:
                run_chunked_xnyzip_decompress(
                    self.tools,
                    self.compressed_artifacts["velocities"],
                    self.output_paths,
                    VELOCITY_FIELDS,
                    self.count,
                    chunk_size,
                    float(
                        self.manifest["error_bounds"][
                            "velocities_xnyzip_abs"
                        ]
                    ),
                    resolve_velocity_chunk_workers(
                        int(getattr(self.args, "vel_chunk_workers", 0))
                    ),
                )
            else:
                run_xnyzip_decompress(
                    self.tools,
                    self.compressed_artifacts["velocities"],
                    self.output_paths,
                    VELOCITY_FIELDS,
                    self.count,
                    float(
                        self.manifest["error_bounds"][
                            "velocities_xnyzip_abs"
                        ]
                    ),
                    self.decompressed_dir / "velocities.xnyzip.f32.raw",
                    self.args.force,
                )
            self.manifest.setdefault("timing", {})[
                "velocity_xnyzip_decompress_wall_seconds"
            ] = time.perf_counter() - started
            decompress_integer_raw(
                self.fields["velocity_order"],
                self.output_paths["velocity_order"],
                self.args.force,
            )
            return
        if self.velocity_codec != "lcp":
            self._decompress_fieldwise_fields(VELOCITY_FIELDS)
            if self.manifest.get("structured_layout", {}).get("enabled", False):
                self._restore_structured_velocities()
            return

        velocity_field = self.fields["velocities"]
        blockwise_order = (
            self.fields.get("velocity_order", {}).get("order_encoding")
            == "lcp_blockwise_packed"
        )
        if blockwise_order:
            if "velocity_block_ids" not in self.fields:
                raise RuntimeError(
                    "Blockwise LCP velocity package is missing its "
                    "velocity_block_ids sidecar metadata."
                )
            decompress_integer_raw(
                self.fields["velocity_order"],
                self.output_paths["velocity_order"],
                self.args.force,
            )
            decompress_integer_raw(
                self.fields["velocity_block_ids"],
                self.output_paths["velocity_block_ids_huffman"],
                self.args.force,
            )
            huffman_started = time.perf_counter()
            huffman_decode_file(
                Path(self.output_paths["velocity_block_ids_huffman"]),
                Path(self.output_paths["velocity_block_ids"]),
                self.args.force,
                expected_count=self.count,
            )
            self.manifest.setdefault("timing", {})[
                "velocity_block_id_huffman_decode_wall_seconds"
            ] = time.perf_counter() - huffman_started
            started = time.perf_counter()
            run_lcp_decompress(
                self.tools,
                self.compressed_artifacts["velocities"],
                self.output_paths,
                VELOCITY_FIELDS,
                self.count,
                float(
                    self.manifest["error_bounds"]["velocities_lcp_abs"]
                ),
                Path(self.output_paths["velocity_order"]),
                Path(self.output_paths["velocity_block_ids"]),
            )
            self.manifest.setdefault("timing", {})[
                "velocity_lcp_decompress_wall_seconds"
            ] = time.perf_counter() - started
            return

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
                resolve_velocity_chunk_workers(
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

    def _decompress_fieldwise(self, logical: str) -> None:
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

    def _restore_structured_velocities(self) -> None:
        if self.position_codec != "xnyzip" or self.velocity_codec != "szo":
            raise RuntimeError("Structured layout requires XnYZip positions and SZO velocities.")
        layout = StructuredParticleLayout.from_metadata(self.manifest["structured_layout"])
        ids = read_raw(self.output_paths["id"], np.dtype(self.fields["id"]["dtype"]), self.count)
        positions = {logical: read_raw(self.output_paths[logical], np.dtype("float32"), self.count)
                     for logical in POSITION_FIELDS}
        permutation = hybrid_velocity_order(ids, positions, layout)
        del ids, positions
        for logical in VELOCITY_FIELDS:
            if self.fields[logical].get("spatial_layout") != HYBRID_VELOCITY_LAYOUT:
                raise RuntimeError(f"Missing hybrid layout metadata for {logical}.")
            values = read_raw(self.output_paths[logical], np.dtype(self.fields[logical]["dtype"]), self.count)
            restored = np.empty_like(values)
            restored[permutation] = values
            del values
            restored.tofile(self.output_paths[logical])

    def _decompress_fieldwise_fields(
        self,
        logical_fields: Sequence[str],
    ) -> None:
        started = time.perf_counter()
        workers = min(self.field_workers, len(logical_fields))
        self.lossy_field_workers_used = max(
            self.lossy_field_workers_used,
            workers,
        )
        if workers == 1:
            for logical in logical_fields:
                self._decompress_fieldwise(logical)
        else:
            jobs = [
                self._field_decompression_job(logical)
                for logical in logical_fields
            ]
            with ProcessPoolExecutor(max_workers=workers) as executor:
                list(executor.map(_decompress_field_job, jobs))
        self.lossy_fields_seconds += time.perf_counter() - started

    def _field_decompression_job(
        self,
        logical: str,
    ) -> FieldDecompressionJob:
        field = dict(self.fields[logical])
        is_lattice = field.get("spatial_layout") == LATTICE_LAYOUT_NAME
        lattice_metadata = (
            dict(self.manifest.get("lattice_layout", {}))
            if is_lattice
            else None
        )
        needs_ids = bool(
            is_lattice
            and lattice_metadata is not None
            and not lattice_metadata.get("implicit_full_lattice", False)
        )
        return FieldDecompressionJob(
            field=field,
            output_path=self.output_paths[logical],
            decompressed_dir=str(self.decompressed_dir),
            force=bool(self.args.force),
            count=self.count,
            lattice_metadata=lattice_metadata,
            id_path=self.output_paths["id"] if needs_ids else None,
            id_dtype=(
                str(self.manifest["fields"]["id"]["dtype"])
                if needs_ids
                else None
            ),
        )

    def _lattice_for_decode(self) -> DenseLatticeLayout:
        if self._decoded_lattice is not None:
            return self._decoded_lattice
        metadata = self.manifest.get("lattice_layout", {})
        if metadata.get("implicit_full_lattice", False):
            sorted_ids = None
        else:
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
        timing["lossy_fields_decompress_wall_seconds"] = (
            self.lossy_fields_seconds
        )
        timing["decompress_and_recombine_wall_seconds"] = (
            time.perf_counter() - started
        )
        timing["recombine_h5_wall_seconds"] = recombine_seconds
        self.manifest["artifacts"]["decompressed"] = self.output_paths
        self.manifest["artifacts"]["reconstructed_h5"] = str(self.output_h5)
        self.manifest.setdefault("sizes", {})[
            "reconstructed_h5_file_bytes"
        ] = self.output_h5.stat().st_size
        self.manifest.setdefault("runtime", {})[
            "decompression_field_workers"
        ] = self.lossy_field_workers_used
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
    "run_chunked_xnyzip_decompress",
    "run_lcp_decompress",
    "run_lcp_decompress_batch",
    "velocity_compressor_from_manifest",
]
