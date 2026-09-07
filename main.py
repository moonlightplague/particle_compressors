"""Particle compressor command-line entry point."""

import argparse
import contextlib
import io
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import h5py

from src.batch import (
    BatchFileResult,
    args_for_batch_file,
    build_batch_metrics,
    discover_particle_files,
    print_batch_summary,
    resolve_file_workers,
)
from src.cli import build_parser
from src.compress import compress
from src.decompress import decompress
from src.hdf5_io import resolve_fields
from src.metrics import (
    compute_metrics,
    print_component_summary,
    print_summary,
)
from src.merge import merge_h5_files
from src.manifest import update_compressed_size_metrics
from src.native_snapshot import AdaptedParticleInput, adapt_particle_inputs
from src.preprocess import preprocess
from src.runtime import read_json, resolve_field_workers, write_json


class PipelineApplication:
    """Execute one CLI command while sharing common stage transitions."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.work_dir = Path(args.work_dir).resolve()

    def run(self) -> int:
        handlers = {
            "preprocess": self._preprocess,
            "compress": self._compress,
            "decompress": self._decompress,
            "roundtrip": self._roundtrip,
        }
        try:
            handler = handlers[self.args.command]
        except KeyError as exc:
            raise RuntimeError(
                f"Unknown command: {self.args.command}"
            ) from exc
        handler()
        return 0

    def _preprocess(self) -> None:
        preprocess(self.args)
        self._print_package_paths()

    def _compress(self) -> None:
        manifest, raw_paths = preprocess(self.args)
        manifest = compress(self.args, manifest, raw_paths)
        self._print_package_paths()
        print(
            "payload_CR = "
            f"{manifest['sizes']['payload_compression_ratio']:.6g}"
        )
        print_component_summary(manifest)

    def _decompress(self) -> None:
        manifest = decompress(self.args)
        self._clean_raw_if_requested()
        print(
            "reconstructed_h5 = "
            f"{manifest['artifacts']['reconstructed_h5']}"
        )

    def _roundtrip(self) -> None:
        inherited_start = getattr(self.args, "roundtrip_started", None)
        started = (
            float(inherited_start)
            if inherited_start is not None
            else time.perf_counter()
        )
        metrics_path = self.work_dir / "metrics.json"
        if not bool(getattr(self.args, "metrics", False)):
            self._remove_stale_metrics(metrics_path)
        manifest, raw_paths = preprocess(self.args)
        manifest = compress(self.args, manifest, raw_paths)
        manifest = decompress(self.args)
        if not bool(getattr(self.args, "metrics", False)):
            self._clean_raw_if_requested()
            manifest.setdefault("timing", {})["roundtrip_wall_seconds"] = (
                time.perf_counter() - started
            )
            update_compressed_size_metrics(manifest, self.work_dir)
            write_json(self.work_dir / "manifest.json", manifest, force=True)
            self._print_roundtrip_summary(manifest)
            return

        metrics = compute_metrics(
            Path(manifest.get("input_h5", self.args.input_h5)).resolve(),
            Path(manifest["artifacts"]["reconstructed_h5"]).resolve(),
            manifest,
        )
        self._clean_raw_if_requested()
        metrics.setdefault("timing", {})["roundtrip_wall_seconds"] = (
            time.perf_counter() - started
        )
        write_json(metrics_path, metrics, force=True)
        print_summary(metrics, metrics_path)
        self._print_runtime_summary(metrics)

    def _print_roundtrip_summary(self, manifest: Dict[str, Any]) -> None:
        print(
            "reconstructed_h5 = "
            f"{manifest['artifacts']['reconstructed_h5']}"
        )
        print(f"manifest = {self.work_dir / 'manifest.json'}")
        sizes = manifest["sizes"]
        print(
            "payload_CR = "
            f"{sizes['payload_compression_ratio']:.6g}"
        )
        print_component_summary(manifest)
        self._print_runtime_summary(manifest)

    @staticmethod
    def _print_runtime_summary(report: Mapping[str, Any]) -> None:
        timing = report.get("timing", {})
        if not isinstance(timing, dict):
            return
        print("runtime_seconds:")
        entries = (
            ("merge", "merge_wall_seconds", False),
            ("preprocess", "preprocess_wall_seconds", False),
            ("compress", "compress_wall_seconds", False),
            ("canonical_order", "canonical_order_wall_seconds", True),
            ("lattice_prepare", "lattice_prepare_wall_seconds", True),
            ("id_compress", "id_compress_wall_seconds", True),
            (
                "lattice_field_prepare",
                "lattice_field_prepare_wall_seconds",
                True,
            ),
            ("lossy_fields_compress", "lossy_fields_wall_seconds", True),
            (
                "decompress_and_recombine",
                "decompress_and_recombine_wall_seconds",
                False,
            ),
            (
                "lossy_fields_decompress",
                "lossy_fields_decompress_wall_seconds",
                True,
            ),
            ("recombine_h5", "recombine_h5_wall_seconds", True),
            ("metrics", "metrics_wall_seconds", False),
            ("roundtrip", "roundtrip_wall_seconds", False),
        )
        for label, key, nested in entries:
            value = timing.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                prefix = "  " if nested else ""
                print(f"  {prefix}{label} = {float(value):.6g}")

    def _remove_stale_metrics(self, metrics_path: Path) -> None:
        if not metrics_path.exists():
            return
        if not self.args.force:
            raise RuntimeError(
                f"{metrics_path} already exists. Use --force to remove the "
                "stale detailed-metrics report."
            )
        metrics_path.unlink()

    def _print_package_paths(self) -> None:
        print(f"package_dir = {self.work_dir}")
        print(f"manifest = {self.work_dir / 'manifest.json'}")

    def _clean_raw_if_requested(self) -> None:
        if self.args.clean_raw:
            clean_raw_directories(self.work_dir)


class DirectoryPipelineApplication:
    """Run isolated file pipelines concurrently for one input directory."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.input_directory = Path(args.input_h5).resolve()
        self.input_files = discover_particle_files(self.input_directory)
        if not self.input_files:
            raise RuntimeError(
                "No HDF5 or native dat_* particle files found in directory: "
                f"{self.input_directory}"
            )
        self.work_dir = Path(args.work_dir).resolve()
        self.workers = resolve_file_workers(
            int(args.file_workers),
            len(self.input_files),
        )

    def run(self) -> int:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        file_args = [
            args_for_batch_file(
                self.args,
                input_h5,
                self.work_dir,
            )
            for input_h5 in self.input_files
        ]
        if int(getattr(self.args, "field_workers", 0)) == 0:
            workers_per_file = resolve_field_workers(
                0,
                concurrent_pipelines=self.workers,
            )
            for args in file_args:
                args.field_workers = workers_per_file
        started = time.perf_counter()
        results_by_input: Dict[str, BatchFileResult] = {}
        with ProcessPoolExecutor(max_workers=self.workers) as executor:
            futures = {
                executor.submit(_run_file_pipeline, args): args
                for args in file_args
            }
            for future in as_completed(futures):
                args = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = BatchFileResult(
                        input_h5=args.input_h5,
                        work_dir=args.work_dir,
                        wall_seconds=0.0,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                results_by_input[result.input_h5] = result
        batch_wall_seconds = time.perf_counter() - started

        results = [
            results_by_input[str(input_h5)]
            for input_h5 in self.input_files
        ]
        self._print_file_results(results)
        metrics = build_batch_metrics(
            self.input_directory,
            self.args.command,
            self.workers,
            results,
            batch_wall_seconds,
        )
        metrics_path = self.work_dir / "batch_metrics.json"
        write_json(metrics_path, metrics, force=True)
        print_batch_summary(metrics, metrics_path)

        failed = [result for result in results if not result.succeeded]
        if failed:
            raise RuntimeError(
                f"{len(failed)} of {len(results)} file pipelines failed; "
                f"see {metrics_path}"
            )
        return 0

    @staticmethod
    def _print_file_results(results: Sequence[BatchFileResult]) -> None:
        for result in results:
            print(f"File: {result.input_h5}")
            if result.console_output:
                print(result.console_output.rstrip())
            if result.error:
                print(f"error: {result.error}", file=sys.stderr)


class MergedDirectoryPipelineApplication:
    """Merge a directory's disjoint particle chunks into one pipeline."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.input_directory = Path(args.input_h5).resolve()
        self.input_files = discover_particle_files(self.input_directory)
        if not self.input_files:
            raise RuntimeError(
                "No HDF5 or native dat_* particle files found in directory: "
                f"{self.input_directory}"
            )
        self.work_dir = Path(args.work_dir).resolve()

    def run(self) -> int:
        started = time.perf_counter()
        adapted_inputs = adapt_particle_inputs(
            self.input_files,
            self.work_dir / "input_adapters",
        )
        result = merge_h5_files(
            [item.h5_path for item in adapted_inputs],
            self.work_dir / "merged" / "merged.h5",
            bool(self.args.force),
            input_directory=self.input_directory,
        )
        _record_native_merge_sources(result.metadata, adapted_inputs)
        merged_args = argparse.Namespace(**vars(self.args))
        merged_args.input_h5 = str(result.output_h5)
        merged_args.merge = False
        merged_args.merge_metadata = result.metadata
        if self.args.command == "roundtrip":
            merged_args.roundtrip_started = started
        print(f"merged_h5 = {result.output_h5}")
        print(
            "merge_wall_seconds = "
            f"{result.metadata['wall_seconds']:.6g}"
        )
        PipelineApplication(merged_args).run()
        print(
            "merged_directory_wall_seconds = "
            f"{time.perf_counter() - started:.6g}"
        )
        return 0


def _record_native_merge_sources(
    metadata: Dict[str, Any],
    adapted_inputs: Sequence[AdaptedParticleInput],
) -> None:
    if not any(item.native_header is not None for item in adapted_inputs):
        return

    metadata["input_files"] = [
        str(item.original_path) for item in adapted_inputs
    ]
    metadata["source_particle_counts"] = {
        item.original_path.name: (
            item.native_header.npart
            if item.native_header is not None
            else _h5_particle_count(item.h5_path)
        )
        for item in adapted_inputs
    }
    metadata["source_file_bytes_total"] = sum(
        item.original_path.stat().st_size for item in adapted_inputs
    )
    metadata.pop("source_h5_file_bytes_total", None)
    metadata["native_sources"] = [
        item.native_header.source_metadata()
        for item in adapted_inputs
        if item.native_header is not None
    ]


def _h5_particle_count(path: Path) -> int:
    with h5py.File(path, "r") as source:
        fields = resolve_fields(source)
        return int(source[fields["id"]].shape[0])


def _run_file_pipeline(args: argparse.Namespace) -> BatchFileResult:
    """Process-pool entry point that preserves each file's normal output."""

    output = io.StringIO()
    started = time.perf_counter()
    try:
        with contextlib.redirect_stdout(output):
            PipelineApplication(args).run()
        report_path = _file_report_path(args)
        return BatchFileResult(
            input_h5=args.input_h5,
            work_dir=args.work_dir,
            wall_seconds=time.perf_counter() - started,
            console_output=output.getvalue(),
            report_path=str(report_path),
            report=read_json(report_path),
        )
    except RuntimeError as exc:
        return BatchFileResult(
            input_h5=args.input_h5,
            work_dir=args.work_dir,
            wall_seconds=time.perf_counter() - started,
            console_output=output.getvalue(),
            error=str(exc),
        )


def _file_report_path(args: argparse.Namespace) -> Path:
    work_dir = Path(args.work_dir).resolve()
    metrics_path = work_dir / "metrics.json"
    return (
        metrics_path
        if args.command == "roundtrip"
        and bool(getattr(args, "metrics", False))
        else work_dir / "manifest.json"
    )


def clean_raw_directories(work_dir: Path) -> None:
    for name in ("preprocessed", "decompressed"):
        path = work_dir / name
        if path.exists():
            shutil.rmtree(path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser(argv)
    args = parser.parse_args(argv)
    try:
        if bool(getattr(args, "merge", False)):
            if not Path(args.input_h5).is_dir():
                raise RuntimeError(
                    "--merge requires input_h5 to be a directory."
                )
            return MergedDirectoryPipelineApplication(args).run()
        if hasattr(args, "input_h5") and Path(args.input_h5).is_dir():
            return DirectoryPipelineApplication(args).run()
        return PipelineApplication(args).run()
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
