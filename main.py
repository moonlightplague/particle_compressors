"""Particle compressor command-line entry point."""

import argparse
import contextlib
import io
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Optional, Sequence

from src.batch import (
    BatchFileResult,
    args_for_batch_file,
    build_batch_metrics,
    discover_h5_files,
    print_batch_summary,
    resolve_file_workers,
)
from src.cli import build_parser, validate_compressor_combination
from src.compress import compress
from src.decompress import decompress
from src.metrics import (
    compute_metrics,
    print_component_summary,
    print_summary,
)
from src.models import ToolPaths
from src.preprocess import preprocess
from src.runtime import read_json, write_json


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
        manifest, raw_paths, tools = preprocess(self.args)
        manifest = compress(self.args, manifest, raw_paths, tools)
        self._print_package_paths()
        print(
            "payload_CR = "
            f"{manifest['sizes']['payload_compression_ratio']:.6g}"
        )
        print_component_summary(manifest)

    def _decompress(self) -> None:
        manifest = decompress(
            self.args,
            ToolPaths(
                lcp=Path(self.args.lcp),
                xnyzip=Path(self.args.xnyzip),
            ),
        )
        self._clean_raw_if_requested()
        print(
            "reconstructed_h5 = "
            f"{manifest['artifacts']['reconstructed_h5']}"
        )

    def _roundtrip(self) -> None:
        manifest, raw_paths, tools = preprocess(self.args)
        manifest = compress(self.args, manifest, raw_paths, tools)
        manifest = decompress(self.args, tools)
        metrics = compute_metrics(
            Path(self.args.input_h5).resolve(),
            Path(manifest["artifacts"]["reconstructed_h5"]).resolve(),
            manifest,
        )
        metrics_path = self.work_dir / "metrics.json"
        write_json(metrics_path, metrics, force=True)
        self._clean_raw_if_requested()
        print_summary(metrics, metrics_path)

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
        self.input_files = discover_h5_files(self.input_directory)
        if not self.input_files:
            raise RuntimeError(
                f"No .h5 files found in directory: {self.input_directory}"
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
        else work_dir / "manifest.json"
    )


def clean_raw_directories(work_dir: Path) -> None:
    for name in ("preprocessed", "decompressed"):
        path = work_dir / name
        if path.exists():
            shutil.rmtree(path)


def default_work_dir_for_args(args: argparse.Namespace) -> Path:
    """Return the legacy data/error-bound-derived work directory."""

    if args.pos_rel_eb is not None or args.vel_rel_eb is not None:
        labels = []
        if args.pos_rel_eb is not None:
            labels.append(f"posrel{args.pos_rel_eb:g}")
        if args.vel_rel_eb is not None:
            labels.append(f"velrel{args.vel_rel_eb:g}")
        suffix = "_".join(labels)
    elif args.rel_eb is not None:
        suffix = f"rel{args.rel_eb:g}"
    else:
        suffix = f"eb{args.abs_eb:g}"
    if args.limit is not None:
        suffix += f"_n{args.limit}"
    return (
        Path("particle_pipeline_runs")
        / f"{Path(args.input_h5).name}.{suffix}"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser(argv)
    args = parser.parse_args(argv)
    try:
        if hasattr(args, "pos_compressor"):
            validate_compressor_combination(
                args.pos_compressor,
                args.vel_compressor,
            )
        if hasattr(args, "input_h5") and Path(args.input_h5).is_dir():
            return DirectoryPipelineApplication(args).run()
        return PipelineApplication(args).run()
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


# Backwards-compatible name.
maybe_clean_raw = clean_raw_directories


if __name__ == "__main__":
    raise SystemExit(main())
