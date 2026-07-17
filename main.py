"""Particle compressor command-line entry point."""

import argparse
import shutil
import sys
from pathlib import Path
from typing import Optional, Sequence

from src.cli import build_parser
from src.compress import compress
from src.decompress import decompress
from src.metrics import (
    compute_metrics,
    print_component_summary,
    print_summary,
)
from src.models import ToolPaths
from src.preprocess import preprocess
from src.runtime import write_json


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
            ToolPaths(lcp=Path(self.args.lcp)),
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
        return PipelineApplication(args).run()
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


# Backwards-compatible name.
maybe_clean_raw = clean_raw_directories


if __name__ == "__main__":
    raise SystemExit(main())
