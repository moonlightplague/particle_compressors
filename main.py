import argparse
import shutil
import sys

from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple
from pathlib import Path

import src.helpers as hp
from src.preprocess import preprocess
from src.compress import compress
from src.decompress import decompress


def default_work_dir_for_args(args: argparse.Namespace) -> Path:
    if args.pos_rel_eb is not None or args.vel_rel_eb is not None:
        labels: List[str] = []
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
    return Path("particle_pipeline_runs") / f"{Path(args.input_h5).name}.{suffix}"


def maybe_clean_raw(work_dir: Path) -> None:
    for name in ("preprocessed", "decompressed"):
        path = work_dir / name
        if path.exists():
            shutil.rmtree(path)


def main(argv: Optional[List[str]] = None) -> int:
    from src.cli import build_parser
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "compress":
            if not args.work_dir:
                args.work_dir = str(default_work_dir_for_args(args))
            manifest, raw_paths, tools = preprocess(args)
            manifest = compress(args, manifest, raw_paths, tools)
            print(f"package_dir = {Path(args.work_dir).resolve()}")
            print(f"manifest = {Path(args.work_dir).resolve() / 'manifest.json'}")
            print(f"payload_CR = {manifest['sizes']['payload_compression_ratio']:.6g}")
            hp.print_component_summary(manifest)
            return 0

        if args.command == "decompress":
            manifest = decompress(args)
            if args.clean_raw:
                maybe_clean_raw(Path(args.work_dir).resolve())
            print(f"reconstructed_h5 = {manifest['artifacts']['reconstructed_h5']}")
            return 0

        if args.command == "roundtrip":
            if not args.work_dir:
                args.work_dir = str(default_work_dir_for_args(args))
            manifest, raw_paths, tools = preprocess(args)
            manifest = compress(args, manifest, raw_paths, tools)
            manifest = decompress(args)
            input_h5 = Path(args.input_h5).resolve()
            output_h5 = Path(manifest["artifacts"]["reconstructed_h5"]).resolve()
            metrics = hp.compute_metrics(input_h5, output_h5, manifest, args.chunk_size)
            metrics_path = Path(args.metrics_json).resolve() if args.metrics_json else Path(args.work_dir).resolve() / "metrics.json"
            hp.write_json(metrics_path, metrics, force=True)
            if args.clean_raw:
                maybe_clean_raw(Path(args.work_dir).resolve())
            hp.print_summary(metrics, metrics_path)
            return 0

        parser.error(f"Unknown command: {args.command}")
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())