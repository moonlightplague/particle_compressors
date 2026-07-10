import argparse

def add_common_tool_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--lcp", help="Path to the LCP executable.")
    parser.add_argument("--sz3", help=argparse.SUPPRESS)
    parser.add_argument("--chunk-size", type=int, default=4_000_000, help="HDF5/raw streaming chunk length.")
    parser.add_argument(
        "--part-size",
        type=int,
        default=0,
        help="Maximum values per pcodec/pysz compressed part; 0 disables part chunking.",
    )
    parser.add_argument(
        "--sz3-block-size",
        type=int,
        default=4_000_000,
        help="Deprecated alias for --part-size.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite pipeline outputs in the work directory.")

def add_compression_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input_h5", help="Input HDF5 particle file.")
    parser.add_argument("--work-dir", help="Pipeline work/package directory.")
    parser.add_argument("--abs-eb", type=float, default=1e-3, help="Default absolute error bound.")
    parser.add_argument(
        "--rel-eb",
        type=float,
        help=(
            "Default relative error bound for lossy fields. It is converted to per-field absolute "
            "bounds from the selected data range unless a class-specific absolute/relative bound is set."
        ),
    )
    parser.add_argument("--pos-abs-eb", type=float, help="Absolute error bound for x/y/z passed to LCP.")
    parser.add_argument(
        "--pos-rel-eb",
        type=float,
        help=(
            "Relative error bound for x/y/z. LCP accepts one absolute bound, so the pipeline passes "
            "the strictest derived x/y/z absolute bound."
        ),
    )
    parser.add_argument("--vel-abs-eb", type=float, help="Absolute error bound for vx/vy/vz passed to pysz/SZ3.")
    parser.add_argument("--vel-rel-eb", type=float, help="Relative error bound for vx/vy/vz passed to pysz/SZ3.")
    parser.add_argument(
        "--id-abs-eb",
        type=float,
        default=0.0,
        help="Expected id absolute error for metrics; pcodec id compression is lossless.",
    )
    parser.add_argument("--limit", type=int, help="Use only the first N particles for a smoke test.")
    parser.add_argument(
        "--position-scale",
        choices=("auto", "raw", "attr", "value"),
        default="auto",
        help="How integer coordinates are mapped to float32 for LCP.",
    )
    parser.add_argument("--position-scale-attr", default="bitwidth", help="Root attribute used by --position-scale attr.")
    parser.add_argument("--position-scale-value", type=float, help="Scale used by --position-scale value.")

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    compress = sub.add_parser("compress", help="Preprocess and compress an HDF5 particle file.")
    add_common_tool_args(compress)
    add_compression_args(compress)

    decompress = sub.add_parser("decompress", help="Decompress a pipeline package and rebuild HDF5.")
    add_common_tool_args(decompress)
    decompress.add_argument("work_dir", help="Pipeline work/package directory containing manifest.json.")
    decompress.add_argument("--output-h5", help="Reconstructed HDF5 output path.")
    decompress.add_argument("--clean-raw", action="store_true", help="Remove raw preprocessed/decompressed files.")

    roundtrip = sub.add_parser("roundtrip", help="Compress, decompress, recombine, and report metrics.")
    add_common_tool_args(roundtrip)
    add_compression_args(roundtrip)
    roundtrip.add_argument("--output-h5", help="Reconstructed HDF5 output path.")
    roundtrip.add_argument("--metrics-json", help="Metrics JSON output path.")
    roundtrip.add_argument("--clean-raw", action="store_true", help="Remove raw preprocessed/decompressed files.")

    return parser