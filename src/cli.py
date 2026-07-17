import argparse

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, List

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
BUILTIN_ADVANCED_DEFAULTS: Dict[str, Any] = {
    "lcp": str(DEFAULT_CONFIG_PATH.parent / "tools" / "LCP" / "build" / "bin" / "lcp"),
    "abs_eb": None,
    "rel_eb": 1e-3,
    "pos_abs_eb": None,
    "pos_rel_eb": None,
    "vel_abs_eb": None,
    "vel_rel_eb": None,
    "vel_chunk_size": 0,
    "vel_chunk_workers": 0,
    "id_abs_eb": 0.0,
    "position_scale": "auto",
    "position_scale_attr": "bitwidth",
    "position_scale_value": None,
    "pos_compressor": "lcp",
    "vel_compressor": "sz3",
    "lossless": "pcodec",
}
AVAILABLE_COMPRESSORS: Dict[str, List[str]] = {
    "pos_compressor": ["lcp", "sz3", "szo"],
    "vel_compressor": ["sz3", "szo", "lcp"],
    "lossless": ["pcodec", "zstd"],
}


def _number_or_none(value: Any, key: str) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"config value advanced.{key} must be a number or null.")
    return float(value)


def _validated_advanced_config(config: Mapping[str, Any], config_path: Path) -> Dict[str, Any]:
    unknown = sorted(set(config) - set(BUILTIN_ADVANCED_DEFAULTS))
    if unknown:
        raise RuntimeError(f"Unknown advanced config key(s): {', '.join(unknown)}")

    defaults = dict(BUILTIN_ADVANCED_DEFAULTS)
    defaults.update(config)

    for key in ("pos_abs_eb", "pos_rel_eb", "vel_abs_eb", "vel_rel_eb", "position_scale_value"):
        defaults[key] = _number_or_none(defaults[key], key)
    defaults["id_abs_eb"] = _number_or_none(defaults["id_abs_eb"], "id_abs_eb")
    if defaults["id_abs_eb"] is None:
        raise RuntimeError("config value advanced.id_abs_eb cannot be null.")
    if isinstance(defaults["vel_chunk_size"], bool) or not isinstance(
        defaults["vel_chunk_size"], int
    ):
        raise RuntimeError("config value advanced.vel_chunk_size must be a non-negative integer.")
    if defaults["vel_chunk_size"] < 0:
        raise RuntimeError("config value advanced.vel_chunk_size must be a non-negative integer.")
    if isinstance(defaults["vel_chunk_workers"], bool) or not isinstance(
        defaults["vel_chunk_workers"], int
    ):
        raise RuntimeError("config value advanced.vel_chunk_workers must be a non-negative integer.")
    if defaults["vel_chunk_workers"] < 0:
        raise RuntimeError("config value advanced.vel_chunk_workers must be a non-negative integer.")

    position_scale = defaults["position_scale"]
    if position_scale not in ("auto", "raw", "attr", "value"):
        raise RuntimeError("config value advanced.position_scale must be auto, raw, attr, or value.")
    if not isinstance(defaults["position_scale_attr"], str):
        raise RuntimeError("config value advanced.position_scale_attr must be a string.")
    if not isinstance(defaults["lcp"], str):
        raise RuntimeError("config value advanced.lcp must be a path string.")
    for key, choices in AVAILABLE_COMPRESSORS.items():
        if defaults[key] not in choices:
            raise RuntimeError(
                f"config value advanced.{key} must be one of: {', '.join(choices)}."
            )

    lcp = Path(defaults["lcp"]).expanduser()
    if not lcp.is_absolute():
        lcp = config_path.parent / lcp
    defaults["lcp"] = str(lcp.resolve())
    return defaults


def load_config(path: str) -> Tuple[Path, Dict[str, Any]]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise RuntimeError(f"Config file does not exist: {config_path}")
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise RuntimeError(f"Could not load config file {config_path}: {exc}") from exc
    if payload is None:
        payload = {}
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"Config file {config_path} must contain a mapping.")
    unknown_sections = sorted(set(payload) - {"advanced"})
    if unknown_sections:
        raise RuntimeError(f"Unknown config section(s): {', '.join(unknown_sections)}")
    advanced = payload.get("advanced", {})
    if not isinstance(advanced, Mapping):
        raise RuntimeError("Config section advanced must be a mapping.")
    return config_path, _validated_advanced_config(advanced, config_path)


def add_config_arg(parser: argparse.ArgumentParser, default: Any) -> None:
    parser.add_argument(
        "--config",
        default=default,
        type=lambda value: str(Path(value).expanduser().resolve()),
        metavar="PATH",
        help="YAML configuration file for advanced defaults.",
    )


def add_common_tool_args(parser: argparse.ArgumentParser, defaults: Mapping[str, Any]) -> None:
    parser.add_argument("--lcp", type=str, default=defaults["lcp"], help="Path to the LCP executable.")
    parser.add_argument(
        "--vel-chunk-workers",
        type=int,
        default=defaults["vel_chunk_workers"],
        help=(
            "Parallel native LCP workers for chunked velocities; 0 selects up to 16 workers "
            "automatically (default: %(default)s)."
        ),
    )
    parser.add_argument("--clean-raw", action="store_true", help="Remove raw preprocessed/decompressed files.")
    parser.add_argument("--force", action="store_true", help="Overwrite pipeline outputs in the work directory.")


def add_compression_args(parser: argparse.ArgumentParser, defaults: Mapping[str, Any]) -> None:
    parser.add_argument("input_h5", help="Input HDF5 particle file.")
    parser.add_argument("--work-dir", default="particle_pipeline_runs", help="Pipeline work/package directory containing manifest.json.")
    parser.add_argument("--abs-eb", type=float, default=defaults["abs_eb"], help="Default absolute error bound.")
    parser.add_argument(
        "--rel-eb",
        type=float,
        default=defaults["rel_eb"],
        help=(
            "Default relative error bound for lossy fields. It is converted to per-field absolute "
            "bounds from the selected data range unless a class-specific absolute/relative bound is set."
        ),
    )
    parser.add_argument(
        "--pos-abs-eb",
        type=float,
        default=defaults["pos_abs_eb"],
        help="Absolute error bound for x/y/z passed to LCP.",
    )
    parser.add_argument(
        "--pos-rel-eb",
        type=float,
        default=defaults["pos_rel_eb"],
        help=(
            "Relative error bound for x/y/z. LCP accepts one absolute bound, so the pipeline passes "
            "the strictest derived x/y/z absolute bound."
        ),
    )
    parser.add_argument(
        "--vel-abs-eb",
        type=float,
        default=defaults["vel_abs_eb"],
        help="Absolute error bound for vx/vy/vz passed to the selected velocity compressor.",
    )
    parser.add_argument(
        "--vel-rel-eb",
        type=float,
        default=defaults["vel_rel_eb"],
        help="Relative error bound for vx/vy/vz passed to the selected velocity compressor.",
    )
    parser.add_argument(
        "--vel-chunk-size",
        type=int,
        default=defaults["vel_chunk_size"],
        help=(
            "Number of position-ordered particles per independent velocity LCP chunk; "
            "0 disables chunking (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--id-abs-eb",
        type=float,
        default=defaults["id_abs_eb"],
        help="Expected ID absolute error for metrics; ID reconstruction is exact.",
    )
    parser.add_argument("--limit", type=int, help="Use only the first N particles for a smoke test.")
    parser.add_argument(
        "--position-scale",
        choices=("auto", "raw", "attr", "value"),
        default=defaults["position_scale"],
        help="How integer coordinates are mapped to float32 for LCP.",
    )
    parser.add_argument(
        "--position-scale-attr",
        default=defaults["position_scale_attr"],
        help="Root attribute used by --position-scale attr.",
    )
    parser.add_argument(
        "--position-scale-value",
        type=float,
        default=defaults["position_scale_value"],
        help="Scale used by --position-scale value.",
    )
    parser.add_argument(
        "--pos-compressor",
        choices=AVAILABLE_COMPRESSORS["pos_compressor"],
        default=defaults["pos_compressor"],
        help="Position triplet compressor (default: %(default)s).",
    )
    parser.add_argument(
        "--vel-compressor",
        choices=AVAILABLE_COMPRESSORS["vel_compressor"],
        default=defaults["vel_compressor"],
        help="Velocity triplet compressor (default: %(default)s).",
    )
    parser.add_argument(
        "--lossless",
        choices=AVAILABLE_COMPRESSORS["lossless"],
        default=defaults["lossless"],
        help="Integer sidecar and ID compressor (default: %(default)s).",
    )



def build_parser(argv: Optional[Sequence[str]] = None) -> argparse.ArgumentParser:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    bootstrap_args, _ = bootstrap.parse_known_args(argv)
    try:
        config_path, defaults = load_config(bootstrap_args.config)
    except RuntimeError as exc:
        bootstrap.error(str(exc))

    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arg(parser, str(config_path))
    sub = parser.add_subparsers(dest="command", required=True)

    preprocess = sub.add_parser("preprocess", help="Preprocess an HDF5 particle file, output the preprocessed data files and calculated manifest")
    add_config_arg(preprocess, argparse.SUPPRESS)
    add_common_tool_args(preprocess, defaults)
    add_compression_args(preprocess, defaults)

    compress = sub.add_parser("compress", help="Preprocess and compress an HDF5 particle file.")
    add_config_arg(compress, argparse.SUPPRESS)
    add_common_tool_args(compress, defaults)
    add_compression_args(compress, defaults)

    decompress = sub.add_parser("decompress", help="Decompress a pipeline package and rebuild HDF5.")
    add_config_arg(decompress, argparse.SUPPRESS)
    add_common_tool_args(decompress, defaults)
    decompress.add_argument("--work-dir", default="particle_pipeline_runs", help="Pipeline work/package directory containing manifest.json.")

    roundtrip = sub.add_parser("roundtrip", help="Compress, decompress, recombine, and report metrics.")
    add_config_arg(roundtrip, argparse.SUPPRESS)
    add_common_tool_args(roundtrip, defaults)
    add_compression_args(roundtrip, defaults)

    return parser
