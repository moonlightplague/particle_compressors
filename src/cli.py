"""Command-line parser and YAML-backed advanced defaults."""

import argparse
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
BUILTIN_ADVANCED_DEFAULTS: Dict[str, Any] = {
    "abs_eb": None,
    "rel_eb": 1e-3,
    "pos_abs_eb": None,
    "pos_rel_eb": None,
    "vel_abs_eb": None,
    "vel_rel_eb": None,
    "id_abs_eb": 0.0,
    "position_scale": "auto",
    "position_scale_attr": "bitwidth",
    "position_scale_value": None,
    "lossy_compressor": "szo",
    "lossless": "pcodec",
    "lattice_layout": False,
    "lattice_min_occupancy": 0.8,
    "lattice_axis_search": True,
}
AVAILABLE_COMPRESSORS: Dict[str, Tuple[str, ...]] = {
    "lossy_compressor": ("szo", "sz3", "sperr", "qoz", "tthresh"),
    "lossless": ("pcodec",),
}
NULLABLE_NUMBER_KEYS = (
    "abs_eb",
    "rel_eb",
    "pos_abs_eb",
    "pos_rel_eb",
    "vel_abs_eb",
    "vel_rel_eb",
    "position_scale_value",
)


def load_config(path: str) -> Tuple[Path, Dict[str, Any]]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise RuntimeError(f"Config file does not exist: {config_path}")
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise RuntimeError(
            f"Could not load config file {config_path}: {exc}"
        ) from exc

    payload = {} if payload is None else payload
    if not isinstance(payload, Mapping):
        raise RuntimeError(
            f"Config file {config_path} must contain a mapping."
        )
    _reject_unknown_keys(payload, {"advanced"}, "config section")
    advanced = payload.get("advanced", {})
    if not isinstance(advanced, Mapping):
        raise RuntimeError("Config section advanced must be a mapping.")
    return config_path, _validated_advanced_config(advanced)


def build_parser(
    argv: Optional[Sequence[str]] = None,
) -> argparse.ArgumentParser:
    config_path, defaults = _bootstrap_config(argv)
    parser = argparse.ArgumentParser(description=__doc__)
    _add_config_argument(parser, str(config_path))
    commands = parser.add_subparsers(dest="command", required=True)

    _add_pipeline_command(
        commands,
        "preprocess",
        "Export compressor-ready raw fields and create a manifest.",
        defaults,
    )
    _add_pipeline_command(
        commands,
        "compress",
        "Preprocess and compress an HDF5 particle file.",
        defaults,
    )
    _add_decompress_command(commands)
    _add_pipeline_command(
        commands,
        "roundtrip",
        "Compress, reconstruct, and report roundtrip metrics.",
        defaults,
    )
    return parser


def _validated_advanced_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    _reject_unknown_keys(
        config,
        set(BUILTIN_ADVANCED_DEFAULTS),
        "advanced config key",
    )
    defaults = {**BUILTIN_ADVANCED_DEFAULTS, **config}
    for key in NULLABLE_NUMBER_KEYS:
        defaults[key] = _number_or_none(defaults[key], key)
    defaults["id_abs_eb"] = _required_number(
        defaults["id_abs_eb"],
        "id_abs_eb",
    )
    for key in ("lattice_layout", "lattice_axis_search"):
        if not isinstance(defaults[key], bool):
            raise RuntimeError(
                f"config value advanced.{key} must be a boolean."
            )
    defaults["lattice_min_occupancy"] = _required_number(
        defaults["lattice_min_occupancy"],
        "lattice_min_occupancy",
    )
    if not 0.0 < defaults["lattice_min_occupancy"] <= 1.0:
        raise RuntimeError(
            "config value advanced.lattice_min_occupancy must be in (0, 1]."
        )
    defaults["position_scale"] = _choice(
        defaults["position_scale"],
        "position_scale",
        ("auto", "raw", "attr", "value"),
    )
    if not isinstance(defaults["position_scale_attr"], str):
        raise RuntimeError(
            "config value advanced.position_scale_attr must be a string."
        )
    for key, choices in AVAILABLE_COMPRESSORS.items():
        defaults[key] = _choice(defaults[key], key, choices)
    return defaults


def _bootstrap_config(
    argv: Optional[Sequence[str]],
) -> Tuple[Path, Dict[str, Any]]:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    bootstrap_args, _ = bootstrap.parse_known_args(argv)
    try:
        return load_config(bootstrap_args.config)
    except RuntimeError as exc:
        bootstrap.error(str(exc))


def _add_pipeline_command(
    commands: Any,
    name: str,
    help_text: str,
    defaults: Mapping[str, Any],
) -> None:
    command = commands.add_parser(name, help=help_text)
    _add_config_argument(command, argparse.SUPPRESS)
    _add_runtime_arguments(command)
    _add_compression_arguments(command, defaults)
    command.add_argument(
        "--file-workers",
        type=int,
        default=0,
        help=(
            "Parallel file processes for directory input; 0 selects up to "
            "128 workers automatically (default: %(default)s)."
        ),
    )


def _add_decompress_command(commands: Any) -> None:
    command = commands.add_parser(
        "decompress",
        help="Decompress a package and rebuild its HDF5 file.",
    )
    _add_config_argument(command, argparse.SUPPRESS)
    _add_runtime_arguments(command)
    command.add_argument(
        "--work-dir",
        default="particle_pipeline_runs",
        help="Pipeline package directory containing manifest.json.",
    )


def _add_config_argument(parser: argparse.ArgumentParser, default: Any) -> None:
    parser.add_argument(
        "--config",
        default=default,
        type=lambda value: str(Path(value).expanduser().resolve()),
        metavar="PATH",
        help="YAML configuration file for advanced defaults.",
    )


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--clean-raw",
        action="store_true",
        help="Remove raw preprocessed and decompressed files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite pipeline outputs in the work directory.",
    )
    parser.add_argument(
        "--field-workers",
        type=int,
        default=0,
        help=(
            "Parallel processes for the six independent floating-point "
            "fields; 0 selects automatically (default: %(default)s)."
        ),
    )


def _add_compression_arguments(
    parser: argparse.ArgumentParser,
    defaults: Mapping[str, Any],
) -> None:
    parser.add_argument(
        "input_h5",
        help="Input HDF5 particle file or directory of .h5 files.",
    )
    parser.add_argument(
        "--work-dir",
        default="particle_pipeline_runs",
        help="Pipeline package directory containing manifest.json.",
    )
    parser.add_argument(
        "--abs-eb",
        type=float,
        default=defaults["abs_eb"],
        help="Default absolute error bound.",
    )
    parser.add_argument(
        "--rel-eb",
        type=float,
        default=defaults["rel_eb"],
        help=(
            "Default relative error bound. The pipeline converts it to "
            "per-field absolute bounds from each selected data range."
        ),
    )
    parser.add_argument(
        "--pos-abs-eb",
        type=float,
        default=defaults["pos_abs_eb"],
        help="Absolute error bound for x/y/z.",
    )
    parser.add_argument(
        "--pos-rel-eb",
        type=float,
        default=defaults["pos_rel_eb"],
        help="Relative error bound for x/y/z.",
    )
    parser.add_argument(
        "--vel-abs-eb",
        type=float,
        default=defaults["vel_abs_eb"],
        help="Absolute error bound for vx/vy/vz.",
    )
    parser.add_argument(
        "--vel-rel-eb",
        type=float,
        default=defaults["vel_rel_eb"],
        help="Relative error bound for vx/vy/vz.",
    )
    parser.add_argument(
        "--id-abs-eb",
        type=float,
        default=defaults["id_abs_eb"],
        help="Expected ID error for metrics; reconstruction remains exact.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Use only the first N particles for a smoke test.",
    )
    parser.add_argument(
        "--position-scale",
        choices=("auto", "raw", "attr", "value"),
        default=defaults["position_scale"],
        help="How positions are mapped to float32 compressor units.",
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
        "--lossy-compressor",
        choices=AVAILABLE_COMPRESSORS["lossy_compressor"],
        default=defaults["lossy_compressor"],
        help=(
            "Lossy compressor used for all position and velocity fields "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--sort",
        action="store_true",
        help="Stably sort particles by ascending ID before compression.",
    )
    parser.add_argument(
        "--lattice-layout",
        action=argparse.BooleanOptionalAction,
        default=defaults["lattice_layout"],
        help=(
            "Use an ID-derived periodic dense 3-D layout; implies ID "
            "sorting and falls back to the normal sorted layout when "
            "occupancy is too low (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--lattice-min-occupancy",
        type=float,
        default=defaults["lattice_min_occupancy"],
        help=(
            "Minimum occupied fraction for --lattice-layout "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--lattice-axis-search",
        action=argparse.BooleanOptionalAction,
        default=defaults["lattice_axis_search"],
        help=(
            "Try all six dense-axis orders and retain the smallest field "
            "payload when using --lattice-layout (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--lossless",
        choices=AVAILABLE_COMPRESSORS["lossless"],
        default=defaults["lossless"],
        help="ID compressor (default: %(default)s).",
    )


def _reject_unknown_keys(
    values: Mapping[str, Any],
    allowed: set[str],
    label: str,
) -> None:
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise RuntimeError(f"Unknown {label}(s): {', '.join(unknown)}")


def _number_or_none(value: Any, key: str) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        raise RuntimeError(
            f"config value advanced.{key} must be a number or null."
        )
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"config value advanced.{key} must be a number or null."
        ) from exc


def _required_number(value: Any, key: str) -> float:
    number = _number_or_none(value, key)
    if number is None:
        raise RuntimeError(f"config value advanced.{key} cannot be null.")
    return number


def _choice(value: Any, key: str, choices: Tuple[str, ...]) -> str:
    if value not in choices:
        raise RuntimeError(
            f"config value advanced.{key} must be one of: "
            f"{', '.join(choices)}."
        )
    return str(value)
