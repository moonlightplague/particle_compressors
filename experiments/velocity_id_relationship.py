"""Test whether a derived velocity scalar is coherent on the ID lattice.

The tested scalar follows the requested definition exactly::

    cbrt(vx**3 + vy**3 + vz**3)

The driver stably sorts particles by ID, infers the production periodic dense
lattice, compresses the scalar both flat and dense, and compares its dense
compression ratio with the three velocity-component streams. A compatible
baseline manifest can supply previously measured component stream sizes.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Optional

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.hdf5_io import resolve_fields  # noqa: E402
from src.lattice_layout import infer_dense_lattice_layout  # noqa: E402
from src.raw_codecs import (  # noqa: E402
    compress_lossy_raw,
    decompress_lossy_raw,
)
from src.runtime import read_json  # noqa: E402
from src.shaped_codecs import (  # noqa: E402
    compress_shaped_lossy_raw,
    decompress_shaped_lossy_raw,
)


VELOCITY_FIELDS = ("vx", "vy", "vz")
FORMULA = "cbrt(vx^3 + vy^3 + vz^3)"
CODEC_EXTENSIONS = {
    "qoz": "qoz",
    "sperr": "sperr",
    "sz3": "psz",
    "szo": "szo",
}
CODEC_METADATA_NAMES = {
    "qoz": "qoz",
    "sperr": "sperr",
    "sz3": "pysz",
    "szo": "szo",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_h5", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--baseline-manifest",
        type=Path,
        help=(
            "Reuse vx/vy/vz sizes from a compatible completed lattice run; "
            "without this option the components are recompressed."
        ),
    )
    parser.add_argument(
        "--codec",
        choices=tuple(CODEC_EXTENSIONS),
        default="szo",
    )
    parser.add_argument("--rel-eb", type=float, default=1e-3)
    parser.add_argument("--min-occupancy", type=float, default=0.8)
    parser.add_argument(
        "--axis-search",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def velocity_cuberoot_sum_cubes(
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
) -> np.ndarray:
    """Evaluate the requested real cube-root scalar as float32."""
    arrays = [np.asarray(values) for values in (vx, vy, vz)]
    if any(values.shape != arrays[0].shape for values in arrays[1:]):
        raise RuntimeError("Velocity components must have matching shapes.")
    cube_sum = np.zeros(arrays[0].shape, dtype=np.float64)
    for values in arrays:
        values64 = values.astype(np.float64, copy=False)
        cube_sum += values64 * values64 * values64
    return np.cbrt(cube_sum).astype(np.float32)


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    input_path = args.input_h5.expanduser().resolve()
    if not input_path.is_file():
        raise RuntimeError(f"Input HDF5 file does not exist: {input_path}")
    if not np.isfinite(args.rel_eb) or args.rel_eb <= 0.0:
        raise RuntimeError("--rel-eb must be finite and positive.")

    baseline_path = (
        args.baseline_manifest.expanduser().resolve()
        if args.baseline_manifest is not None
        else None
    )
    baseline = read_json(baseline_path) if baseline_path is not None else None
    components: dict[str, np.ndarray] = {}
    with h5py.File(input_path, "r") as source:
        fields = resolve_fields(source)
        ids = np.asarray(source[fields["id"]][:])
        order = np.argsort(ids, kind="stable")
        sorted_ids = ids[order]
        side = int(source.attrs["nsidemesh"])
        scale = float(source.attrs["bitwidth"])
        sorted_positions = {
            logical: (
                source[fields[logical]][:].astype(np.float64) / scale
            ).astype(np.float32)[order]
            for logical in ("x", "y", "z")
        }
        layout = infer_dense_lattice_layout(
            sorted_ids,
            sorted_positions,
            side,
            float(args.min_occupancy),
        )
        del sorted_positions
        print(
            f"lattice shape={layout.shape}, occupancy={layout.occupancy:.6f}",
            flush=True,
        )

        cube_sum = np.zeros(ids.size, dtype=np.float64)
        keep_components = baseline is None
        for logical in VELOCITY_FIELDS:
            values = np.asarray(
                source[fields[logical]][:],
                dtype=np.float32,
            )[order]
            values64 = values.astype(np.float64)
            cube_sum += values64 * values64 * values64
            if keep_components:
                components[logical] = values
            del values64
        scalar = np.cbrt(cube_sum).astype(np.float32)
        del cube_sum
        print(f"derived {FORMULA} for {scalar.size} rows", flush=True)

    if not np.all(np.isfinite(scalar)):
        raise RuntimeError("The derived velocity scalar contains non-finite values.")
    scalar_range = float(np.ptp(scalar.astype(np.float64)))
    scalar_bound = float(args.rel_eb) * scalar_range
    raw_bytes = int(scalar.size * scalar.dtype.itemsize)

    baseline_rows = (
        _baseline_component_rows(
            baseline,
            input_path,
            args.codec,
            float(args.rel_eb),
            layout.shape,
            scalar.size,
            bool(args.axis_search),
            baseline_path,
        )
        if baseline is not None
        else None
    )

    with tempfile.TemporaryDirectory(prefix="velocity_id_relationship_") as tmp:
        temp_dir = Path(tmp)
        derived_rows = _compress_derived_scalar(
            scalar,
            scalar_bound,
            layout,
            args.codec,
            bool(args.axis_search),
            temp_dir,
        )
        component_rows = (
            baseline_rows
            if baseline_rows is not None
            else _compress_components(
                components,
                layout,
                args.codec,
                float(args.rel_eb),
                bool(args.axis_search),
                temp_dir,
            )
        )

    dense_derived = next(
        row for row in derived_rows if row["layout"] == "periodic_dense_lattice"
    )
    flat_derived = next(
        row for row in derived_rows if row["layout"] == "id_sorted_flat"
    )
    component_crs = {
        str(row["field"]): float(row["particle_compression_ratio"])
        for row in component_rows
    }
    derived_cr = float(dense_derived["particle_compression_ratio"])
    correlation = _relationship_statistics(sorted_ids, scalar)
    conclusion = {
        "dense_scalar_beats_all_components": bool(
            derived_cr > max(component_crs.values())
        ),
        "dense_scalar_rank_among_four": int(
            1
            + sum(
                component_cr > derived_cr
                for component_cr in component_crs.values()
            )
        ),
        "dense_vs_flat_size_ratio": (
            int(flat_derived["compressed_bytes"])
            / int(dense_derived["compressed_bytes"])
        ),
        "dense_scalar_cr": derived_cr,
        "best_component_cr": max(component_crs.values()),
        "component_crs": component_crs,
    }
    payload = {
        "input_h5": str(input_path),
        "formula": FORMULA,
        "interpretation": (
            "This is the real cube root of the signed sum of cubes, exactly "
            "as requested; it is not the conventional nonnegative L3 norm."
        ),
        "codec": args.codec,
        "relative_error_bound": float(args.rel_eb),
        "count": int(scalar.size),
        "particle_raw_bytes_per_field": raw_bytes,
        "derived_scalar": {
            "dtype": str(scalar.dtype),
            "minimum": float(scalar.min()),
            "maximum": float(scalar.max()),
            "range": scalar_range,
            "absolute_error_bound": scalar_bound,
            "mean": float(np.mean(scalar, dtype=np.float64)),
            "standard_deviation": float(np.std(scalar, dtype=np.float64)),
        },
        "lattice": layout.manifest_metadata(),
        "relationship_statistics": correlation,
        "results": [*component_rows, *derived_rows],
        "conclusion": conclusion,
        "seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def _compress_derived_scalar(
    scalar: np.ndarray,
    error_bound: float,
    layout: Any,
    codec: str,
    axis_search: bool,
    temp_dir: Path,
) -> list[dict[str, Any]]:
    flat = _compress_flat(
        "velocity_cuberoot_sum_cubes",
        scalar,
        error_bound,
        codec,
        temp_dir,
    )
    dense, _, _ = layout.encode_field(
        scalar,
        "velocity_cuberoot_sum_cubes",
        position_residual=False,
    )
    shaped = _compress_dense(
        "velocity_cuberoot_sum_cubes",
        dense,
        scalar.size,
        error_bound,
        codec,
        axis_search,
        temp_dir,
    )
    return [flat, shaped]


def _compress_components(
    components: Mapping[str, np.ndarray],
    layout: Any,
    codec: str,
    rel_eb: float,
    axis_search: bool,
    temp_dir: Path,
) -> list[dict[str, Any]]:
    rows = []
    for logical in VELOCITY_FIELDS:
        values = components[logical]
        error_bound = rel_eb * float(np.ptp(values.astype(np.float64)))
        dense, _, _ = layout.encode_field(
            values,
            logical,
            position_residual=False,
        )
        rows.append(
            _compress_dense(
                logical,
                dense,
                values.size,
                error_bound,
                codec,
                axis_search,
                temp_dir,
            )
        )
    return rows


def _compress_flat(
    field: str,
    values: np.ndarray,
    error_bound: float,
    codec: str,
    temp_dir: Path,
) -> dict[str, Any]:
    raw_path = temp_dir / f"{field}.flat.raw"
    compressed_path = temp_dir / f"{field}.flat.{CODEC_EXTENSIONS[codec]}"
    decoded_path = temp_dir / f"{field}.flat.decoded.raw"
    values.tofile(raw_path)
    print(f"compressing {field} as flat ID-sorted data", flush=True)
    started = time.perf_counter()
    metadata = compress_lossy_raw(
        codec,
        str(raw_path),
        str(values.dtype),
        str(compressed_path),
        field,
        values.size,
        error_bound,
        False,
    )
    seconds = time.perf_counter() - started
    decompress_lossy_raw(metadata, str(decoded_path), False)
    decoded = np.fromfile(decoded_path, dtype=values.dtype)
    max_error = _max_error(values, decoded)
    _require_bound(field, max_error, error_bound)
    return _result_row(
        field,
        "id_sorted_flat",
        values.size,
        values.size,
        error_bound,
        int(metadata["bytes"]),
        max_error,
        seconds,
        metadata,
        reused=False,
    )


def _compress_dense(
    field: str,
    dense: np.ndarray,
    particle_count: int,
    error_bound: float,
    codec: str,
    axis_search: bool,
    temp_dir: Path,
) -> dict[str, Any]:
    raw_path = temp_dir / f"{field}.dense.raw"
    compressed_path = temp_dir / f"{field}.dense.{CODEC_EXTENSIONS[codec]}"
    decoded_path = temp_dir / f"{field}.dense.decoded.raw"
    dense.tofile(raw_path)
    print(
        f"compressing {field} as dense lattice; axis_search={axis_search}",
        flush=True,
    )
    started = time.perf_counter()
    metadata = compress_shaped_lossy_raw(
        codec,
        str(raw_path),
        str(dense.dtype),
        str(compressed_path),
        field,
        particle_count,
        error_bound,
        False,
        dense.shape,
        axis_search,
    )
    seconds = time.perf_counter() - started
    decompress_shaped_lossy_raw(metadata, str(decoded_path), False)
    decoded = np.fromfile(decoded_path, dtype=dense.dtype).reshape(dense.shape)
    max_error = _max_error(dense, decoded)
    _require_bound(field, max_error, error_bound)
    return _result_row(
        field,
        "periodic_dense_lattice",
        particle_count,
        dense.size,
        error_bound,
        int(metadata["bytes"]),
        max_error,
        seconds,
        metadata,
        reused=False,
    )


def _baseline_component_rows(
    manifest: Mapping[str, Any],
    input_path: Path,
    codec: str,
    rel_eb: float,
    dense_shape: tuple[int, ...],
    count: int,
    axis_search: bool,
    baseline_path: Optional[Path],
) -> list[dict[str, Any]]:
    recorded_input = Path(str(manifest["input_h5"])).expanduser().resolve()
    if not recorded_input.is_file() or not recorded_input.samefile(input_path):
        raise RuntimeError("Baseline manifest describes a different input file.")
    if int(manifest["count"]) != count:
        raise RuntimeError("Baseline manifest particle count does not match.")
    configured = manifest.get("compressors", {}).get(
        "lossy",
        manifest.get("compressors", {}).get("velocities"),
    )
    if configured != codec:
        raise RuntimeError(
            f"Baseline uses {configured!r}, but --codec selected {codec!r}."
        )
    lattice = manifest.get("lattice_layout", {})
    if not lattice.get("enabled") or tuple(lattice["dense_shape"]) != tuple(
        dense_shape
    ):
        raise RuntimeError("Baseline lattice shape does not match this inference.")
    if bool(lattice.get("axis_search", False)) != axis_search:
        raise RuntimeError("Baseline and requested lattice axis-search differ.")

    raw_bytes = count * np.dtype("float32").itemsize
    metrics_path = (
        baseline_path.parent / "metrics.json"
        if baseline_path is not None
        else Path("metrics.json")
    )
    metrics = read_json(metrics_path) if metrics_path.is_file() else {}
    rows = []
    for logical in VELOCITY_FIELDS:
        field = manifest["compressed_fields"][logical]
        if field.get("codec") != CODEC_METADATA_NAMES[codec]:
            raise RuntimeError(f"Baseline field {logical} uses another codec.")
        bound = manifest["field_error_bounds"][logical]
        if not math.isclose(
            float(bound.get("relative")),
            rel_eb,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise RuntimeError(
                f"Baseline field {logical} uses a different relative bound."
            )
        observed = (
            metrics.get("error_bound_consistency", {})
            .get(logical, {})
            .get("observed_max_absolute_error")
        )
        rows.append(
            {
                "field": logical,
                "layout": "periodic_dense_lattice",
                "codec": codec,
                "particle_count": count,
                "encoded_count": int(field["encoded_count"]),
                "particle_raw_bytes": raw_bytes,
                "encoded_raw_bytes": int(field["encoded_count"]) * 4,
                "error_bound": float(field["abs_error_bound"]),
                "compressed_bytes": int(field["bytes"]),
                "particle_compression_ratio": raw_bytes / int(field["bytes"]),
                "encoded_compression_ratio": (
                    int(field["encoded_count"]) * 4 / int(field["bytes"])
                ),
                "max_absolute_error": observed,
                "seconds": None,
                "axis_permutation": field.get("axis_permutation"),
                "axis_flips": field.get("axis_flips"),
                "reused_baseline": True,
            }
        )
    return rows


def _result_row(
    field: str,
    layout: str,
    particle_count: int,
    encoded_count: int,
    error_bound: float,
    compressed_bytes: int,
    max_error: float,
    seconds: float,
    metadata: Mapping[str, Any],
    reused: bool,
) -> dict[str, Any]:
    particle_raw_bytes = particle_count * np.dtype("float32").itemsize
    encoded_raw_bytes = encoded_count * np.dtype("float32").itemsize
    return {
        "field": field,
        "layout": layout,
        "codec": str(metadata["codec"]),
        "particle_count": particle_count,
        "encoded_count": encoded_count,
        "particle_raw_bytes": particle_raw_bytes,
        "encoded_raw_bytes": encoded_raw_bytes,
        "error_bound": error_bound,
        "compressed_bytes": compressed_bytes,
        "particle_compression_ratio": particle_raw_bytes / compressed_bytes,
        "encoded_compression_ratio": encoded_raw_bytes / compressed_bytes,
        "max_absolute_error": max_error,
        "seconds": seconds,
        "axis_permutation": metadata.get("axis_permutation"),
        "axis_flips": metadata.get("axis_flips"),
        "reused_baseline": reused,
    }


def _relationship_statistics(
    sorted_ids: np.ndarray,
    values: np.ndarray,
    maximum_pairs: int = 1_000_000,
) -> dict[str, float]:
    samples = np.linspace(
        0,
        values.size - 1,
        min(values.size, maximum_pairs),
        dtype=np.intp,
    )
    pair_starts = np.linspace(
        0,
        values.size - 2,
        min(max(0, values.size - 1), maximum_pairs),
        dtype=np.intp,
    )
    return {
        "pearson_r_with_numeric_id": _pearson(
            sorted_ids[samples].astype(np.float64),
            values[samples].astype(np.float64),
        ),
        "id_sorted_adjacent_value_pearson_r": _pearson(
            values[pair_starts].astype(np.float64),
            values[pair_starts + 1].astype(np.float64),
        ),
        "id_sorted_adjacent_mean_abs_difference": float(
            np.mean(
                np.abs(
                    values[pair_starts + 1].astype(np.float64)
                    - values[pair_starts].astype(np.float64)
                )
            )
        ),
    }


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or float(np.ptp(left)) == 0.0 or float(np.ptp(right)) == 0.0:
        return math.nan
    return float(np.corrcoef(left, right)[0, 1])


def _max_error(original: np.ndarray, decoded: np.ndarray) -> float:
    return float(
        np.max(
            np.abs(
                decoded.astype(np.float64) - original.astype(np.float64)
            ),
            initial=0.0,
        )
    )


def _require_bound(field: str, observed: float, bound: float) -> None:
    tolerance = 1e-12 + 1e-5 * bound
    if observed > bound + tolerance:
        raise RuntimeError(
            f"{field} exceeded error bound {bound:g}: observed {observed:g}."
        )


def main() -> int:
    args = _parse_args()
    payload = run(args)
    for row in payload["results"]:
        print(
            f"{row['field']} {row['layout']}: "
            f"bytes={row['compressed_bytes']}, "
            f"CR={row['particle_compression_ratio']:.6g}"
        )
    conclusion = payload["conclusion"]
    print(
        "dense scalar beats all components = "
        f"{conclusion['dense_scalar_beats_all_components']}"
    )
    print(f"results = {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
