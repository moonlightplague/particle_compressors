"""Search reversible lattice layouts and predictors for particle payloads.

This is intentionally a benchmark, not part of the package format.  It reads a
complete lattice partition, sorts it by particle ID, and compares the existing
one-dimensional field layout with dense three-dimensional SZO layouts.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.hdf5_io import resolve_fields
from src.runtime import load_pyszo, load_pysz


POSITION_FIELDS = ("x", "y", "z")
VELOCITY_FIELDS = ("vx", "vy", "vz")


def _compress_szo(values: np.ndarray, error_bound: float, algorithm: int) -> int:
    szo, config_type, error_bound_mode, algorithms = load_pyszo()
    contiguous = np.ascontiguousarray(values, dtype=np.float32)
    config = config_type(contiguous.shape)
    config.errorBoundMode = error_bound_mode.ABS
    config.absErrorBound = float(error_bound)
    config.cmprAlgo = algorithm
    payload, _ = szo.compress(contiguous, config, copy=True)
    decoded, _ = szo.decompress(payload, np.float32, contiguous.shape)
    error = float(
        np.max(
            np.abs(
                np.asarray(decoded, dtype=np.float32).astype(np.float64)
                - contiguous.astype(np.float64)
            ),
            initial=0.0,
        )
    )
    if error > error_bound * (1.0 + 1e-5) + 1e-12:
        raise RuntimeError(
            f"SZO violated {error_bound:g} bound with max error {error:g}."
        )
    return int(np.asarray(payload, dtype=np.uint8).size)


def _compress_sz3(values: np.ndarray, error_bound: float) -> int:
    pysz, config_type, error_bound_mode = load_pysz()
    contiguous = np.ascontiguousarray(values, dtype=np.float32)
    config = config_type(contiguous.shape)
    config.errorBoundMode = error_bound_mode.ABS
    config.absErrorBound = float(error_bound)
    payload, _ = pysz.compress(contiguous, config)
    return int(np.asarray(payload, dtype=np.uint8).size)


def _lattice_coordinates(ids: np.ndarray, side: int, id_base: int) -> np.ndarray:
    linear = ids.astype(np.uint64, copy=False) - np.uint64(id_base)
    plane = np.uint64(side * side)
    coordinates = np.empty((ids.size, 3), dtype=np.int32)
    coordinates[:, 0] = (linear // plane).astype(np.int32)
    coordinates[:, 1] = ((linear // np.uint64(side)) % side).astype(np.int32)
    coordinates[:, 2] = (linear % side).astype(np.int32)
    return coordinates


def _dense_layout(coordinates: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    lower = coordinates.min(axis=0)
    upper = coordinates.max(axis=0)
    shape_array = upper.astype(np.int64) - lower.astype(np.int64) + 1
    shape = tuple(int(value) for value in shape_array)
    local = coordinates.astype(np.int64) - lower
    indices = np.ravel_multi_index(local.T, shape)
    if np.unique(indices).size != indices.size:
        raise RuntimeError("Particle IDs do not map uniquely to lattice cells.")
    return indices.astype(np.intp, copy=False), shape


def _dense_field(
    sorted_values: np.ndarray,
    dense_indices: np.ndarray,
    dense_count: int,
) -> np.ndarray:
    dense = np.empty(dense_count, dtype=np.float32)
    dense[dense_indices] = sorted_values
    if dense_count == dense_indices.size:
        return dense

    occupied = np.zeros(dense_count, dtype=bool)
    occupied[dense_indices] = True
    missing_indices = np.flatnonzero(~occupied)
    # Linear fill keeps holes from injecting artificial discontinuities into
    # the predictor.  Missing cells are never emitted by the roundtrip.
    dense[missing_indices] = np.interp(
        missing_indices,
        dense_indices,
        sorted_values,
    ).astype(np.float32)
    return dense


def _fit_affine(features: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sample_step = max(1, features.shape[0] // 1_000_000)
    sampled_features = features[::sample_step].astype(np.float64)
    sampled_targets = targets[::sample_step].astype(np.float64)
    design = np.column_stack(
        (np.ones(sampled_features.shape[0], dtype=np.float64), sampled_features)
    )
    coefficients = np.linalg.lstsq(design, sampled_targets, rcond=None)[0]
    prediction = coefficients[0] + features.astype(np.float64) @ coefficients[1:]
    residual = targets.astype(np.float64) - prediction
    return residual.astype(np.float32), coefficients


def _record(
    rows: list[dict[str, object]],
    group: str,
    field: str,
    layout: str,
    codec: str,
    values: np.ndarray,
    error_bound: float,
    algorithm: int,
) -> None:
    started = time.perf_counter()
    if codec == "szo":
        size = _compress_szo(values, error_bound, algorithm)
    else:
        size = _compress_sz3(values, error_bound)
    raw_bytes = int(values.size * values.dtype.itemsize)
    row = {
        "group": group,
        "field": field,
        "layout": layout,
        "codec": codec,
        "shape": list(values.shape),
        "error_bound": error_bound,
        "compressed_bytes": size,
        "encoded_raw_bytes": raw_bytes,
        "encoded_compression_ratio": raw_bytes / size,
        "seconds": time.perf_counter() - started,
    }
    rows.append(row)
    print(json.dumps(row), flush=True)


def run(args: argparse.Namespace) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    with h5py.File(args.input_h5, "r") as source:
        fields = resolve_fields(source)
        count = int(source[fields["id"]].shape[0])
        ids = source[fields["id"]][:]
        order = np.argsort(ids, kind="stable")
        sorted_ids = ids[order]
        side = int(source.attrs[args.mesh_side_attr])
        scale = float(source.attrs[args.position_scale_attr])
        coordinates = _lattice_coordinates(sorted_ids, side, args.id_base)
        dense_indices, dense_shape = _dense_layout(coordinates)
        dense_count = math.prod(dense_shape)

        algorithms = load_pyszo()[3]
        algorithm_values = {
            "auto": algorithms.INTERP_LORENZO,
            "interp": algorithms.INTERP,
            "lorenzo": algorithms.LORENZO_REG,
        }
        selected_algorithms = {
            name: algorithm_values[name] for name in args.algorithms
        }

        sorted_positions: dict[str, np.ndarray] = {}
        for axis, logical in enumerate(POSITION_FIELDS):
            values = (
                source[fields[logical]][:].astype(np.float64) / scale
            ).astype(np.float32)[order]
            sorted_positions[logical] = values
            error_bound = args.rel_eb * float(
                values.astype(np.float64).max()
                - values.astype(np.float64).min()
            )
            predictor = coordinates[:, axis].astype(np.float64) / side
            residual = (values.astype(np.float64) - predictor).astype(np.float32)
            dense_raw = _dense_field(values, dense_indices, dense_count).reshape(
                dense_shape
            )
            dense_residual = _dense_field(
                residual,
                dense_indices,
                dense_count,
            ).reshape(dense_shape)
            for name, algorithm in selected_algorithms.items():
                _record(
                    rows,
                    "positions",
                    logical,
                    f"id_sorted_1d:{name}",
                    "szo",
                    values,
                    error_bound,
                    algorithm,
                )
                _record(
                    rows,
                    "positions",
                    logical,
                    f"dense_3d:{name}",
                    "szo",
                    dense_raw,
                    error_bound,
                    algorithm,
                )
                _record(
                    rows,
                    "positions",
                    logical,
                    f"lattice_residual_dense_3d:{name}",
                    "szo",
                    dense_residual,
                    error_bound,
                    algorithm,
                )
            if args.sz3:
                _record(
                    rows,
                    "positions",
                    logical,
                    "id_sorted_1d",
                    "sz3",
                    values,
                    error_bound,
                    algorithm_values["auto"],
                )
                _record(
                    rows,
                    "positions",
                    logical,
                    "lattice_residual_dense_3d",
                    "sz3",
                    dense_residual,
                    error_bound,
                    algorithm_values["auto"],
                )

        coordinate_features = coordinates.astype(np.float64) / side
        position_features = np.column_stack(
            [sorted_positions[field] for field in POSITION_FIELDS]
        )
        displacement_features = position_features - coordinate_features
        for logical in VELOCITY_FIELDS:
            values = source[fields[logical]][:].astype(np.float32, copy=False)[order]
            error_bound = args.rel_eb * float(
                values.astype(np.float64).max()
                - values.astype(np.float64).min()
            )
            residual_lattice, lattice_coefficients = _fit_affine(
                coordinate_features,
                values,
            )
            residual_displacement, displacement_coefficients = _fit_affine(
                displacement_features,
                values,
            )
            candidates = {
                "dense_3d": _dense_field(values, dense_indices, dense_count),
                "lattice_affine_residual_dense_3d": _dense_field(
                    residual_lattice,
                    dense_indices,
                    dense_count,
                ),
                "displacement_affine_residual_dense_3d": _dense_field(
                    residual_displacement,
                    dense_indices,
                    dense_count,
                ),
            }
            coefficient_bytes = (
                lattice_coefficients.nbytes + displacement_coefficients.nbytes
            )
            for name, algorithm in selected_algorithms.items():
                _record(
                    rows,
                    "velocities",
                    logical,
                    f"id_sorted_1d:{name}",
                    "szo",
                    values,
                    error_bound,
                    algorithm,
                )
                for layout, dense in candidates.items():
                    _record(
                        rows,
                        "velocities",
                        logical,
                        f"{layout}:{name}",
                        "szo",
                        dense.reshape(dense_shape),
                        error_bound,
                        algorithm,
                    )
            if args.sz3:
                _record(
                    rows,
                    "velocities",
                    logical,
                    "id_sorted_1d",
                    "sz3",
                    values,
                    error_bound,
                    algorithm_values["auto"],
                )
                _record(
                    rows,
                    "velocities",
                    logical,
                    "dense_3d",
                    "sz3",
                    candidates["dense_3d"].reshape(dense_shape),
                    error_bound,
                    algorithm_values["auto"],
                )

    result = {
        "input_h5": str(Path(args.input_h5).resolve()),
        "count": count,
        "lattice": {
            "side": side,
            "id_base": args.id_base,
            "dense_shape": list(dense_shape),
            "dense_count": dense_count,
            "occupancy": count / dense_count,
        },
        "predictor_metadata_upper_bound_bytes": coefficient_bytes,
        "results": rows,
    }
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_h5")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rel-eb", type=float, default=1e-3)
    parser.add_argument("--id-base", type=int, choices=(0, 1), default=1)
    parser.add_argument("--mesh-side-attr", default="nsidemesh")
    parser.add_argument("--position-scale-attr", default="bitwidth")
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=("auto", "interp", "lorenzo"),
        default=("auto",),
    )
    parser.add_argument("--sz3", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run(args)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(f"results = {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
