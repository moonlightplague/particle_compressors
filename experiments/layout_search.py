"""Fine-grained SZO layout search for dense particle lattice fields."""

from __future__ import annotations

import argparse
import itertools
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

from experiments.payload_search import (  # noqa: E402
    POSITION_FIELDS,
    VELOCITY_FIELDS,
    _compress_szo,
    _dense_field,
    _dense_layout,
    _lattice_coordinates,
)
from src.hdf5_io import resolve_fields  # noqa: E402
from src.runtime import load_pyszo  # noqa: E402


def _fill_dense(
    values: np.ndarray,
    indices: np.ndarray,
    dense_count: int,
    method: str,
) -> np.ndarray:
    if method == "linear":
        return _dense_field(values, indices, dense_count)
    dense = np.empty(dense_count, dtype=np.float32)
    dense[indices] = values
    occupied = np.zeros(dense_count, dtype=bool)
    occupied[indices] = True
    missing = np.flatnonzero(~occupied)
    if method == "zero":
        dense[missing] = 0.0
    elif method == "mean":
        dense[missing] = float(np.mean(values, dtype=np.float64))
    elif method == "nearest":
        insertion = np.searchsorted(indices, missing)
        right_pos = np.minimum(insertion, indices.size - 1)
        left_pos = np.maximum(insertion - 1, 0)
        left = indices[left_pos]
        right = indices[right_pos]
        choose_right = (right - missing) < (missing - left)
        source_pos = np.where(choose_right, right_pos, left_pos)
        dense[missing] = values[source_pos]
    else:
        raise ValueError(method)
    return dense


def _result(
    rows: list[dict[str, object]],
    group: str,
    field: str,
    layout: str,
    values: np.ndarray,
    error_bound: float,
    algorithm: int,
    metadata_bytes: int = 0,
) -> None:
    started = time.perf_counter()
    size = _compress_szo(values, error_bound, algorithm) + metadata_bytes
    row = {
        "group": group,
        "field": field,
        "layout": layout,
        "shape": list(values.shape),
        "compressed_bytes": size,
        "seconds": time.perf_counter() - started,
    }
    rows.append(row)
    print(json.dumps(row), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_h5")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rel-eb", type=float, default=1e-3)
    parser.add_argument("--id-base", type=int, choices=(0, 1), default=1)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    with h5py.File(args.input_h5, "r") as source:
        fields = resolve_fields(source)
        ids = source[fields["id"]][:]
        order = np.argsort(ids, kind="stable")
        sorted_ids = ids[order]
        side = int(source.attrs["nsidemesh"])
        scale = float(source.attrs["bitwidth"])
        coordinates = _lattice_coordinates(sorted_ids, side, args.id_base)
        dense_indices, shape = _dense_layout(coordinates)
        dense_count = math.prod(shape)
        algorithm = load_pyszo()[3].INTERP_LORENZO

        position_values: dict[str, np.ndarray] = {}
        position_bounds: dict[str, float] = {}
        position_residuals: dict[str, np.ndarray] = {}
        velocity_values: dict[str, np.ndarray] = {}
        velocity_bounds: dict[str, float] = {}
        for axis, logical in enumerate(POSITION_FIELDS):
            values = (
                source[fields[logical]][:].astype(np.float64) / scale
            ).astype(np.float32)[order]
            position_values[logical] = values
            position_bounds[logical] = args.rel_eb * float(np.ptp(values))
            predictor = coordinates[:, axis].astype(np.float64) / side
            position_residuals[logical] = (
                values.astype(np.float64) - predictor
            ).astype(np.float32)
        for logical in VELOCITY_FIELDS:
            values = source[fields[logical]][:].astype(np.float32, copy=False)[order]
            velocity_values[logical] = values
            velocity_bounds[logical] = args.rel_eb * float(np.ptp(values))

        # Hole-fill sensitivity on the native axis order.
        for method in ("zero", "mean", "nearest", "linear"):
            for logical in VELOCITY_FIELDS:
                dense = _fill_dense(
                    velocity_values[logical],
                    dense_indices,
                    dense_count,
                    method,
                ).reshape(shape)
                _result(
                    rows,
                    "velocities",
                    logical,
                    f"dense_3d_fill_{method}",
                    dense,
                    velocity_bounds[logical],
                    algorithm,
                )

        # Axis order affects interpolation traversal and is effectively free:
        # the chosen permutation is a few manifest integers.
        for permutation in itertools.permutations(range(3)):
            permutation_label = "".join(str(axis) for axis in permutation)
            for logical in POSITION_FIELDS:
                dense = _dense_field(
                    position_residuals[logical],
                    dense_indices,
                    dense_count,
                ).reshape(shape)
                transposed = np.ascontiguousarray(np.transpose(dense, permutation))
                _result(
                    rows,
                    "positions",
                    logical,
                    f"lattice_residual_axes_{permutation_label}",
                    transposed,
                    position_bounds[logical],
                    algorithm,
                )
            for logical in VELOCITY_FIELDS:
                dense = _dense_field(
                    velocity_values[logical],
                    dense_indices,
                    dense_count,
                ).reshape(shape)
                transposed = np.ascontiguousarray(np.transpose(dense, permutation))
                _result(
                    rows,
                    "velocities",
                    logical,
                    f"dense_axes_{permutation_label}",
                    transposed,
                    velocity_bounds[logical],
                    algorithm,
                )

        # Normalize by per-field error bounds so one stream can preserve every
        # component's individual tolerance.
        for group, names, values_by_name, bounds_by_name in (
            (
                "positions",
                POSITION_FIELDS,
                position_residuals,
                position_bounds,
            ),
            (
                "velocities",
                VELOCITY_FIELDS,
                velocity_values,
                velocity_bounds,
            ),
        ):
            dense_fields = [
                _dense_field(values_by_name[name], dense_indices, dense_count)
                .reshape(shape)
                / bounds_by_name[name]
                for name in names
            ]
            for component_axis in (0, 3):
                stacked = np.ascontiguousarray(
                    np.stack(dense_fields, axis=component_axis),
                    dtype=np.float32,
                )
                _result(
                    rows,
                    group,
                    "xyz" if group == "positions" else "vxyz",
                    f"normalized_stacked_axis_{component_axis}",
                    stacked,
                    1.0,
                    algorithm,
                )

        # Independent slabs can sometimes outperform a monolithic stream by
        # adapting entropy tables to local flow regimes.
        for slab_depth in (1, 2, 4, 8, 16, 32, 64):
            for logical in VELOCITY_FIELDS:
                dense = _dense_field(
                    velocity_values[logical],
                    dense_indices,
                    dense_count,
                ).reshape(shape)
                size = 0
                started = time.perf_counter()
                for start in range(0, shape[0], slab_depth):
                    slab = np.ascontiguousarray(dense[start : start + slab_depth])
                    size += _compress_szo(
                        slab,
                        velocity_bounds[logical],
                        algorithm,
                    )
                row = {
                    "group": "velocities",
                    "field": logical,
                    "layout": f"dense_slabs_{slab_depth}",
                    "shape": list(shape),
                    "compressed_bytes": size,
                    "seconds": time.perf_counter() - started,
                }
                rows.append(row)
                print(json.dumps(row), flush=True)

    payload = {
        "input_h5": str(Path(args.input_h5).resolve()),
        "shape": list(shape),
        "count": int(ids.size),
        "dense_count": dense_count,
        "results": rows,
    }
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"results = {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
