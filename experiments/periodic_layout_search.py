"""Benchmark a periodically unwrapped dense lattice on a particle partition."""

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

from experiments.payload_search import (  # noqa: E402
    POSITION_FIELDS,
    VELOCITY_FIELDS,
    _compress_szo,
    _lattice_coordinates,
)
from src.hdf5_io import resolve_fields  # noqa: E402
from src.runtime import load_pyszo  # noqa: E402


def _periodic_start(values: np.ndarray, side: int) -> tuple[int, int]:
    present = np.unique(values)
    if present.size == side:
        return 0, side
    following = np.roll(present, -1)
    gaps = (following.astype(np.int64) - present.astype(np.int64)) % side
    cut = int(np.argmax(gaps))
    start = int(following[cut])
    unwrapped = (present.astype(np.int64) - start) % side
    return start, int(unwrapped.max()) + 1


def _periodic_layout(
    coordinates: np.ndarray,
    side: int,
) -> tuple[np.ndarray, tuple[int, ...], tuple[int, ...]]:
    starts_and_spans = [
        _periodic_start(coordinates[:, axis], side) for axis in range(3)
    ]
    starts = tuple(item[0] for item in starts_and_spans)
    shape = tuple(item[1] for item in starts_and_spans)
    local = np.empty_like(coordinates, dtype=np.int64)
    for axis in range(3):
        local[:, axis] = (
            coordinates[:, axis].astype(np.int64) - starts[axis]
        ) % side
    indices = np.ravel_multi_index(local.T, shape)
    if np.unique(indices).size != indices.size:
        raise RuntimeError("Particle IDs do not map uniquely to lattice cells.")
    return indices.astype(np.intp, copy=False), shape, starts


def _dense_field(
    values: np.ndarray,
    indices: np.ndarray,
    dense_count: int,
) -> np.ndarray:
    dense = np.empty(dense_count, dtype=np.float32)
    dense[indices] = values
    if indices.size == dense_count:
        return dense
    ordering = np.argsort(indices, kind="stable")
    sorted_indices = indices[ordering]
    sorted_values = values[ordering]
    occupied = np.zeros(dense_count, dtype=bool)
    occupied[sorted_indices] = True
    missing = np.flatnonzero(~occupied)
    dense[missing] = np.interp(
        missing,
        sorted_indices,
        sorted_values,
    ).astype(np.float32)
    return dense


def _record(
    rows: list[dict[str, object]],
    group: str,
    field: str,
    layout: str,
    values: np.ndarray,
    error_bound: float,
    algorithm: int,
) -> None:
    started = time.perf_counter()
    size = _compress_szo(values, error_bound, algorithm)
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
    parser.add_argument("--id-base", type=int, choices=(0, 1), default=0)
    parser.add_argument("--axes", action="store_true")
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
        dense_indices, shape, starts = _periodic_layout(coordinates, side)
        dense_count = math.prod(shape)
        algorithm = load_pyszo()[3].INTERP_LORENZO
        permutations = (
            tuple(__import__("itertools").permutations(range(3)))
            if args.axes
            else ((0, 1, 2),)
        )

        for axis, logical in enumerate(POSITION_FIELDS):
            values = (
                source[fields[logical]][:].astype(np.float64) / scale
            ).astype(np.float32)[order]
            error_bound = args.rel_eb * float(np.ptp(values))
            predictor = coordinates[:, axis].astype(np.float64) / side
            residual = (
                (values.astype(np.float64) - predictor + 0.5) % 1.0 - 0.5
            ).astype(np.float32)
            _record(
                rows,
                "positions",
                logical,
                "id_sorted_1d",
                values,
                error_bound,
                algorithm,
            )
            dense = _dense_field(residual, dense_indices, dense_count).reshape(shape)
            for permutation in permutations:
                encoded = np.ascontiguousarray(np.transpose(dense, permutation))
                _record(
                    rows,
                    "positions",
                    logical,
                    "periodic_residual_dense_axes_"
                    + "".join(str(item) for item in permutation),
                    encoded,
                    error_bound,
                    algorithm,
                )

        for logical in VELOCITY_FIELDS:
            values = source[fields[logical]][:].astype(np.float32, copy=False)[order]
            error_bound = args.rel_eb * float(np.ptp(values))
            _record(
                rows,
                "velocities",
                logical,
                "id_sorted_1d",
                values,
                error_bound,
                algorithm,
            )
            dense = _dense_field(values, dense_indices, dense_count).reshape(shape)
            for permutation in permutations:
                encoded = np.ascontiguousarray(np.transpose(dense, permutation))
                _record(
                    rows,
                    "velocities",
                    logical,
                    "periodic_dense_axes_"
                    + "".join(str(item) for item in permutation),
                    encoded,
                    error_bound,
                    algorithm,
                )

    payload = {
        "input_h5": str(Path(args.input_h5).resolve()),
        "side": side,
        "starts": list(starts),
        "shape": list(shape),
        "count": int(ids.size),
        "dense_count": dense_count,
        "occupancy": int(ids.size) / dense_count,
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
