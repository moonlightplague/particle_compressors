"""Search the rate tradeoff for an error-safe position/velocity predictor."""

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
    _dense_field,
    _dense_layout,
    _lattice_coordinates,
)
from src.hdf5_io import resolve_fields  # noqa: E402
from src.runtime import load_pyszo  # noqa: E402


def _roundtrip(
    values: np.ndarray,
    error_bound: float,
) -> tuple[int, np.ndarray]:
    szo, config_type, error_bound_mode, algorithms = load_pyszo()
    values = np.ascontiguousarray(values, dtype=np.float32)
    config = config_type(values.shape)
    config.errorBoundMode = error_bound_mode.ABS
    config.absErrorBound = float(error_bound)
    config.cmprAlgo = algorithms.INTERP_LORENZO
    payload, _ = szo.compress(values, config, copy=True)
    decoded, _ = szo.decompress(payload, np.float32, values.shape)
    return int(np.asarray(payload, dtype=np.uint8).size), np.asarray(decoded)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_h5")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rel-eb", type=float, default=1e-3)
    parser.add_argument("--id-base", type=int, choices=(0, 1), default=1)
    args = parser.parse_args()

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

        x = (
            source[fields["x"]][:].astype(np.float64) / scale
        ).astype(np.float32)[order]
        vx = source[fields["vx"]][:].astype(np.float32, copy=False)[order]
        qx = coordinates[:, 0].astype(np.float64) / side
        x_residual = (x.astype(np.float64) - qx).astype(np.float32)
        x_bound = args.rel_eb * float(np.ptp(x))
        vx_bound = args.rel_eb * float(np.ptp(vx))

        sample_step = max(1, x.size // 2_000_000)
        design = np.column_stack(
            (
                np.ones(x[::sample_step].size, dtype=np.float64),
                x_residual[::sample_step].astype(np.float64),
            )
        )
        coefficients = np.linalg.lstsq(
            design,
            vx[::sample_step].astype(np.float64),
            rcond=None,
        )[0]
        intercept, slope = (float(value) for value in coefficients)
        prediction = intercept + slope * x_residual.astype(np.float64)
        vx_residual = (vx.astype(np.float64) - prediction).astype(np.float32)

        dense_x = _dense_field(
            x_residual,
            dense_indices,
            dense_count,
        ).reshape(shape)
        dense_vx_residual = _dense_field(
            vx_residual,
            dense_indices,
            dense_count,
        ).reshape(shape)

        rows: list[dict[str, object]] = []
        for fraction in (1.0, 0.5, 0.25, 0.1, 0.05, 0.025, 0.01, 0.005):
            position_codec_bound = x_bound * fraction
            propagated_budget = abs(slope) * position_codec_bound
            residual_codec_bound = vx_bound - propagated_budget
            if residual_codec_bound <= 0:
                continue
            started = time.perf_counter()
            x_size, decoded_dense_x = _roundtrip(
                dense_x,
                position_codec_bound,
            )
            vx_size, decoded_dense_vx = _roundtrip(
                dense_vx_residual,
                residual_codec_bound,
            )
            decoded_x_residual = decoded_dense_x.reshape(-1)[dense_indices]
            decoded_x = decoded_x_residual.astype(np.float64) + qx
            decoded_prediction = intercept + slope * decoded_x_residual.astype(
                np.float64
            )
            decoded_vx = (
                decoded_dense_vx.reshape(-1)[dense_indices].astype(np.float64)
                + decoded_prediction
            )
            row = {
                "position_bound_fraction": fraction,
                "position_codec_bound": position_codec_bound,
                "velocity_requested_bound": vx_bound,
                "velocity_residual_codec_bound": residual_codec_bound,
                "propagated_velocity_budget": propagated_budget,
                "x_compressed_bytes": x_size,
                "vx_residual_compressed_bytes": vx_size,
                "combined_bytes": x_size + vx_size + 16,
                "x_max_error_scaled": float(
                    np.max(np.abs(decoded_x - x.astype(np.float64)), initial=0.0)
                ),
                "vx_max_error": float(
                    np.max(np.abs(decoded_vx - vx.astype(np.float64)), initial=0.0)
                ),
                "seconds": time.perf_counter() - started,
            }
            rows.append(row)
            print(json.dumps(row), flush=True)

    payload = {
        "input_h5": str(Path(args.input_h5).resolve()),
        "shape": list(shape),
        "count": int(ids.size),
        "x_requested_bound": x_bound,
        "vx_requested_bound": vx_bound,
        "predictor": {"intercept": intercept, "slope": slope},
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
