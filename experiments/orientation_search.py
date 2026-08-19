"""Search dense-axis permutations and reversals for lattice velocity fields."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.payload_search import _compress_szo  # noqa: E402
from src.constants import POSITION_FIELDS, VELOCITY_FIELDS  # noqa: E402
from src.hdf5_io import resolve_fields  # noqa: E402
from src.lattice_layout import infer_dense_lattice_layout  # noqa: E402
from src.runtime import load_pyszo  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_h5")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rel-eb", type=float, default=1e-3)
    parser.add_argument("--min-occupancy", type=float, default=0.8)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    with h5py.File(args.input_h5, "r") as source:
        fields = resolve_fields(source)
        ids = source[fields["id"]][:]
        order = np.argsort(ids, kind="stable")
        sorted_ids = ids[order]
        side = int(source.attrs["nsidemesh"])
        scale = float(source.attrs["bitwidth"])
        positions = {
            logical: (
                source[fields[logical]][:].astype(np.float64) / scale
            ).astype(np.float32)[order]
            for logical in POSITION_FIELDS
        }
        layout = infer_dense_lattice_layout(
            sorted_ids,
            positions,
            side,
            args.min_occupancy,
        )
        algorithm = load_pyszo()[3].INTERP_LORENZO

        for logical in VELOCITY_FIELDS:
            values = source[fields[logical]][:].astype(
                np.float32,
                copy=False,
            )[order]
            bound = args.rel_eb * float(np.ptp(values))
            dense, _, _ = layout.encode_field(
                values,
                logical,
                position_residual=False,
            )
            for permutation in itertools.permutations(range(3)):
                permuted = np.transpose(dense, permutation)
                for flips in itertools.product((False, True), repeat=3):
                    slices = tuple(
                        slice(None, None, -1) if flipped else slice(None)
                        for flipped in flips
                    )
                    candidate = permuted[slices]
                    started = time.perf_counter()
                    size = _compress_szo(candidate, bound, algorithm)
                    row = {
                        "field": logical,
                        "permutation": list(permutation),
                        "flips": list(flips),
                        "compressed_bytes": size,
                        "seconds": time.perf_counter() - started,
                    }
                    rows.append(row)
                    print(json.dumps(row), flush=True)

    best = {
        logical: min(
            (row for row in rows if row["field"] == logical),
            key=lambda row: int(row["compressed_bytes"]),
        )
        for logical in VELOCITY_FIELDS
    }
    payload = {
        "input_h5": str(Path(args.input_h5).resolve()),
        "dense_shape": list(layout.shape),
        "occupancy": layout.occupancy,
        "best": best,
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"results = {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
