"""Compare velocity tie orders on an existing XnYZip/SZO roundtrip package.

Run the baseline without --clean-raw or --xnyzip-tie-sort. This search measures
velocity streams only; use main.py roundtrip for complete packaged ratios.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from src.constants import POSITION_FIELDS, VELOCITY_FIELDS
from src.runtime import load_pyszo
from src.structured_layout import morton_encode_3d, morton_to_hilbert_3d


def run(work: Path, limit: int) -> dict:
    manifest = json.loads((work / "manifest.json").read_text())
    if manifest["compressors"]["positions"] != "xnyzip":
        raise ValueError("The baseline must use XnYZip positions.")
    if manifest.get("position_tie_sort", {}).get("enabled"):
        raise ValueError("Use a baseline without tie sorting.")
    count = min(limit, manifest["count"])
    if count < 1:
        raise ValueError("Search limit must be positive.")
    positions = [np.fromfile(work / "decompressed" / f"{key}.f32.raw", dtype="float32", count=count)
                 for key in POSITION_FIELDS]
    velocities = [np.fromfile(manifest["artifacts"]["preprocessed"][f"{key}_canonical_ordered"],
                              dtype=manifest["fields"][key]["dtype"], count=count)
                  for key in VELOCITY_FIELDS]
    if any(len(a) != count for a in positions + velocities):
        raise ValueError("Baseline raw arrays are incomplete.")
    bounds = [manifest["field_error_bounds"][key]["abs"] for key in VELOCITY_FIELDS]
    changed = np.zeros(count - 1, dtype=bool)
    for values in positions:
        bits = values.view("uint32")
        changed |= bits[1:] != bits[:-1]
    groups = np.r_[np.uint32(0), np.cumsum(changed, dtype="uint32")]
    szo, Config, EB, Algorithm = load_pyszo()
    results = []

    def measure(name, permutation):
        for profile in ("default", "lorenzo_1d"):
            started = time.perf_counter()
            sizes, errors = [], []
            for values, bound in zip(velocities, bounds):
                ordered = np.ascontiguousarray(values[permutation])
                config = Config(ordered.shape)
                config.errorBoundMode = EB.ABS
                config.absErrorBound = bound
                if profile == "lorenzo_1d":
                    config.cmprAlgo = Algorithm.LORENZO_REG
                    config.lorenzo = True
                    config.lorenzo2 = False
                    config.regression = False
                    config.regression2 = False
                payload, _ = szo.compress(ordered, config, copy=True)
                decoded, _ = szo.decompress(payload, ordered.dtype, ordered.shape)
                error = float(np.max(np.abs(decoded.astype("float64") - ordered.astype("float64"))))
                if error > bound:
                    raise RuntimeError(f"{name}/{profile} exceeded its bound: {error} > {bound}")
                sizes.append(int(payload.size))
                errors.append(error)
            row = {"order": name, "profile": profile, "velocity_bytes": sum(sizes),
                   "field_bytes": sizes, "max_abs_errors": errors,
                   "seconds": time.perf_counter() - started}
            results.append(row)
            print(json.dumps(row), flush=True)

    measure("native", slice(None))
    measure("vx_ties", np.lexsort((velocities[0], groups)))
    for bits in (6, 8, 10):
        coordinates = []
        for values in velocities:
            values = values.astype("float64")
            span = float(np.ptp(values))
            coordinates.append(np.floor((values - values.min()) / (span or 1)
                                        * ((1 << bits) - 1)).astype("uint32"))
        morton = morton_encode_3d(*coordinates)
        measure(f"morton{bits}_ties", np.lexsort((morton, groups)))
        hilbert = morton_to_hilbert_3d(morton, bits)
        measure(f"hilbert{bits}_ties", np.lexsort((hilbert, groups)))
    return {"baseline": str(work.resolve()), "count": count,
            "decoded_position_runs": int(groups[-1]) + 1,
            "velocity_abs_bounds": bounds, "results": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work_dir", type=Path)
    parser.add_argument("--limit", type=int, default=1_000_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.work_dir, args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
