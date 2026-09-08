"""Audit equal-bound baseline/structured runs and write a reproducible report.

Usage: python -m experiments.compare_structured BASELINE_DIR OPTIMIZED_DIR
       [--output docs/snapshot7_structure_aware.json]

Accepts one package or a directory of per-partition packages. Optimized runs
must have completed with --metrics. Reads every package's actual archive and
manifest sizes; never includes reconstructed HDF5 or temporary raw data in CR.
"""

import argparse
import json
from pathlib import Path

from src.manifest import compressed_sizes
from src.runtime import read_json, write_json


def packages(root, batch=False):
    paths = [root / "manifest.json"] if not batch and (root / "manifest.json").is_file() else sorted(root.glob("*/manifest.json"))
    if not paths:
        raise RuntimeError(f"No packages in {root}")
    result = {}
    for path in paths:
        manifest = read_json(path)
        key = "merged" if manifest.get("merge", {}).get("enabled") else Path(
            manifest.get("input_file", manifest.get("input_h5", path.parent.name))).name
        if key in result:
            raise RuntimeError(f"Duplicate input {key}")
        result[key] = (path.parent, manifest)
    return result


def measure(root, manifest):
    sizes = compressed_sizes(root)
    return {
        "original_bytes": manifest["sizes"]["selected_original_payload_bytes"],
        "compressed_bytes": sum(sizes.values()),
        "id_bytes": sizes["compressed/id.pco"],
        "position_bytes": sizes["compressed/positions.xnyzip"],
        "velocity_bytes": sum(sizes[f"compressed/{key}.szo"] for key in ("vx", "vy", "vz")),
        "manifest_bytes": sizes["manifest.json"],
    }


def compare(baseline_root, optimized_root, batch=False):
    baseline, optimized = packages(baseline_root, batch), packages(optimized_root, batch)
    if baseline.keys() != optimized.keys():
        raise RuntimeError("Baseline and optimized input sets differ")
    rows = []
    for key in baseline:
        old_root, old = baseline[key]
        new_root, new = optimized[key]
        if old["count"] != new["count"] or old["field_error_bounds"] != new["field_error_bounds"]:
            raise RuntimeError(f"Particle count or error bounds differ for {key}")
        if not new.get("structured_layout", {}).get("enabled"):
            raise RuntimeError(f"Structured layout was not enabled for {key}")
        quality = read_json(new_root / "metrics.json")
        if not quality["fields"]["id"]["exact_match"]:
            raise RuntimeError(f"IDs are not exact for {key}")
        checks = quality["error_bound_consistency"]
        vector = quality["xnyzip_l2_error_bound_consistency"]["positions"]
        if set(checks) != {"id", "x", "y", "z", "vx", "vy", "vz"} or not all(
            item["satisfied"] for item in checks.values()
        ) or not vector["satisfied"]:
            raise RuntimeError(f"Error-bound check failed for {key}")
        rows.append({
            "input": key, "particles": new["count"],
            "baseline": measure(old_root, old), "optimized": measure(new_root, new),
            "ids_exact": True, "all_bounds_satisfied": True,
            "position_l2_error": vector["observed_max_l2_error"],
            "position_l2_bound": vector["requested_l2_bound"],
            "velocity_errors": {field: checks[field]["observed_max_absolute_error"] for field in ("vx", "vy", "vz")},
            "velocity_bounds": {field: checks[field]["requested_abs_bound"] for field in ("vx", "vy", "vz")},
            "timing": quality["timing"],
        })
    totals = {mode: {field: sum(row[mode][field] for row in rows)
                     for field in rows[0][mode]} for mode in ("baseline", "optimized")}
    for values in totals.values():
        values["compression_ratio"] = values["original_bytes"] / values["compressed_bytes"]
    return {
        "baseline_directory": str(baseline_root.resolve()),
        "optimized_directory": str(optimized_root.resolve()),
        "baseline_provenance": "Existing baseline artifacts; optimized quality freshly verified",
        "particle_count": sum(row["particles"] for row in rows),
        "package_count": len(rows), "totals": totals,
        "compressed_byte_reduction_percent": 100 * (1 - totals["optimized"]["compressed_bytes"] / totals["baseline"]["compressed_bytes"]),
        "packages": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("optimized", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch", action="store_true", help="Compare child packages even if a root merged manifest exists")
    args = parser.parse_args()
    report = compare(args.baseline, args.optimized, args.batch)
    if args.output:
        write_json(args.output, report, force=False)
    print(json.dumps({key: value for key, value in report.items() if key != "packages"}, indent=2))


if __name__ == "__main__":
    main()
