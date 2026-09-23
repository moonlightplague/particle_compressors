#!/usr/bin/env bash
set -euo pipefail

INPUT_H5="data/EXASKY-HACC-data-medium-size"
WORK_DIR="particle_pipeline_runs/$(basename -- "${INPUT_H5}")"

python main.py roundtrip "${INPUT_H5}" \
  --config "config.yaml" \
  --xnyzip-tie-sort \
  --work-dir "${WORK_DIR}" \
  --rel-eb "1e-3" \
  --force --clean-raw --metrics
