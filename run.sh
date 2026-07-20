#!/usr/bin/env bash
set -euo pipefail

INPUT_H5="data/dat_2.1.h5"
WORK_DIR="particle_pipeline_runs/$(basename -- "${INPUT_H5}")"

python main.py roundtrip "${INPUT_H5}" \
  --config "config.yaml" \
  --work-dir "${WORK_DIR}" \
  --rel-eb 1e-3 \
  --force --clean-raw
