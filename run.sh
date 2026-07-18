#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
INPUT_H5="${1:-${PROJECT_ROOT}/data/new_data/dat_2.13.h5}"
WORK_DIR="${2:-${PROJECT_ROOT}/particle_pipeline_runs/$(basename -- "${INPUT_H5}")}"

python "${PROJECT_ROOT}/main.py" roundtrip "${INPUT_H5}" \
  --config "${PROJECT_ROOT}/config.yaml" \
  --work-dir "${WORK_DIR}" \
  --rel-eb 1e-5 \
  --force --clean-raw
