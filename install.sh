#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LCP_SOURCE="${PROJECT_ROOT}/tools/LCP"
LCP_BUILD="${LCP_SOURCE}/build"

cmake \
  -S "${LCP_SOURCE}" \
  -B "${LCP_BUILD}" \
  -DCMAKE_INSTALL_PREFIX:PATH="${LCP_BUILD}"
cmake --build "${LCP_BUILD}" --target lcp
cmake --install "${LCP_BUILD}"

python -m pip install -r "${PROJECT_ROOT}/requirements.txt"
