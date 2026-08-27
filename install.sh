#!/usr/bin/env bash
set -euo pipefail

cmake -S tools/SPERR -B tools/SPERR/build \
  -DBUILD_SHARED_LIBS=ON \
  -DBUILD_UNIT_TESTS=OFF \
  -DBUILD_CLI_UTILITIES=OFF \
  -DUSE_OMP=OFF \
  -DCMAKE_BUILD_TYPE=Release
cmake --build tools/SPERR/build --parallel

mkdir -p tools/SPERR/.pybuild
python -m pip install -r requirements.txt
python -m pip install tools/QoZ
