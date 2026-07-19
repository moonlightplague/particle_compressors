#!/usr/bin/env bash
set -euo pipefail

cd tools/LCP
cmake -B build -DCMAKE_INSTALL_PREFIX:PATH=build
cmake --build build --target lcp
cmake --install build
cd ../..

python -m pip install -r requirements.txt

cd tools/XnYZip
conan install . -s build_type=Release -b missing
cmake -S . -B build \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -D CMAKE_TOOLCHAIN_FILE=conan/conan_toolchain.cmake
cmake --build build --parallel
cd ../..