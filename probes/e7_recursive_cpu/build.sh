#!/bin/bash
# E7 probe build: recursive LU / Cholesky on Accelerate CBLAS (new LAPACK ABI).
set -euo pipefail
cd "$(dirname "$0")"
xcrun clang++ -std=c++17 -O3 -fno-fast-math -fPIC -dynamiclib \
  -DACCELERATE_NEW_LAPACK \
  e7.cpp -framework Accelerate -o libe7.dylib
echo "built: $(pwd)/libe7.dylib"
