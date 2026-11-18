#!/bin/bash
# E6 probe build: hybrid float32 LU (CPU panels + GPU steel GEMM) + refinement.
# Reuses the E1 runtime source and metal kernel file and the E7 recursive
# panel (header-only); produces its own metallib + dylib in this directory.
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(cd ../.. && pwd)"
MLX_INC="$ROOT/.venv/lib/python3.14/site-packages/mlx/include"

xcrun -sdk macosx metal -std=metal3.1 -O2 -fno-fast-math -Wno-c++17-extensions \
  -I "$MLX_INC" -c ../e1_gemm/steel_gemm.metal -o steel_gemm.air
xcrun -sdk macosx metallib steel_gemm.air -o steel_gemm.metallib

xcrun clang++ -std=c++17 -O3 -fno-fast-math -fobjc-arc -fPIC -dynamiclib \
  -DACCELERATE_NEW_LAPACK -I "$MLX_INC" -I ../e1_gemm \
  hybrid_lu.cpp ../e1_gemm/gemm_runtime.mm \
  -framework Metal -framework Foundation -framework Accelerate \
  -o libe6.dylib
echo "built: $(pwd)/steel_gemm.metallib $(pwd)/libe6.dylib"
