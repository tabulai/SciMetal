#!/bin/bash
# E1 probe build: offline-compiled metallib + standalone runtime dylib.
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(cd ../.. && pwd)"
MLX_INC="$ROOT/.venv/lib/python3.14/site-packages/mlx/include"

# Metal kernels, compiled offline, no fast-math.
xcrun -sdk macosx metal -std=metal3.1 -O2 -fno-fast-math -Wno-c++17-extensions \
  -I "$MLX_INC" -c steel_gemm.metal -o steel_gemm.air
xcrun -sdk macosx metallib steel_gemm.air -o steel_gemm.metallib

# Runtime (Objective-C++, ARC) + Accelerate references. No libmlx.
xcrun clang++ -std=c++17 -O2 -fno-fast-math -fobjc-arc -fPIC -dynamiclib \
  -DACCELERATE_NEW_LAPACK -I "$MLX_INC" \
  gemm_runtime.mm cpu_ref.cpp \
  -framework Metal -framework Foundation -framework Accelerate \
  -o libe1gemm.dylib
echo "built: $(pwd)/steel_gemm.metallib $(pwd)/libe1gemm.dylib"
