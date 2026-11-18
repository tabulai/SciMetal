#!/bin/bash
# metal-sci probe E5 -- build the standalone FFT runtime.
#
#   vendor/mlx/backend/metal/kernels/{fft.h,defines.h,fft/radix.h,
#       fft/readwrite.h,steel/defines.h}   byte-identical copies from the
#       mlx 0.31.1 wheel (.venv/.../mlx/include/mlx/backend/metal/kernels)
#   vendor/mlx/backend/metal/kernels/fft.metal   byte-identical copy from
#       third_party/mlx-ref-0.31.1 (the kernel instantiations)
#   vendor/LICENSE.mlx   MLX's MIT licence (Copyright (c) 2023 Apple Inc.)
#
# Metal flags mirror MLX's own kernel build (-fno-fast-math, metal3.1).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build

MFLAGS=(-std=metal3.1 -O2 -fno-fast-math -Wno-c++17-extensions -Wno-c++20-extensions -I vendor)

xcrun -sdk macosx metal "${MFLAGS[@]}" -c vendor/mlx/backend/metal/kernels/fft.metal -o build/mlx_fft.air
xcrun -sdk macosx metallib build/mlx_fft.air -o build/mlx_fft.metallib

xcrun -sdk macosx metal "${MFLAGS[@]}" -c e5_ext.metal -o build/e5_ext.air
xcrun -sdk macosx metallib build/e5_ext.air -o build/e5_ext.metallib

xcrun -sdk macosx clang++ -std=c++17 -O2 -fobjc-arc -fvisibility=hidden \
  -Wall -Wno-unused-function -dynamiclib e5fft.mm \
  -framework Metal -framework Foundation -framework Accelerate \
  -o build/libe5fft.dylib

echo "built: build/mlx_fft.metallib build/e5_ext.metallib build/libe5fft.dylib"
