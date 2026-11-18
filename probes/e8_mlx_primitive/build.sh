#!/bin/sh
# E8 probe build: (1) kernel -> metallib (offline, our own toolchain flags),
# (2) standalone ObjC++ runner (no libmlx), (3) MLX extension (CMake + nanobind
# against the installed mlx wheel).
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
VENV="$HERE/../../.venv"
PY="$VENV/bin/python"
B="$HERE/build"
mkdir -p "$B"

echo "[e8] kernel -> metallib"
xcrun -sdk macosx metal -std=metal3.1 -O2 -fno-fast-math -Wall -Wextra \
  -c "$HERE/kernels/e8_correlate1d.metal" -o "$B/e8_correlate1d.air"
xcrun -sdk macosx metallib "$B/e8_correlate1d.air" -o "$B/e8_kernels.metallib"

echo "[e8] standalone runner"
xcrun -sdk macosx clang++ -std=c++17 -O2 -fobjc-arc -fvisibility=hidden \
  -dynamiclib "$HERE/standalone/e8_runner.mm" \
  -framework Metal -framework Foundation -o "$B/libe8_runner.dylib"

echo "[e8] MLX extension"
PATH="$VENV/bin:$PATH" cmake -G Ninja -S "$HERE/mlx_ext" -B "$B/ext" \
  -DPython_EXECUTABLE="$PY" -DCMAKE_BUILD_TYPE=Release >"$B/cmake_configure.log" 2>&1 \
  || { cat "$B/cmake_configure.log"; exit 1; }
PATH="$VENV/bin:$PATH" cmake --build "$B/ext" >"$B/cmake_build.log" 2>&1 \
  || { cat "$B/cmake_build.log"; exit 1; }
echo "[e8] build ok"
