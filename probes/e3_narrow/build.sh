#!/bin/bash
# Build the E3 probe: offline-compiled Metal kernels + a small ObjC++ dylib.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build
xcrun -sdk macosx metal -std=metal3.1 -O2 -fno-fast-math -c kernels.metal -o build/kernels.air
xcrun -sdk macosx metallib build/kernels.air -o build/kernels.metallib
xcrun -sdk macosx clang++ -std=c++20 -O3 -fobjc-arc -fno-fast-math -Wall -Wextra \
    -dynamiclib narrow.mm -o build/libe3.dylib \
    -framework Metal -framework Foundation -framework Accelerate
echo "built build/libe3.dylib build/kernels.metallib"
