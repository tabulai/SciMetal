#!/bin/bash
# Re-vendors the MLX Metal FFT kernels into third_party/mlx-kernels, byte for
# byte, keeping MLX's directory layout so the kernels' own #include lines
# resolve with `-I third_party/mlx-kernels`.
#
#   tools/vendor_mlx_kernels.sh [--check] [mlx wheel include dir] [mlx source checkout]
#
# The headers come from the installed MLX wheel (they ship in its include
# directory); fft.metal, which only instantiates the kernels, is not in the
# wheel and comes from a source checkout of the same release. --check copies
# nothing and fails if the vendored files differ from those sources.
#
# After a re-sync to a new MLX release: update third_party/mlx-kernels/PROVENANCE.md,
# re-read mlx/backend/metal/fft.cpp against src/fft/plan.cpp (the plan logic is
# a port of it), and run the fft tests.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
CHECK=0
if [[ ${1:-} == --check ]]; then
  CHECK=1
  shift
fi
WHEEL=${1:-$(ls -d "$ROOT"/.venv/lib/python3.*/site-packages/mlx/include 2>/dev/null | head -1)}
SOURCE=${2:-$ROOT/third_party/mlx-ref-0.31.1}
DST=$ROOT/third_party/mlx-kernels
KERNELS=mlx/backend/metal/kernels

if [[ -z $WHEEL || ! -d $WHEEL/$KERNELS ]]; then
  echo "MLX wheel include directory not found; pass it as the first argument" >&2
  exit 1
fi
if [[ ! -f $SOURCE/$KERNELS/fft.metal ]]; then
  echo "MLX source checkout with $KERNELS/fft.metal not found; pass it as the second argument" >&2
  exit 1
fi

status=0
vendor() { # <source file> <path under third_party/mlx-kernels>
  if [[ $CHECK == 1 ]]; then
    if cmp -s "$1" "$DST/$2"; then
      echo "identical  $2"
    else
      echo "DIFFERS    $2 (from $1)" >&2
      status=1
    fi
  else
    mkdir -p "$(dirname "$DST/$2")"
    cp "$1" "$DST/$2"
    cmp "$1" "$DST/$2"
    echo "vendored   $2"
  fi
}

for header in fft.h defines.h fft/radix.h fft/readwrite.h steel/defines.h; do
  vendor "$WHEEL/$KERNELS/$header" "$KERNELS/$header"
done
vendor "$SOURCE/$KERNELS/fft.metal" "$KERNELS/fft.metal"
vendor "$SOURCE/LICENSE" "LICENSE.mlx"

version=$(sed -n 's/^Version: //p' "$(dirname "$WHEEL")"/../mlx-*.dist-info/METADATA 2>/dev/null | head -1)
echo "MLX wheel version: ${version:-unknown} ($WHEEL)"
shasum -a 256 "$DST/$KERNELS"/fft.metal "$DST/$KERNELS"/fft.h "$DST/$KERNELS"/defines.h \
  "$DST/$KERNELS"/fft/radix.h "$DST/$KERNELS"/fft/readwrite.h "$DST/$KERNELS"/steel/defines.h "$DST/LICENSE.mlx"
exit $status
