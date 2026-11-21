#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [[ -d "$ROOT/.venv/bin" ]]; then
    export PATH="$ROOT/.venv/bin:$PATH"
fi

if [[ -z "${PYTHON:-}" ]]; then
    if [[ -x "$ROOT/.venv/bin/python" ]]; then
        PYTHON="$ROOT/.venv/bin/python"
    else
        PYTHON="$(command -v python3)"
    fi
fi

WITH_WHEEL=0
for arg in "$@"; do
    case "$arg" in
        --wheel) WITH_WHEEL=1 ;;
        -h|--help)
            echo "Usage: $0 [--wheel]"
            echo "  Runs metal and cpu-only CTest suites, Python tests against both builds,"
            echo "  and (with --wheel) builds and tests a wheel in a fresh temporary venv."
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg" >&2
            exit 2
            ;;
    esac
done

cd "$ROOT"

echo "==> 1/4: Metal build and CTest"
cmake --preset metal
cmake --build --preset metal
ctest --preset metal --output-on-failure

echo "==> 2/4: CPU-only build and CTest"
cmake --preset cpu-only
cmake --build --preset cpu-only
ctest --preset cpu-only --output-on-failure

echo "==> 3/4: Python suite (Metal build)"
env -u METAL_SCI_LIBRARY "$PYTHON" -m pytest tests/python -q -p no:warnings

echo "==> 4/4: Python suite (CPU-only build)"
METAL_SCI_LIBRARY="$ROOT/.build/cpu-only/libmetalsci.dylib" "$PYTHON" -m pytest tests/python -q -p no:warnings

if [[ "$WITH_WHEEL" -eq 1 ]]; then
    echo "==> 5/5: Clean-venv wheel build and verification"
    TMP_WHEEL_ROOT="$(mktemp -d)"
    trap 'rm -rf "$TMP_WHEEL_ROOT"' EXIT
    MACOSX_DEPLOYMENT_TARGET=15.0 "$PYTHON" -m pip wheel . -w "$TMP_WHEEL_ROOT/dist" --no-deps --no-build-isolation
    WHEEL_PATH="$(ls "$TMP_WHEEL_ROOT"/dist/metal_sci-*-macosx_15_0_arm64.whl)"
    "$PYTHON" -m venv "$TMP_WHEEL_ROOT/venv"
    WHEEL_PY="$TMP_WHEEL_ROOT/venv/bin/python"
    "$WHEEL_PY" -m pip install -q "$WHEEL_PATH" numpy scipy pytest hypothesis
    (
        cd /tmp
        unset METAL_SCI_LIBRARY
        "$WHEEL_PY" -c "
import pathlib, metal_sci
from metal_sci import _native
root = pathlib.Path('$TMP_WHEEL_ROOT/venv').resolve()
pkg = pathlib.Path(metal_sci.__file__).resolve()
lib = pathlib.Path(_native._library_path()).resolve()
assert root in pkg.parents, f'{pkg} not inside {root}'
assert root in lib.parents and lib.name == 'libmetalsci.dylib', f'unexpected lib: {lib}'
print('verified installed package:', pkg)
print('verified bundled library:', lib)
"
        METAL_SCI_TEST_INSTALLED=1 "$WHEEL_PY" -m pytest "$ROOT/tests/python" -q -p no:warnings
    )
fi

echo "==> All checks passed."
