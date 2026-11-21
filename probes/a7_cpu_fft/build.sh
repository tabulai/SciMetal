#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
clang++ -O3 -std=c++20 -shared -fPIC -Wall -Wextra -framework Accelerate \
    -o "$HERE/liba7.dylib" "$HERE/a7.cpp"
