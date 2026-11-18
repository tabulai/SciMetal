#!/bin/bash
# Runs each Phase 0 probe alone, one after another, and stores its RESULT JSON
# in probes/results/<probe>.json (full console output in <probe>.log).
#   probes/run_all.sh [probe dir names...]   (default: all eight)
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY="$ROOT/.venv/bin/python"
OUT="$ROOT/probes/results"
mkdir -p "$OUT"
PROBES=("$@")
[ ${#PROBES[@]} -eq 0 ] && PROBES=(e1_gemm e7_recursive_cpu e6_hybrid_lu e2_nocopy e3_narrow e4_stencil e5_fft e8_mlx_primitive)
for probe in "${PROBES[@]}"; do
  dir="$ROOT/probes/$probe"
  echo "== $probe (load: $(sysctl -n vm.loadavg))"
  start=$(date +%s)
  ( cd "$dir" && { [ ! -x ./build.sh ] || ./build.sh; } && "$PY" run.py ) > "$OUT/$probe.log" 2>&1
  status=$?
  grep '^RESULT ' "$OUT/$probe.log" | tail -1 | sed 's/^RESULT //' > "$OUT/$probe.json"
  echo "   exit=$status, $(( $(date +%s) - start )) s, result bytes: $(wc -c < "$OUT/$probe.json")"
done
