#!/bin/bash
# Re-vendors the generic runtime subset of metal-arrow into SciMetal, renaming
# the `ma` prefix to `sm`. Files are copied verbatim apart from that rename;
# local changes on top are listed in src/runtime/PROVENANCE.md and must be
# re-applied (or kept, by vendoring into a scratch dir and diffing) on re-sync.
#
#   tools/vendor_runtime.sh [metal-arrow checkout] [destination root]
set -euo pipefail
SRC=${1:-/Users/tunguz/Programming/metal-arrow}
DST=${2:-$(cd "$(dirname "$0")/.." && pwd)}

rename() { # <source file> <destination file>
  mkdir -p "$(dirname "$2")"
  perl -pe 's/\bma_/sm_/g; s/\bMA_/SM_/g; s/\bma::/sm::/g; s/\bnamespace ma\b/namespace sm/g;
            s#(["<])ma/#$1sm/#g; s/\bma\.h\b/sm.h/g; s/\bMA_H\b/SM_H/g;
            s/\bmacore\b/smcore/g' "$1" > "$2"
}

# Public descriptor contract and the storage/context part of the runtime ABI.
rename "$SRC/include/ma.h" "$DST/include/sm.h"
rename "$SRC/include/ma_runtime.h" "$DST/include/sm_runtime.h.full"
# Keep the context/buffer API only: everything up to the Arrow/primitive section.
awk '/^\/\* Execution for the primitives and Arrow operations/ {exit} {print}' \
  "$DST/include/sm_runtime.h.full" > "$DST/include/sm_runtime.h"
printf '#ifdef __cplusplus\n}\n#endif\n#endif\n' >> "$DST/include/sm_runtime.h"
rm "$DST/include/sm_runtime.h.full"

for f in backend.hpp storage.hpp parallel.hpp; do
  rename "$SRC/include/ma/detail/$f" "$DST/include/sm/detail/$f"
done
for f in storage.cpp metal_storage.cpp metal_device.hpp metal_cache_policy.hpp pipeline_cache.hpp \
         metal_compute.hpp metal_compute.cpp buffer_contract.cpp; do
  rename "$SRC/src/runtime/$f" "$DST/src/runtime/$f"
done
rename "$SRC/cmake/EmbedMetallib.cmake" "$DST/cmake/EmbedMetallib.cmake"
for f in contract_test.cpp pipeline_cache_test.cpp storage_test.cpp metal_storage_test.cpp \
         buffer_cache_test.cpp metal_cache_policy_test.cpp; do
  rename "$SRC/tests/native/$f" "$DST/tests/native/$f"
done

mkdir -p "$DST/third_party"
rm -rf "$DST/third_party/metal-cpp"
cp -R "$SRC/third_party/metal-cpp" "$DST/third_party/metal-cpp"

echo "vendored from $SRC at $(git -C "$SRC" rev-parse HEAD)"
