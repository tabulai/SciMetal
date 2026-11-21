# Runtime provenance

The generic storage/compute runtime is vendored from the author's `metal-arrow`
project by `tools/vendor_runtime.sh`, which copies the files below and renames the
`ma` prefix to `sm` (`ma_` → `sm_`, `MA_` → `SM_`, `ma::` → `sm::`, `ma/` → `sm/`).

- Source: `/Users/tunguz/Programming/metal-arrow` at commit `f578b4e5cb2809b93aaedfd342f388734dc242f9`
- Vendored on 2026-11-18

| SciMetal file | metal-arrow file |
|---|---|
| `include/sm.h` | `include/ma.h` |
| `include/sm_runtime.h` | `include/ma_runtime.h`, context/buffer API only (cut before the Arrow/primitive section) |
| `include/sm/detail/{backend,storage,parallel}.hpp` | `include/ma/detail/…` |
| `src/runtime/{storage,metal_storage,metal_compute,buffer_contract}.cpp` | `src/runtime/…` |
| `src/runtime/{metal_device,metal_cache_policy,pipeline_cache,metal_compute}.hpp` | `src/runtime/…` |
| `cmake/EmbedMetallib.cmake` | `cmake/EmbedMetallib.cmake` |
| `tests/native/{contract,pipeline_cache,storage,metal_storage,buffer_cache,metal_cache_policy}_test.cpp` | `tests/native/…` |
| `third_party/metal-cpp/` | `third_party/metal-cpp/` (Apple, see its LICENSE.txt and PINNED_VERSION.md) |

## Local changes on top of the vendored files

Re-running the script overwrites these; re-apply them (or vendor into a scratch
directory and diff) when re-syncing.

- `src/runtime/metal_compute.{hpp,cpp}`
  - `Grid`: dispatches take an N-D threadgroup grid with per-axis threadgroup
    sizes; the 1-D `threadgroups` overloads remain as `Grid::linear` (256 threads).
  - The pipeline-wide requirement of 32-lane SIMD groups in 256-thread
    threadgroups is replaced by a per-dispatch threadgroup-capacity check.
  - `dispatch_indexed`, `Specialization`, `warm_pipeline`: arguments at explicit
    indices and function-constant pipeline specializations, for the vendored
    MLX kernels.
  - The `static_assert` over metal-arrow's kernel parameter structs is removed;
    SciMetal's own layouts are asserted in `src/kernels/sm_kernels.h`.
- `include/sm/detail/parallel.hpp`
  - `hardware_lanes()` counts performance cores (`hw.perflevel0.physicalcpu`)
    instead of all cores.
- `tests/native/buffer_cache_test.cpp`
  - Dropped the two cases that call metal-arrow primitives (CPU scan/sort in the
    poison test; `metal_export_and_import_with_completion_event`, which needs an
    asynchronous GPU producer).
- Not vendored: `src/runtime/metal_disabled.cpp` is rewritten (backend stubs only),
  `src/kernels/sm_kernels.h` is new.
