# MLX Metal kernels: provenance

The FFT kernels of Apple's [MLX](https://github.com/ml-explore/mlx), vendored
byte for byte with MLX's directory layout so that their own `#include` lines
resolve with `-I third_party/mlx-kernels`. They are compiled into SciMetal's
embedded metallib by `src/fft/fft_kernels.metal`; the native core does not
link libmlx.

- MLX release: **0.31.1** (the `mlx` / `mlx-metal` 0.31.1 wheels)
- Licence: MIT, Copyright © 2023 Apple Inc. — `LICENSE.mlx` (MLX's `LICENSE`, byte identical)
- Vendored on 2026-11-18 by `tools/vendor_mlx_kernels.sh`

| File (under `mlx/backend/metal/kernels/`) | Source | SHA-256 |
|---|---|---|
| `fft.h` | wheel, `mlx/include/mlx/backend/metal/kernels/fft.h` | `b56daa26efce084b42c0839c980d36fd6abd20abd771da5ad5654693d1a58370` |
| `fft/radix.h` | wheel, same directory | `3f3e8a9f42f464c6f39893fe972c2d7bfacaefc4b47e5ef8ea0205b662da869e` |
| `fft/readwrite.h` | wheel, same directory | `e26d49d0c9a86fefef643fd3a7c24c6f13b5a92cd0aac2352c8b03a07b7c1f35` |
| `defines.h` | wheel, same directory | `a2930dbd644c69c4b66a511a094034217f3c03f48e29a1613f601532150f9163` |
| `steel/defines.h` | wheel, same directory | `b03cea6a7d5cfe814e6838d8536e2aacdcafd2744cb408e954fda336ce21759a` |
| `fft.metal` | MLX 0.31.1 source tree (`third_party/mlx-ref-0.31.1`); the wheel ships only the compiled metallib | `3a1fbb38ed64f50a49a20d0c5adb1748d9d06ea20e5931e99aa26be543cb7825` |
| `../../../../LICENSE.mlx` | MLX 0.31.1 source tree, `LICENSE` | `ccfab7ccb2ea306f71531c8ca77bb55507606cd90768b1e32b8b52ab5b48cf01` |

`tools/vendor_mlx_kernels.sh --check` compares every file with its source
(`cmp`) and prints these hashes; it exits non-zero on any difference.

## What is used, and what is ours

- Used unmodified: the radix codelets and `perform_fft`, the Stockham, Rader
  and fused Bluestein kernels (`fft`, `rader_fft`, `bluestein_fft`), and the
  four-step kernels (`four_step_fft`) as the first and last pass of
  power-of-two plans.
- Not used: MLX's own large-size plan. Its four-step stops at 2^20; MLX 0.31.1
  raises at 2^21–2^22 and returns wrong results at 2^23–2^24
  (`probes/README.md`, E5).
- Ours, in `src/fft/` (not part of MLX): the strided pass `sm_fft_strided`
  (built on MLX's `perform_fft`), which the planner chains for long and
  non-power-of-two lengths and for transforms along other axes; the planner
  itself; and the engine. `src/fft/plan.cpp` contains a port of the planning
  logic of MLX's `mlx/backend/metal/fft.cpp` (radix decomposition,
  thread-layout heuristics, Rader and Bluestein constants) and carries the MIT
  notice for it.

The build compiles these kernels with SciMetal's flags (`-std=metal3.1 -O2
-fno-fast-math -ffp-contract=off`), where MLX's own build leaves contraction
on, so results can differ from MLX's in the last bit.

## Re-syncing to a newer MLX

1. `tools/vendor_mlx_kernels.sh [wheel include dir] [source checkout]`
2. Update the release, date and hashes above.
3. Diff the new `mlx/backend/metal/fft.cpp` against the port in
   `src/fft/plan.cpp`: kernel names, argument order, function-constant indices
   and the thread-layout heuristics must match the kernels.
4. Run the FFT tests, including the exhaustive length sweep
   (`SCIMETAL_FFT_EXHAUSTIVE=1 pytest tests/python/test_fft_sizes.py`).
