# SciMetal — execution plan

Hand-off document for whoever (human or AI) continues this project. It is meant to be
executed top to bottom without access to the conversation that produced it.

Written 2026-11-18; updated 2026-11-19 after Stage A0–A1. Everything under "State of the
repository" was true on that date; verify it with the commands given before building on it.

---

## 0. How to use this document

1. Read sections 1–4 once. They are the decisions and the evidence; do not reopen them
   unless a gate fails.
2. Run the checks in section 5.4. If any fails, fix that before starting new work.
3. Follow the engineering rules in section 6 for every function you add. They exist
   because each one was learned by measurement or by a bug in this project.
4. Execute the work packages in section 7 in the order given in section 8. Each package
   has a gate. A failed gate changes the plan (the package says how); it is never hidden.
5. Section 9 lists known defects, unverified claims and questions that belong to the user.
6. Section 10 is reference material: commands, internal APIs, kernel facts.

Words used throughout:

- **Native route**: our implementation of a SciPy function (C++ core behind a C ABI,
  reached from Python through ctypes).
- **Decline / fall through**: the native route refuses a call and SciPy's own function
  runs with the caller's arguments untouched.
- **Engine**: where the arithmetic runs. *CPU engine* = threaded C++/NEON/Accelerate on
  the performance cores; *GPU engine* = Metal compute kernels, always single precision.
- **Oracle**: SciPy's own test suite run with our functions patched into SciPy's
  namespaces.
- **Gate**: a measurable pass/fail condition that ends a work package.

---

## 1. Mission, decisions, hard constraints

### 1.1 Mission

Build `SciMetal`: a SciPy-compatible library that extracts the largest possible gains
from Apple Silicon, the GPU in particular. The fundamental algorithms live in native code
(C++20 behind a C ABI, Metal Shading Language kernels); the public surface is Python and
mirrors SciPy's namespaces.

### 1.2 Decisions the user made (binding)

| Topic | Decision |
|---|---|
| Form | **Standalone package.** No SciPy fork, no patches to SciPy, no monkeypatching of SciPy internals. The only hook into SciPy is its public `scipy.fft` backend protocol. |
| float64 policy | **Speed first.** By default float64 inputs are computed in float32 on the GPU and returned as float64 arrays with about 7 significant digits. A strict mode gives SciPy's exact results. |
| Priorities | **Signal, FFT and image** first, then **linear algebra and sparse**. |
| Audience | **The user's own machines first** (M1 Ultra, macOS 26, one pinned MLX version). Generalise later. |
| Name and place | Project name **SciMetal**: install and import name `scimetal`, C prefix `sm_`, library `libscimetal.dylib`, environment variables `SCIMETAL_*`. Renamed on 2026-11-21 from `metal-sci` (import `metal_sci`, prefix `ms_`): one word for install and import, and METAL-SCI is already the name of a public Apple Silicon Metal benchmark. The folder is `/Users/tunguz/Programming/SciMetal` (renamed from `metal-sci` the same day). |

### 1.3 Hard constraints

- **Never modify** `/Users/tunguz/Programming/scipy` (SciPy 2.0.0.dev0 checkout, reference
  only) or `/Users/tunguz/Programming/metal-arrow` (the user's own project, source of the
  vendored runtime). Read them freely.
- **Version control:** a local git repository on `main` with no remote. Commit only when
  the user asks; never add a remote or push unless asked.
- **Privacy:** do not send the user's name, email address or project details to any
  external service.
- **Never slower than SciPy.** A call the native route cannot win declines to SciPy.
- **Never silently wrong.** Speed-first precision is allowed to lose digits; it is not
  allowed to return garbage (overflow to infinity, a wrong plan, a diverged solve)
  without either falling back or raising.
- **Report honestly.** A number measured on a loaded machine is labelled as such. A gate
  that fails is reported as failed.

---

## 2. Environment

| Item | Value |
|---|---|
| Machine | Mac Studio, M1 Ultra: 16 performance + 4 efficiency cores, 64-core GPU (`applegpu_g13d`), 128 GB unified memory, no SME |
| OS | macOS 26.5.1 |
| Toolchain | Xcode 16.3, SDK 15.4, Apple clang 17, cmake 4.2.1, ninja (from the venv) |
| Metal | Offline `xcrun metal` / `metallib`, MSL up to 3.1. **No Metal 4 headers** in this SDK. |
| Python | `.venv` in the repo: Python 3.14.2, numpy 2.5.5, scipy 1.18.1 (Accelerate build), mlx 0.31.1, nanobind 3.1.0, scikit-build-core, pytest, hypothesis |
| MLX pin | 0.31.1. It was built with nanobind 2.10.2; a C++ bridge must use that version (nanobind 3.x cannot share `mx.array`). |

`PATH` must contain `.venv/bin` for cmake to find ninja:
`export PATH=/Users/tunguz/Programming/SciMetal/.venv/bin:$PATH`.

The machine is shared with other work; load averages of 3–11 were common while this was
written. See the benchmark protocol (6.9).

---

## 3. Strategy: Metal-first native core, MLX beside it

The user asked whether to build on MLX or on Metal. The answer, backed by section 4:

- **The core is our own native library**: C++20 behind a C ABI, MSL kernels compiled
  offline into one metallib that is embedded in the library, Accelerate plus NEON and
  threads for the CPU. It never includes an MLX header and does not link libmlx.
- **MLX has three supporting roles**:
  1. donor of best-in-class kernel *sources* (its GEMM and FFT Metal kernels, MIT), which
     we vendor into our metallib;
  2. the GPU-resident array type, through an optional bridge (MLX array in, MLX array out);
  3. autodiff and vmap for later features (ensembles of ODE/optimisation problems).
- **Why not MLX as the base:** no float64 on the GPU, no complex128 anywhere, every
  factorisation (qr/svd/inv/cholesky/eigh/lu/solve) runs on the CPU, no data-dependent
  output shapes, no sparse, 200–260 µs per GPU evaluation, `mx.array(numpy)` copies at
  about 9 GB/s, no stable C++ ABI, and its GPU FFT is wrong above 2^20 points in 0.31.1.
- **Why not Metal alone:** MLX's GEMM and FFT kernels are far ahead of a first attempt,
  and `mx.array` is the natural device array; we should not write an array library.

### 3.1 Layers

| Layer | What it is | Where |
|---|---|---|
| Runtime | Context, shared-storage Metal buffers, buffer cache, command-buffer `Batch`, embedded metallib, performance-core thread lanes | `src/runtime/`, `include/sm_runtime.h`, `include/sm/detail/` (vendored from metal-arrow, see `src/runtime/PROVENANCE.md`) |
| Core | Array descriptors, pooled outputs, threaded precision conversion, router | `include/sm_array.h`, `src/core/` |
| Families | One directory per SciPy area: C ABI, CPU engine, GPU engine, kernels | `include/sm_<family>.h`, `src/<family>/` |
| Python | ctypes binding, precision/device policy, SciPy-mirroring namespaces with fall-through | `python/scimetal/` |
| Oracle and tests | Native tests (ctest), Python tests (pytest), SciPy's suite against our routes | `tests/` |

The Python binding is **ctypes over the C ABI**, not a compiled extension module. That is
a deliberate deviation from the first plan (an abi3 extension): one wheel serves every
Python 3, and there is nothing to rebuild per interpreter. The cost is about 7 µs of
hand-off per call, which only matters for inputs small enough to decline anyway.

### 3.2 Precision policy

`scimetal.set_precision(...)` or `with scimetal.precision(...)`:

- **`fast`** (default): double-precision inputs may run on the GPU engine in single
  precision; results come back in the dtype SciPy would return, with about 6–7 digits.
  Below the GPU crossover the CPU engine runs, in the input's own precision.
- **`refined`**: single-precision work plus a double-precision correction where one
  exists (solves, least squares, Krylov). Elsewhere it behaves like `fast`.
- **`strict`**: SciPy's result. Double precision never goes to the GPU. For the ndimage
  filters the CPU engine is bit-identical to SciPy 1.18.1.

`scimetal.set_device('auto' | 'cpu' | 'gpu')` forces an engine; `gpu` never falls back to
the CPU engine (it declines to SciPy instead). `scimetal.explain()` and
`scimetal.last_route()` say how the last call ran.

**Not building:** float-float ("double-double") emulation on the GPU. By the arithmetic
it lands at 0.3–1.0 TFLOPS-equivalent against a measured 0.8 TFLOPS for real `dgemm`.

---

## 4. Evidence: Phase 0 probes

Eight throwaway probes tested the design before production code was written. Each ran
alone on the M1 Ultra. Sources and raw results: `probes/`, `probes/results/`,
`probes/README.md`. Re-run with `probes/run_all.sh` (about 3.5 minutes).

### 4.1 Gates

| # | Probe | Gate | Result | Verdict |
|---|---|---|---|---|
| E1 | MLX steel GEMM in our metallib and runtime | ≥ 80% of MLX at n = 2048–8192; tiny call ≤ 0.3 ms | 102% / 99% / 93% of MLX (10.3 / 16.1 / 11.5 TFLOPS); tiny call 0.25 ms | Pass |
| E2 | No-copy wrap of NumPy pages | wrap ≤ 5% of memcpy | the wrap call is ≤ 1% of a memcpy, but the first GPU use of a fresh wrap costs 11–12 ms/GB | **Not met** |
| E3 | float64 ⇄ float32 conversion | bit-exact | CPU 16 threads 164–179 GB/s; GPU in-kernel 317–464 GB/s on warm buffers; bit-identical to `numpy.astype` | — |
| E4 | `gaussian_filter` 4096², float64 in and out | fast ≥ 20×, strict ≥ 8× | fast 133× / 140× (σ = 3 / 10); strict 49× / 41×, bit-identical | Pass |
| E5 | MLX FFT kernels in our runtime | ≥ 80% of MLX at 2^24 | 2.47 ms and correct; MLX takes 3.12 ms and returns a **wrong result** | Pass |
| E6 | Hybrid float32 LU + float64 refinement | n = 8192 ≤ 250 ms, backward error ≤ 1e-14 for κ ≤ 1e4, break-even n ≤ 2000 | 180–204 ms, backward error ≤ 3.4e-16, break-even n = 512 | Pass |
| E7 | Recursive factorisations on Accelerate GEMM (exact float64, CPU) | ≥ 4× at 8192, ≥ 1.5× at 2048 | Cholesky 5.1× / 1.55×; LU 1.98× / 1.05× | Cholesky pass, **LU fail** |
| E8 | Out-of-tree MLX primitive running our metallib | ≤ 30 µs marginal cost, no extra commit | −1.3 to +0.5 µs against a built-in op; no extra commit | Pass |

### 4.2 Headline numbers (probes, not the product)

| Workload | Stock | Probe | Speed-up |
|---|---|---|---|
| `gaussian_filter`, 4096², σ = 3, float64 | SciPy 385 ms | fast 2.9 ms; strict 7.9 ms | 133×; 49× |
| FFT, complex64, 2^24 points | `scipy.fft` 300 ms | 2.47 ms | 122× |
| FFT, complex64, 2^20 points | `scipy.fft` 10.0 ms | 0.39 ms | 25× |
| `solve`, n = 8192, float64, κ = 1e4 | `scipy.linalg.solve` 1467 ms | 204 ms, backward error 4e-17 | 7× |
| Matmul, float32, n = 4096 | `sgemm` 3.0 TFLOPS (`dgemm` 0.8) | 16.1 TFLOPS | 5.4× (20× against float64) |
| Cholesky, float64, n = 8192, exact, CPU | `dpotrf` 2522 ms | 497 ms | 5.1× |
| LU, float64, n = 8192, exact, CPU | `dgetrf` 1567 ms | 790 ms | 2.0× |

Caveat on the filter figure: 133× assumes result buffers are recycled through the pool;
if the caller keeps every result it is about 53×.

### 4.3 Design rules that came out of the probes

1. **Copy in, pool, hand buffers out. Do not wrap one-shot inputs.** A fresh no-copy wrap
   of host pages pays about 12 ms/GB at its first GPU use; a 16-thread copy or narrow into
   a pooled buffer costs 5–6 ms/GB. Results are pooled buffers that NumPy views without a
   copy.
2. **Never read float64 inside a multi-tap kernel.** Narrow once (CPU, threaded). Fuse
   only the final widen into the last kernel, where it is free.
3. **FFT uses our own plan** on MLX's kernels. MLX 0.31.1's large-size plan raises at
   2^21–2^22 and returns wrong answers at 2^23–2^24.
4. **Hybrid LU is bound by the CPU panel**, not the GPU: the GPU was busy 45 ms of a
   127 ms factorisation. The next lever is the panel (a tall-panel kernel; swaps and TRSM
   in the update's command buffer).
5. **Strict linear algebra gains are uneven.** Cholesky 5× (Accelerate's `dsyrk`
   collapses on rank-k updates; a GEMM-based update avoids it). LU only 2×. Use
   `getrf` + `getrs`; Accelerate's `gesv` drivers are 1.3–3× slower.
6. **Mixed precision pays even without the GPU**: float32 factor plus float64 refinement
   on the CPU beats `dgesv` from n = 512.
7. **MLX bridge is cheap, but pin nanobind** to the version MLX was built with.
8. **Fixed GPU cost is about 0.25 ms per synchronous command buffer** (0.035–0.055 ms per
   call when several are pipelined). One command buffer per public call, always.

---

## 5. State of the repository

### 5.1 Layout

```
SciMetal/
  PLAN.md                     this document
  CMakeLists.txt              targets sm_contract (static), smcore (shared, libscimetal.dylib), tests
  CMakePresets.json           presets: metal, metal-asan-ubsan, metal-tsan, cpu-only
  pyproject.toml              scikit-build-core, py3 wheel that carries libscimetal.dylib
  cmake/EmbedMetallib.cmake   metallib -> C array
  include/
    sm.h, sm_runtime.h        runtime ABI (contexts, buffers)            [vendored]
    sm_array.h                array descriptors (ABI v2), routing structs (reason[256]), sm_array_astype, sm_array_capabilities
    sm_ndimage.h              sm_ndimage_separable_filter[_to], sm_ndimage_dense_filter, sm_ndimage_composite_filter
    sm_fft.h                  sm_fft_transform, sm_fft_plan, sm_fft_auto_min_points
    sm_signal.h               sm_signal_convolve, sm_signal_fftconvolve, sm_signal_choose_conv_method
    sm/detail/                storage.hpp, backend.hpp, parallel.hpp     [vendored, private]
  src/
    runtime/                  storage, Metal storage, Batch (metal_compute), pipeline cache  [vendored + local changes]
    core/                     array.{hpp,cpp}, router.{hpp,cpp}, dense_correlate.{hpp,cpp,_gpu.cpp,_kernels.h,.metal}
    kernels/                  sm_kernels.h (host<->kernel structs), smoke.metal
    ndimage/                  filters.{hpp,cpp}, filters_cpu.cpp, filters_gpu.cpp, ndimage_kernels.h, ndimage_filters.metal
    fft/                      plan.{hpp,cpp}, fft.cpp, engine.hpp, gpu_engine.cpp, fft_kernels.{h,metal}
    signal/                   convolve.cpp, fft_conv.hpp, fft_conv_gpu.cpp, signal_kernels.{h,metal}
    <family>/sources.cmake    appends the family's sources to the build
    <family>/tests.cmake      registers the family's native tests
  python/scimetal/
    __init__.py               policy API, PrecisionWarning, explain(), accelerated(), namespace loading
    _config.py                precision / device / small-input policy
    _dispatch.py              @accelerated decorator, NotAccelerated (with context hook), PrecisionWarning, Route, counters
    _fallback.py              fall-through namespaces and the SciPy alias finder
    _native.py                ctypes structs, library(), context(), as_native(), to_numpy(), check()
    ndimage/, fft/, signal/   native routes
    linalg/, sparse/          fall-through only so far
  tests/
    native/                   ctest executables
    python/                   pytest (conftest.py puts python/ on sys.path and keeps small inputs native)
    oracle/scimetal_oracle.py   pytest plugin: SciPy's suite against our routes
  bench/                      ndimage_filters.py, fft.py, signal_convolve.py
  probes/                     Phase 0 probes and results
  tools/                      vendor_runtime.sh, vendor_mlx_kernels.sh
  third_party/
    metal-cpp/                Apple's C++ Metal headers
    mlx-kernels/              MLX FFT kernels, byte for byte, with PROVENANCE.md and LICENSE.mlx
    mlx-ref-0.31.1/           MLX reference sources (matmul.cpp, fft.cpp, GEMM and FFT kernels)
```

### 5.2 What exists

**Runtime (vendored from metal-arrow at commit `f578b4e5`, renamed `ma_` → `sm_`).**
Local changes are listed in `src/runtime/PROVENANCE.md`: N-D dispatch grids, indexed
bindings and function-constant specialisations (needed by MLX's kernels), a per-dispatch
threadgroup-capacity check, `max_threads_per_group`, performance-core lane count.
Re-vendor with `tools/vendor_runtime.sh`, then re-apply the listed changes.

**Core.** `sm_array` descriptors (C-contiguous real `float32`/`float64`/`bool`/`int8`–`int64`/`uint8`–`uint64`
and `complex64`/`complex128`, up to 8 dimensions), `sm_route_options` (execution, precision),
`sm_route_info` (ABI v3: engine, compute dtype, `reason[256]`), pooled result buffers, single-pass
NEON threaded `array::convert` (`ConvertReport` tracking `overflowed`, `inexact_int` for `|x| >= 2^24`
32/64-bit ints, `has_nonfinite`, `max_finite`, `max_magnitude` at `>= 150 GB/s`), a router that picks
the engine from the policy and a per-family work threshold, and the shared N-D stencil primitive
`sm::detail::dense_correlate` (CPU + GPU `encode_gpu`, supporting arbitrary `out_shape`, `before[8]`,
all real & complex float/int/bool dtypes, `AccumMode`, and zero-tap pruning).

**Python package.** Drop-in namespaces (`import scimetal.ndimage as ndimage`): a name
with a native route is ours, every other name is SciPy's own object. Sub-modules without
a file of ours alias SciPy's module object (never a second copy). Results are NumPy
arrays that view the native buffer and return it to the pool when dropped. Calling any
fast-mode route on double-precision inputs emits `scimetal.PrecisionWarning` once by
default (naming `set_precision('strict')` and `warnings.filterwarnings`), and
`explain()` / `last_route()` surface re-route reasons.

**`ndimage` filters** — separable (`correlate1d`, `convolve1d`, `gaussian_filter1d`,
`gaussian_filter` of any order, `uniform_filter1d`, `uniform_filter`, `prewitt`,
`sobel`), dense N-D (`correlate`, `convolve`), and composite (`laplace`,
`gaussian_laplace`, `gaussian_gradient_magnitude`). SciPy runs the callbacks for
`generic_laplace` and `generic_gradient_magnitude` on real arrays.
- Three C primitives: `sm_ndimage_separable_filter[_to]`, `sm_ndimage_dense_filter`, and
  `sm_ndimage_composite_filter` across all 11 real dtypes (`float32`, `float64`, `bool`,
  `int8`–`int64`, `uint8`–`uint64`), including cross-dtype `output=` among real dtypes.
- CPU engine: double accumulation, SciPy's operation order and integer/boolean conversion
  rules (`NI_LineBufferToArray` and `CASE_FILTER_OUT_SAFE`, raw `bool` bytes, wrapping
  composite combines), bit-identical to the SciPy 1.18.1 arm64 wheel across all 11 real
  dtypes. Innermost-axis uniform passes use interleaved 4/8-line NEON `float64x2_t`
  recurrences (fixing the 2-core tile bottleneck).
- GPU engine: float32 (`fast`/`refined`), one command buffer per call, specialised separable
  kernels (any/even/odd × float32/float64 store, overhanging windows `taps > length` handled
  without modulo division), dense N-D stencil via `dense_correlate::encode_gpu` (and rank-1
  outer-product detection into separable passes in `fast` with adjusted `cval_k`), and a
  single-`Batch` composite combine kernel. Float64 widening and threaded P-core post-cast
  (`ndimage_cast_from_float`) convert output to any real `out_dtype`.
- Guard rails (A5, A6): `AUTO` double inputs whose narrowing or filter gain overflows float32,
  floating inputs whose peak finite magnitude is nonzero and below the float32 normal
  range, or 32/64-bit integers with `|x| >= 2^24` re-route to the CPU engine in
  `float64` with the reason in `sm_route_info::reason`. Forced `GPU` declines
  with `SM_ENOTSUP`.
- Routing: separable work = elements × CPU operations over all passes (GPU from `80·2^20`);
  dense work = elements × non-zero taps (GPU from `128·2^20`). `strict` with `auto`
  always uses the CPU engine, float32 and int/bool included.
- Declines: anything that is not a plain aligned native real `ndarray` (`float32`, `float64`,
  `bool`, `int8`–`int64`, `uint8`–`uint64`) of rank 1–8; `output` of a complex/`float16`/
  `longdouble`/non-native-endian dtype, wrong shape, read-only, or partly overlapping;
  complex or non-1-D/rank-incompatible weights; anything SciPy would reject;
  fewer than 16384 elements (unless `scimetal.set_small_inputs('native')`).

**`fft`** — `fft`, `ifft`, `rfft`, `irfft`, `fft2`, `ifft2`, `fftn`, `ifftn`, `rfft2`,
`irfft2`, `rfftn`, `irfftn`, plus `scimetal.fft.register()` (the `scipy.fft` uarray
backend, so unmodified `scipy.fft` calls are accelerated) and
`scimetal.fft.prepare(...)` (calls `sm_fft_plan(..., prepare=1)` to pre-compile and
cache GPU pipelines ahead of first use). Internal `fft::Scratch`, `fft::encode(...)`,
and `fft::next_fast_len(...)` allow `signal` to compose forward/inverse FFT stages into
its own `gpu::Batch`.
- GPU engine only, single precision: MLX's kernels (vendored byte for byte), our own
  planner (`src/fft/plan.cpp`, which also ports MLX's plan logic under its MIT notice)
  and our own strided pass `sm_fft_strided` for long, non-power-of-two and off-axis
  transforms. One command buffer per call.
- Coverage by transform length n:

  | n | Runs as |
  |---|---|
  | 2–4096 with prime factors ≤ 13 | one Stockham kernel |
  | ≤ 2048 with one prime factor > 13 whose p − 1 is smooth | Rader |
  | other n ≤ 2048 | fused Bluestein |
  | powers of two, 4096 < n ≤ 2^20 | MLX's two-pass four-step |
  | powers of two above 2^20 | our multi-pass plan (checked at 2^21–2^24) |
  | other n > 4096 with prime factors ≤ 13 | chain of strided passes; `rfft`/`irfft` run as a complex transform with a host crop, about twice the GPU work |
  | any axis that is not the last | strided chain, complex only, prime factors ≤ 13 |

- Metal grants some pipelines fewer threads per threadgroup than MLX's plan assumes (14
  length/transform combinations up to 4096). Those are re-planned before running; MLX
  itself does not check this.
- Declines to SciPy: n = 1; non-smooth n in (2048, 4096] and any n > 4096 with a prime
  factor > 13 (multi-pass Bluestein is not written); non-smooth lengths off the last
  axis; real transforms whose half-spectrum axis is not the array's last; `n`/`s`
  resizing off the last axis; batches of 2^31 points or more; double-precision input
  under `strict`; device `cpu`; `auto` below the crossover; a double input whose
  magnitudes float32 cannot carry (above 2^96, all below 2^-100, infinities, NaNs);
  inputs that are not plain arrays; `plan=`.
- There is **no native CPU FFT**. A declined call goes to SciPy's pocketfft; when the
  caller left `workers` unset and the call is large (a batch, or one 1-D transform of
  2^17 points or more), the fall-through sets `workers` to the core count. That result
  is double-precision SciPy but **not always bit-identical to single-worker SciPy**:
  pocketfft's SIMD and scalar paths round differently depending on how rows split
  between threads (about 1e-16 relative). `SCIMETAL_FFT_THREADED_FALLBACK=0` turns the
  threading off.
- Routing is a cost model in nanoseconds (`router::Family{"fft", 375000}`): host cost
  from points × ns/point per axis (scaled for double, real transforms and the threads
  SciPy would use), device cost 0.3 ms + 0.35 ns/point (1.0 for double), and the GPU
  must be predicted 1.25× faster. Crossovers: 2^17 points for a single complex64
  transform, 2^16 for complex128, 170k–500k points for batches. The FFT route has its
  own small-input floor (`sm_fft_auto_min_points`, 12461 points) and does not consult
  `set_small_inputs`.
- `unregister()` leaves the uarray backend registered but inert: uarray has no public
  call that removes one backend.
- Accuracy against SciPy complex128 (relative L2): worst 2.6e-6 over every length
  2–4096 and all four transforms (Rader, n = 2029); 1.0–1.2e-6 at 2^20–2^22;
  4e-7 at 2^23–2^24. Tests allow 5e-6 per transformed axis.

**`signal` convolution and correlation** — `convolve`, `correlate`, `fftconvolve`,
`oaconvolve`, `convolve2d`, `correlate2d`, and `choose_conv_method` across `mode`
(`full`/`same`/`valid`), `method` (`auto`/`direct`/`fft`), `axes` (for FFT forms), and
2-D `boundary` (`fill`/`wrap`/`symm`) / `fillvalue` for plain C-contiguous real and
complex float arrays (`float32`, `float64`, `complex64`, `complex128`, rank 1–8).
- Direct path (`sm_signal_convolve`): reuses `sm::detail::dense_correlate` (`signal.convolve.direct`
  crossover `128·2^20`). The CPU path preserves `0.0 * NaN = NaN`. The GPU
  uses single precision (`float32` / `complex64`). Its range and gain checks reroute
  automatic calls to the CPU when needed and set `sm_route_info::reason`.
- Strict Python `convolve2d` verifies the installed SciPy 1.18.1 CPython 3.14 macOS
  arm64 FIR binary before selecting `sm_signal_convolve_scipy_fir`. That CPU path
  groups each kernel row in the same order with the same FMA contractions and
  vectorizes over independent output pixels. SciPy handles unsupported inputs,
  unverified builds, nondefault floating-point settings, and nonfinite results.
- FFT & overlap-add GPU path (`sm_signal_fftconvolve`, `fft_gpu_min_elements = 2^15`):
  single precision (`fast` / `refined` only); in one `gpu::Batch` per call, pads/chunks
  inputs (`fft::next_fast_len` and Lambert-W OA block sizing), runs forward FFTs via
  `fft::encode` (r2c on the last axis for real inputs), multiplies spectra, runs the
  inverse FFT, and overlap-adds/crops/widens to the promoted output dtype. Guards against
  double inputs outside `[2^-100, 2^96]` or product overflow (`peak1 * peak2 *
  overlap_elems > 2^124`).
- `strict` FFT (`fftconvolve`, `oaconvolve`, `method='fft'`, and double `method='auto'`
  when FFT wins): declines to SciPy wrapped in `scipy.fft.set_workers(performance_cores())`
  for calls `>= 2^17` elements when `workers == 1` (`SCIMETAL_FFT_THREADED_FALLBACK=0`
  disables).
- Declines: integer/boolean/`float16`/`longdouble` inputs, 0-D/empty/rank>8 or
  mismatched-rank inputs, non-NumPy/unaligned/invalid arguments, 1-D/N-D non-`fill`
  boundary or non-zero `fillvalue` outside `convolve2d`/`correlate2d`, and small calls
  below break-even (`2^16` 1-D direct, `2^12` N-D direct, `2^15` FFT unless
  `scimetal.set_small_inputs('native')`).

### 5.3 Measured results of the product so far

Measured 2026-11-19 alone on a quiet M1 Ultra (sequential runs with debug/override
env vars unset; pre/post `uptime` load averages `1.32–2.85`, recorded in each JSON),
through the public Python API (plus `sm_array_test --bench`), median of warm runs. Raw
output: `bench/results/m1-ultra/ndimage_filters.json`,
`ndimage_filters_calibrate.json`, `fft.json`, `fft_crossover.json`, and
`signal_convolve.json` (`fft.baseline.json` and `fft_crossover.baseline.json` are
preserved as M1 pre-A1-optimization baselines; `*.quick.log` are the 2026-11-18
Phase-1-handoff logs).

**Core `array::convert` (double → single narrowing, `./.build/metal/sm_array_test --bench`):**
256 MB: `165.17 GB/s` (`no_report`), `172.14 GB/s` (`with_report`); 1024 MB:
`170.35 GB/s` (`no_report`), `179.44 GB/s` (`with_report`).

**`ndimage` filters** (`bench/ndimage_filters.py`, `uptime` loadavg `2.15 2.23 2.17`),
float64 in and out plus integer image rows (`bench/results/m1-ultra/ndimage_filters.json`):

| Workload | SciPy ms | fast (GPU) ms | × | strict (CPU) ms | × |
|---|---|---|---|---|---|
| gaussian 4096², σ = 3 (`float64`) | 390.7 | 3.70 | 105.6 | 7.20 | 54.3 |
| gaussian 4096², σ = 3 (`uint8`) | 308.5 | 4.79 | 64.5 | 7.95 | 38.8 |
| gaussian 4096², σ = 3 (`uint16`) | 320.0 | 4.59 | 69.7 | 7.65 | 41.8 |
| gaussian 4096², σ = 10 (`float64`) | 730.2 | 7.74 | 94.4 | 17.03 | 42.9 |
| gaussian 512³, σ = 2 (`float64`) | 3795.6 | 22.40 | 169.5 | 70.72 | 53.7 |
| uniform 4096², size 9 (`float64`) | 447.7 | 2.43 | 184.2 | 3.69 | 121.5 |
| correlate1d 9 taps, last axis (`float64`) | 67.1 | 2.03 | 33.1 | 1.78 | 37.7 |
| correlate1d 9 taps, axis 0 (`float64`) | 270.1 | 2.08 | 130.2 | 2.98 | 90.5 |
| correlate1d 8192×64, 513 taps overhang (`float64`) | 314.2 | 1.56 | 201.5 | 2.08 | 151.1 |
| gaussian_laplace 4096², σ = 3 (`float64`) | 771.1 | 4.00 | 192.8 | 14.62 | 52.7 |
| correlate 4096², 5×5 dense (`float64`) | 275.3 | 3.44 | 79.9 | 4.56 | 60.4 |
| correlate 4096², 5×5 dense (`uint8`) | 300.2 | 4.71 | 63.8 | 11.12 | 27.0 |

float32 inputs run faster (3.05 ms for gaussian 4096², σ = 3, 114.3×; 19.72 ms for
512³, 166.3×; 2.24 ms strict CPU for uniform 4096², size 9, 106.2×; 3.40 ms fast for
gaussian_laplace 4096², 206.2×; 2.88 ms fast / 3.83 ms strict for 5×5 dense correlate,
100.2× / 75.5×). Fast-mode error is `1.8e-7–1.4e-6` of the data scale across all float64
workloads; strict is bit-identical to SciPy in every row. Quiet calibration
(`bench/results/m1-ultra/ndimage_filters_calibrate.json`, 116 rows across 9 filter
families × 2 dtypes) confirmed `gpu_min_work = 80·2^20`.

Small inputs (CPU engine, either policy): a native call has a floor of about 16 µs
against SciPy's 3 µs. Break-even is 96²–128² in 2-D and 16384 elements in 1-D, hence the
16384-element decline in `python/scimetal/ndimage/_filters.py`.

**FFT** (`bench/fft.py`, `uptime` loadavg before `1.77 1.88 2.42`, after `1.69 1.85 2.40`),
default policy (`fast`, `auto`):

| Case | dtype | SciMetal ms | SciPy 1 worker ms | × | SciPy 16 workers ms | × | first call ms |
|---|---|---|---|---|---|---|---|
| fft 2^16 | complex128 | 0.54 | 0.69 | 1.3 | 0.27 | **0.50** | 58.1 |
| fft 2^16 | complex64 | 0.27 (SciPy) | 0.25 | 0.9 | 0.25 | 0.92 | — |
| fft 2^20 | complex128 | 1.49 | 13.4 | 9.0 | 2.14 | 1.4 | 3.9 |
| fft 2^20 | complex64 | 1.11 | 7.27 | 6.5 | 1.29 | 1.2 | 1.3 |
| fft 2^22 | complex128 | 2.72 | 67.6 | 24.9 | 7.13 | 2.6 | 14.7 |
| fft 2^22 | complex64 | 1.96 | 43.2 | 22.1 | 4.83 | 2.5 | 2.1 |
| fft 2^24 | complex128 | 8.01 | 299.9 | 37.4 | 29.9 | 3.7 | 56.9 |
| fft 2^24 | complex64 | 5.35 | 206.1 | 38.5 | 21.6 | 4.0 | 6.0 |
| fft 10^6 | complex128 | 1.60 | 10.9 | 6.8 | 1.82 | 1.1 | 2.5 |
| rfft 2^22 | float64 | 2.19 | 47.8 | 21.9 | 9.04 | 4.1 | 6.2 |
| fft 4096 × 4096 | complex128 | 5.09 | 55.9 | 11.0 | 5.40 | 1.1 | 12.2 |
| fft 4096 × 4096 | complex64 | 2.55 | 31.6 | 12.4 | 3.36 | 1.3 | 2.6 |
| fft 1024 × 65536 | complex128 | 20.6 | 644.8 | 31.3 | 52.4 | 2.5 | 193.2 |

What these say, plainly:

- Against SciPy as most code calls it (one worker), the GPU FFT is 7–38× faster from
  2^20 points. Against SciPy with 16 workers it is 1.1–4.0× faster on these cases, and
  at 2^16 complex128 it is **slower** (0.52×); the router compares against the worker
  count the caller asked for, which defaults to one.
- Stage A1 profiled the FFT host path (`SCIMETAL_FFT_PROFILE=1`, 15 warm runs on
  `2^24 complex64`, `four-step 16×32×32×32×32`) and cut non-GPU overhead via a bounded
  `ResolvedPlan` cache keyed by monotonic `Plan::id`, `commandBufferWithUnretainedReferences()`,
  `__restrict__` / `memcpy` load/store fast paths, and `ctz` power-of-2 shifts in
  `sm_fft_strided`. Remaining non-GPU overhead is ~1.32 ms (~27 µs planning/alloc/encode,
  ~14.5 µs Python wrapper, ~0.95 ms 16-thread `memcpy` copy-in, ~0.32–0.35 ms wait
  overhead, 0.04 µs publish):

  | Stage (µs) | Before (med) | After (med) | After (min) |
  |---|---|---|---|
  | `parse_route` + `stage` + `refused` | 9.80 | 0.96 | 0.45 |
  | `alloc_in` + `alloc_scratch` + `encode` | 66.25 | 27.34 | 14.42 |
  | `copy_in` (256 MiB host → shared) | 1159.62 | 947.33 | 936.29 |
  | `gpu` | 3553.54 | 3183.38 | 2506.87 |
  | `wait_overhead` + `publish` | 427.16 | 346.92 | 259.67 |
  | C++ `total` / public `mfft.fft` | 5236 / 5362 | 4727 / 4806 | 3818 / 3930 |

  At boosted GPU clock (`gpu` ~2.51 ms), `2^24 complex64` reaches 3.82 ms C++ / 3.93 ms
  Python (`≤ 4.0 ms`), whereas interleaved CPU/GPU loops (such as `fft.json`'s 5.51 ms
  median / 5.05 ms min) often catch mid-frequency GPU DVFS P-states (`gpu` ~3.55 ms).
- First calls pay for pipeline compilation (up to ~198 ms for some plans with Metal's
  on-disk shader cache already warm). `scimetal.fft.prepare(...)` now calls
  `sm_fft_plan(..., prepare=1)` ahead of time so subsequent first calls avoid pipeline
  compilation latency.
- Quiet FFT crossover sweep (`bench/results/m1-ultra/fft_crossover.json`, 118 rows across
  59 shapes × 2 dtypes) verified against `fft_crossover.baseline.json`: 0 / 118 false
  negatives (`ratio ≥ 1.25`) and 1 / 118 borderline false positive (`fft [8192, 64] complex128`,
  0.96×, improved from 4 / 104 in baseline). `src/fft/fft.cpp` routing thresholds
  (`device_fixed_ns = 300e3`, `gpu_min_work = 375000`, `auto_min_points = 12461`) were kept.

**`signal` convolution and correlation** (`bench/signal_convolve.py`, `uptime` loadavg
before `1.97 1.91 2.41`, after `2.85 2.11 2.47`), float64 in and out:

| Workload | SciPy w1 ms | SciPy w16 ms | fast (GPU) ms | × vs w1 | × vs w16 | strict ms (engine) | err / scale |
|---|---|---|---|---|---|---|---|
| `fftconvolve` 4096² × 4096² (`full`) | 1393.9 | 176.6 | 25.21 | 55.3 | 7.0 | — (SciPy) | 1.3e-06 |
| `convolve` 2^24 × 1025 (`same`) | 518.7 | 129.4 | 3.97 | 130.7 | 32.6 | 128.34 (`scipy`) | 1.1e-06 |
| `oaconvolve` 2^24 × 1025 (`same`) | 299.0 | 44.6 | 3.93 | 76.2 | 11.4 | — (SciPy) | 1.1e-06 |
| `oaconvolve` 4096² × 65² (`same`) | 234.3 | 67.6 | 7.04 | 33.3 | 9.6 | — (SciPy) | 1.8e-06 |
| `convolve2d` 4096² × 5² (`same`) | 572.5 | — | 3.35 | 171.1 | — | See below | 1.9e-07 |

On the same `convolve2d` workload, the verified strict CPU route delivered
byte-identical results in 4.77 ms median versus 588 ms for SciPy, measured in a
separate warm run.

**SciPy's own tests through the oracle** (integrated build):

| Suite | strict / auto | fast / forced GPU |
|---|---|---|
| `scipy.ndimage.tests.test_filters` | 2479 passed | 2448 passed, 31 failed |
| `scipy.fft.tests.test_basic` | 171 passed | 148 passed, 23 failed |
| `scipy.signal.tests.test_signaltools` (`Convolve or Correlate or choose_conv_method`) | 11690 passed, 1 skipped | 428 passed, 928 failed (`-m "not slow"`) |

- Filters, native calls per function (strict / auto after A6): `convolve` 701, `convolve1d`
  863, `correlate` 749, `correlate1d` 1034, `gaussian_filter` 135, `gaussian_filter1d` 7,
  `gaussian_gradient_magnitude` 100, `gaussian_laplace` 109,
  `generic_gradient_magnitude` 1, `generic_laplace` 10, `laplace` 98, `prewitt` 53,
  `sobel` 53, `uniform_filter` 171, `uniform_filter1d` 18 (all 15 functions > 0, with
  integer/bool fall-throughs eliminated). Remaining fall-throughs are complex/`float16`/
  `longdouble`/non-native-endian inputs, non-real `output=`, complex weights,
  non-traceable `generic_*` callbacks, and invalid-argument tests. The 31 forced-GPU
  failures are 15 float32-vs-float64 accuracy mismatches (`< 1e-3`) plus 16 integer
  intermediate-quantization mismatches (`<= 20` on `prewitt`/`sobel` uint32/uint64 and
  `gaussian_laplace`/`gaussian_gradient_magnitude` int32, where SciPy's reference calls
  quantize intermediate passes to integer dtypes while GPU `fast` keeps intermediates in
  `float32`), pinned in `tests/python/test_ndimage_oracle.py`.
- **FFT in strict / auto runs zero native calls**: SciPy's FFT tests use small
  double-precision arrays, and strict keeps double precision off the GPU. That run only
  proves the fall-through is transparent. The GPU run is the meaningful one: 808 native
  calls, and its 23 failures are tolerance failures (22 compare against double
  references at rtol 6e-8 to 1e-7 with atol 0; one is `test_multiprocess[irfft]`, where
  spawned workers run SciPy and the parent the GPU). The suite uses unseeded random
  inputs, so the failing set can vary slightly. The FFT's real validation is its own
  tests: about 770 transforms against a double-precision reference DFT in
  `tests/native/fft_gpu_test.cpp`, and every length 2–4096 in the Python tests.
- **Signal** (`test_signaltools`, `-k "Convolve or Correlate or choose_conv_method"`):
  in `strict / auto`, `11690 passed, 1 skipped` (`choose_conv_method` 12 native,
  `convolve` 18717, `convolve2d` 14, `correlate` 64, `correlate2d` 3; `fftconvolve`
  12178 and `oaconvolve` 10839 fall through to threaded SciPy). Under `fast / GPU`
  (`-m "not slow"`), all 7 functions run natively (`choose_conv_method` 12, `convolve`
  3455, `convolve2d` 14, `correlate` 65, `correlate2d` 3, `fftconvolve` 1132,
  `oaconvolve` 1056) with `428 passed, 928 failed`; `tests/python/test_signal_oracle.py`
  verifies every failure is a float32 vs float64 numerical tolerance mismatch.

### 5.4 Checks to run before starting

Run the entire C++ and Python ladder (both `metal` and `cpu-only`) in one command:

```bash
./tools/check.sh           # add --wheel to also build & verify the wheel in a clean venv
```

Or run each step individually:

```bash
cd /Users/tunguz/Programming/SciMetal
export PATH=$PWD/.venv/bin:$PATH
cmake --preset metal && cmake --build --preset metal && ctest --preset metal --output-on-failure
cmake --preset cpu-only && cmake --build --preset cpu-only && ctest --preset cpu-only --output-on-failure
.venv/bin/python -m pytest tests/python -q -p no:warnings
PYTHONPATH=python:tests/oracle .venv/bin/python -m pytest -p scimetal_oracle \
    --pyargs scipy.ndimage.tests.test_filters -q -p no:cacheprovider
PYTHONPATH=python:tests/oracle .venv/bin/python -m pytest -p scimetal_oracle \
    --pyargs scipy.fft.tests.test_basic -q -p no:cacheprovider
PYTHONPATH=python:tests/oracle .venv/bin/python -m pytest -p scimetal_oracle \
    --pyargs scipy.signal.tests.test_signaltools -k "Convolve or Correlate or choose_conv_method" -q -p no:cacheprovider
```

The expected results are recorded in section 5.5. The oracle prints, per function, how
many calls ran natively and how many fell through; a suite that passes only because
everything fell through shows up there.

### 5.5 Verified status at hand-off

Run on 2026-11-19 against the integrated builds (`.build/metal` and `.build/cpu-only`,
all families):

| Check | Result |
|---|---|
| `cmake --build --preset metal && ctest --preset metal --output-on-failure` | **20 / 20 passed** in 57.18s (runtime 10, array 1, ndimage 4, fft 3, signal 2; `_validated` targets run under Metal API and shader validation) |
| `cmake --build --preset cpu-only && ctest --preset cpu-only --output-on-failure` | **11 / 11 passed** in 3.86s (`SM_ENABLE_METAL=OFF`) |
| `pytest tests/python` | **728 passed** (verified on wheel + `./tools/check.sh`; includes A6 int/bool dtypes, A5 guard rails, A3/A4 ndimage, A2 signal, and strict + forced-GPU oracle tests) |
| `SCIMETAL_LIBRARY=$PWD/.build/cpu-only/libscimetal.dylib pytest tests/python` | **371 passed, 357 skipped** in 22.98s (`0 failed, 0 errors`; GPU-only FFT/GPU tests skip cleanly when `not _native.has_metal()`) |
| `SCIMETAL_FFT_EXHAUSTIVE=1 pytest tests/python/test_fft_sizes.py` | **3 passed in 31.38s** (41,424 calls) |
| `./.build/metal/sm_array_test --bench` | `with_report` **172.14 GB/s** (256 MB), **179.44 GB/s** (1024 MB) (`no_report` 165.17 / 170.35 GB/s) |
| Oracle, filters, strict / auto | **2479 passed** (all 15 functions native > 0, integer fall-throughs eliminated) |
| Oracle, filters, fast / GPU | **2448 passed, 31 failed** (15 float32 accuracy `< 1e-3` + 16 int intermediate quantization `<= 20`, pinned in `test_ndimage_oracle.py`) |
| Oracle, FFT, strict / auto | **171 passed**, **0 native calls** (see 5.3) |
| Oracle, FFT, fast / GPU | **148 passed, 23 failed** (808 native calls; float32 vs float64 tolerances, exercised in `test_fft_oracle.py`) |
| Oracle, signal, strict / auto | **11690 passed, 1 skipped** (`Convolve or Correlate or choose_conv_method`; 5 CPU-capable functions native > 0, `fftconvolve`/`oaconvolve` fall through to threaded SciPy) |
| Oracle, signal, fast / GPU | **428 passed, 928 failed** (`-m "not slow"`; all 7 functions native > 0, all failures float32 numerical tolerance, verified in `test_signal_oracle.py`) |
| Wheel (`macosx_15_0_arm64`) | `MACOSX_DEPLOYMENT_TARGET=15.0 .venv/bin/python -m pip wheel . -w dist --no-deps --no-build-isolation` builds `scimetal-0.1.0.dev0-py3-none-macosx_15_0_arm64.whl` (2.2 MB, bundles `libscimetal.dylib`); clean-venv install passes **728 / 728** (`SCIMETAL_TEST_INSTALLED=1`) from `/tmp` |
| `tools/vendor_mlx_kernels.sh --check` | **identical** to MLX 0.31.1 sources (verified 2026-11-19) |

Milestone status:
- **Milestone M1 reached** (all work packages `A0`–`A8` complete and verified; `./tools/check.sh` runs the full CTest + Python Metal + CPU-only ladder, with `--wheel` for isolated wheel verification).
- Sanitizers: AddressSanitizer / ThreadSanitizer remain environment-blocked on this host; see 6.8 and 9.2.
- The repository is local only (no git remote configured).

Safe-to-delete stale build directories if still present: `.build/ndimage`, `.build/fft`,
`.build/fft-all`, `.build/fft-cpu`, `.build/fft-asan`, `.build/wheel` (all ignored by
`.gitignore`).

---

## 6. Engineering rules

### 6.1 Recipe: adding a native function

Follow the two existing families; `ndimage` is the model for a family with both engines.

1. **Read SciPy first.** Find the function in the checkout
   (`/Users/tunguz/Programming/scipy/scipy/<area>/`), its C or Cython core, and its test
   module. Write down: accepted dtypes, argument validation order and messages, the
   arithmetic's precision and *order of operations*, boundary behaviour, output dtype
   rules, in-place behaviour. The strict engine has to reproduce these.
2. **C ABI** in `include/sm_<family>.h`: plain C, opaque handles, option structs that
   start with `struct_size` and `abi_version` and end with zeroed `reserved` fields,
   `sm_status` return, results through `sm_array_result`, route through `sm_route_info`.
   Document precisely what each engine computes, its error bound, and every refusal.
3. **Validation and routing** in `src/<family>/<name>.cpp`: build a `Plan` from the
   arguments (all checks here), compute the router's work figure, call `router::decide`,
   run one engine, `array::publish`, `array::report`. Wrap the body in
   `sm::detail::api(context, [&] { ... })`, which turns `Error(status, message)` into a
   status code and stores the message on the context.
4. **CPU engine** in `src/<family>/<name>_cpu.cpp`: threads through
   `parallel::parallel_for` / `parallel_ranges` (performance cores only); arithmetic in
   the precision and order SciPy uses; output into `array::allocate` storage.
5. **GPU engine** in `src/<family>/<name>_gpu.cpp` (only built with Metal): narrow the
   input once with `array::load_single`, encode every pass into **one** `gpu::Batch`,
   `commit_and_wait()` once. Bracket with `GpuOperationScope`.
6. **Kernels** in `src/<family>/<name>.metal`, parameter structs in a `.h` shared with
   the host (`static_assert` the layout on the host side).
7. **Build wiring**: `src/<family>/sources.cmake` appends to `SM_FAMILY_SOURCES`,
   `SM_FAMILY_GPU_SOURCES`, `SM_METAL_SOURCES`, `SM_METAL_INCLUDE_DIRS`;
   `src/<family>/tests.cmake` registers tests with `sm_add_gpu_test(... PLAIN VALIDATED)`.
   Do not edit the top-level `CMakeLists.txt` for a family.
8. **Native test** in `tests/native/<family>_<name>_test.cpp`: CPU engine against an
   independent reference written in the test; forced-GPU engine against the CPU engine
   within the documented bound; argument validation; routing; exactly one command buffer
   per call (`sm_context_execution_info`); buffer reuse; runs clean under Metal API and
   shader validation.
9. **Python binding** in `python/scimetal/<family>/_native_<family>.py`: ctypes
   structs and prototypes declared lazily; raise `NotAccelerated` if the loaded library
   lacks the symbol.
10. **Python route** in `python/scimetal/<family>/_<name>.py`: SciPy's exact signature,
    decorated with `@accelerated("<family>.<function>")`. Validate the way SciPy does and
    **decline instead of raising**, so the exception a caller sees is SciPy's own. Call
    `_native.decline_small(elements, break_even)` with a measured break-even.
11. **Python tests** in `tests/python/test_<family>_*.py`: bit-for-bit (strict) or
    bounded (fast) agreement with SciPy over generated shapes, dtypes, modes and axes
    (hypothesis); every decline path; in-place and `output=` behaviour; threads; an
    oracle test that runs SciPy's suite in a subprocess and pins the outcome.
12. **Benchmark** in `bench/<family>_*.py` with `--quick` and a calibration mode that
    sweeps forced-CPU against forced-GPU around the routing threshold.
13. **Record** thresholds, accuracy and declines in the C header and the Python module
    docstring. Those are the documentation of record.

### 6.2 C ABI conventions

- Status codes: `SM_OK`, `SM_EINVAL` (caller error), `SM_ENOTSUP` (valid but not handled
  here: Python turns it into a fall-through), `SM_EOVERFLOW`, `SM_EOOM`, `SM_ETIMEOUT`,
  `SM_EBACKEND`, `SM_ESTATE` (for example after fork: also a fall-through).
- Inputs are borrowed C-contiguous host memory, valid for the duration of the call. The
  core never retains input pointers.
- Results are owned buffers with one retained reference (`sm_array_result.buffer`).
- Every struct is versioned. Add fields only by bumping the family's ABI version.
- No exceptions cross the ABI. No Objective-C or Metal types in public headers.
- `sm_dtype` currently has four codes (float32, float64, complex64, complex128). Integer
  and boolean codes are work package A6.

### 6.3 CPU engine rules

- Lanes come from `parallel::hardware_lanes()` (performance cores, 16 here). Never use
  the efficiency cores for data-parallel work: they lower the throughput of the lane set.
- Give each lane at least about 2^14–2^18 scalar operations, or stay single-threaded.
- **Strict means SciPy's bits where SciPy's arithmetic is deterministic.** Reproduce the
  precision (SciPy's C filters accumulate every dtype in double), the summation order and
  the fused multiply-adds of the reference binary. The ndimage engine got to bit identity
  by reading the wheel's disassembly; when the SciPy wheel changes, expect 1–2 ulp
  differences and update the tests' expectation, not the claim.
- Where SciPy's result is not deterministic across thread counts or SIMD paths (FFT), say
  so in the docstring and test to SciPy's own tolerance.
- Prefer Accelerate for BLAS/LAPACK/vDSP (`ACCELERATE_NEW_LAPACK=1` is defined). Known
  slow spots in Accelerate on this machine: `dsyrk` rank-k updates, the `gesv` drivers.

### 6.4 GPU engine and kernel rules

- One public call → one command buffer → one `commit_and_wait()`. Tests assert the count.
- Kernels are compiled with `-std=metal3.1 -O2 -fno-fast-math -ffp-contract=off`. Do not
  loosen these: results must not depend on reassociation.
- Never read or write float64 in a kernel except the final store, which writes the
  widened bit pattern as `ulong` (see `sm_ndimage_widen`, bit-exact against a C cast).
- Index arithmetic in kernels is 32-bit. Decline (AUTO → CPU, forced GPU → `SM_ENOTSUP`)
  when element counts reach 2^31, and bound every intermediate product.
- Threadgroup shape matters by 1.5–8×. Measured on 4096²: along the contiguous axis flat
  groups of 256–1024 threads; along a strided axis 32 × 32. Ask the pipeline for its
  capacity with `gpu::max_threads_per_group` (shader validation lowers it to 512).
- Specialise with templates or function constants for anything that changes the inner
  loop (symmetry, store type): one kernel branching on a parameter measured 10% slower.
  Keep rarely-read parameters (boundary mode) as runtime parameters.
- A command buffer should not hold the GPU for more than about a second (the window
  server shares it). Split or decline to the CPU above that.
- Every kernel must run clean under `MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1`
  (`sm_add_gpu_test(... VALIDATED)`).
- State an error bound for the GPU engine in the header and test against it. Derive it;
  the first closed-form bound written for the filters was wrong and a property test
  caught it.

### 6.5 Routing and calibration

- `router::decide(ctx, family, options, input_dtype, work)`: forced CPU → CPU engine in
  the input's dtype; forced GPU → GPU or `SM_ENOTSUP` (no Metal, or strict with a double
  input); AUTO → GPU iff a GPU exists, the policy allows it and `work >= gpu_min_work`.
- The work unit is the *CPU engine's* cost in its own operations, so that one threshold
  covers the family. Return 0 work for shapes where the GPU would do disproportionately
  more (for example long uniform windows, where the CPU's running sum does not grow).
- Calibrate with the benchmark's sweep of forced-CPU against forced-GPU. Put the
  constant and the measurement that justifies it next to each other in the source.
- Python declines calls below a measured break-even size before touching the core
  (`_native.decline_small`). The oracle and the test suite switch that off with
  `scimetal.set_small_inputs('native')` so small arrays still reach native code.
- Thresholds are for `applegpu_g13d` with 16 performance cores. A table keyed by GPU and
  core count, and `scimetal.calibrate()`, are work package X1.

### 6.6 Precision rules

- The policy decides only whether a double-precision input may go to the GPU. The CPU
  engine always computes in the precision SciPy uses.
- Under `fast`, the FFT and ndimage routes use magnitude checks to decline inputs
  outside the supported single-precision range. FFT promotion checks 32/64-bit integer
  bounds before casting.
- Under `strict`, forced GPU with a double input is a decline, never a silent downgrade.
- Solve-type functions get a double-precision residual check and refinement (Phase 3);
  when refinement stalls, fall back to the double-precision CPU path and say so in the
  route's reason.

### 6.7 Python layer rules

- A route either returns a result after recording a `Route`, or raises `NotAccelerated`
  with a reason. Nothing else. `ValueError`/`MemoryError` from the core become declines
  in the family binding, so SciPy decides what the caller sees.
- Accept exactly `type(x) is np.ndarray` (no subclasses, lists, masked arrays, other
  libraries) unless the family has a reason to do more; SciPy handles the rest.
- Compare dtypes by type character where it matters: on arm64 macOS `long double` equals
  float64 in size, yet SciPy rejects it in places.
- `output=` arrays: fill and return the caller's array for same dtype and shape; decline
  for partial overlap; handle exact in-place.
- Fall-through namespaces never forward dunder attributes (forwarding `__path__` once
  made Python load a second copy of a SciPy sub-module).
- After `fork`, the native context is unusable in the child: `_native.context()` raises
  `NotAccelerated` and buffers inherited from the parent are never released there.

### 6.8 Testing ladder

A function is done when all of these hold:

1. Native test passes plain and under Metal validation.
2. Python tests pass, including property tests against SciPy.
3. SciPy's own test module passes through the oracle in `strict` / `auto` with a
   non-zero native count for the function.
4. The forced-GPU / `fast` oracle run fails only tests that need more than float32, and
   that set is pinned in a test.
5. The CPU-only build (`cmake --preset cpu-only`) builds and its tests pass.
6. The benchmark reproduces the claimed speed-up when run alone.

Sanitizers: the `metal-asan-ubsan` and `metal-tsan` presets exist, but on this machine
ASan was reported to hang at start-up and TSan to crash even for the smallest test
(reported by a builder, not re-verified). UBSan alone and Guard Malloc
(`DYLD_INSERT_LIBRARIES=/usr/lib/libgmalloc.dylib`) were reported to work. Treat
"sanitizers do not run here" as an open defect (9.2), not as a pass.

### 6.9 Benchmark protocol

- Run benchmarks **alone and one at a time**. Parallel builds or other agents on the
  machine moved results by 2× in this project. Check `uptime` first and record the load
  average with the result.
- Report medians of warm runs; exclude the first call (pipeline compilation, buffer
  pool warm-up) and report it separately if it matters.
- Always measure through the public Python API, float64 in and float64 out, against
  stock SciPy called the way a user would call it. For FFT also report SciPy with
  `workers=-1`.
- State whether results are dropped between calls (buffers recycle) or kept.
- Store results per chip under `bench/results/<chip>/<benchmark>.json`.

### 6.10 Working in parallel

Independent families can be built concurrently by separate agents if:

- each agent owns only `include/sm_<family>.h`, `src/<family>/**`,
  `tests/native/<family>_*`, `python/scimetal/<family>/**`,
  `tests/python/test_<family>_*.py`, `bench/<family>*.py`;
- shared files (`CMakeLists.txt`, `src/core`, `src/runtime`, `python/scimetal/_*.py`,
  `tests/oracle`, `tests/python/conftest.py`) are changed only by the lead, on request;
- each agent builds in its own tree with only its family:
  `cmake -S . -B .build/<family> -G Ninja -DMS_FAMILIES=<family>` and points Python at it
  with `SCIMETAL_LIBRARY=$PWD/.build/<family>/libscimetal.dylib`;
- the lead reviews the code (not just the report), integrates, runs the full ladder,
  and re-runs every benchmark alone.

Agent reports are claims, not facts. Re-run the commands they cite.

---

## 7. Work packages

Format: **Goal** / **Scope** / **Design** / **Gate** / **Notes**. Sizes are relative
(S ≈ hours, M ≈ a day, L ≈ several days of focused agent work).

### Stage A — finish Phase 1 (the first vertical slice)

#### A0. Shared-code clean-up left by the first slice (S)

Small, do first:

- FFT route: use `_native.decline_small` and `scimetal.set_small_inputs` like ndimage
  does, instead of its private floor, so one switch controls small inputs everywhere.
- `NotAccelerated` should be able to carry keyword overrides for the fall-through (the
  FFT route wants to add `workers=`); today `fft/_basic.py` calls SciPy itself and
  compensates the counters by hand. Move that into `_dispatch.accelerated`.
- One shared `performance_cores()` helper in `_native.py` (the FFT module has its own).
- Remove the redundant plugin hook from `tests/python/test_fft_oracle.py`.
- The FFT wrappers set `__module__`/`__qualname__` themselves (`_public` in
  `fft/_basic.py`); `_dispatch.accelerated` now does this for every wrapper, so drop the
  local copy.
- Decide, with the user's precision policy in mind, whether the threaded FFT
  fall-through should stay on by default: it makes `strict` results differ from stock
  single-worker SciPy in the last place for some shapes.
**Gate.** Section 5.4 still passes.
**Status.** Done (M1). Section 5.4 passes; `fft/_basic.py` uses `_native.decline_small`,
`_native.performance_cores()`, and `NotAccelerated(..., **overrides)` via
`_dispatch.accelerated`; the redundant plugin hook was removed from
`tests/python/test_fft_oracle.py`; and `strict` keeps single-worker SciPy bit-identity
while `fast`/`refined` thread eligible declines (`workers=16`).

#### A1. Integration, measured alone; FFT host path (M)

**Goal.** Benchmark numbers that can be quoted, and the probe's FFT figure recovered.
**Do.**
1. Run section 5.4. Then, with nothing else running (check `uptime`):
   `.venv/bin/python bench/ndimage_filters.py`, `.venv/bin/python bench/fft.py`, and the
   calibration modes (`--calibrate`, `--crossover`). Store the JSON under
   `bench/results/m1-ultra/`. Move the routing thresholds if the quiet sweep says so.
2. Profile the FFT's host path. 2^24 complex64 takes 5.4 ms through the API against
   2.47 ms of GPU time in probe E5. Instrument `src/fft/fft.cpp` and
   `src/fft/gpu_engine.cpp` (copy-in, narrow, encode, GPU, widen, publish) and remove
   what is not GPU time: for complex64 the input copy is one threaded memcpy of 256 MB
   (about 1.5 ms at measured bandwidth); the output is already a pooled buffer.
3. Re-run the exhaustive length sweep and the FFT crossover grid in the final state.
4. Call `sm_fft_plan(..., prepare=1)` from somewhere useful (for example a
   `scimetal.fft.prepare(shape, dtype)` helper) so first calls need not pay 50–180 ms.
**Gate.** Section 5.4 passes (`15/15` ctest, `369` pytest, `2479` ndimage oracle, `171`
FFT oracle); gaussian 4096² σ = 3 fast ≤ 3.5 ms (**met**: `3.425 ms` float64 / `2.816 ms`
float32); FFT 2^24 complex64 ≤ 4 ms through the public API (**reached at boosted GPU
clock**, `3.82 ms` C++ / `3.93 ms` Python min when `gpu` ~2.51 ms, vs `5.51 ms` median /
`5.05 ms` min in interleaved `bench/fft.py` at mid-DVFS clock `gpu` ~3.55 ms; non-GPU
overhead cut to ~1.32 ms; full stage breakdown recorded in 5.3, moving on per gate rule).
**Status.** Done. Quiet benchmarks (`ndimage_filters.json`, `ndimage_filters_calibrate.json`,
`fft.json`, `fft_crossover.json`), `SCIMETAL_FFT_PROFILE=1` instrumentation,
`scimetal.fft.prepare(...)`, `cpu-only` (`8/8`), and `SCIMETAL_FFT_EXHAUSTIVE=1` (`3 passed`,
41,424 calls) all passed; crossover thresholds kept (`0 / 118` false negatives).

#### A2. `signal` convolution and correlation (L)

**Goal.** `scipy.signal.convolve`, `correlate`, `fftconvolve`, `oaconvolve`,
`convolve2d`, `correlate2d`, `choose_conv_method` with native routes.
**Scope.** `mode` full/valid/same; `method` auto/direct/fft; `axes` for the FFT forms;
`boundary` fill/wrap/symm and `fillvalue` for the 2-D forms; real and complex float
inputs. Integer inputs decline at first (SciPy rounds FFT results to integers; float32
cannot guarantee exactness above 2^24).
**Design.**
- Reference: `scipy/signal/_signaltools.py` (`convolve`, `correlate`, `fftconvolve`,
  `_freq_domain_conv`, `_apply_conv_mode`, `_centered`, `_inputs_swap_needed`,
  `_reverse_and_conj`, `_fftconv_faster`, `_np_conv_ok`) and
  `scipy/signal/_correlate_nd.cc`.
- *FFT method, GPU:* a native `sm_signal_fftconvolve` that, in one command buffer, pads
  both inputs to a transform shape the FFT planner likes, runs the forward transforms
  (r2c along the last axis for real inputs), multiplies the spectra in a pointwise
  kernel, runs the inverse and crops to the requested mode. This needs the FFT family's
  plan to be encodable into a caller's `Batch` (an internal C++ entry next to
  `fft::run`), not a round trip through the C ABI per transform. Pick transform lengths
  from the planner's own notion of a fast length, not `scipy.fft.next_fast_len`.
- *FFT method, strict:* decline to SciPy, wrapped in `scipy.fft.set_workers(cores)` so
  the fall-through is at least threaded. Document that threaded pocketfft can differ
  from the single-worker result in the last place.
- *Direct method:* a dense N-D correlation primitive shared with A3. CPU: threaded,
  double accumulation in `_correlate_nd.cc`'s order. GPU: float32 stencil over an
  (offset, weight) table. For 1-D, SciPy calls `np.convolve`/`np.correlate`; match their
  results in strict.
- *`method='auto'`:* SciPy times nothing, it predicts with `_fftconv_faster`. Our choice
  should come from our own crossover (direct work = out elements × kernel elements;
  FFT work ≈ padded elements × log). Calibrate both and choose the cheaper native path.
- `correlate` semantics that are easy to get wrong: inputs are swapped when the second
  is larger in every dimension for `valid`; the result is then reversed and conjugated;
  `same` centres on the first input.
- `oaconvolve` matters when one input is much longer than the other: on the GPU, chunk
  the long input and batch the block FFTs in one command buffer.
**Gate.** The convolution and correlation classes of
`scipy/signal/tests/test_signaltools.py` pass through the oracle in strict with native
counts above zero; `fftconvolve` of two 4096² float64 images in `fast` ≥ 20× SciPy;
`convolve` of a 2^24-point signal with a 1025-tap kernel ≥ 20×.
**Notes.** Benchmark against SciPy the way it is called (`fftconvolve` uses one worker
unless the caller wraps it in `scipy.fft.set_workers`), and report the threaded figure
too. FFT convolution in float32 has absolute error about `eps32 × max|output|`,
which can exceed small output values entirely. State it; consider a `refined` variant
that splits inputs into high and low parts only if a user needs it.
**Status.** Done (M4–M5). Strict `test_signaltools` oracle passes (`11690 passed, 1 skipped`,
5 CPU-capable routes native > 0); `fftconvolve` 4096²×4096² `fast` = `25.21 ms` (**55.3×** vs
w1, **7.0×** vs w16); `convolve` 2^24×1025 `fast` = `3.97 ms` (**130.7×** vs w1, **32.6×** vs
w16).

#### A3. `ndimage` dense filters and the composite filters (M)

**Goal.** `correlate`, `convolve` (N-D weights), `laplace`, `gaussian_laplace`,
`gaussian_gradient_magnitude`, `generic_laplace` / `generic_gradient_magnitude` when the
derivative is one of ours.
**Design.**
- Reference: `NI_Correlate` in `scipy/ndimage/src/ni_filters.c` (it iterates the
  non-zero weights in C order and accumulates in double) and the Python wrappers in
  `scipy/ndimage/_filters.py` (`_correlate_or_convolve` reverses weights and shifts
  origins for even sizes).
- CPU: threaded over the outermost axis, same order of accumulation, bit-identical.
- GPU: one kernel looping over a table of non-zero taps; interior threads take a fast
  path with plain strided loads, threads near a face go through the per-axis extension
  function already in `ndimage_filters.metal`.
- In `fast`, detect rank-1 weights (an outer product) and run separable passes instead.
  Not in `strict`: the rounding differs.
- Composite filters are sums of separable chains (Laplace) or the root of a sum of
  squares (gradient magnitude). Add the combination as a kernel so the whole filter is
  still one command buffer; on the CPU fuse the combination into the last pass.
**Gate.** `test_filters.py` through the oracle: all pass in strict; native counts for
the new functions above zero; `gaussian_laplace` 4096² σ = 3 fast ≥ 60×; dense 5×5
`correlate` 4096² fast ≥ 40×, strict ≥ 15×.
**Status.** Done (M2, M5). Strict `test_filters` oracle passes (`2479 passed`, all 15
functions native > 0); `gaussian_laplace` 4096² σ = 3 `fast` = `4.18 ms` (**185.8×**);
dense 5×5 `correlate` 4096² `fast` = `2.55 ms` (**108.7×**), `strict` = `4.70 ms`
(**58.9×**).

#### A4. `ndimage` output dtypes and remaining declines (S)

**Goal.** Shrink the fall-through set of the filters.
**Do.** `output=` of the other float dtype (compute, then convert in the last pass or
with `array::convert`); the innermost-axis uniform filter on the CPU, currently a scalar
recurrence per line (7.5 ms for 4096²), interleaved across lines with NEON; windows
longer than the axis on the GPU (they take the slow boundary path for every tap).
**Gate.** The "output of another dtype" fall-throughs for float outputs disappear from
the oracle counters; CPU uniform 4096² size 9 ≤ 4 ms.
**Status.** Done (M3, M5). Cross-dtype float `output=` routes natively across separable,
dense, and composite filters; CPU `uniform_filter` 4096² size 9 = `3.65 ms` (float64) /
`2.41 ms` (float32); GPU overhanging `correlate1d` 8192×64 (513 taps) = `1.56 ms`
(**202.1×**).

#### A5. Fast-mode guard rails (M)

**Goal.** Speed-first never returns silent garbage.
**Do.**
- Make `array::convert` (double → single) report, at no extra pass, whether any finite
  value overflowed to infinity and the largest magnitude seen. A call whose narrowing
  overflowed re-routes to the CPU engine and records the reason.
- A one-time `scimetal.PrecisionWarning` the first time a double-precision input is
  computed in single precision, naming `set_precision('strict')` and how to silence it.
- `explain()` already reports engine and compute dtype; add the reason for a re-route.
- Tests: a float64 array with values near 1e39 through every fast route returns finite,
  correct values.
**Gate.** The tests above, and no measurable slowdown of the narrowing (≥ 150 GB/s).
**Status.** Done (M1, M5). Overflowing `1e39` float64 inputs re-route to CPU/SciPy with
`sm_route_info::reason` surfaced in `explain()` / `last_route()`; one-time
`PrecisionWarning` verified in `test_precision_guard.py`; `sm_array_test --bench`
`with_report` = `175.17 GB/s` (256 MB), `176.85 GB/s` (1024 MB).

#### A6. Integer and boolean dtypes (M)

**Goal.** Most real images are uint8/uint16; about 1950 of the oracle's filter calls fall
through for that reason alone.
**Do.** Extend `sm_dtype` with bool, int8–int64, uint8–uint64 (bump the array ABI
version). Filters: SciPy converts every input to double per line, filters in double and
casts back with its own rules; read the conversion macros in
`scipy/ndimage/src/ni_support.h` and `ni_filters.c` and reproduce them exactly in the
CPU engine (pay attention to negative values into unsigned types and to out-of-range
values). GPU: convert to float32 on the way in, and only in `fast`, and only when the
input range is exactly representable (|x| < 2^24).
**Gate.** Oracle strict passes with the integer fall-throughs gone; uint8 gaussian 4096²
strict ≥ 30×.
**Status.** Done (M1). ABI bumped to `SM_ARRAY_ABI_VERSION = 3u` across `bool` and
`int8..64`/`uint8..64`; CPU `ndimage` passes `scipy.ndimage.tests.test_filters` strict
oracle (**2479 passed**, integer fall-throughs gone) and bit-exact native C++ & Python
tests across all 15 filters; `uint8` `gaussian_filter` 4096² σ = 3 achieves **38.8×**
strict (`7.95 ms`) and **64.5×** fast GPU (`4.79 ms`); `|x| >= 2^24` guard rail
verified via `ConvertReport::inexact_int`; `array::convert` `with_report` measures
**172.14 / 179.44 GB/s** (`>= 150 GB/s`).

#### A7. CPU FFT engine — measure first (M)

**Goal.** Decide whether a native CPU FFT is worth having.
**Why.** Today anything the GPU does not take goes to pocketfft (threaded where
possible). The original plan assumed vDSP below the crossover. Threaded pocketfft is a
strong baseline on this machine: with 16 workers it is within 1.2–5× of the GPU at every
size measured (5.3), and it is exact double precision.
**Do.** A probe (in `probes/`) comparing, on the performance cores: pocketfft with
`workers`, `vDSP_DFT_Execute` / `vDSP_fft_zop` threaded over batch lines, for float32 and
float64, lengths 2^8–2^24 and batches. vDSP supports lengths `f·2^n` with f ∈ {1, 3, 5,
15} only.
**Gate.** If vDSP is ≥ 1.5× pocketfft for supported lengths in some size range, add a CPU
engine for exactly that range (and make it the engine for `strict` float32). Otherwise
record the negative result and keep the SciPy fall-through.
**Status.** Done (negative result recorded in `probes/README.md` and
`probes/results/a7_cpu_fft.{json,log}`). `probes/a7_cpu_fft` measured 16-thread
end-to-end interleaved vDSP at **0.27×–1.20×** vs `scipy.fft.fft(..., workers=16)` on
batches and `0.14×–0.89×` on large 1-D (`>= 2^17` `complex64`, `>= 2^16` `complex128`),
with Python + alloc + ctypes handoff erasing gains below `<= 16384`; keeping the SciPy
(`pocketfft`) fall-through avoids `strict` rounding divergence across a narrow
`49152–65536` `f·2^n` band.

#### A8. Packaging and documentation (S)

**Do.**
- `README.md` at the repo root exists but is minimal; expand it with install, usage,
  the precision policy, the accuracy table and the coverage list
  (`scimetal.accelerated()`).
- The wheel builds (`.venv/bin/python -m pip wheel . -w dist --no-deps
  --no-build-isolation`) and works when unpacked outside the tree. Still to do: fix the
  platform tag (it says macOS 26; export `MACOSX_DEPLOYMENT_TARGET=15.0`), install it in
  a fresh venv, and run `tests/python` against the installed package (the tests'
  `conftest.py` puts `python/` first on `sys.path`; add an environment switch to skip
  that).
- Verify the `cpu-only` preset end to end, including the Python tests with
  `SCIMETAL_LIBRARY` pointing at it (they should run the CPU engines and skip GPU
  cases).
- `tools/check.sh`: the whole ladder in one command.
**Gate.** A clean venv with only the wheel installed passes the Python tests.
**Status.** Done (M1 reached). Expanded `README.md`, created executable `tools/check.sh`
(with `--wheel`), skipped GPU-only FFT tests cleanly when `not _native.has_metal()`
(`cpu-only` Python suite **371 passed, 357 skipped**), added `SCIMETAL_TEST_INSTALLED`
in `tests/python/conftest.py` and all subprocess `PYTHONPATH` setups, built
`dist/scimetal-0.1.0.dev0-py3-none-macosx_15_0_arm64.whl`, and verified **728 passed**
against the installed wheel inside an isolated venv from `/tmp`.

### Stage B — Phase 2: signal, FFT and image breadth

#### B0. FFT coverage gaps (M)

**Do**, in order of how often users hit them:
1. Real transforms (`rfftn`, `irfftn`, `rfft2`, `irfft2`) whose half-spectrum axis is
   not the array's last, and `n`/`s` resizing along any axis.
2. `rfft`/`irfft` of long non-power-of-two lengths as true real transforms (today a
   complex transform with a host crop: twice the GPU work).
3. Multi-pass Bluestein, so that lengths with a large prime factor above 2048 run
   natively (today they decline). Bluestein of length n needs a smooth transform of
   length ≥ 2n − 1: reuse the multi-pass power-of-two plan.
4. Non-smooth lengths along axes other than the last.
**Gate.** The exhaustive sweep extended to these cases passes at the same tolerance;
the fall-through counters of the forced-GPU oracle run drop accordingly.

#### B1. Geometric interpolation (L)

**Scope.** `ndimage.map_coordinates`, `affine_transform`, `zoom`, `rotate`, `shift`,
`spline_filter`, `spline_filter1d`.
**Design.**
- Reference: `scipy/ndimage/src/ni_interpolation.c` (`NI_GeometricTransform`,
  `NI_ZoomShift`, `map_coordinate`), `ni_splines.c` (prefilter poles, boundary
  initialisation per mode), and `_interpolation.py` (padding for `grid-*` modes,
  `prefilter`, `output_shape`, how `rotate` and `zoom` reduce to the two C routines).
- Prefilter (orders 2–5): causal and anti-causal recurrences per line. Sequential along
  a line, independent across lines → CPU, threaded per line, in the array's precision.
  Orders 0 and 1 need none.
- Evaluation on the GPU: per output element compute the source coordinate, the
  `order + 1` B-spline weights per axis, map each tap through the boundary mode, and sum
  the tensor product. Up to 6^ndim taps; fine for 2-D and 3-D.
- Coordinates in float32 lose sub-pixel accuracy on long axes (2^-24 relative is
  2.4e-4 px at 4096). Compute the integer part exactly (integer arithmetic on the output
  index for affine maps with a per-row double-precision base passed in a small buffer)
  and only the fractional part in float32.
- CPU engine: threaded over output rows, SciPy's order of operations. Target bit
  identity; if the weight evaluation cannot be reproduced exactly, target ≤ 2 ulp and
  say so.
**Gate.** `scipy/ndimage/tests/test_interpolation.py` passes through the oracle in
strict; `affine_transform` 4096² order 1 and order 3 fast ≥ 30×, strict ≥ 10×;
`zoom` 2048² → 4096² order 3 fast ≥ 30×.

#### B2. Morphology and rank filters (L)

**Scope.** `minimum_filter`, `maximum_filter` (and `1d`), `grey_erosion`,
`grey_dilation`, `grey_opening`, `grey_closing`, `morphological_gradient`,
`white_tophat`, `black_tophat`, `binary_erosion`, `binary_dilation`, `binary_opening`,
`binary_closing`, `median_filter`, `rank_filter`, `percentile_filter`, `signal.medfilt`,
`signal.medfilt2d`, `signal.order_filter`. Needs A6 (integer dtypes).
**Design.**
- Rectangular min/max are separable. CPU: van Herk / Gil-Werman (three comparisons per
  element whatever the window). GPU: direct window scan for windows up to about 32,
  otherwise a doubling scheme (minima over power-of-two windows in log2(w) passes, then
  combine two overlapping windows).
- Min/max are exact in any precision, so the GPU result equals the CPU result on the
  narrowed data; for float64 input in `fast` the values are float32-rounded.
- Arbitrary footprints and non-flat structuring elements: dense tap table as in A3.
- Median and rank on the GPU: selection networks for 3×3 and 5×5; per-thread partial
  sort up to 7×7; larger footprints on the CPU (for uint8 the constant-time histogram
  median; for floats a per-row sliding structure, threaded over rows).
- Binary morphology: uint8 on the GPU with iteration inside one command buffer; for
  `iterations=-1` (until stable) sync every k iterations to test for change.
**Gate.** `scipy/ndimage/tests/test_morphology.py` and the rank-filter tests of
`test_filters.py` pass through the oracle in strict; `median_filter` 4096² 3×3 and 5×5
fast ≥ 40×; `maximum_filter` 4096² size 15 fast ≥ 40×, strict ≥ 10×.

#### B3. Resampling (M)

**Scope.** `signal.upfirdn`, `resample_poly`, `decimate` (FIR path), `resample`.
**Design.** Reference `scipy/signal/_upfirdn.py` and `_upfirdn_apply.pyx`. Polyphase FIR
is parallel in the output index: GPU kernel `out[k] = Σ h[phase(k) + j·up] · x[base(k) − j]`;
CPU: threaded blocks of one long line as well as across lines. `resample` is FFT-based:
compose with the FFT family (forward, spectrum resize, inverse) in one command buffer.
`decimate` with the IIR default goes through `filtfilt` (B6).
**Gate.** `scipy/signal/tests/test_upfirdn.py` and the resample tests pass in strict;
`resample_poly` of 2^24 samples (up 3, down 2) fast ≥ 20×.

#### B4. DCT and DST (M)

**Scope.** `fft.dct`, `idct`, `dst`, `idst`, `dctn`, `idctn`, `dstn`, `idstn`, types
1–4, `norm` and `orthogonalize`.
**Design.** Each type is a real FFT of the same length with a reorder before and a
twiddle multiply after (type 2: even/odd interleave, r2c, multiply by `2·exp(−iπk/2N)`,
take the real part; type 3 is its inverse; types 1 and 4 via the standard length-N or
length-2N reductions). Add the pre- and post-kernels to the FFT family and register the
functions in the uarray backend.
**Gate.** `scipy/fft/tests/test_real_transforms.py` passes through the oracle with
native counts above zero; `dctn` 4096² fast ≥ 20× over SciPy with `workers=-1`.

#### B5. Spectral analysis (L)

**Scope.** `signal.stft`, `istft`, `ShortTimeFFT` (`stft`, `istft`, `spectrogram`),
`spectrogram`, `welch`, `csd`, `coherence`, `periodogram`, `hilbert`, `hilbert2`,
`lombscargle`.
**Design.** One fused pipeline per call: frame gather with padding and detrend → window
multiply → batched r2c FFT (the frames are the batch) → magnitude and scaling → mean or
median over segments. Reference `scipy/signal/_spectral_py.py` and
`_short_time_fft.py`. `lombscargle` is an O(N·M) pairwise sum: a GPU pairwise kernel,
with the phase `ω·t` reduced modulo 2π in double precision on the CPU side per frequency
block, or float32 will lose the phase for long time spans.
**Gate.** `scipy/signal/tests/test_spectral.py` and `test_short_time_fft.py` pass in
strict; `spectrogram` of 2^26 samples (nperseg 1024) fast ≥ 30×; `welch` likewise.

#### B6. IIR filtering — CPU only (M)

**Scope.** `signal.lfilter`, `sosfilt`, `filtfilt`, `sosfiltfilt`, `lfilter_zi` users.
**Design.** A recurrence cannot be parallelised along its own axis; the gain is
threading across channels and lines. Strict: SciPy's exact recurrence (transposed
direct form II for `lfilter`; the section loop of `_sosfilt.pyx`) in C, threaded over
lines. Fast: `vDSP_biquad` where its form applies. Never on the GPU.
**Gate.** `test_signaltools.py` filter classes pass in strict; `sosfiltfilt` on 64
channels × 2^20 samples ≥ 8×.

### Stage C — Phase 3: dense linear algebra

#### C1. GPU BLAS-3 layer (L)

**Goal.** Internal `blas::gemm`, `trsm`, `syrk` that encode into a caller's `Batch`.
**Design.**
- Vendor MLX's steel GEMM kernels into `third_party/mlx-kernels` (extend
  `tools/vendor_mlx_kernels.sh` and the provenance table). Facts from probe E1
  (`probes/e1_gemm/`, `third_party/mlx-ref-0.31.1/mlx/backend/metal/matmul.cpp`): kernel
  `steel_gemm_fused` with block sizes bm 64, bn 64, bk 16 and warps wm 2, wn 2; buffers
  0 = A, 1 = B, 2 = C (addend), 3 = D (output), 4 = `GEMMParams`, 5 = `GEMMAddMMParams`;
  function constants at indices 10, 100, 110, 200, 201, 202 (transposes, alignment
  flags, axpby, block shape). Read `matmul.cpp` for how MLX chooses the tile grid and
  the swizzle; reproduce those choices.
- MLX matrices are row-major and LAPACK's are column-major. A column-major matrix is the
  row-major transpose, so `C = A·B` in column-major is `Cᵀ = Bᵀ·Aᵀ` in row-major: swap
  the operands rather than copying.
- TRSM: blocked. Invert the diagonal blocks (64–128) on the CPU, then each block step is
  a GEMM. SYRK: GEMM first (twice the work, correct), a triangle-only tile grid later.
- Operands are views (buffer, offset, rows, columns, leading dimension) so that
  factorisations update sub-matrices in place.
**Gate.** GEMM ≥ 90% of MLX's rate at n = 2048–8192 through `blas::gemm`; TRSM with 4096
right-hand sides against an 8192 triangle ≥ 4× `strsm`; results agree with Accelerate to
float32 rounding.

#### C2. `solve`, `lu_factor`, `lu_solve`, `lu`, `inv`, `det` (L)

**Design** (from probe E6, `probes/e6_hybrid_lu/`):
- Float32 right-looking blocked LU, block 256. The panel is factored on the CPU with
  Accelerate (`sgetrf` on the panel, in the shared buffer: unified memory means the CPU
  works on the same bytes). The GPU applies row swaps, the triangular solve and the
  trailing GEMM. **Look-ahead**: the next panel is updated and factored on the CPU while
  the GPU updates the rest. Finish on the CPU when the remainder is ≤ 1024.
- The CPU and the GPU may touch the same buffer concurrently only in disjoint regions;
  keep the partition explicit in the code.
- Refinement: `r = b − A·x` in float64 (threaded), correction with the float32 factors,
  until the backward error stops improving or reaches about `n·eps64`; at most five
  steps. If it does not converge (roughly κ > 1e7), redo with `dgetrf` + `dgetrs` and
  record the reason.
- SciPy's `solve` warns (`LinAlgWarning`) from a reciprocal condition estimate and
  raises `LinAlgError` for exact singularity. Preserve both: estimate with `sgecon` on
  the float32 factors; if that estimate is below a float32-scale threshold, take the
  float64 path, which then reports exactly as SciPy does.
- `assume_a`: `gen` → LU; `pos` → C3; `sym`/`her` → LU in `fast`, SciPy in `strict`;
  banded and tridiagonal → SciPy. Honour `lower`, `transposed`, `overwrite_*` (as
  permissions only), `check_finite`. SciPy ≥ 1.15 accepts stacks of matrices: see C5.
- Strict: recursive LU over `dgemm` gave only 2×; ship it only if it is ≥ 1.5× at the
  size in question, else decline.
**Gate.** `solve` n = 8192 float64 `fast` ≤ 250 ms with backward error ≤ 1e-14 for
κ ≤ 1e4 (stock 1467 ms); break-even with SciPy at n ≤ 1000; the `solve`/`lu`/`inv`/`det`
classes of `scipy/linalg/tests/test_basic.py` and `test_decomp_lu.py` pass through the
oracle in strict *and* in `refined` at SciPy's tolerances.

#### C3. Cholesky (M)

**Design.** Fast: hybrid float32 (CPU `spotrf` on diagonal blocks, GPU TRSM and rank-k
update) plus refinement. Strict: recursive Cholesky over `dgemm` (E7: 5.1× at 8192,
1.55× at 2048). Recursion: factor A11; `A21 ← A21·L11⁻ᵀ` (`dtrsm`); `A22 ← A22 − A21·A21ᵀ`
with **`dgemm`, not `dsyrk`**; recurse on A22; leaves ≤ 256 use `dpotrf`. Report a
non-positive-definite matrix with LAPACK's leading-minor index so SciPy's message is
reproduced.
**Gate.** `cholesky` n = 8192 strict ≥ 4×, fast ≥ 8×; `test_decomp_cholesky.py` passes.

#### C4. QR and least squares (L)

**Design.** Blocked Householder: panel on the CPU (`sgeqrf` + `slarft`), trailing update
`C ← C − V·Tᵀ·(Vᵀ·C)` as two GEMMs on the GPU. For tall, well-conditioned matrices,
CholeskyQR2 is GEMM-bound and very fast on the GPU, but only valid for κ below about
1e3 in float32; guard it with a condition estimate and fall back to Householder.
`lstsq`'s default driver (`gelsd`) returns singular values and handles rank deficiency,
which a QR path cannot reproduce: accelerate `lapack_driver='gelsy'` semantics natively
and leave the default to SciPy, unless the user decides otherwise (9.3).
**Gate.** `qr` 8192 × 2048 fast ≥ 5×; `test_decomp.py` QR classes pass.

#### C5. Stacked small matrices (M)

**Design.** For inputs of shape `(B, n, n)`: CPU first (thread the batch over Accelerate
per matrix: an easy 10–16×), then a GPU kernel with one matrix per thread for n ≤ 32
(partial pivoting in float32, refinement on the CPU). Check which functions batch in the
installed SciPy (`scipy/linalg/src/batched_linalg/` in the checkout) and match their
broadcasting and error reporting.
**Gate.** `solve` of 10^5 systems of size 8 ≥ 10× SciPy.

#### C6. `expm` (M)

**Design.** SciPy's scaling-and-squaring (reference `scipy/linalg/_matfuncs.py` and its
C core) is GEMM-bound: Padé evaluation, one solve, s squarings. Run the same algorithm
with the GPU BLAS in `fast`; choose the Padé order and s from the float64 norm estimate
as SciPy does; refine nothing (there is no cheap residual), so compare against the
float64 result in tests and document the bound.
**Gate.** `expm` n = 2048 fast ≥ 5×; `test_matfuncs.py` expm tests pass in strict.

#### C7. Eigenvalues and SVD — measure first (L)

**Do.** A probe: Accelerate float32 against float64 drivers (`syevd`, `gesdd`) at
n = 1024–8192; Ogita–Aishima refinement of a float32 eigendecomposition to float64
accuracy (GEMM-bound, about two iterations) against `dsyevd`.
**Gate.** Implement `eigh`/`eigvalsh`/`svd`/`svdvals` fast paths only where the probe
shows ≥ 2×. A two-stage GPU eigensolver is out of scope until profiling justifies it.

#### C8. Remaining GEMM-bound functions (S–M)

`solve_triangular` with many right-hand sides (GPU TRSM), `inv` (LU + TRSM),
`pinv` only if C7 delivers. Schur-type functions (`eig`, `schur`, `sqrtm`, `logm`,
`funm`) stay with SciPy.

### Stage D — Phase 4: sparse

#### D1. Types and SpMV (L)

**Design.**
- `scimetal.sparse.csr_array` / `csr_matrix` as **subclasses** of SciPy's classes that
  override the matrix-vector and matrix-matrix products (`_matmul_vector`,
  `_matmul_multivector`), so `isinstance` checks inside SciPy keep working and nothing
  in SciPy is patched. `scimetal.sparse.accelerate(A)` re-types an existing matrix
  without copying its arrays.
- CPU SpMV: threaded CSR with rows partitioned by non-zero count; same per-row
  summation order as `scipy/sparse/sparsetools/csr.h` (`csr_matvec`), so strict is
  bit-identical.
- GPU SpMV: SELL-C-σ (slice height 32, sorting window a few thousand rows), float32.
  SciPy sparse arrays are mutable in place and a stale device copy would be a silent
  wrong answer, so build the device copy **per solver call**, not per matrix. A
  standalone `A @ x` uses the CPU engine.
**Gate.** SpMV on a 3-D Poisson matrix with 10^7 rows: CPU ≥ 8× SciPy; GPU (inside a
solver) ≥ 3× the CPU engine.

#### D2. Krylov solvers (L)

**Scope.** `cg`, `gmres`, `bicgstab`, `minres`, `lsmr`, `lsqr`, `lgmres`, `tfqmr`.
**Design.** Vectors live on the device; vector updates and dot products are fused
kernels; scalars (α, β) stay in a small device buffer so that k iterations fit in one
command buffer and the host reads the residual every k iterations only. Mixed precision:
an outer float64 residual loop on the CPU around a float32 inner solve on the GPU, until
SciPy's own `rtol`/`atol` test passes in float64. A user `callback`, a `LinearOperator`
or a Python preconditioner forces the CPU engine (threaded SpMV, SciPy's iteration).
Return `(x, info)` with SciPy's meaning; iteration counts may differ.
**Gate.** `cg` on the 10^7-row Poisson matrix to `rtol=1e-8` ≥ 10× SciPy;
`scipy/sparse/linalg/_isolve/tests` pass through the oracle in strict.

#### D3. Eigen and singular solvers (M)

`lobpcg` (SpMM plus GEMM-bound Rayleigh–Ritz: GPU), `eigsh`/`svds` (ARPACK's reverse
communication stays; the gain is the threaded SpMV).

#### D4. Direct solvers (M)

`spsolve`, `factorized` for symmetric systems through Accelerate's Sparse Solvers
(detect symmetry in O(nnz), try Cholesky, fall back to LDLᵀ). Check the SDK header
`Accelerate/Sparse/Solve.h` for an LU factorisation before deciding that unsymmetric
systems stay with SuperLU.
**Gate.** `spsolve` on a 2-D Poisson system with 10^6 unknowns ≥ 3× SuperLU.

#### D5. Construction and conversion (M)

COO → CSR with duplicate summing, CSR ↔ CSC, index sorting, `A + B`. metal-arrow has
radix sort, scans and COO → CSR primitives (`ma_runtime.h` lines 232–354, deliberately
left out of the vendored header); vendor them with `tools/vendor_runtime.sh` when this
starts.

### Stage E — Phase 5: the rest, by measured gain

Probe each before building; keep only what shows ≥ 3×.

- `spatial.distance.cdist` / `pdist`: Euclidean-type metrics through GEMM, the others
  through a pairwise kernel. Brute-force k-nearest neighbours on the GPU.
- `stats.gaussian_kde.evaluate`, bootstrap and permutation resampling.
- `cluster.vq.kmeans`, `vq`.
- `special`: threaded CPU evaluation; a curated float32 GPU tier for the common
  functions (`erf`, `gammaln`, Bessel `j0`/`j1`), each with a stated error bound.
- `interpolate` evaluation (`RegularGridInterpolator`, `BSpline`, `RBFInterpolator`).
- Ensembles of ODE or optimisation problems with MLX autodiff (the only place MLX is a
  runtime dependency of a feature).

Not on the GPU, ever: sequential recurrences, KD-trees, Qhull, graph traversals, sparse
factorisation, Schur-type algorithms, callback-driven solvers.

### Stage X — cross-cutting

#### X1. Cost-model router and `calibrate()` (M)
Replace per-family constants with `t_call + n_cb·t_cb + work/rate + bytes/bandwidth`,
a table keyed by GPU architecture and performance-core count, and
`scimetal.calibrate()` that measures and stores a table for the current machine. Do
this when a second machine enters the picture, not before.

#### X2. MLX bridge (M)
Tier 1 (pure Python, any MLX ≥ 0.31): accept `mx.array`, evaluate it, view it as NumPy
without a copy, run the core, return `mx.array`. Tier 2 (C++ with nanobind **2.10.2**,
exact MLX pin): one generic MLX primitive that encodes our plans into MLX's lazy command
buffer, so there is no extra synchronisation (probe E8, `probes/e8_mlx_primitive/`).

#### X3. Wrap cache (S)
Inputs that come back unchanged (the same array filtered with several parameters) can be
wrapped once and reused. Only for arrays the caller marks read-only, or with an explicit
`scimetal.pin(array)`: an unmarked array can change under us.

#### X4. Memory behaviour (S)
The context's limit is half of physical memory (`SCIMETAL_MEMORY_LIMIT_GB`). Add
`scimetal.trim()` (calls `sm_context_trim`) and an idle trim. Document that a kept
result holds a page-aligned Metal buffer.

#### X5. Thread safety (S)
ctypes releases the GIL during native calls, so the core is entered concurrently. Add a
stress test (many Python threads, mixed families, both engines) and run it under the
thread sanitizer once that works (9.2).

#### X6. Generalising beyond the M1 Ultra (M)
Other chips need thresholds (X1), a look at threadgroup shapes, and MLX's newer GEMM
variants. Metal 4 paths need a newer Xcode and are deferred.

---

## 8. Order of execution

```
A0  shared-code clean-up
A1  integration, quiet benchmarks, FFT host path
A5  guard rails                      (before more fast-mode surface exists)
A2  signal convolution   ─┐
A3  ndimage dense        ─┤ parallel (disjoint families; A2 and A3 share the dense
A4  ndimage declines     ─┘ stencil: build it in A3, reuse in A2)
A6  integer dtypes
A7  CPU FFT probe
A8  packaging            → milestone M1: Phase 1 gate met, wheel installable
B0  FFT coverage gaps
B4  DCT/DST   B3 resampling   B1 interpolation   B2 morphology   (parallel, by family)
B5  spectral  (needs B4's FFT hooks)   B6 IIR
                         → milestone M2: signal/FFT/image breadth
C1  GPU BLAS-3
C2  solve/LU   C3 Cholesky   (parallel after C1)
C5  batched    C4 QR         C6 expm     C7 probe, then maybe eigh/svd   C8
                         → milestone M3: dense linear algebra
D1 → D2 → D3;  D4 and D5 independent
                         → milestone M4: sparse
E   by measured gain;  X items when their trigger occurs
```

At every milestone: run the full ladder, re-run all benchmarks alone, update section 5
of this document, and report to the user with the gates that passed and failed.

---

## 9. Defects, unverified claims, open questions

### 9.1 Known gaps in what exists

- `ndimage` filters: `output` of a different float dtype declines; integer, boolean and
  complex inputs decline; arrays of 2^31 elements or more are untested (largest run
  512³); windows longer than the axis are slow on the GPU; the GPU uniform filter does
  not reproduce SciPy's propagation of a NaN along the rest of a line (the CPU engine
  does); the CPU uniform filter along the innermost axis is a scalar recurrence.
- `ndimage` fast mode has no float32 range check (A5).
- No `PrecisionWarning` exists yet (A5).
- FFT has no native CPU engine (A7); its coverage gaps are listed in 5.2 (B0); its
  threaded fall-through is not bit-identical to single-worker SciPy; against SciPy with
  16 workers its gain is 1.1–4.0×, and at 2^16 complex128 it loses (0.52×); the 2^31-point
  limit was exercised only through `sm_fft_plan`, never at real size; under shader
  validation plans use smaller threadgroups, so the validated test covers a reduced set
  of Rader and Bluestein sizes.
- The Phase 1 gate "SciPy's FFT tests pass in strict" is met only trivially (zero native
  calls). SciPy's FFT suite cannot validate a single-precision engine at its
  double-precision tolerances.
- `signal` convolution, the last third of the Phase 1 slice, is not started (A2).
- `signal`, `linalg`, `sparse` namespaces are pure fall-through.
- The MLX bridge does not exist (X2).
- Bit identity of the strict CPU filters is tied to the SciPy 1.18.1 arm64 wheel.

### 9.2 Unverified or environment problems

- Benchmarks in section 5.3 were re-measured alone on a quiet machine (2026-11-19,
  load average ~2.0–2.9; pre/post `uptime` recorded in `bench/results/m1-ultra/*.json`).
- GPU FFT time is bimodal with the GPU's clock state (2^24 complex64: ~2.51–3.55 ms of
  GPU time plus ~1.32 ms host/sync overhead from Stage A1 profiling).
- AddressSanitizer and ThreadSanitizer builds reportedly do not run usefully on this
  machine (ASan hangs or crawls, TSan crashes at start-up, even for the smallest test).
  Not investigated. UBSan alone and Guard Malloc were reported clean for ndimage.
- The cause of the first-use cost of wrapped host memory (E2) is unknown; only its size
  was measured.
- Thresholds and threadgroup shapes were tuned on the inputs they are reported for.

### 9.3 Questions for the user

1. **Remote.** The repository is local only. Push it somewhere, and where?
2. **`lstsq`.** The default driver returns singular values, which a fast QR path cannot.
   Accelerate only `gelsy`, or change the default in `scimetal.linalg.lstsq`?
3. **Warning by default.** Should the one-time `PrecisionWarning` be on by default under
   the speed-first policy?
4. **Integer images in fast mode.** Compute uint8/uint16 filters in float32 on the GPU
   (exact for these ranges) by default?
5. **Non-SciPy extras.** NumPy's `A @ B` cannot be intercepted. Offer
   `scimetal.extras.matmul` (GPU, 5× `sgemm`, 20× `dgemm`) even though it is not SciPy
   API?
6. **Threaded fall-through.** Declined FFT calls are handed to SciPy with all cores,
   which is faster but not bit-identical to stock SciPy for some shapes. Keep it on by
   default, or only outside `strict`?
7. **Sharing.** When the audience widens beyond the user's machines: licence, name on
   PyPI, supported chips.

---

## 10. Reference

### 10.1 Commands

```bash
cd /Users/tunguz/Programming/SciMetal && export PATH=$PWD/.venv/bin:$PATH

# build and native tests
cmake --preset metal && cmake --build --preset metal && ctest --preset metal
cmake --preset cpu-only && cmake --build --preset cpu-only && ctest --preset cpu-only

# one family in its own tree
cmake -S . -B .build/ndimage -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DMS_FAMILIES=ndimage
cmake --build .build/ndimage && ctest --test-dir .build/ndimage --output-on-failure
SCIMETAL_LIBRARY=$PWD/.build/ndimage/libscimetal.dylib .venv/bin/python -m pytest tests/python/test_ndimage_filters.py -q

# Python tests (source tree; finds .build/metal/libscimetal.dylib)
.venv/bin/python -m pytest tests/python -q -p no:warnings

# oracle: SciPy's suite against our routes
PYTHONPATH=python:tests/oracle .venv/bin/python -m pytest -p scimetal_oracle \
    --pyargs scipy.ndimage.tests.test_filters -q -p no:cacheprovider
SCIMETAL_ORACLE_PRECISION=fast SCIMETAL_ORACLE_DEVICE=gpu PYTHONPATH=python:tests/oracle \
    .venv/bin/python -m pytest -p scimetal_oracle --pyargs scipy.fft.tests.test_basic -q -p no:cacheprovider

# benchmarks (run alone)
uptime
.venv/bin/python bench/ndimage_filters.py            # --quick, --calibrate
.venv/bin/python bench/fft.py                        # --quick, --crossover

# probes
probes/run_all.sh                                    # or: probes/run_all.sh e4_stencil

# vendoring
tools/vendor_runtime.sh                              # then re-apply src/runtime/PROVENANCE.md
tools/vendor_mlx_kernels.sh --check
```

Environment variables: `SCIMETAL_LIBRARY` (path to `libscimetal.dylib`),
`SCIMETAL_PRECISION`, `SCIMETAL_DEVICE`, `SCIMETAL_SMALL_INPUTS` (`scipy` | `native`),
`SCIMETAL_MEMORY_LIMIT_GB`, `SCIMETAL_ORACLE_PRECISION`, `SCIMETAL_ORACLE_DEVICE`,
`SCIMETAL_FFT_THREADED_FALLBACK`, `SCIMETAL_FFT_EXHAUSTIVE`, `SCIMETAL_FFT_PROFILE`.

### 10.2 Python API

```python
import scimetal
import scimetal.ndimage as ndimage        # drop-in for scipy.ndimage
import scimetal.fft                        # drop-in for scipy.fft
scimetal.fft.register()                    # make scipy.fft itself use the native routes
scimetal.fft.prepare(shape, dtype, ...)    # pre-compile and cache GPU FFT pipelines

scimetal.set_precision("fast" | "refined" | "strict");  with scimetal.precision("strict"): ...
scimetal.set_device("auto" | "cpu" | "gpu");            with scimetal.device("gpu"): ...
scimetal.set_small_inputs("scipy" | "native")
scimetal.explain()        # 'ndimage.gaussian_filter: engine=gpu, computed in float32, policy=fast'
scimetal.last_route()     # Route(function, engine, compute_dtype, policy, reason)
scimetal.accelerated()    # {'fft': [...], 'ndimage': [...]}
```

### 10.3 Internal C++ API (namespace `sm::detail`)

```cpp
// error handling: every C entry point
return sm::detail::api(context, [&] { ...; throw Error(SM_EINVAL, "message"); });

// arrays (src/core/array.hpp)
array::Shape shape = array::checked_shape(input);          // validates the descriptor
array::check_options(options);
BufferPtr out = array::allocate(context, bytes);           // pooled; write every byte
uint8_t* p = mutable_data(out.get());
array::convert(src, src_dtype, dst, dst_dtype, elements);  // threaded, bit-exact
BufferPtr single = array::load_single(context, *input, shape);
array::publish(result, std::move(out), dtype, shape);
array::report(info, engine, compute_dtype);

// routing (src/core/router.hpp)
constexpr router::Family family{"family.name", gpu_min_work};
router::Decision d = router::decide(context, family, *options, input_dtype, work);

// threads (include/sm/detail/parallel.hpp): performance cores only
parallel::parallel_for(count, min_items_per_lane, [&](uint64_t begin, uint64_t end) { ... });

// GPU (src/runtime/metal_compute.hpp)
GpuOperationScope operation(context, true);
gpu::Batch batch(context);
gpu::Grid grid;  grid.groups[0] = ...; grid.threads[0] = ...;        // up to 3 axes
batch.dispatch("kernel_name", params, {gpu::bind(a, context), gpu::bind(b, context)}, grid);
//   params struct at buffer 0, buffers from index 1
batch.dispatch_indexed("kernel", &specialization, {{3, binding}}, {{2, &value, sizeof value}}, grid);
//   explicit indices and function constants: what MLX's kernels need
uint32_t capacity = gpu::max_threads_per_group("kernel_name");
gpu::warm_pipeline("kernel_name");
batch.commit_and_wait();
operation.complete();
```

Read these two files before writing a GPU engine: `src/ndimage/filters_gpu.cpp` (155
lines, the simplest complete engine) and `tests/native/compute_smoke_test.cpp`.

### 10.4 SciPy sources to read (in `/Users/tunguz/Programming/scipy`)

| Area | Files |
|---|---|
| ndimage filters | `scipy/ndimage/src/ni_filters.c`, `ni_support.c`, `ni_support.h`, `scipy/ndimage/_filters.py` |
| ndimage interpolation | `scipy/ndimage/src/ni_interpolation.c`, `ni_splines.c`, `scipy/ndimage/_interpolation.py` |
| ndimage morphology | `scipy/ndimage/src/ni_morphology.c`, `scipy/ndimage/_morphology.py` |
| signal convolution | `scipy/signal/_signaltools.py`, `scipy/signal/_correlate_nd.cc` |
| signal resampling, spectral | `scipy/signal/_upfirdn.py`, `_upfirdn_apply.pyx`, `_spectral_py.py`, `_short_time_fft.py`, `_sosfilt.pyx` |
| fft | `scipy/fft/_backend.py`, `_basic.py`, `_realtransforms.py`, `_pocketfft/` |
| linalg | `scipy/linalg/_basic.py`, `_decomp_lu.py`, `_decomp_cholesky.py`, `_decomp_qr.py`, `_matfuncs.py`, `scipy/linalg/src/batched_linalg/` |
| sparse | `scipy/sparse/sparsetools/csr.h`, `scipy/sparse/_compressed.py`, `scipy/sparse/linalg/_isolve/`, `_dsolve/` |
| benchmarks to mirror | `benchmarks/benchmarks/{signal,signal_filtering,fft_basic,ndimage_interpolation,linalg,sparse,sparse_linalg_solve}.py` |

The installed SciPy (1.18.1) is what the oracle runs and what strict must match; the
checkout (2.0.0.dev0) is for reading. Where they differ, the installed wheel wins.

### 10.5 metal-arrow material worth reusing (read-only)

`/Users/tunguz/Programming/metal-arrow`: `tests/native/route_harness.hpp` (CPU route
against forced-GPU route harness), `include/ma_runtime.h` lines 232–354 (radix sort,
scans, COO → CSR), ADR 0009 (command-buffer fusion), ADR 0010 (no monkeypatching).
