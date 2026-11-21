# Phase 0 probes — results

Eight throwaway probes that test the design before the production code is written.
Numbers below are from `probes/run_all.sh` on 2026-11-18: each probe run alone, one
after another, on a Mac Studio M1 Ultra (16 P-cores, 64-core GPU, 128 GB, macOS 26.5),
load average 2–6. Full results: `probes/results/<probe>.json` (console output in `.log`).

Re-run everything: `probes/run_all.sh` (about 3.5 minutes), or one probe:
`probes/run_all.sh e4_stencil`.

## Gates

| # | Probe | Gate | Result | Verdict |
|---|---|---|---|---|
| E1 | MLX steel GEMM in our own metallib + runtime | ≥ 80% of MLX's rate at n = 2048–8192; tiny call ≤ 0.3 ms | 102% / 99% / 93% of MLX (10.3 / 16.1 / 11.5 TFLOPS); tiny call 0.25 ms | **Pass** |
| E2 | No-copy wrap of NumPy pages | Wrap cost ≤ 5% of memcpy at ≥ 64 MB | The wrap call is ≤ 1% of a memcpy, but the first GPU use of a fresh wrap costs ~11–12 ms/GB, 44–50% of a single-threaded memcpy | **Not met** |
| E3 | float64 ⇄ float32 conversion | (no gate) bit-exact; throughput | CPU 16 threads 164–179 GB/s; GPU in-kernel 317–464 GB/s on warm buffers, ~50 GB/s on freshly wrapped ones; all bit-identical to `numpy.astype` | — |
| E4 | Gaussian filter 4096², float64 in → float64 out | fast ≥ 20× SciPy; strict CPU ≥ 8× | fast 133× / 140× (σ = 3 / 10); strict 49× / 41×, bit-identical to SciPy | **Pass** |
| E5 | MLX FFT kernels in our runtime | ≥ 80% of MLX's speed at 2^24 | 2.47 ms, correct, vs MLX 3.12 ms, **wrong result**; same plan as MLX runs at 96% of its speed | **Pass** (see note) |
| E6 | Hybrid float32 LU + float64 refinement | n = 8192 ≤ 250 ms; backward error ≤ 1e-14 for κ ≤ 1e4; break-even n ≤ 2000 | 180 ms (κ = 1e2) / 204 ms (κ = 1e4); backward error ≤ 3.4e-16; break-even n = 512 | **Pass** |
| E7 | Recursive factorizations on Accelerate GEMM | ≥ 4× over LAPACK at 8192, ≥ 1.5× at 2048 | Cholesky 5.1× / 1.55×; LU only 1.98× / 1.05× | Cholesky **pass**, LU **fail** |
| E8 | Out-of-tree MLX primitive on our metallib | ≤ 30 µs marginal cost in a lazy graph, no extra commit | −1.3 to +0.5 µs vs a comparable built-in op; +12 µs vs no op at all (a built-in costs the same); no extra commit or sync | **Pass** |
| A7 | Accelerate vDSP CPU FFT (`vDSP_fft_zop` / `vDSP_DFT_Execute`) vs `scipy.fft` | ≥ 1.5× end-to-end interleaved over a meaningful range | Batches (16t vs `workers=16`): 0.27–1.20×; 1-D ≥ 2^17 (`complex64`) / ≥ 2^16 (`complex128`) vs `w16`: 0.14–0.89×; ≤ 16384 (`complex64`) with Python+alloc+handoff (~15–20 µs): ≤ 1.4×; narrow 49152–65536 1-D band (~1.7–1.8×) breaks `strict` bit-identity and limits to `f·2^n`, `f∈{1,3,5,15}` | **Negative (keep SciPy fall-through)** |

## Headline numbers

| Workload | Stock | SciMetal probe | Speed-up |
|---|---|---|---|
| `gaussian_filter`, 4096², σ = 3, float64 in/out | SciPy 385 ms | fast 2.9 ms (4.1 ms into a plain NumPy array; 7.3 ms if every result is kept); strict 7.9 ms | 133× (93×; 53×); strict 49× |
| `gaussian_filter`, 4096², σ = 10, float64 in/out | SciPy 738 ms | fast 5.3 ms; strict 18.2 ms | 140×; strict 41× |
| FFT, complex64, n = 2^24 | `scipy.fft` 300 ms | 2.47 ms (relative error 6e-7) | 122× |
| FFT, complex64, n = 2^20 | `scipy.fft` 10.0 ms | 0.39 ms | 25× |
| FFT, complex64, 4096 × 4096 along last axis | `scipy.fft` 31.3 ms (1 worker), 3.4 ms (16) | 0.69 ms | 45× / 5× |
| `solve`, n = 8192, float64, κ = 1e4 | `scipy.linalg.solve` 1467 ms; `dgetrf`+`dgetrs` 1628 ms | 204 ms, backward error 4e-17 | 7–8× |
| `solve`, n = 2048, float64 | `scipy.linalg.solve` 52 ms; `dgetrf`+`dgetrs` 28 ms | 15–16 ms | 3.3× / 1.8× |
| Matmul, float32, n = 4096 | Accelerate `sgemm` 3.0 TFLOPS (`dgemm` 0.8) | 16.1 TFLOPS | 5.4× (20× vs float64) |
| Cholesky, float64, n = 8192 (CPU only, exact) | `dpotrf` 2522 ms | 497 ms | 5.1× |
| LU, float64, n = 8192 (CPU only, exact) | `dgetrf` 1567 ms | 790 ms | 2.0× |

## What changes in the design

1. **Copy in, pool, and hand buffers out — do not wrap one-shot inputs.** Fresh no-copy
   wraps pay ~12 ms/GB at first GPU use, more than a 16-thread narrow or copy into a pooled
   buffer (5–6 ms/GB). Inputs are copied/narrowed into pooled shared buffers on the P-cores;
   outputs are pooled buffers exposed to NumPy without a copy. Wrapping stays for inputs
   that come back (a wrap cache) and for MLX arrays.
2. **Never read float64 inside a multi-tap kernel.** Fusing the narrow into the first
   filter pass converts once per tap (4× slower pass). Narrow once, on the CPU or in its own
   GPU pass; fuse only the widen into the last pass, where it is free.
3. **FFT uses our own plan.** MLX 0.31.1's GPU FFT raises at 2^21–2^22 and returns wrong
   answers at 2^23–2^24. The vendored kernels are sound; the large-n plan (≤ 32-point
   passes, one command buffer) is ours. The GPU wins from about 2^17–2^18 points; below
   that, vDSP.
4. **Hybrid LU is CPU-bound.** The GPU is busy 45 ms of a 127 ms factorization; the panel
   (73 ms) dominates. Next lever: a hand-written tall-panel kernel, and row swaps + TRSM on
   the GPU in the update's command buffer.
5. **`strict` linear algebra gains are uneven.** Exact float64 Cholesky gets 5× (Accelerate's
   `dsyrk` collapses on rank-k updates; a GEMM-based update avoids it). Exact LU gets only
   2×: `dgetrf` is at 1.5 s natively, not the 3.15 s an earlier Python-level probe showed.
   Use `getrf`+`getrs` rather than the `gesv` drivers, which are 1.3–3× slower in Accelerate.
6. **Mixed precision pays even without the GPU**: float32 factor + float64 refinement on
   the CPU beats `dgesv` from n = 512.
7. **MLX bridge is cheap, but pin nanobind** to the version MLX was built with (2.10.2 for
   MLX 0.31.1); 3.x cannot share `mx.array`.
8. **Fixed GPU cost** is 0.25 ms per synchronous command buffer (0.035–0.055 ms per call
   when several are pipelined), so inputs below roughly 1e5 elements stay on the CPU.

## Caveats

- Speed-ups are for one machine and one input shape per workload; thresholds and
  threadgroup shapes were tuned on the same inputs they are reported for.
- E4's 133–140× assumes output buffers recycle through the pool. Keeping every result
  costs a fresh buffer per call (53–65×).
- The strict Gaussian path is bit-identical to this SciPy wheel (1.18.1, arm64) because it
  replicates that binary's summation order; a plain IEEE order lands within 1 ulp.
- E6 covers one right-hand side, power-of-two n in the timed runs, and κ ≤ 1e4; there is no
  float64 fallback yet for ill-conditioned systems.
- The cause of the first-use cost of wrapped memory (E2) is unverified; only its size and
  shape were measured.

## A7 CPU FFT probe (follow-up)

Run alone on the same M1 Ultra host via `probes/run_all.sh a7_cpu_fft` (`uptime` load `2.92 2.74 2.31`; full tables in `probes/results/a7_cpu_fft.{json,log}`):

- **Batches (`batch > 1`, 16 P-core threads vs `scipy.fft.fft(..., workers=16)`)**: end-to-end interleaved vDSP (`vDSP_ctoz[D]` + transform + `vDSP_ztoc[D]`) measures **0.27×–1.20×** across `(64, 256)`, `(256, 1024)`, `(64, 4096)`, `(256, 4096)`, `(1024, 4096)`, `(1024, 1024)`, `(256, 3072)`, and `(64, 15360)` in both `complex64` and `complex128` — never reaching the 1.5× gate (even split-only vDSP tops out at ~1.3–1.4×).
- **Large 1-D (`≥ 2^17` `complex64`, `≥ 2^16` `complex128`)**: single-threaded vDSP trails 16-worker pocketfft (`0.14×–0.89×`), and Metal GPU already handles `complex64` cleanly in `fast`/`auto` above ~`2^17`.
- **Small 1-D (`≤ 16384`)**: realistic Python call + output allocation + `SciMetal` ctypes hand-off (`~15–20 µs`, which is also why `auto_min_points = 12461`) erases raw C++ microsecond gains (`≤ 1.4×`, slower than SciPy at `≤ 4096`).
- **Narrow band**: only 1-D `complex64` `f·2^n` around `49152–65536` (`~0.14 ms` vs `0.25 ms`) hits ~1.7–1.8× end-to-end, while supporting only `f ∈ {1, 3, 5, 15}` and introducing ~1e-7 rounding differences that break `strict` bit-identity with SciPy.
- **Decision**: Negative result for a native CPU FFT engine; keep the SciPy (`pocketfft`, threaded `workers=cores` under `fast`/`refined` where beneficial, single-worker under `strict`) fall-through.
