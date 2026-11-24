# SciMetal

[![PyPI version](https://img.shields.io/pypi/v/scimetal)](https://pypi.org/project/scimetal/)

SciPy's API, accelerated on Apple Silicon: Metal compute kernels and a multithreaded
Accelerate/NEON CPU engine behind a versioned C ABI (`SM_ARRAY_ABI_VERSION` 3), with
Python namespaces mirroring `scipy`.

Three families have native routes today (`scimetal.accelerated()`). Every other function
and declined call delegates transparently to SciPy with caller arguments untouched.
Design rationale, probes, and milestones live in [PLAN.md](PLAN.md).

## Install from PyPI

The wheel supports Apple Silicon Macs running macOS 15 or later with Python 3.11
or later. pip installs NumPy and SciPy as dependencies.

```sh
pip install scimetal
```

## Requirements

- Apple Silicon (`arm64`), macOS 15 or later
- Xcode / Metal toolchain, CMake, Ninja (for building from source)
- Python 3.11+, NumPy 2+, SciPy 1.15+

## Build, install, and verify

### CMake presets (`metal` and `cpu-only`)

```bash
export PATH=$PWD/.venv/bin:$PATH
cmake --preset metal && cmake --build --preset metal && ctest --preset metal --output-on-failure
cmake --preset cpu-only && cmake --build --preset cpu-only && ctest --preset cpu-only --output-on-failure
```

- The `metal` preset builds `.build/metal/libscimetal.dylib` with Metal GPU kernels and
  runs all 22 native tests (including `_validated` runs under Metal API & shader validation).
- The `cpu-only` preset builds `.build/cpu-only/libscimetal.dylib` (`SM_ENABLE_METAL=OFF`)
  and runs all 12 CPU-only native tests.

### Wheel build

```bash
MACOSX_DEPLOYMENT_TARGET=15.0 .venv/bin/python -m pip wheel . -w dist --no-deps --no-build-isolation
```

This builds `dist/scimetal-<version>-py3-none-macosx_15_0_arm64.whl` bundling
`libscimetal.dylib`.

### Publish a release

Update the version in `pyproject.toml`, `python/scimetal/__init__.py`, and
`CMakeLists.txt` and merge the version bump into `main`. Push a matching tag
such as `v0.2.0`. The [release workflow](.github/workflows/publish-pypi.yml)
builds and tests the Apple Silicon wheel, then uploads it to PyPI using
Trusted Publishing. A manual run in GitHub Actions builds and tests the wheel
without publishing it.

### Verification ladder

```bash
./tools/check.sh           # C++ CTests (metal + cpu-only) and Python suites for both builds
./tools/check.sh --wheel   # also builds and tests a fresh-venv wheel install
```

Run SciPy's own test suites through the oracle directly:

```bash
PYTHONPATH=python:tests/oracle .venv/bin/python -m pytest -p scimetal_oracle --pyargs scipy.ndimage.tests.test_filters -q
PYTHONPATH=python:tests/oracle .venv/bin/python -m pytest -p scimetal_oracle --pyargs scipy.ndimage.tests.test_morphology -q
PYTHONPATH=python:tests/oracle .venv/bin/python -m pytest -p scimetal_oracle --pyargs scipy.fft.tests.test_basic -q
PYTHONPATH=python:tests/oracle .venv/bin/python -m pytest -p scimetal_oracle --pyargs scipy.signal.tests.test_signaltools -k "Convolve or Correlate or choose_conv_method" -q
```

## Usage

```python
import numpy as np
import scipy.fft
import scimetal
import scimetal.ndimage as ndimage          # drop-in replacement for scipy.ndimage
import scimetal.signal as signal            # drop-in replacement for scipy.signal
import scimetal.fft as mfft                 # drop-in replacement for scipy.fft

# Inspect native coverage
print(scimetal.accelerated())

# fast policy (default): float64 & representable integer images use float32 on Metal
image = np.random.default_rng(0).standard_normal((2048, 2048))
smoothed = ndimage.gaussian_filter(image, 3.0)
print(scimetal.explain())                   # engine, compute_dtype, policy, any re-route reason
print(scimetal.last_route())                # structured Route namedtuple

# strict policy: bit-identical to SciPy 1.18.1 arm64 (runs on CPU engine / SciPy)
with scimetal.precision("strict"):
    exact = ndimage.gaussian_filter(image, 3.0)

# Use scimetal.fft directly or as a scipy.fft backend
mfft.prepare((4096, 4096), np.complex64)     # pre-compiles pipelines to avoid first-call latency
mfft.register()                              # plain scipy.fft calls now use SciMetal too
spectrum = scipy.fft.fft(np.ones((4096, 4096), dtype=np.complex64))
mfft.unregister()
```

Context managers (`with scimetal.precision(...)`, `with scimetal.device("auto" | "cpu" | "gpu")`)
and setters (`set_precision`, `set_device`, `set_small_inputs`) scope per-thread/task state cleanly.

## Accelerated coverage (`scimetal.accelerated()`)

- **`ndimage`** (28 filters):
  - Linear & derivative filters (13): `convolve1d`, `correlate1d`, `gaussian_filter1d`,
    `gaussian_filter`, `uniform_filter1d`, `uniform_filter`, `prewitt`, `sobel`,
    `convolve`, `correlate`, `laplace`, `gaussian_laplace`,
    `gaussian_gradient_magnitude`
  - Min/max, rank/median, and grayscale morphology (15): `minimum_filter1d`,
    `maximum_filter1d`, `minimum_filter`, `maximum_filter`, `rank_filter`,
    `median_filter`, `percentile_filter`, `grey_erosion`, `grey_dilation`,
    `grey_opening`, `grey_closing`, `morphological_gradient`,
    `morphological_laplace`, `white_tophat`, `black_tophat`
  - Supports `float32`, `float64`, `bool`, `int8`–`int64`, `uint8`–`uint64`, plus cross-dtype
    `output=` among real dtypes; CPU `strict` is bit-identical to SciPy 1.18.1 arm64.
    SciPy runs the derivative callbacks for `generic_laplace` and `generic_gradient_magnitude`.
- **`fft`** (12 transforms + backend + `prepare`): `fft`, `ifft`, `rfft`, `irfft`,
  `fft2`, `ifft2`, `fftn`, `ifftn`, `rfft2`, `irfft2`, `rfftn`, `irfftn`, plus
  `scimetal.fft.register()` / `unregister()` and `scimetal.fft.prepare()`.
- **`signal`** (7 functions): `choose_conv_method`, `convolve`, `convolve2d`,
  `correlate`, `correlate2d`, `fftconvolve`, `oaconvolve`.

## Precision policies and guard rails

- **`fast` (default)**: computes float64 and representable (`|x| < 2^24`) integer/bool
  inputs in single precision (`float32`/`complex64`) on the Metal GPU when above the
  cost crossover, widening outputs to the target dtype (~6–7 significant digits on floats).
  Emits a one-time `scimetal.PrecisionWarning` when narrowing a double-precision input;
  silence with `warnings.filterwarnings("ignore", category=scimetal.PrecisionWarning)`.
- **`refined`**: single-precision compute plus a double-precision correction where one
  exists; otherwise behaves as `fast`.
- **`strict`** keeps double-precision work on the CPU. Native `ndimage` uses double
  accumulation. `signal.convolve2d` and `signal.correlate2d` can use the native CPU
  engine for matching `float32` or `float64` NumPy inputs. SciMetal recognizes the FIR binaries from
  all 12 official SciPy 1.18.1 macOS arm64 wheels, including free-threaded
  CPython builds. SciMetal requires macOS 15 or later.
  Unverified builds, unsupported arguments, nondefault floating-point settings,
  and nonfinite results go through SciPy. Double-precision FFTs delegate with the
  caller's worker settings.
  Large signal FFT convolutions can delegate to multithreaded SciPy.
  [SciPy 1.18.1 release files](https://pypi.org/project/scipy/1.18.1/#files)
- **Guard rails (`auto` re-routing / forced `gpu` decline)**:
  - Double-precision inputs and predicted filter gains outside the GPU's supported
    range can reroute to the native CPU engine or SciPy with a reason in `last_route()`.
    Native `ndimage` filters also reroute when the largest finite input magnitude is
    nonzero and below the smallest normal float32 value.
  - For 32/64-bit integer inputs containing `|x| >= 2^24`, `ndimage` uses the native
    CPU engine or SciPy, and FFTs delegate to SciPy before converting the input.
    Forced `device("gpu")` calls that hit these guards delegate to SciPy.

## Benchmarks and accuracy (Apple M1 Ultra, 16 P-cores, 64-core GPU, 128 GB)

Measured standalone on macOS 26.5.1 (SciPy 1.18.1, NumPy 2.5.5; median times in `bench/results/m1-ultra/`):

| Family | Workload | SciPy (1w / 16w) | `fast` (GPU) | Speedup | `strict` (CPU) | Error / Scale |
|---|---|---|---|---|---|---|
| `ndimage` | `gaussian_filter` 4096², σ=3 (`float64`) | 390.7 ms | 3.70 ms | **105.6×** | 7.20 ms (**54.3×**, bit-identical) | 2.0e-07 |
| `ndimage` | `gaussian_filter` 4096², σ=3 (`uint8`) | 308.5 ms | 4.79 ms | **64.5×** | 7.95 ms (**38.8×**, bit-identical) | 5.1e-03 |
| `ndimage` | `gaussian_filter` 4096², σ=3 (`uint16`) | 320.0 ms | 4.59 ms | **69.7×** | 7.65 ms (**41.8×**, bit-identical) | 2.0e-05 |
| `ndimage` | `gaussian_laplace` 4096², σ=3 (`float64`) | 771.1 ms | 4.00 ms | **192.8×** | 14.62 ms (**52.7×**, bit-identical) | 1.4e-06 |
| `ndimage` | `correlate` 4096², 5×5 dense (`float64`) | 275.3 ms | 3.44 ms | **79.9×** | 4.56 ms (**60.4×**, bit-identical) | 3.0e-07 |
| `ndimage` | `correlate` 4096², 5×5 dense (`uint8`) | 300.2 ms | 4.71 ms | **63.8×** | 11.12 ms (**27.0×**, bit-identical) | 4.1e-03 |
| `ndimage` | `median_filter` 4096², 3×3 (`float32` / `float64`) | 1605.0 / 1541.8 ms | 1.66 / 2.11 ms | **966.2× / 731.9×** | 81.54 / 82.36 ms (**19.7× / 18.7×**, bit-identical) | 0.0 / 3.5e-08 |
| `ndimage` | `median_filter` 4096², 5×5 (`float32` / `float64`) | 5420.4 / 5336.3 ms | 2.70 / 2.95 ms | **2,010.0× / 1,811.5×** | 411.44 / 414.99 ms (**13.2× / 12.9×**, bit-identical) | 0.0 / 3.6e-08 |
| `ndimage` | `median_filter` 4096², 7×7 (`float32` / `float64`) | 9834.7 / 9693.9 ms | 3.79 / 3.91 ms | **2,598.3× / 2,479.9×** | 604.86 / 603.19 ms (**16.3× / 16.1×**, bit-identical) | 0.0 / 3.7e-08 |
| `ndimage` | `median_filter` 256³, 3×3×3 (`float32` / `float64`) | 5804.6 / 5708.7 ms | 21.53 / 22.30 ms | **269.6× / 256.1×** | 457.19 / 445.68 ms (**12.7× / 12.8×**, bit-identical) | 0.0 / 3.6e-08 |
| `ndimage` | `percentile_filter` 4096², p=25, 5×5 (`float64`) | 4998.9 ms | 16.44 ms | **304.0×** | 426.07 ms (**11.7×**, bit-identical) | 3.8e-08 |
| `ndimage` | `minimum_filter` 4096², size=15 (`float64`) | 483.2 ms | 2.76 ms | **175.0×** | 6.68 ms (**72.3×**, bit-identical) | 4.7e-08 |
| `ndimage` | `maximum_filter` 4096², size=101 (`float64`) | 489.5 ms | 5.68 ms | **86.2×** | 10.33 ms (**47.4×**, bit-identical) | 3.4e-08 |
| `ndimage` | `maximum_filter` 4096², 5×5 disk (`float32`) | 596.7 ms | 2.37 ms | **252.1×** | 8.65 ms (**69.0×**, bit-identical) | 0.0 |
| `ndimage` | `grey_opening` 4096², size=15 (`float64`) | 907.9 ms | 3.07 ms | **295.9×** | 12.32 ms (**73.7×**, bit-identical) | 4.7e-08 |
| `ndimage` | `morphological_gradient` 4096², size=15 (`float64`) | 994.8 ms | 3.35 ms | **297.4×** | 12.59 ms (**79.0×**, bit-identical) | 1.9e-07 |
| `ndimage` | `white_tophat` 4096², size=15 (`float64` / `uint8`) | 921.2 / 635.3 ms | 3.30 / 4.73 ms | **278.8× / 134.2×** | 13.54 / 10.76 ms (**68.0× / 59.1×**, bit-identical) | 2.2e-07 / 0.0 |
| `ndimage` | `morphological_laplace` 4096², size=15 (`float64`) | 1020.8 ms | 3.31 ms | **308.4×** | 15.20 ms (**67.2×**, bit-identical) | 6.5e-07 |
| `fft` | `fft` 2^24 (`complex64`) | 206.1 / 21.6 ms | 5.35 ms | **38.5× / 4.0×** | SciPy | 4.2e-07 |
| `fft` | `fft` 2^24 (`complex128`) | 299.9 / 29.9 ms | 8.01 ms | **37.4× / 3.7×** | SciPy | 4.2e-07 |
| `fft` | `rfft` 2^22 (`float64`) | 47.8 / 9.0 ms | 2.19 ms | **21.9× / 4.1×** | SciPy | 1.2e-06 |
| `fft` | `fft` 1024×65536 (`complex128`) | 644.8 / 52.4 ms | 20.61 ms | **31.3× / 2.5×** | SciPy | 6.7e-07 |
| `signal` | `fftconvolve` 4096² × 4096² (`float64`) | 1393.9 / 176.6 ms | 25.21 ms | **55.3× / 7.0×** | SciPy | 1.3e-06 |
| `signal` | `convolve` 2^24 × 1025 (`float64`) | 518.7 / 129.4 ms | 3.97 ms | **130.7× / 32.6×** | SciPy (threaded) | 1.1e-06 |
| `signal` | `convolve2d` 4096² × 5² (`float64`) | 572.5 ms | 3.35 ms | **171.1×** | See below | 1.9e-07 |

A separate warm run of the same `convolve2d` workload on the M1 Ultra measured
**4.77 ms** for the verified strict CPU route and **588 ms** for SciPy. The outputs
were byte-identical.

For `correlate2d` with the same shapes and dtype, the strict CPU route measured
**5.34 ms** versus **588 ms** in SciPy (**110×**), with byte-identical output.
These medians are from five warm calls under a concurrent system load (one-minute
load average 5.29–5.43).

The morphology and rank/median rows were measured standalone via
`bench/ndimage_morphology.py` (saved in `bench/results/m1-ultra/ndimage_morphology.json`,
one-minute load average 4.95–5.60) on full-precision inputs; `float32`, `uint8`, and
`uint16` GPU outputs are bit-identical (`0.0` error) to SciPy.

## Provenance

- `src/runtime/` is derived from `metal-arrow` (see `src/runtime/PROVENANCE.md`).
- `third_party/mlx-kernels/` vendors Apple MLX FFT Metal kernels (MIT license; see `third_party/mlx-kernels/PROVENANCE.md`).
- `third_party/metal-cpp/` vendors Apple's `metal-cpp` headers.
- Reference behaviour and oracle test suites verify against SciPy 1.18.1 (`arm64`).
