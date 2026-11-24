# SciMetal

SciMetal accelerates selected SciPy image filters, Fourier transforms, and signal
convolutions on Apple Silicon. Its SciPy-compatible namespaces can run a call on
the Metal GPU, on a multithreaded Accelerate/NEON CPU engine, or in SciPy.

## Install

The wheel supports Apple Silicon Macs running macOS 15 or later with Python
3.11 or later. pip installs NumPy and SciPy as dependencies.

```sh
python -m pip install --upgrade scimetal
```

## Native coverage

**SciMetal 0.1.1** provides native routes for 28 `ndimage` filters, 12 `fft`
transforms, and 7 `signal` functions. This release adds 15 image filters to the
13 in 0.1.0. The additions cover minimum and maximum filters, rank, median, and
percentile filters, and grayscale morphology such as `grey_opening`,
`grey_closing`, and `white_tophat`. SciMetal chooses an engine for each call
based on the input, filter window, and precision policy.
[Full coverage and benchmarks](https://github.com/tabulai/SciMetal/tree/v0.1.1#readme)

## Usage

```python
import numpy as np
import scimetal
from scimetal.ndimage import median_filter

native_filters = scimetal.accelerated()["ndimage"]
print(native_filters)                     # Registered filters in this installation

image = np.random.default_rng(0).standard_normal((1024, 1024), dtype=np.float32)
denoised = median_filter(image, size=7)
print(scimetal.explain())                 # Engine and compute dtype for this call
```

The default `fast` policy can use single-precision GPU compute for
double-precision inputs. Use `with scimetal.precision("strict"):` to retain
double-precision computation. Under `strict`, supported `signal.convolve2d`
and `signal.correlate2d` calls use a bit-exact native CPU implementation with
the verified SciPy 1.18.1 Apple Silicon wheels. Other SciPy builds handle those
calls directly.
