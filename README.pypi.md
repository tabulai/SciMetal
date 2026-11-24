# SciMetal

SciMetal accelerates selected SciPy image filters, Fourier transforms, and signal
convolutions on Apple Silicon. Its SciPy-compatible namespaces can run a call on
the Metal GPU, on a multithreaded Accelerate/NEON CPU engine, or in SciPy.

## Install

The wheel supports Apple Silicon Macs running macOS 15 or later with Python
3.11 or later. pip installs NumPy and SciPy as dependencies.

```sh
python -m pip install scimetal
```

## Native coverage

The **0.1.0 wheel** provides 13 native `ndimage` filters, including Gaussian,
uniform, Sobel, and Laplace filters. It also includes 12 `fft` transforms and
7 `signal` functions. Other functions, including `ndimage.median_filter`, remain
available through SciPy. [PyPI release files](https://pypi.org/project/scimetal/0.1.0/#files)

The **development checkout** adds 15 native `ndimage` functions for minimum and
maximum filters, rank, median and percentile filters, and grayscale morphology
such as `grey_opening`, `grey_closing`, and `white_tophat`.
[Project documentation](https://github.com/tabulai/SciMetal#readme)

## Usage

```python
import numpy as np
import scimetal
from scimetal.ndimage import gaussian_filter

native_filters = scimetal.accelerated()["ndimage"]
print(native_filters)                     # Registered filters in this installation
print("median_filter" in native_filters)  # False in the 0.1.0 wheel

image = np.random.default_rng(0).standard_normal((1024, 1024), dtype=np.float32)
smoothed = gaussian_filter(image, sigma=3.0)
print(scimetal.explain())                 # Engine and compute dtype for this call
```

The default `fast` policy can use single-precision GPU compute for
double-precision inputs. Use `with scimetal.precision("strict"):` to retain
double-precision computation. Under `strict`, supported `signal.convolve2d`
and `signal.correlate2d` calls use a bit-exact native CPU implementation with
the verified SciPy 1.18.1 Apple Silicon wheels. Other SciPy builds handle those
calls directly.
