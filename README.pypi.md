# SciMetal

SciMetal accelerates selected SciPy `ndimage`, `fft`, and `signal` operations on
Apple Silicon using Metal GPU kernels and a multithreaded Accelerate and NEON
CPU engine. Other calls use SciPy directly. Call `scimetal.accelerated()` to see
which functions have native implementations.

## Install

The wheel supports Apple Silicon Macs running macOS 15 or later with Python
3.11 or later. pip installs NumPy and SciPy as dependencies.

```sh
python -m pip install scimetal
```

## Usage

```python
import numpy as np
import scimetal
from scimetal.ndimage import gaussian_filter

image = np.random.default_rng(0).standard_normal((1024, 1024), dtype=np.float32)
smoothed = gaussian_filter(image, sigma=3.0)
print(scimetal.explain())
```

The default `fast` policy can use single-precision GPU compute for
double-precision inputs. Use `with scimetal.precision("strict"):` to retain
double-precision computation. Under `strict`, supported `signal.convolve2d`
and `signal.correlate2d` calls use a bit-exact native CPU implementation with
the verified SciPy 1.18.1 Apple Silicon wheels. Other SciPy builds handle those
calls directly.
