# metal-sci

SciPy's API, accelerated on Apple Silicon: Metal GPU kernels and a multithreaded
Accelerate/NEON CPU engine behind a C ABI, with Python namespaces that mirror SciPy's.

Status: early. Two families have native routes (`ndimage` separable filters and `fft`);
everything else is SciPy's own function. The roadmap, the evidence behind the design and
the rules for extending it are in [PLAN.md](PLAN.md).

## Use

```python
import metal_sci
import metal_sci.ndimage as ndimage          # instead of scipy.ndimage

y = ndimage.gaussian_filter(image, 3.0)      # GPU, float32 arithmetic, float64 result
print(metal_sci.explain())                   # how that call ran

with metal_sci.precision("strict"):          # SciPy's exact result, on the CPU engine
    y = ndimage.gaussian_filter(image, 3.0)

import metal_sci.fft
metal_sci.fft.register()                     # plain scipy.fft calls now use the GPU too
```

Precision policy:

- `fast` (default): float64 inputs may be computed in float32 on the GPU and are returned
  as float64 with about 7 significant digits.
- `refined`: float32 work plus a float64 correction where one exists.
- `strict`: SciPy's result; double precision never goes to the GPU.

A call the native code does not handle, or cannot win (small inputs), runs SciPy's own
function with the caller's arguments untouched.

## Build and test

Requires Apple Silicon, macOS 15 or later, Xcode with the Metal toolchain, cmake, ninja.

```bash
export PATH=$PWD/.venv/bin:$PATH
cmake --preset metal && cmake --build --preset metal && ctest --preset metal
.venv/bin/python -m pytest tests/python -q
PYTHONPATH=python:tests/oracle .venv/bin/python -m pytest -p metal_sci_oracle \
    --pyargs scipy.ndimage.tests.test_filters -q
```

The last command runs SciPy's own test suite against the native routes.

## Third-party code

- `src/runtime/` is vendored from metal-arrow (see `src/runtime/PROVENANCE.md`).
- `third_party/mlx-kernels/` holds FFT kernels from Apple's MLX, MIT licence
  (see `third_party/mlx-kernels/PROVENANCE.md`).
- `third_party/metal-cpp/` is Apple's metal-cpp.
