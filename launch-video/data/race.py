import time, warnings
import numpy as np
import scipy.ndimage, scipy.signal
import scimetal, scimetal.ndimage, scimetal.signal

warnings.filterwarnings("ignore", category=scimetal.PrecisionWarning)

def median_ms(f, *args, n=7, **kw):
    f(*args, **kw)
    ts = []
    for _ in range(n):
        t = time.perf_counter(); f(*args, **kw); ts.append(time.perf_counter() - t)
    return sorted(ts)[n // 2] * 1e3

img = np.random.default_rng(0).standard_normal((4096, 4096))
k5 = np.random.default_rng(1).standard_normal((5, 5))

cases = [
    ("gaussian_filter  sigma=3", scipy.ndimage.gaussian_filter, scimetal.ndimage.gaussian_filter, (img, 3.0)),
    ("gaussian_laplace sigma=3", scipy.ndimage.gaussian_laplace, scimetal.ndimage.gaussian_laplace, (img, 3.0)),
    ("convolve2d       5x5    ", scipy.signal.convolve2d, scimetal.signal.convolve2d, (img, k5)),
]
print("4096 x 4096 float64, Apple M1 Ultra\n")
for name, a, b, args in cases:
    s = median_ms(a, *args); m = median_ms(b, *args)
    print(f"{name}   scipy {s:7.1f} ms   scimetal {m:5.1f} ms   {s/m:5.0f}x")

with scimetal.precision("strict"):
    exact = scimetal.ndimage.gaussian_filter(img, 3.0)
    ms = median_ms(scimetal.ndimage.gaussian_filter, img, 3.0)
ref = scipy.ndimage.gaussian_filter(img, 3.0)
print(f"\nstrict: {ms:.1f} ms, identical to scipy: {np.array_equal(exact, ref)}")
