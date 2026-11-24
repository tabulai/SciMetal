import time, warnings, json
import numpy as np
import scipy.ndimage, scimetal, scimetal.ndimage
warnings.filterwarnings("ignore", category=scimetal.PrecisionWarning)
img = np.random.default_rng(0).standard_normal((4096, 4096))
out = {}
for name, f in [("scimetal", scimetal.ndimage.gaussian_filter), ("scipy", scipy.ndimage.gaussian_filter)]:
    f(img, 3.0)
    stamps = []
    t0 = time.perf_counter()
    for _ in range(100):
        f(img, 3.0); stamps.append(time.perf_counter() - t0)
    out[name] = stamps
    print(name, f"{stamps[-1]:.3f} s")
json.dump(out, open("hundred.json", "w"))
