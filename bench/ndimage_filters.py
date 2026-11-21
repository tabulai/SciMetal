#!/usr/bin/env python
"""Benchmark of scimetal.ndimage's separable filters against scipy.ndimage.

    .venv/bin/python bench/ndimage_filters.py              # about a minute
    .venv/bin/python bench/ndimage_filters.py --quick      # one SciPy run per workload
    .venv/bin/python bench/ndimage_filters.py --calibrate  # CPU engine vs GPU engine sweep

The native library is found as in scimetal._native (SCIMETAL_LIBRARY, or the
source tree's .build/metal).

Every timing is the median wall time of warm calls, end to end as a user sees
it: Python wrapper, input copy or narrowing, the filter, and the result as a
NumPy array. Results are dropped between calls, so result storage recycles
through the buffer cache as it does in a loop that overwrites its variable.

The last line printed is `RESULT {json}`.

--calibrate prints, for a ladder of sizes, the forced-CPU and forced-GPU times
of representative calls next to the router's work figure (see routing_work in
src/ndimage/filters.cpp); the AUTO threshold is where the ratio crosses 1.
"""
import argparse
import functools
import json
import pathlib
import platform
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))

import scipy  # noqa: E402
import scipy.ndimage as sn  # noqa: E402

import scimetal  # noqa: E402
import scimetal.ndimage as mn  # noqa: E402


def timed(function, repeats, warmup):
    """Median and minimum wall time in ms of function(); each result is dropped before the next call."""
    for _ in range(warmup):
        function()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = function()
        times.append(time.perf_counter() - start)
        del result
    return statistics.median(times) * 1e3, min(times) * 1e3


def image(shape, dtype, seed=20261118):
    """Positive, image-like data: smooth structure plus noise."""
    rng = np.random.default_rng(seed)
    grids = np.meshgrid(*[np.arange(n, dtype=np.float32) for n in shape], indexing="ij", sparse=True)
    smooth = functools.reduce(np.multiply, [np.sin(g / (61.0 + 18 * i)) for i, g in enumerate(grids)])
    smooth = 0.5 + 0.25 * smooth
    val = smooth + 0.25 * (rng.random(shape, dtype=np.float32) - 0.5)
    dt = np.dtype(dtype)
    if np.issubdtype(dt, np.integer):
        info = np.iinfo(dt)
        low = 0.0
        high = float(min(info.max, 65535))
        return ((high - low) * val + low).astype(dt)
    return val.astype(dt)


WEIGHTS = np.array([0.02, -0.05, 0.11, 0.24, 0.31, 0.22, 0.13, -0.04, 0.06])  # no symmetry
OVERHANG_513 = np.linspace(0.001, 0.004, 513)
OVERHANG_513 /= OVERHANG_513.sum()
DENSE_5X5 = np.array([
    [0.01, -0.02, 0.04, -0.02, 0.01],
    [-0.03, 0.07, 0.11, 0.05, -0.01],
    [0.02, 0.12, 0.35, 0.14, 0.03],
    [-0.01, 0.06, 0.10, 0.08, -0.02],
    [0.01, -0.03, 0.05, -0.01, 0.02],
])  # non-separable 5x5

# name, shape, call(module, array)
WORKLOADS = [
    ("gaussian_filter 4096x4096 sigma=3", (4096, 4096), lambda m, a: m.gaussian_filter(a, 3)),
    ("gaussian_filter 4096x4096 sigma=10", (4096, 4096), lambda m, a: m.gaussian_filter(a, 10)),
    ("gaussian_filter 512x512x512 sigma=2", (512, 512, 512), lambda m, a: m.gaussian_filter(a, 2)),
    ("uniform_filter 4096x4096 size=9", (4096, 4096), lambda m, a: m.uniform_filter(a, 9)),
    ("correlate1d 4096x4096 9 taps axis=-1", (4096, 4096), lambda m, a: m.correlate1d(a, WEIGHTS)),
    ("correlate1d 4096x4096 9 taps axis=0", (4096, 4096), lambda m, a: m.correlate1d(a, WEIGHTS, axis=0)),
    ("correlate1d 8192x64 513 taps overhang", (8192, 64), lambda m, a: m.correlate1d(a, OVERHANG_513, axis=-1)),
    ("gaussian_laplace 4096x4096 sigma=3", (4096, 4096), lambda m, a: m.gaussian_laplace(a, 3)),
    ("correlate 4096x4096 5x5 dense", (4096, 4096), lambda m, a: m.correlate(a, DENSE_5X5)),
]


def benchmark(quick):
    rows = []
    header = f"{'workload':38s} {'dtype':8s} {'scipy':>9s}  {'fast':>8s} {'':>7s} {'engine':6s} {'err/scale':>9s}  " \
             f"{'strict':>8s} {'':>7s} {'engine':6s} {'identical'}"
    print(header)
    for name, shape, call in WORKLOADS:
        dtypes = [np.float64, np.float32]
        if name == "gaussian_filter 4096x4096 sigma=3":
            dtypes += [np.uint8, np.uint16]
        elif name == "correlate 4096x4096 5x5 dense":
            dtypes += [np.uint8]
        for dtype in dtypes:
            a = image(shape, dtype)
            scipy_ms, scipy_min = timed(lambda: call(sn, a), 1 if quick else 3, 0)
            reference = call(sn, a)
            scale = float(np.abs(reference).max())
            row = {"workload": name, "dtype": np.dtype(dtype).name, "elements": int(a.size),
                   "scipy_ms": round(scipy_ms, 2), "scipy_min_ms": round(scipy_min, 2)}
            for policy in ("fast", "strict"):
                with scimetal.precision(policy):
                    ms, fastest = timed(lambda: call(mn, a), 5 if quick else 9, 2)
                    result = call(mn, a)
                    route = scimetal.last_route()
                error = float(np.abs(result.astype(np.float64) - reference.astype(np.float64)).max())
                row[policy] = {"ms": round(ms, 3), "min_ms": round(fastest, 3), "engine": route.engine,
                               "computed_in": route.compute_dtype, "speedup": round(scipy_ms / ms, 1),
                               "max_error_over_scale": error / scale,
                               "bit_identical": bool(result.tobytes() == reference.tobytes())}
                del result
            del reference
            fast, strict = row["fast"], row["strict"]
            print(f"{name:38s} {row['dtype']:8s} {scipy_ms:7.1f}ms  {fast['ms']:6.2f}ms {fast['speedup']:6.1f}x "
                  f"{fast['engine']:6s} {fast['max_error_over_scale']:9.1e}  {strict['ms']:6.2f}ms "
                  f"{strict['speedup']:6.1f}x {strict['engine']:6s} {strict['bit_identical']}")
            rows.append(row)
    return rows


def work_units(taps, symmetric, ndim, uniform=False):
    """routing_work per element for a filter applied along every axis of an ndim-D array."""
    arithmetic = 16 if uniform else (taps // 2 + 1 if symmetric else taps)
    return (ndim - 1) * (arithmetic + 12) + arithmetic


def calibrate():
    sides2d = (256, 362, 512, 724, 1024, 1448, 2048, 2896, 4096)
    cases = [(f"gaussian s=1 {n}^2", (n, n), lambda a: mn.gaussian_filter(a, 1), work_units(9, True, 2)) for n in sides2d]
    cases += [(f"gaussian s=3 {n}^2", (n, n), lambda a: mn.gaussian_filter(a, 3), work_units(25, True, 2)) for n in sides2d]
    cases += [(f"gaussian s=10 {n}^2", (n, n), lambda a: mn.gaussian_filter(a, 10), work_units(81, True, 2)) for n in sides2d[:-2]]
    cases += [(f"sobel {n}^2", (n, n), lambda a: mn.sobel(a), work_units(3, True, 2)) for n in sides2d[2:]]
    cases += [(f"uniform 5 {n}^2", (n, n), lambda a: mn.uniform_filter(a, 5), work_units(5, False, 2, True)) for n in sides2d[2:]]
    cases += [(f"correlate1d 9 taps axis=-1 {n}^2", (n, n), lambda a: mn.correlate1d(a, WEIGHTS), 9) for n in sides2d[4:]]
    cases += [(f"correlate1d 9 taps axis=0 {n}^2", (n, n), lambda a: mn.correlate1d(a, WEIGHTS, axis=0), 9 + 12) for n in sides2d[4:]]
    cases += [(f"gaussian s=2 {n}^3", (n, n, n), lambda a: mn.gaussian_filter(a, 2), work_units(17, True, 3)) for n in (64, 96, 128, 192, 256)]
    cases += [(f"gaussian1d s=3 n=2^{k}", (2 ** k,), lambda a: mn.gaussian_filter1d(a, 3), 13) for k in (18, 20, 22, 24)]
    cases += [(f"correlate 3x3 {n}^2", (n, n), lambda a: mn.correlate(a, DENSE_5X5[:3, :3]), 9) for n in sides2d[3:]]
    cases += [(f"correlate 5x5 {n}^2", (n, n), lambda a: mn.correlate(a, DENSE_5X5), 25) for n in sides2d[2:]]
    rows = []
    for dtype in (np.float64, np.float32):
        print(f"\n{np.dtype(dtype).name}: {'case':36s} {'elements':>9s} {'work/2^20':>10s} {'cpu ms':>8s} {'gpu ms':>8s} {'cpu/gpu':>8s}  auto")
        for name, shape, call, per_element in cases:
            a = image(shape, dtype)
            with scimetal.precision("fast"):
                with scimetal.device("cpu"):
                    cpu, _ = timed(lambda: call(a), 9, 2)
                with scimetal.device("gpu"):
                    gpu, _ = timed(lambda: call(a), 9, 2)
                    engine = scimetal.last_route().engine
                call(a)
                auto = scimetal.last_route().engine
            if engine != "gpu":
                raise SystemExit("the GPU engine is not available: nothing to calibrate")
            work = a.size * per_element
            print(f"         {name:36s} {a.size:9d} {work / 2 ** 20:10.1f} {cpu:8.3f} {gpu:8.3f} {cpu / gpu:8.2f}  {auto}")
            rows.append({"case": name, "dtype": np.dtype(dtype).name, "elements": int(a.size), "work": int(work),
                         "cpu_ms": round(cpu, 3), "gpu_ms": round(gpu, 3), "auto": auto})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--quick", action="store_true", help="one SciPy run and five native runs per workload")
    parser.add_argument("--calibrate", action="store_true", help="sweep the CPU and GPU engines around the routing threshold")
    options = parser.parse_args()
    started = time.perf_counter()
    result = {"benchmark": "ndimage_filters", "numpy": np.__version__, "scipy": scipy.__version__,
              "scimetal": scimetal.__version__, "machine": platform.machine(), "macos": platform.mac_ver()[0]}
    if options.calibrate:
        result["calibration"] = calibrate()
    else:
        result["workloads"] = benchmark(options.quick)
    result["wall_seconds"] = round(time.perf_counter() - started, 1)
    print("RESULT " + json.dumps(result))


if __name__ == "__main__":
    main()
