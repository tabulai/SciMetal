#!/usr/bin/env python
"""Benchmark of scimetal.ndimage's morphology, min/max, and rank/median filters against scipy.ndimage.

    .venv/bin/python bench/ndimage_morphology.py              # full benchmark suite
    .venv/bin/python bench/ndimage_morphology.py --quick      # one timed SciPy call per case
    .venv/bin/python bench/ndimage_morphology.py --routing    # signals and thin images vs SciPy
    .venv/bin/python bench/ndimage_morphology.py --calibrate  # CPU engine vs GPU engine sweep

Every timing is the median wall time of warm calls, end to end as a user sees it:
Python wrapper, input copy or narrowing, the filter, and the result as a NumPy
array. Results are dropped between calls so storage recycles through the buffer
cache. The last line printed is ``RESULT {json}``.
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import pathlib
import platform
import statistics
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))

import scipy  # noqa: E402
import scipy.ndimage as sn  # noqa: E402

import scimetal  # noqa: E402
import scimetal.ndimage as mn  # noqa: E402


def timed(function, repeats, warmup):
    """Median and minimum wall time in ms of function(); drops each result before the next call."""
    for _ in range(warmup):
        result = function()
        del result
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = function()
        times.append(time.perf_counter() - start)
        del result
    return statistics.median(times) * 1e3, min(times) * 1e3


def image(shape, dtype, seed=20261123):
    """Positive, image-like data: smooth structure plus noise (full precision for float64)."""
    rng = np.random.default_rng(seed)
    dt = np.dtype(dtype)
    work_dt = np.float64 if dt == np.float64 else np.float32
    grids = np.meshgrid(*[np.arange(n, dtype=work_dt) for n in shape], indexing="ij", sparse=True)
    smooth = functools.reduce(np.multiply, [np.sin(g / work_dt(61.0 + 18 * i)) for i, g in enumerate(grids)])
    smooth = work_dt(0.5) + work_dt(0.25) * smooth
    val = smooth + work_dt(0.25) * (rng.random(shape, dtype=work_dt) - work_dt(0.5))
    if np.issubdtype(dt, np.integer):
        info = np.iinfo(dt)
        high = float(min(info.max, 65535))
        return (high * val).astype(dt)
    return val.astype(dt)


def disk(radius: int) -> np.ndarray:
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (x * x + y * y) <= radius * radius


DISK_5X5 = disk(2)  # 13 active taps, non-separable
DISK_7X7 = disk(3)  # 29 active taps, non-separable
DISK_15X15 = disk(7)

# (name, shape, dtypes, call(module, array))
WORKLOADS = [
    ("median_filter 4096x4096 3x3", (4096, 4096), (np.float64, np.float32, np.uint8, np.uint16),
     lambda m, a: m.median_filter(a, size=3)),
    ("median_filter 4096x4096 5x5", (4096, 4096), (np.float64, np.float32, np.uint8),
     lambda m, a: m.median_filter(a, size=5)),
    ("median_filter 4096x4096 7x7", (4096, 4096), (np.float64, np.float32),
     lambda m, a: m.median_filter(a, size=7)),
    ("median_filter 256x256x256 3x3x3", (256, 256, 256), (np.float64, np.float32),
     lambda m, a: m.median_filter(a, size=3)),
    ("percentile_filter 4096x4096 p=25 5x5", (4096, 4096), (np.float64, np.float32),
     lambda m, a: m.percentile_filter(a, 25, size=5)),
    ("minimum_filter 4096x4096 size=15", (4096, 4096), (np.float64, np.float32, np.uint8),
     lambda m, a: m.minimum_filter(a, size=15)),
    ("maximum_filter 4096x4096 size=101", (4096, 4096), (np.float64, np.float32),
     lambda m, a: m.maximum_filter(a, size=101)),
    ("maximum_filter 4096x4096 5x5 disk", (4096, 4096), (np.float64, np.float32),
     lambda m, a: m.maximum_filter(a, footprint=DISK_5X5)),
    ("grey_opening 4096x4096 size=15", (4096, 4096), (np.float64, np.float32),
     lambda m, a: m.grey_opening(a, size=15)),
    ("morphological_gradient 4096x4096 s=15", (4096, 4096), (np.float64, np.float32),
     lambda m, a: m.morphological_gradient(a, size=15)),
    ("white_tophat 4096x4096 size=15", (4096, 4096), (np.float64, np.float32, np.uint8),
     lambda m, a: m.white_tophat(a, size=15)),
    ("morphological_laplace 4096x4096 s=15", (4096, 4096), (np.float64, np.float32),
     lambda m, a: m.morphological_laplace(a, size=15)),
]


# Wide 1-D windows, skinny images and small windows on the GPU/CPU crossover.
ROUTING_WORKLOADS = [
    ("median_filter 65536 size=31", (65536,), (np.float32,), lambda m, a: m.median_filter(a, 31)),
    ("median_filter 65536 size=127", (65536,), (np.float32,), lambda m, a: m.median_filter(a, 127)),
    ("median_filter 65536 size=1023", (65536,), (np.float32,), lambda m, a: m.median_filter(a, 1023)),
    ("rank_filter 65536 rank=2 size=1023", (65536,), (np.float32,), lambda m, a: m.rank_filter(a, 2, 1023)),
    ("percentile_filter 65536 p=25 size=255", (65536,), (np.uint16,),
     lambda m, a: m.percentile_filter(a, 25, 255)),
    ("minimum_filter 1x16384 15x15 disk", (1, 16384), (np.float32,),
     lambda m, a: m.minimum_filter(a, footprint=DISK_15X15)),
    ("median_filter 1x16384 15x15 disk", (1, 16384), (np.float32,),
     lambda m, a: m.median_filter(a, footprint=DISK_15X15)),
    ("maximum_filter1d 4096x4 size=511", (4096, 4), (np.float32,),
     lambda m, a: m.maximum_filter1d(a, 511)),
]


def load_averages() -> list[float]:
    return [round(x, 2) for x in os.getloadavg()]


def uptime_line() -> str:
    try:
        return subprocess.check_output(["uptime"], text=True).strip()
    except Exception:
        return ""


def benchmark(quick: bool, workloads=WORKLOADS) -> list[dict]:
    rows = []
    header = (
        f"{'workload':39s} {'dtype':8s} {'scipy':>9s}  {'fast':>8s} {'':>8s} {'engine':6s} {'err/scale':>9s}  "
        f"{'strict':>8s} {'':>7s} {'engine':6s} {'identical'}"
    )
    print(header)
    for name, shape, dtypes, call in workloads:
        for dtype in dtypes:
            a = image(shape, dtype)
            reference = call(sn, a)  # warm-up and correctness oracle
            scipy_ms, scipy_min = timed(lambda: call(sn, a), 1 if quick else 5, 0)
            scale = float(np.abs(reference).max()) or 1.0
            row = {
                "workload": name,
                "dtype": np.dtype(dtype).name,
                "elements": int(a.size),
                "scipy_ms": round(scipy_ms, 2),
                "scipy_min_ms": round(scipy_min, 2),
            }
            for policy in ("fast", "strict"):
                with scimetal.precision(policy):
                    # Strict rank/median on 16M elements can take ~300ms; use fewer repeats when slow.
                    first_start = time.perf_counter()
                    first_res = call(mn, a)
                    first_sec = time.perf_counter() - first_start
                    del first_res
                    if first_sec > 0.25:
                        repeats = 3 if quick else 5
                        warmup = 1
                    else:
                        repeats = 5 if quick else 9
                        warmup = 2
                    ms, fastest = timed(lambda: call(mn, a), repeats, warmup)
                    result = call(mn, a)
                    route = scimetal.last_route()
                error = float(np.abs(result.astype(np.float64) - reference.astype(np.float64)).max())
                row[policy] = {
                    "ms": round(ms, 3),
                    "min_ms": round(fastest, 3),
                    "engine": route.engine,
                    "computed_in": route.compute_dtype,
                    "speedup": round(scipy_ms / ms, 1),
                    "max_error_over_scale": error / scale,
                    "bit_identical": bool(result.tobytes() == reference.tobytes()),
                }
                del result
            del reference
            fast, strict = row["fast"], row["strict"]
            print(
                f"{name:39s} {row['dtype']:8s} {scipy_ms:7.1f}ms  {fast['ms']:6.2f}ms {fast['speedup']:7.1f}x "
                f"{fast['engine']:6s} {fast['max_error_over_scale']:9.1e}  {strict['ms']:6.2f}ms "
                f"{strict['speedup']:6.1f}x {strict['engine']:6s} {strict['bit_identical']}"
            )
            rows.append(row)
    return rows


def calibrate() -> list[dict]:
    sides2d = (128, 256, 362, 512, 724, 1024, 1448, 2048, 2896, 4096)
    cases = []
    # Rank / median family (work = elements * k)
    cases += [(f"median 3x3 {n}^2", (n, n), lambda a: mn.median_filter(a, 3), 9) for n in sides2d[:7]]
    cases += [(f"median 5x5 {n}^2", (n, n), lambda a: mn.median_filter(a, 5), 25) for n in sides2d[:6]]
    cases += [(f"median 7x7 {n}^2", (n, n), lambda a: mn.median_filter(a, 7), 49) for n in sides2d[:6]]
    cases += [(f"percentile25 5x5 {n}^2", (n, n), lambda a: mn.percentile_filter(a, 25, 5), 25) for n in sides2d[:6]]
    cases += [(f"median 3x3x3 {n}^3", (n, n, n), lambda a: mn.median_filter(a, 3), 27) for n in (32, 48, 64, 96, 128)]
    # Dense min/max family (work = elements * k)
    cases += [(f"max disk5x5 {n}^2", (n, n), lambda a: mn.maximum_filter(a, footprint=DISK_5X5), int(DISK_5X5.sum()))
              for n in sides2d[3:]]
    cases += [(f"max disk7x7 {n}^2", (n, n), lambda a: mn.maximum_filter(a, footprint=DISK_7X7), int(DISK_7X7.sum()))
              for n in sides2d[2:]]
    # Separable min/max family (uses separable routing_work: 16 + 12 for non-last + 16 for last = 44)
    cases += [(f"max sep s=15 {n}^2", (n, n), lambda a: mn.maximum_filter(a, 15), 44) for n in sides2d[4:]]
    cases += [(f"max sep s=101 {n}^2", (n, n), lambda a: mn.maximum_filter(a, 101), 28) for n in sides2d[4:]]
    cases += [(f"white_tophat s=15 {n}^2", (n, n), lambda a: mn.white_tophat(a, 15), 88) for n in sides2d[4:]]

    rows = []
    for dtype in (np.float64, np.float32):
        print(
            f"\n{np.dtype(dtype).name}: {'case':28s} {'elements':>9s} {'work/2^20':>10s} "
            f"{'cpu ms':>8s} {'gpu ms':>8s} {'cpu/gpu':>8s}  auto"
        )
        for name, shape, call, per_element in cases:
            a = image(shape, dtype)
            with scimetal.precision("fast"):
                with scimetal.device("cpu"):
                    cpu, _ = timed(lambda: call(a), 7, 2)
                with scimetal.device("gpu"):
                    gpu, _ = timed(lambda: call(a), 7, 2)
                    engine = scimetal.last_route().engine
                call(a)
                auto = scimetal.last_route().engine
            if engine != "gpu":
                raise SystemExit("the GPU engine is not available: nothing to calibrate")
            work = a.size * per_element
            print(
                f"         {name:28s} {a.size:9d} {work / 2 ** 20:10.2f} "
                f"{cpu:8.3f} {gpu:8.3f} {cpu / gpu:8.2f}  {auto}"
            )
            rows.append({
                "case": name,
                "dtype": np.dtype(dtype).name,
                "elements": int(a.size),
                "work": int(work),
                "cpu_ms": round(cpu, 3),
                "gpu_ms": round(gpu, 3),
                "cpu_over_gpu": round(cpu / gpu, 2),
                "auto": auto,
            })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--quick", action="store_true", help="one timed SciPy call and fewer native repeats per workload")
    suite = parser.add_mutually_exclusive_group()
    suite.add_argument("--calibrate", action="store_true", help="sweep CPU and GPU engines around routing thresholds")
    suite.add_argument("--routing", action="store_true", help="compare 1-D and thin-image routes with SciPy")
    parser.add_argument("--out", type=pathlib.Path, default=None, help="optional JSON file to save the RESULT payload")
    options = parser.parse_args()
    started = time.perf_counter()
    load_before = load_averages()
    up_before = uptime_line()
    result = {
        "benchmark": "ndimage_morphology",
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scimetal": scimetal.__version__,
        "machine": platform.machine(),
        "macos": platform.mac_ver()[0],
        "load_before": load_before,
        "uptime_before": up_before,
    }
    if options.calibrate:
        result["calibration"] = calibrate()
    else:
        result["workloads"] = benchmark(options.quick, ROUTING_WORKLOADS if options.routing else WORKLOADS)
    result["load_after"] = load_averages()
    result["uptime_after"] = uptime_line()
    result["wall_seconds"] = round(time.perf_counter() - started, 1)
    if options.out is not None:
        options.out.parent.mkdir(parents=True, exist_ok=True)
        options.out.write_text(json.dumps(result, indent=2) + "\n")
    print("RESULT " + json.dumps(result))


if __name__ == "__main__":
    main()
