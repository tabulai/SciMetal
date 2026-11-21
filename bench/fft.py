#!/usr/bin/env python
"""Benchmark: scimetal.fft against scipy.fft.

    SCIMETAL_LIBRARY=.build/fft/libscimetal.dylib .venv/bin/python bench/fft.py [--quick] [--crossover]

Times the public Python API (``scimetal.fft.fft`` and ``rfft`` under the
default policies: fast precision, automatic device) against ``scipy.fft`` with
one worker (its default) and with 16, for complex128 and complex64 inputs
(float64 and float32 for rfft). Each timing is the median of warm runs taken
in interleaved rounds, so drifting background load hits all three alike; the
first call of each case, which builds the plan and compiles its pipelines the
first time a machine sees them, is reported separately. The last line printed
is ``RESULT {json}``.

``--crossover`` instead measures where the GPU route starts to win, which is
what the router's thresholds in src/fft/fft.cpp are calibrated from: for a
grid of shapes it prints SciPy's time, the forced-GPU time and the router's
work measure.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import platform
import subprocess
import sys
import time

import numpy as np
import scipy
import scipy.fft

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))

import scimetal  # noqa: E402
import scimetal.fft as mfft  # noqa: E402
from scimetal.fft import _native_fft  # noqa: E402

WORKERS = 16

# (label, function, shape)
CASES = [
    ("fft 2^16", "fft", (1 << 16,)),
    ("fft 2^20", "fft", (1 << 20,)),
    ("fft 2^22", "fft", (1 << 22,)),
    ("fft 2^24", "fft", (1 << 24,)),
    ("fft 10^6 (2^6 5^6)", "fft", (10 ** 6,)),
    ("rfft 2^22", "rfft", (1 << 22,)),
    ("fft 4096x4096", "fft", (4096, 4096)),
    ("fft 1024x65536", "fft", (1024, 65536)),
]


def make(shape, dtype, seed=0):
    rng = np.random.default_rng(seed)
    real = np.float32 if np.dtype(dtype) in (np.dtype(np.float32), np.dtype(np.complex64)) else np.float64
    x = rng.standard_normal(shape, dtype=real)
    if np.dtype(dtype).kind == "c":
        x = x + 1j * rng.standard_normal(shape, dtype=real)
    return np.ascontiguousarray(x.astype(dtype, copy=False))


def clock(function):
    start = time.perf_counter()
    result = function()
    return time.perf_counter() - start, result


def median_ms(samples):
    return round(1e3 * float(np.median(samples)), 4)


def measure(candidates, budget, min_rounds=3, max_rounds=40):
    """Interleaved rounds of every candidate until the time budget is spent."""
    samples = {name: [] for name in candidates}
    started = time.perf_counter()
    rounds = 0
    while rounds < min_rounds or (rounds < max_rounds and time.perf_counter() - started < budget):
        for name, function in candidates.items():
            elapsed, result = clock(function)
            del result
            samples[name].append(elapsed)
        rounds += 1
    return samples


def run_case(label, function, shape, dtype, budget):
    x = make(shape, dtype)
    ours = getattr(mfft, function)
    theirs = getattr(scipy.fft, function)
    kind = _native_fft.R2C if function == "rfft" else _native_fft.FORWARD
    plan = _native_fft.plan(x.dtype, x.shape, _native_fft.options(kind, 0, [x.ndim - 1]))

    first, result = clock(lambda: ours(x))
    route = scimetal.last_route()
    reference = theirs(x.astype(np.complex128 if x.dtype.kind == "c" else np.float64, copy=False))
    relative_error = float(np.linalg.norm(result - reference) / np.linalg.norm(reference))
    del result, reference
    for _ in range(2):
        ours(x)
    samples = measure({"scimetal": lambda: ours(x),
                       "scipy_workers_1": lambda: theirs(x),
                       f"scipy_workers_{WORKERS}": lambda: theirs(x, workers=WORKERS)}, budget)
    record = {
        "case": label, "function": function, "shape": list(shape), "dtype": x.dtype.name,
        "engine": route.engine, "plan": plan.description.decode() if plan.status == 0 else None,
        "route_reason": route.reason or None,
        "scimetal_ms": median_ms(samples["scimetal"]),
        "scimetal_min_ms": round(1e3 * min(samples["scimetal"]), 4),
        "scimetal_first_call_ms": round(1e3 * first, 3),
        "scipy_workers_1_ms": median_ms(samples["scipy_workers_1"]),
        f"scipy_workers_{WORKERS}_ms": median_ms(samples[f"scipy_workers_{WORKERS}"]),
        "rounds": len(samples["scimetal"]),
        "relative_error_vs_scipy_complex128": relative_error,
    }
    record["speedup_vs_workers_1"] = round(record["scipy_workers_1_ms"] / record["scimetal_ms"], 2)
    record[f"speedup_vs_workers_{WORKERS}"] = round(record[f"scipy_workers_{WORKERS}_ms"] / record["scimetal_ms"], 2)
    return record


def machine():
    def sysctl(name):
        try:
            return subprocess.check_output(["sysctl", "-n", name], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    return {"cpu": sysctl("machdep.cpu.brand_string"), "performance_cores": sysctl("hw.perflevel0.physicalcpu"),
            "memory_gb": round(int(sysctl("hw.memsize") or 0) / 2 ** 30), "macos": platform.mac_ver()[0],
            "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "loadavg": [round(value, 2) for value in os.getloadavg()]}


def benchmark(quick):
    budget = 0.25 if quick else 1.0
    results = []
    for label, function, shape in CASES:
        for dtype in (np.complex128, np.complex64):
            if function == "rfft":
                dtype = np.float64 if dtype == np.complex128 else np.float32
            record = run_case(label, function, shape, dtype, budget)
            results.append(record)
            print("%-22s %-10s %-5s  SciMetal %9.3f ms (first call %8.2f)  scipy w1 %9.3f  w%d %9.3f  x%-7.1f x%-6.1f err %.1e  %s"
                  % (label, record["dtype"], record["engine"], record["scimetal_ms"], record["scimetal_first_call_ms"],
                     record["scipy_workers_1_ms"], WORKERS, record[f"scipy_workers_{WORKERS}_ms"],
                     record["speedup_vs_workers_1"], record[f"speedup_vs_workers_{WORKERS}"],
                     record["relative_error_vs_scipy_complex128"], record["plan"] or record["route_reason"]), flush=True)
    return {"benchmark": "fft", "machine": machine(), "policy": {"precision": scimetal.get_precision(),
            "device": scimetal.get_device()}, "timing": "median of warm runs, interleaved rounds", "cases": results}


def crossover():
    """SciPy (one worker) against the forced-GPU route around the break-even."""
    rows = []
    shapes = [(1 << k,) for k in range(12, 21)]
    for n in (16, 64, 256, 1024, 4096):
        shapes += [(max((1 << k) // n, 1), n) for k in range(13, 22)]
    shapes += [(1 << k, 1 << k) for k in range(6, 11)]
    for dtype in (np.complex64, np.complex128):
        for shape in shapes:
            x = make(shape, dtype)
            function = "fft2" if len(shape) == 2 and shape[0] == shape[1] else "fft"
            ours, theirs = getattr(mfft, function), getattr(scipy.fft, function)
            with scimetal.device("gpu"):
                ours(x)
                samples = measure({"gpu": lambda: ours(x), "scipy": lambda: theirs(x)}, 0.15, min_rounds=7)
            lengths = shape if function == "fft2" else shape[-1:]
            work = sum(x.size * max(int(length - 1).bit_length(), 0) for length in lengths)
            gpu_ms, scipy_ms = median_ms(samples["gpu"]), median_ms(samples["scipy"])
            rows.append({"function": function, "shape": list(shape), "dtype": x.dtype.name, "points": x.size,
                         "work": work, "gpu_ms": gpu_ms, "scipy_ms": scipy_ms, "ratio": round(scipy_ms / gpu_ms, 2)})
            print("%-5s %-14s %-10s points %9d  work %11d  gpu %8.3f ms  scipy %8.3f ms  scipy/gpu %6.2f"
                  % (function, "x".join(map(str, shape)), x.dtype.name, x.size, work, gpu_ms, scipy_ms, scipy_ms / gpu_ms),
                  flush=True)
    return {"benchmark": "fft crossover", "machine": machine(), "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--quick", action="store_true", help="a quarter of the time per case")
    parser.add_argument("--crossover", action="store_true", help="measure the GPU break-even instead")
    arguments = parser.parse_args()
    result = crossover() if arguments.crossover else benchmark(arguments.quick)
    print("RESULT " + json.dumps(result))


if __name__ == "__main__":
    main()
