#!/usr/bin/env python
"""Benchmark: metal_sci.signal convolution and correlation against scipy.signal.

    .venv/bin/python bench/signal_convolve.py [--quick] [--calibrate]

Run ALONE on a quiet machine with `uptime` before and after per PLAN.md Section 6.9.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import platform
import statistics
import subprocess
import sys
import time

import numpy as np
import scipy
import scipy.fft
import scipy.signal as ss

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))

import metal_sci  # noqa: E402
import metal_sci.signal as msig  # noqa: E402

WORKERS = 16


def timed(fn, repeats, warmup):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        res = fn()
        times.append(time.perf_counter() - t0)
        del res
    return statistics.median(times) * 1e3, min(times) * 1e3


def make_data(shape, dtype, seed=42):
    rng = np.random.default_rng(seed)
    real_dt = np.float32 if np.dtype(dtype) in (np.float32, np.complex64) else np.float64
    x = rng.standard_normal(shape).astype(real_dt)
    if np.dtype(dtype).kind == "c":
        x = (x + 1j * rng.standard_normal(shape).astype(real_dt)).astype(dtype)
    return np.ascontiguousarray(x.astype(dtype, copy=False))


def machine_info():
    def sysctl(name):
        try:
            return subprocess.check_output(["sysctl", "-n", name], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    return {
        "cpu": sysctl("machdep.cpu.brand_string"),
        "performance_cores": sysctl("hw.perflevel0.physicalcpu"),
        "memory_gb": round(int(sysctl("hw.memsize") or 0) / 2 ** 30),
        "macos": platform.mac_ver()[0],
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "loadavg": [round(v, 2) for v in os.getloadavg()],
    }


def benchmark(quick: bool):
    workloads = [
        ("fftconvolve 4096^2 x 4096^2", "fftconvolve", (4096, 4096), (4096, 4096), {"mode": "full"}, True),
        ("convolve 2^24 x 1025", "convolve", (1 << 24,), (1025,), {"mode": "same"}, True),
        ("oaconvolve 2^24 x 1025", "oaconvolve", (1 << 24,), (1025,), {"mode": "same"}, True),
        ("oaconvolve 4096^2 x 65^2", "oaconvolve", (4096, 4096), (65, 65), {"mode": "same"}, True),
        ("convolve2d 4096^2 x 5^2", "convolve2d", (4096, 4096), (5, 5), {"mode": "same"}, False),
    ]

    rows = []
    repeats = 3 if quick else 7
    warmup = 1 if quick else 2
    scipy_repeats = 1 if quick else 3

    for label, func, s1, s2, kw, has_fft_workers in workloads:
        for dtype in (np.float64,):
            a = make_data(s1, dtype, seed=1)
            b = make_data(s2, dtype, seed=2)
            ours = getattr(msig, func)
            theirs = getattr(ss, func)

            scipy_1_ms, _ = timed(lambda: theirs(a, b, **kw), scipy_repeats, 0)
            if has_fft_workers:
                def run_w16():
                    with scipy.fft.set_workers(WORKERS):
                        return theirs(a, b, **kw)
                scipy_w16_ms, _ = timed(run_w16, scipy_repeats, 0)
            else:
                scipy_w16_ms = None

            with metal_sci.precision("fast"):
                t0 = time.perf_counter()
                first_res = ours(a, b, **kw)
                first_ms = (time.perf_counter() - t0) * 1e3
                route = metal_sci.last_route()
                del first_res
                fast_ms, fast_min = timed(lambda: ours(a, b, **kw), repeats, warmup)

            ref = theirs(a, b, **kw)
            res = ours(a, b, **kw)
            scale = max(1.0, float(np.abs(ref).max()))
            rel_err = float(np.abs(res - ref).max()) / scale
            del res, ref

            row = {
                "workload": label,
                "function": func,
                "dtype": np.dtype(dtype).name,
                "scipy_w1_ms": round(scipy_1_ms, 2),
                "scipy_w16_ms": round(scipy_w16_ms, 2) if scipy_w16_ms is not None else None,
                "fast_ms": round(fast_ms, 3),
                "fast_min_ms": round(fast_min, 3),
                "first_call_ms": round(first_ms, 2),
                "fast_engine": route.engine,
                "speedup_vs_w1": round(scipy_1_ms / fast_ms, 2),
                "speedup_vs_w16": round(scipy_w16_ms / fast_ms, 2) if scipy_w16_ms else None,
                "max_err_over_scale": rel_err,
            }

            if func in ("convolve2d", "convolve"):
                with metal_sci.precision("strict"):
                    strict_ms, _ = timed(lambda: ours(a, b, **kw), scipy_repeats if func == "convolve2d" else repeats, warmup)
                    s_route = metal_sci.last_route()
                row["strict_ms"] = round(strict_ms, 2)
                row["strict_engine"] = s_route.engine
                row["strict_speedup_vs_w1"] = round(scipy_1_ms / strict_ms, 2)

            rows.append(row)
            w16_str = f"w16 {scipy_w16_ms:7.1f}ms ({row['speedup_vs_w16']:5.1f}x)" if scipy_w16_ms else ""
            print(
                f"{label:28s} scipy {scipy_1_ms:7.1f}ms  fast {fast_ms:7.2f}ms "
                f"({row['speedup_vs_w1']:5.1f}x vs w1, engine={route.engine}) {w16_str}  err={rel_err:.1e}",
                flush=True,
            )
    return rows


def calibrate():
    print("=== Direct CPU vs GPU crossover (5x5 and 9x9 kernels) ===")
    direct_rows = []
    for k in (5, 9):
        ker = make_data((k, k), np.float64, seed=99)
        for n in (512, 1024, 1448, 2048, 2896, 4096):
            img = make_data((n, n), np.float64, seed=n)
            work = img.size * ker.size
            with metal_sci.precision("fast"):
                with metal_sci.device("cpu"):
                    cpu_ms, _ = timed(lambda: msig.convolve2d(img, ker, mode="same"), 5, 1)
                with metal_sci.device("gpu"):
                    gpu_ms, _ = timed(lambda: msig.convolve2d(img, ker, mode="same"), 5, 1)
                msig.convolve2d(img, ker, mode="same")
                auto_eng = metal_sci.last_route().engine
            print(f"  {k}x{k} on {n}x{n} (work={work/2**20:6.1f}M): cpu={cpu_ms:6.2f}ms gpu={gpu_ms:6.2f}ms ratio={cpu_ms/gpu_ms:5.2f}x auto={auto_eng}")
            direct_rows.append({"k": k, "n": n, "work": work, "cpu_ms": round(cpu_ms, 3), "gpu_ms": round(gpu_ms, 3), "auto": auto_eng})

    print("\n=== FFT GPU vs SciPy w1 crossover ===")
    fft_rows = []
    for n in (4096, 8192, 16384, 32768, 65536, 131072, 262144, 1048576):
        a = make_data((n,), np.float64, seed=1)
        b = make_data((n // 2,), np.float64, seed=2)
        sp_ms, _ = timed(lambda: ss.fftconvolve(a, b), 5, 1)
        with metal_sci.precision("fast"), metal_sci.device("gpu"):
            gpu_ms, _ = timed(lambda: msig.fftconvolve(a, b), 5, 1)
        with metal_sci.precision("fast"), metal_sci.device("auto"):
            msig.fftconvolve(a, b)
            auto_eng = metal_sci.last_route().engine
        print(f"  1D n={n:7d} (padded~{2*n:7d}): scipy={sp_ms:6.3f}ms gpu={gpu_ms:6.3f}ms ratio={sp_ms/gpu_ms:5.2f}x auto={auto_eng}")
        fft_rows.append({"n": n, "scipy_ms": round(sp_ms, 3), "gpu_ms": round(gpu_ms, 3), "auto": auto_eng})

    return {"direct": direct_rows, "fft": fft_rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--calibrate", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    out = {
        "benchmark": "signal_convolve",
        "metal_sci": metal_sci.__version__,
        "machine": machine_info(),
    }
    if args.calibrate:
        out["calibration"] = calibrate()
    else:
        out["workloads"] = benchmark(args.quick)
    out["wall_seconds"] = round(time.perf_counter() - started, 1)
    print("RESULT " + json.dumps(out))


if __name__ == "__main__":
    main()
