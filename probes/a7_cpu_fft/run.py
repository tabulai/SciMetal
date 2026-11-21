#!/usr/bin/env python
"""A7 probe: Apple Accelerate vDSP (vDSP_fft_zop / vDSP_DFT_Execute) vs scipy.fft (pocketfft)."""
import ctypes as C
import json
import os
import statistics
import time

import numpy as np
import scipy.fft as sfft

HERE = os.path.dirname(os.path.abspath(__file__))
LIB_PATH = os.path.join(HERE, "liba7.dylib")
lib = C.CDLL(LIB_PATH)

lib.a7_create.argtypes = [C.c_size_t, C.c_size_t, C.c_int, C.c_int]
lib.a7_create.restype = C.c_void_p
lib.a7_destroy.argtypes = [C.c_void_p]
lib.a7_destroy.restype = None
lib.a7_run_interleaved.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_size_t]
lib.a7_run_interleaved.restype = None
lib.a7_run_split.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p, C.c_size_t]
lib.a7_run_split.restype = None
lib.a7_bench_interleaved.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_size_t, C.c_int, C.c_int]
lib.a7_bench_interleaved.restype = C.c_double
lib.a7_bench_split.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p, C.c_size_t, C.c_int, C.c_int]
lib.a7_bench_split.restype = C.c_double


def rel_err(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def verify_accuracy():
    rng = np.random.default_rng(42)
    checks = [
        ((1, 256), np.complex64, 0),
        ((64, 1024), np.complex64, 0),
        ((1, 15360), np.complex64, 1),
        ((16, 3072), np.complex64, 1),
        ((1, 256), np.complex128, 0),
        ((64, 1024), np.complex128, 0),
        ((1, 15360), np.complex128, 1),
        ((16, 3072), np.complex128, 1),
    ]
    for shape, dtype, use_dft in checks:
        batch, n = shape
        real_dt = np.float64 if dtype == np.complex128 else np.float32
        x = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(dtype)
        if batch == 1:
            x = x.reshape(n)
        ref = sfft.fft(x, axis=-1)
        out = np.empty_like(x)
        h = lib.a7_create(batch, n, int(dtype == np.complex128), use_dft)
        assert h, f"failed to create vDSP handle for {shape} {dtype} dft={use_dft}"
        try:
            lib.a7_run_interleaved(h, x.ctypes.data, out.ctypes.data, 16)
            err = rel_err(out, ref)
            tol = 1e-12 if dtype == np.complex128 else 1e-5
            assert err < tol, f"interleaved mismatch {shape} {dtype} dft={use_dft}: {err}"

            ir = np.ascontiguousarray(x.real, dtype=real_dt)
            ii = np.ascontiguousarray(x.imag, dtype=real_dt)
            orr = np.empty_like(ir)
            oi = np.empty_like(ii)
            lib.a7_run_split(h, ir.ctypes.data, ii.ctypes.data, orr.ctypes.data, oi.ctypes.data, 16)
            out_sp = orr + 1j * oi
            err_sp = rel_err(out_sp, ref)
            assert err_sp < tol, f"split mismatch {shape} {dtype} dft={use_dft}: {err_sp}"
        finally:
            lib.a7_destroy(h)
    print("accuracy checks passed")


def bench_scipy(x, workers, warm, reps):
    for _ in range(warm):
        _ = sfft.fft(x, axis=-1, workers=workers)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        _ = sfft.fft(x, axis=-1, workers=workers)
        ts.append((time.perf_counter() - t0) * 1e3)
    return float(statistics.median(ts))


def reps_for(total_elems):
    if total_elems <= (1 << 16):
        return 3, 15
    if total_elems <= (1 << 20):
        return 2, 9
    if total_elems <= (1 << 22):
        return 1, 5
    return 1, 3


def main():
    verify_accuracy()
    rng = np.random.default_rng(20261119)

    cases = []
    # 1-D powers of two 2^8..2^24 (zop + DFT where valid/reasonable)
    for k in (8, 10, 12, 14, 16, 18, 20, 22, 24):
        n = 1 << k
        cases.append((f"1d_p2_2^{k}", (n,), False, 0))
        if k <= 22:
            cases.append((f"1d_dft_2^{k}", (n,), False, 1))
    # Non-power-of-two f * 2^n (DFT)
    for n in (3072, 5120, 15360, 49152, 245760, 786432):
        cases.append((f"1d_dft_{n}", (n,), False, 1))
    # Batched transforms
    for batch, n, use_dft in (
        (64, 256, 0),
        (256, 1024, 0),
        (64, 4096, 0),
        (256, 4096, 0),
        (1024, 4096, 0),
        (1024, 1024, 0),
        (256, 3072, 1),
        (64, 15360, 1),
        (64, 4096, 1),
    ):
        kind = "dft" if use_dft else "zop"
        cases.append((f"batch_{batch}x{n}_{kind}", (batch, n), True, use_dft))

    rows = []
    print(f"{'case':20s} {'dtype':10s} {'sp_w1':>8s} {'sp_w16':>8s} {'v_1t':>8s} {'v_16t':>8s} "
          f"{'split16':>8s} {'vs_w1':>7s} {'vs_w16':>7s}")
    for dtype in (np.complex64, np.complex128):
        real_dt = np.float64 if dtype == np.complex128 else np.float32
        for name, shape, is_batch, use_dft in cases:
            batch = shape[0] if is_batch else 1
            n = shape[-1]
            total = batch * n
            warm, reps = reps_for(total)
            x = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(dtype)
            out = np.empty_like(x)
            ir = np.ascontiguousarray(x.real, dtype=real_dt)
            ii = np.ascontiguousarray(x.imag, dtype=real_dt)
            orr = np.empty_like(ir)
            oi = np.empty_like(ii)

            sp_w1 = bench_scipy(x, 1, warm, reps)
            sp_w16 = bench_scipy(x, 16, warm, reps) if total >= 16384 or batch > 1 else sp_w1

            h = lib.a7_create(batch, n, int(dtype == np.complex128), use_dft)
            if not h:
                print(f"{name:20s} {np.dtype(dtype).name:10s} UNSUPPORTED by vDSP")
                continue
            try:
                v_1t = float(lib.a7_bench_interleaved(h, x.ctypes.data, out.ctypes.data, 1, warm, reps))
                v_16t = (float(lib.a7_bench_interleaved(h, x.ctypes.data, out.ctypes.data, 16, warm, reps))
                         if is_batch else v_1t)
                split16 = float(lib.a7_bench_split(
                    h, ir.ctypes.data, ii.ctypes.data, orr.ctypes.data, oi.ctypes.data,
                    16 if is_batch else 1, warm, reps))
            finally:
                lib.a7_destroy(h)

            vs_w1 = sp_w1 / v_16t
            vs_w16 = sp_w16 / v_16t
            print(f"{name:20s} {np.dtype(dtype).name:10s} {sp_w1:7.3f}m {sp_w16:7.3f}m {v_1t:7.3f}m {v_16t:7.3f}m "
                  f"{split16:7.3f}m {vs_w1:6.2f}x {vs_w16:6.2f}x")
            rows.append({
                "case": name,
                "shape": list(shape),
                "dtype": np.dtype(dtype).name,
                "vdsp": "dft" if use_dft else "fft_zop",
                "scipy_w1_ms": round(sp_w1, 4),
                "scipy_w16_ms": round(sp_w16, 4),
                "vdsp_interleaved_1t_ms": round(v_1t, 4),
                "vdsp_interleaved_16t_ms": round(v_16t, 4),
                "vdsp_split_ms": round(split16, 4),
                "speedup_vs_w1": round(vs_w1, 2),
                "speedup_vs_w16": round(vs_w16, 2),
            })

    max_w1 = max(r["speedup_vs_w1"] for r in rows)
    max_w16 = max(r["speedup_vs_w16"] for r in rows)
    result = {
        "probe": "a7_cpu_fft",
        "numpy": np.__version__,
        "scipy": np.__version__ if False else __import__("scipy").__version__,
        "max_speedup_vs_w1": max_w1,
        "max_speedup_vs_w16": max_w16,
        "rows": rows,
    }
    print("RESULT " + json.dumps(result))


if __name__ == "__main__":
    main()
