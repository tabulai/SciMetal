#!/usr/bin/env python
"""SciMetal probe E5: can MLX's GPU FFT kernels be vendored into our own
Metal runtime (no libmlx) and keep their speed?

Re-run with:
    cd probes/e5_fft && ./build.sh && ../../.venv/bin/python run.py

(run.py builds automatically if build/ is missing or stale.)  The last line
of output is `RESULT {json}`; the same JSON is saved to build/last_result.json.
"""
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import scipy
import scipy.fft

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def ensure_built():
    srcs = ["e5fft.mm", "e5_ext.metal", "build.sh",
            "vendor/mlx/backend/metal/kernels/fft.h",
            "vendor/mlx/backend/metal/kernels/fft/readwrite.h",
            "vendor/mlx/backend/metal/kernels/fft/radix.h",
            "vendor/mlx/backend/metal/kernels/fft.metal"]
    outs = ["build/libe5fft.dylib", "build/mlx_fft.metallib", "build/e5_ext.metallib"]
    newest_src = max(os.path.getmtime(os.path.join(HERE, s)) for s in srcs)
    stale = any(not os.path.exists(os.path.join(HERE, o)) or
                os.path.getmtime(os.path.join(HERE, o)) < newest_src for o in outs)
    if stale:
        print("[build] running build.sh", flush=True)
        subprocess.check_call(["/bin/bash", os.path.join(HERE, "build.sh")])


ensure_built()

import e5lib  # noqa: E402
from e5lib import C2C, C2R, R2C, Plan  # noqa: E402

import mlx.core as mx  # noqa: E402  (reference / competitor only -- never linked into the runtime)

RNG = np.random.default_rng(1234)
T_START = time.perf_counter()


# ---------------------------------------------------------------------------
# Probe-level planner for power-of-two c2c / rfft / irfft
# ---------------------------------------------------------------------------
def choose_plan(n):
    """Returns (factors, coalesce) for a SciMetal multi-step plan, or None to
    use MLX's own plan unchanged (bitwise-identical to mx.fft).

    n <= 2^20 : MLX's plan (1 kernel for n <= 4096, else 2-pass four-step).
    2^21,2^22 : still 2 passes, but with sub-FFTs of 2048 (MLX refuses these
                because it insists on 4 FFTs per threadgroup).
    >= 2^23   : radix-16/32 passes. Once the array no longer fits the GPU's
                caches, a strided pass is only cheap if each threadgroup
                touches <= ~32 streams, so many small passes beat few big ones.
    """
    lg = n.bit_length() - 1
    if n != 1 << lg:
        return None
    if lg <= 20:
        return None
    if lg == 21:
        return [1024, 2048], 4
    if lg == 22:
        return [2048, 2048], 4
    bits = lg - 4
    k = math.ceil(bits / 5)
    base, extra = divmod(bits, k)
    parts = [base + 1] * extra + [base] * (k - extra)
    return [16] + [1 << b for b in parts], 16


def make_plan(n, rows=1, inverse=False, kind=C2C):
    ch = choose_plan(n)
    if ch is None:
        return Plan(n, rows=rows, inverse=inverse, kind=kind)
    return Plan(n, rows=rows, inverse=inverse, kind=kind, factors=ch[0], coalesce=ch[1])


def plan_label(p):
    if p.factors is None:
        return "mlx-plan"
    return "ms" + "x".join(str(f) for f in p.factors)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def crandn(shape):
    out = np.empty(shape, dtype=np.complex64)
    out.real = RNG.standard_normal(shape, dtype=np.float32)
    out.imag = RNG.standard_normal(shape, dtype=np.float32)
    return out


def relerr(a, ref):
    return float(np.linalg.norm(a.astype(ref.dtype, copy=False) - ref) / np.linalg.norm(ref))


def bit_compare(a, b):
    """(number of differing float32 words, relative L2 difference)."""
    a = np.ascontiguousarray(a)
    b = np.ascontiguousarray(b)
    nd = int(np.count_nonzero(a.view(np.uint32) != b.view(np.uint32)))
    l2 = float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))
    return nd, l2


def med_ms(ts):
    return round(1e3 * float(np.median(ts)), 4)


def min_ms(ts):
    return round(1e3 * float(np.min(ts)), 4)


def timeit(fn, reps, warm=2):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return ts


def mlx_fft_np(x, inverse=False):
    xm = mx.array(x)
    y = mx.fft.ifft(xm) if inverse else mx.fft.fft(xm)
    mx.eval(y)
    return np.array(y)


def log(msg):
    print("[%6.1fs] %s" % (time.perf_counter() - T_START, msg), flush=True)


# ---------------------------------------------------------------------------
# 1. correctness, c2c power-of-two 2^10 .. 2^24, forward + inverse
# ---------------------------------------------------------------------------
def correctness_pow2():
    rows_out = []
    for lg in range(10, 25):
        n = 1 << lg
        x = crandn((1, n))
        x128 = x.astype(np.complex128)
        for inverse in (False, True):
            ref = (scipy.fft.ifft if inverse else scipy.fft.fft)(x128, axis=-1)
            p = make_plan(n, inverse=inverse)
            p.x[:] = x
            p.exec()
            rec = {
                "log2n": lg, "dir": "inv" if inverse else "fwd",
                "plan": plan_label(p), "passes": p.num_passes, "temps": p.num_temps,
                "relerr_vs_scipy_c128": relerr(p.y, ref),
            }
            # MLX's own answer
            try:
                ym = mlx_fft_np(x, inverse)
                merr = relerr(ym, ref)
                rec["mlx_relerr_vs_scipy_c128"] = merr
                rec["mlx_status"] = "ok" if merr < 1e-4 else "WRONG_RESULT"
                nd, l2 = bit_compare(p.y, ym)
                if rec["mlx_status"] == "ok":
                    rec["bitwise_equal_to_mlx"] = nd == 0
                    rec["l2_reldiff_vs_mlx"] = l2
            except Exception as e:  # MLX raises for 2^21 and 2^22
                rec["mlx_status"] = "RAISES"
                rec["mlx_error"] = str(e).strip().splitlines()[0][:120]
                ym = None
            # Our faithful port of MLX's plan: must agree with MLX bit for bit,
            # including where MLX is broken.
            try:
                pc = Plan(n, inverse=inverse)
                pc.x[:] = x
                pc.exec()
                rec["port_of_mlx_plan"] = {
                    "relerr_vs_scipy_c128": relerr(pc.y, ref),
                    "bitwise_equal_to_mlx": (bit_compare(pc.y, ym)[0] == 0) if ym is not None else None,
                }
                pc.close()
            except RuntimeError as e:
                rec["port_of_mlx_plan"] = {"error": str(e).splitlines()[0][:120]}
            p.close()
            rows_out.append(rec)
    return rows_out


# ---------------------------------------------------------------------------
# 2. stretch coverage
# ---------------------------------------------------------------------------
def stretch_case(kind, n, rows, inverse=False, factors=None, coalesce=4, real_dc=False):
    half = n // 2 + 1
    name = {C2C: "ifft" if inverse else "fft", R2C: "rfft", C2R: "irfft"}[kind]
    rec = {"op": name, "n": n, "rows": rows}
    if kind == C2C:
        x = crandn((rows, n))
        ref = (scipy.fft.ifft if inverse else scipy.fft.fft)(x.astype(np.complex128), axis=-1)
        mfn = (lambda a: mx.fft.ifft(a)) if inverse else (lambda a: mx.fft.fft(a))
    elif kind == R2C:
        x = RNG.standard_normal((rows, n), dtype=np.float32)
        ref = scipy.fft.rfft(x.astype(np.float64), axis=-1)
        mfn = lambda a: mx.fft.rfft(a)  # noqa: E731
    else:
        x = crandn((rows, half))
        if real_dc:
            x[:, 0] = x[:, 0].real
            rec["input"] = "x[0] real"
        ref = scipy.fft.irfft(x.astype(np.complex128), n=n, axis=-1)
        mfn = lambda a: mx.fft.irfft(a, n=n)  # noqa: E731
    try:
        if factors is None:
            p = Plan(n, rows=rows, inverse=(inverse or kind == C2R), kind=kind)
        else:
            p = Plan(n, rows=rows, inverse=(inverse or kind == C2R), kind=kind,
                     factors=factors, coalesce=coalesce)
    except RuntimeError as e:
        rec["status"] = "NOT_COVERED"
        rec["error"] = str(e).splitlines()[0][:110]
        return rec
    p.x[:] = x
    p.exec()
    rec["plan"] = plan_label(p)
    rec["kernel"] = p.describe()[0]["kernel"].split("_mem_")[0]
    rec["passes"] = p.num_passes
    rec["relerr_vs_scipy"] = relerr(p.y, ref)
    try:
        ymx = mfn(mx.array(x))
        mx.eval(ymx)  # explicit eval: an implicit one (np.array) aborts the process when MLX throws
        ym = np.array(ymx)
        rec["mlx_relerr_vs_scipy"] = relerr(ym, ref)
        nd, l2 = bit_compare(p.y, ym)
        rec["bitwise_equal_to_mlx"] = nd == 0
        rec["l2_reldiff_vs_mlx"] = l2
    except Exception as e:
        rec["mlx_error"] = str(e).strip().splitlines()[0][:100]
    rec["status"] = "ok" if rec["relerr_vs_scipy"] < 1e-4 else "WRONG_RESULT"
    if rec["status"] != "ok" and rec.get("mlx_relerr_vs_scipy", 0) > 1e-4 and rec.get("l2_reldiff_vs_mlx", 1) < 1e-5:
        rec["note"] = ("same answer as MLX: MLX's Bluestein irfft kernel does not drop imag(x[0]) "
                       "the way NumPy/SciPy do, so non-Hermitian-consistent input differs")
    p.close()
    return rec


def stretch():
    out = []
    # rfft / irfft through MLX's kernels (plan port), pow2
    for n in (1024, 4096, 1 << 16, 1 << 20):
        out.append(stretch_case(R2C, n, 3))
        out.append(stretch_case(C2R, n, 3))
    # rfft / irfft beyond MLX's limit through the multi-step plan
    for n in (1 << 22, 1 << 24):
        f, w = choose_plan(n)
        out.append(stretch_case(R2C, n, 1, factors=f, coalesce=w))
        out.append(stretch_case(C2R, n, 1, factors=f, coalesce=w))
    # non-power-of-two sizes MLX handles with its radix set (n <= 4096)
    for n in (1000, 2187, 3003, 4095):
        out.append(stretch_case(C2C, n, 5))
        out.append(stretch_case(C2C, n, 2, inverse=True))
    out.append(stretch_case(R2C, 1000, 3))
    out.append(stretch_case(C2R, 1000, 3))
    # Rader (single prime factor p > 13 with smooth p-1, n <= 2048)
    for n in (17, 1009, 2003):
        out.append(stretch_case(C2C, n, 5))
        out.append(stretch_case(C2C, n, 2, inverse=True))
    out.append(stretch_case(R2C, 1009, 3))
    # Bluestein, fused single-kernel form (n <= 2048)
    for n in (47, 1021, 2038):
        out.append(stretch_case(C2C, n, 5))
        out.append(stretch_case(C2C, n, 2, inverse=True))
    out.append(stretch_case(R2C, 1021, 3))
    out.append(stretch_case(C2R, 1021, 3))               # general input: MLX's kernel disagrees with NumPy/SciPy
    out.append(stretch_case(C2R, 1021, 3, real_dc=True))  # Hermitian-consistent input: fine
    # MLX's "multi-upload Bluestein" sizes: not ported
    for n in (3001, 5000):
        out.append(stretch_case(C2C, n, 2))
    # beyond MLX: smooth non-power-of-two n > 4096 through the multi-step plan
    out.append(stretch_case(C2C, 6000, 3, factors=[60, 100], coalesce=4))
    out.append(stretch_case(C2C, 10 ** 6, 1, factors=[1000, 1000], coalesce=4))
    out.append(stretch_case(C2C, 10 ** 6, 1, inverse=True, factors=[100, 100, 100], coalesce=4))
    return out


# ---------------------------------------------------------------------------
# 3. speed
# ---------------------------------------------------------------------------
def vdsp_bench(x2d, ref, reps=5):
    rows, n = x2d.shape
    lg = n.bit_length() - 1
    L = e5lib.lib()
    h = L.e5_vdsp_create(lg)
    out = np.empty_like(x2d)
    res = {}
    names = {0: "fft_zop_split", 1: "ctoz+fft_zop+ztoc", 2: "DFT_zop_split", 3: "DFT_interleaved"}
    try:
        if rows == 1:
            for v, nm in names.items():
                if not L.e5_vdsp_available(h, v):
                    res[nm] = {"status": "unsupported length (setup returned NULL)"}
                    continue
                ts = []
                for i in range(reps + 1):
                    t = L.e5_vdsp_run(h, v, x2d.ctypes.data, out.ctypes.data)
                    if i:
                        ts.append(t)
                res[nm] = {"ms": med_ms(ts), "relerr": relerr(out, ref)}
        else:
            for v in (1, 3):
                if not L.e5_vdsp_available(h, v):
                    continue
                for threads in (1, 16):
                    ts = []
                    for i in range(reps + 1):
                        t = L.e5_vdsp_run_batch(h, v, rows, threads, x2d.ctypes.data, out.ctypes.data)
                        if i:
                            ts.append(t)
                    res["%s_rowloop_%dthr" % (names[v], threads)] = {
                        "ms": med_ms(ts), "relerr": relerr(out, ref)}
    finally:
        L.e5_vdsp_destroy(h)
    return res


def speed_case(label, rows, n, rounds=15, inner=2):
    x = crandn((rows, n))
    ref = scipy.fft.fft(x.astype(np.complex128), axis=-1)
    rec = {"rows": rows, "n": n}

    # -- our runtime (probe planner) --
    p = make_plan(n, rows=rows)
    p.x[:] = x
    for _ in range(3):
        p.exec()
    rec["ours"] = {"plan": plan_label(p), "passes": p.num_passes, "temps": p.num_temps,
                   "relerr": relerr(p.y, ref)}

    # -- our runtime running MLX's exact plan (port fidelity / dispatch parity) --
    pc = None
    if p.factors is not None:
        try:
            pc = Plan(n, rows=rows)
            pc.x[:] = x
            for _ in range(3):
                pc.exec()
            rec["ours_port_of_mlx_plan"] = {"passes": pc.num_passes, "temps": pc.num_temps,
                                            "relerr": relerr(pc.y, ref)}
        except RuntimeError as e:
            rec["ours_port_of_mlx_plan"] = {"error": str(e).splitlines()[0][:120]}
            pc = None

    # -- MLX --
    xm = mx.array(x)
    mx.eval(xm)
    mlx_ok = True
    try:
        y = mx.fft.fft(xm)
        mx.eval(y)
        ym = np.array(y)
        merr = relerr(ym, ref)
        rec["mlx"] = {"relerr": merr, "status": "ok" if merr < 1e-4 else "WRONG_RESULT"}
        rec["ours"]["bitwise_equal_to_mlx"] = bit_compare(p.y, ym)[0] == 0
        if pc is not None:
            rec["ours_port_of_mlx_plan"]["bitwise_equal_to_mlx"] = bit_compare(pc.y, ym)[0] == 0
        for _ in range(2):
            y = None
            y = mx.fft.fft(xm)
            mx.eval(y)
    except Exception as e:
        mlx_ok = False
        rec["mlx"] = {"status": "RAISES", "error": str(e).strip().splitlines()[0][:120]}

    # -- interleaved timing rounds (robust against drifting background load) --
    t_ours, g_ours, t_pc, g_pc, t_mlx = [], [], [], [], []
    for _ in range(rounds):
        for _ in range(inner):
            t0 = time.perf_counter()
            p.exec()
            t_ours.append(time.perf_counter() - t0)
            g_ours.append(p.gpu_seconds())
        if pc is not None:
            for _ in range(inner):
                t0 = time.perf_counter()
                pc.exec()
                t_pc.append(time.perf_counter() - t0)
                g_pc.append(pc.gpu_seconds())
        if mlx_ok:
            for _ in range(inner):
                y = None
                y = mx.fft.fft(xm)          # lazy graph node; not timed
                t0 = time.perf_counter()
                mx.eval(y)                  # timed: schedule + encode + commit + wait
                t_mlx.append(time.perf_counter() - t0)
    rec["ours"].update(ms=med_ms(t_ours), min_ms=min_ms(t_ours), gpu_ms=med_ms(g_ours))
    if pc is not None:
        rec["ours_port_of_mlx_plan"].update(ms=med_ms(t_pc), min_ms=min_ms(t_pc), gpu_ms=med_ms(g_pc))
    if mlx_ok:
        rec["mlx"].update(ms=med_ms(t_mlx), min_ms=min_ms(t_mlx))

    # per-pass GPU time (one command buffer per pass; diagnostic only)
    p.set_split_cb(1)
    pp = []
    for _ in range(5):
        p.exec()
        pp.append(p.pass_gpu_seconds())
    p.set_split_cb(0)
    rec["ours"]["per_pass_gpu_ms"] = [round(1e3 * float(v), 3) for v in np.median(np.array(pp), axis=0)]
    rec["ours"]["dispatches"] = [
        {"kernel": d["kernel"], "grid": d["grid"], "group": d["group"]} for d in p.describe()]

    y = None
    xm = None
    p.close()
    if pc is not None:
        pc.close()

    # -- scipy (pocketfft), complex64 --
    for w in (1, 16):
        ts = timeit(lambda: scipy.fft.fft(x, axis=-1, workers=w), reps=5, warm=1)
        rec["scipy_workers%d" % w] = {"ms": med_ms(ts)}
    rec["scipy_workers1"]["relerr"] = relerr(scipy.fft.fft(x, axis=-1, workers=1), ref)

    # -- Accelerate vDSP --
    rec["vdsp"] = vdsp_bench(x, ref)
    log("speed %-10s ours %.3f ms (%s) | mlx %s | scipy w1 %.2f ms" % (
        label, rec["ours"]["ms"], rec["ours"]["plan"],
        ("%.3f ms [%s]" % (rec["mlx"]["ms"], rec["mlx"]["status"])) if mlx_ok else "RAISES",
        rec["scipy_workers1"]["ms"]))
    return rec


def fixed_overhead():
    n = 1024
    x = crandn((1, n))
    p = Plan(n)
    p.x[:] = x
    xm = mx.array(x)
    mx.eval(xm)
    for _ in range(5):
        p.exec()
        y = mx.fft.fft(xm)
        mx.eval(y)
    t_o, g_o, t_m, t_e = [], [], [], []
    L = e5lib.lib()
    for _ in range(8):
        for _ in range(5):
            t0 = time.perf_counter()
            p.exec()
            t_o.append(time.perf_counter() - t0)
            g_o.append(p.gpu_seconds())
        for _ in range(5):
            y = None
            y = mx.fft.fft(xm)
            t0 = time.perf_counter()
            mx.eval(y)
            t_m.append(time.perf_counter() - t0)
        for _ in range(5):
            t_e.append(L.e5_empty_roundtrip())
    ts = timeit(lambda: scipy.fft.fft(x, axis=-1, workers=1), reps=20, warm=3)
    p.close()
    return {"n": n,
            "ours_ms": med_ms(t_o), "ours_min_ms": min_ms(t_o), "ours_gpu_ms": med_ms(g_o),
            "mlx_ms": med_ms(t_m), "mlx_min_ms": min_ms(t_m),
            "empty_command_buffer_roundtrip_ms": med_ms(t_e),
            "scipy_workers1_ms": med_ms(ts)}


def plan_sweep_2p24():
    """Why many small passes: alternatives at n = 2^24 (interleaved)."""
    n = 1 << 24
    x = crandn((1, n))
    cands = [("port of MLX plan (WRONG RESULT)", None, 0),
             ("16x32x32x32x32", [16, 32, 32, 32, 32], 16),
             ("16^6", [16] * 6, 16),
             ("16x2048x512", [16, 2048, 512], 16),
             ("256x256x256", [256, 256, 256], 16),
             ("4096x4096 (coalesce 1)", [4096, 4096], 4)]
    plans = []
    for name, f, w in cands:
        p = Plan(n) if f is None else Plan(n, factors=f, coalesce=w)
        p.x[:] = x
        for _ in range(2):
            p.exec()
        plans.append((name, p, [], []))
    for _ in range(4):
        for name, p, ts, gs in plans:
            for _ in range(2):
                t0 = time.perf_counter()
                p.exec()
                ts.append(time.perf_counter() - t0)
                gs.append(p.gpu_seconds())
    out = []
    for name, p, ts, gs in plans:
        p.set_split_cb(1)
        pp = []
        for _ in range(3):
            p.exec()
            pp.append(p.pass_gpu_seconds())
        out.append({"plan": name, "passes": p.num_passes, "ms": med_ms(ts), "gpu_ms": med_ms(gs),
                    "per_pass_gpu_ms": [round(1e3 * float(v), 2) for v in np.median(np.array(pp), axis=0)]})
        p.close()
    return out


def shipped_metallib_check():
    """A/B: run the same plans against the metallib shipped in the MLX wheel
    (a data file, not libmlx) instead of our own offline compile."""
    path = os.path.join(os.path.dirname(mx.__file__), "lib", "mlx.metallib")
    if not os.path.exists(path):
        return {"status": "mlx.metallib not found"}
    out = {"path_suffix": "site-packages/mlx/lib/mlx.metallib"}
    n = 1 << 20
    x = crandn((1, n))
    plans = {}
    for tag, lib in (("ours", e5lib.FFT_METALLIB), ("wheel", path)):
        e5lib.init(fft_metallib=lib)   # plans keep their own pipelines alive
        p = Plan(n)
        p.x[:] = x
        for _ in range(3):
            p.exec()
        plans[tag] = (p, [])
    e5lib.init()  # back to our own build
    for _ in range(15):
        for tag, (p, ts) in plans.items():
            for _ in range(2):
                t0 = time.perf_counter()
                p.exec()
                ts.append(time.perf_counter() - t0)
    out["n"] = n
    out["bitwise_equal_outputs"] = bit_compare(plans["ours"][0].y, plans["wheel"][0].y)[0] == 0
    out["our_offline_compile_ms"] = med_ms(plans["ours"][1])
    out["wheel_metallib_ms"] = med_ms(plans["wheel"][1])
    for p, _ in plans.values():
        p.close()
    return out


def sig(v, digits=4):
    if isinstance(v, bool) or not isinstance(v, float):
        return v
    if v != v or v in (float("inf"), float("-inf")):
        return None
    return float("%.*e" % (digits - 1, v))


def rounded(o):
    if isinstance(o, dict):
        return {k: rounded(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [rounded(v) for v in o]
    return sig(o)


def compact(result):
    """Same measurements, tabular form for the long per-size lists."""
    out = dict(result)
    cols = ["log2n", "dir", "plan", "passes", "temps", "relerr_vs_scipy_c128",
            "mlx_status", "mlx_relerr_vs_scipy_c128", "ours_bitwise_equal_to_mlx",
            "port_of_mlx_plan"]
    rows = []
    for r in result["correctness_c2c_pow2"]:
        port = r["port_of_mlx_plan"]
        if "error" in port:
            pm = "same_error_as_mlx" if r["mlx_status"] == "RAISES" else "error"
        elif port["bitwise_equal_to_mlx"]:
            pm = "bitwise_equal_to_mlx" + ("" if r["mlx_status"] == "ok" else "(both wrong)")
        else:
            pm = "differs_from_mlx"
        rows.append([r["log2n"], r["dir"], r["plan"], r["passes"], r["temps"],
                     r["relerr_vs_scipy_c128"], r["mlx_status"],
                     r.get("mlx_relerr_vs_scipy_c128"), r.get("bitwise_equal_to_mlx"), pm])
    out["correctness_c2c_pow2"] = {"columns": cols, "rows": rows}
    cols = ["op", "n", "rows", "status", "plan", "kernel", "passes", "relerr_vs_scipy",
            "mlx_relerr_vs_scipy", "bitwise_equal_to_mlx", "l2_reldiff_vs_mlx", "note"]
    rows = []
    for r in result["stretch"]:
        note = r.get("error") or r.get("note") or r.get("input") or \
            ("mlx raises" if "mlx_error" in r else None)
        rows.append([r["op"], r["n"], r["rows"], r["status"], r.get("plan"), r.get("kernel"),
                     r.get("passes"), r.get("relerr_vs_scipy"), r.get("mlx_relerr_vs_scipy"),
                     r.get("bitwise_equal_to_mlx"), r.get("l2_reldiff_vs_mlx"), note])
    out["stretch"] = {"columns": cols, "rows": rows}
    speed = {}
    for k, v in result["speed"].items():
        v = dict(v)
        v["ours"] = dict(v["ours"])
        if k != "n=2^24":
            v["ours"].pop("dispatches", None)
        speed[k] = v
    out["speed"] = speed
    return rounded(out)


# ---------------------------------------------------------------------------
def main():
    e5lib.init()
    result = {
        "probe": "e5_fft",
        "device": e5lib.device_name(),
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__, "mlx": mx.__version__,
                     "python": sys.version.split()[0]},
        "loadavg_start": [round(v, 2) for v in os.getloadavg()],
    }

    log("correctness: c2c power-of-two 2^10..2^24 (fwd + inv)")
    result["correctness_c2c_pow2"] = correctness_pow2()
    log("stretch coverage")
    result["stretch"] = stretch()

    log("speed")
    speed = {}
    speed["n=2^16"] = speed_case("n=2^16", 1, 1 << 16)
    speed["n=2^20"] = speed_case("n=2^20", 1, 1 << 20)
    speed["n=2^22"] = speed_case("n=2^22", 1, 1 << 22)
    speed["n=2^24"] = speed_case("n=2^24", 1, 1 << 24)
    speed["4096x4096"] = speed_case("4096x4096", 4096, 4096)
    result["speed"] = speed
    result["fixed_overhead_n1024"] = fixed_overhead()
    log("plan alternatives at 2^24")
    result["plan_alternatives_2p24"] = plan_sweep_2p24()
    try:
        result["shipped_metallib_check"] = shipped_metallib_check()
    except Exception as e:  # diagnostic only
        result["shipped_metallib_check"] = {"error": str(e)[:200]}
    result["pipelines"] = {"built": e5lib.lib().e5_pso_built(),
                           "total_build_s": round(e5lib.lib().e5_pso_build_seconds(), 3)}

    # ---------------- gate ----------------
    s24, s20 = speed["n=2^24"], speed["n=2^20"]
    gate = {"threshold": 0.8, "definition": "speed ratio = mlx_ms / ours_ms (medians)"}
    mlx24 = s24["mlx"]
    gate["n=2^24"] = {
        "ours_ms": s24["ours"]["ms"], "ours_relerr": s24["ours"]["relerr"], "ours_plan": s24["ours"]["plan"],
        "mlx_status": mlx24["status"], "mlx_ms": mlx24.get("ms"), "mlx_relerr": mlx24.get("relerr"),
    }
    if mlx24.get("ms"):
        r = mlx24["ms"] / s24["ours"]["ms"]
        gate["n=2^24"]["ratio_ours_correct_vs_mlx"] = round(r, 3)
        gate["n=2^24"]["ratio_min_based"] = round(mlx24["min_ms"] / s24["ours"]["min_ms"], 3)
    pc24 = s24.get("ours_port_of_mlx_plan", {})
    if pc24.get("ms") and mlx24.get("ms"):
        gate["n=2^24"]["ratio_same_plan_as_mlx"] = round(mlx24["ms"] / pc24["ms"], 3)
        gate["n=2^24"]["same_plan_bitwise_equal_to_mlx"] = pc24.get("bitwise_equal_to_mlx")
    gate["n=2^20_largest_size_mlx_gets_right"] = {
        "ours_ms": s20["ours"]["ms"], "mlx_ms": s20["mlx"]["ms"],
        "ratio": round(s20["mlx"]["ms"] / s20["ours"]["ms"], 3),
        "bitwise_equal_to_mlx": s20["ours"].get("bitwise_equal_to_mlx"),
    }
    r24 = gate["n=2^24"].get("ratio_ours_correct_vs_mlx")
    mlx_correct_24 = mlx24["status"] == "ok"
    if r24 is None:
        verdict = "NOT_EVALUABLE (MLX raises at 2^24)"
    elif r24 >= 0.8:
        verdict = "PASS" if mlx_correct_24 else \
            "PASS on speed (but MLX's own 2^24 output is wrong, so this compares our correct transform with MLX's incorrect one)"
    else:
        verdict = "FAIL" if mlx_correct_24 else \
            "FAIL on speed vs MLX's timing (but MLX's own 2^24 output is wrong)"
    gate["verdict"] = verdict
    result["gate"] = gate
    result["loadavg_end"] = [round(v, 2) for v in os.getloadavg()]
    result["wall_s"] = round(time.perf_counter() - T_START, 1)

    # ---------------- human-readable summary ----------------
    print()
    print("== correctness: c2c power-of-two, relative L2 error vs scipy.fft (complex128) ==")
    print("%-5s %-4s %-20s %-6s %-10s | %-28s | %s" % (
        "log2n", "dir", "our plan", "passes", "ours", "MLX mx.fft", "ours vs MLX"))
    for r in result["correctness_c2c_pow2"]:
        if r["mlx_status"] == "RAISES":
            m = "RAISES"
            cmpm = "port of MLX plan: " + ("same error" if "error" in r["port_of_mlx_plan"] else "?")
        else:
            m = "%.2e %s" % (r["mlx_relerr_vs_scipy_c128"], "" if r["mlx_status"] == "ok" else "WRONG")
            if r["mlx_status"] == "ok":
                cmpm = "bitwise identical" if r.get("bitwise_equal_to_mlx") else \
                    "L2 reldiff %.1e" % r.get("l2_reldiff_vs_mlx", float("nan"))
            else:
                cmpm = "port of MLX plan reproduces MLX's wrong output bitwise: %s" % \
                    r["port_of_mlx_plan"].get("bitwise_equal_to_mlx")
        print("%-5d %-4s %-20s %-6d %-10.2e | %-28s | %s" % (
            r["log2n"], r["dir"], r["plan"], r["passes"], r["relerr_vs_scipy_c128"], m, cmpm))

    print()
    print("== stretch coverage ==")
    for r in result["stretch"]:
        if r["status"] == "NOT_COVERED":
            print("%-5s n=%-8d rows=%d NOT COVERED (%s)" % (r["op"], r["n"], r["rows"], r["error"][:60]))
            continue
        if "mlx_error" in r:
            vs = "MLX raises"
        elif r.get("bitwise_equal_to_mlx"):
            vs = "bitwise identical to MLX"
        elif r["mlx_relerr_vs_scipy"] > 1e-4 and r["status"] == "ok":
            vs = "MLX result WRONG (relerr %.1e)" % r["mlx_relerr_vs_scipy"]
        else:
            vs = "vs MLX: L2 reldiff %.1e (MLX relerr %.1e)" % (r["l2_reldiff_vs_mlx"], r["mlx_relerr_vs_scipy"])
        print("%-5s n=%-8d rows=%d %-17s %-13s passes=%d relerr %.2e %-12s %s %s" % (
            r["op"], r["n"], r["rows"], r["plan"], r["kernel"], r["passes"], r["relerr_vs_scipy"],
            r["status"], vs, r.get("input", "") or ("[" + r["note"][:60] + "...]" if "note" in r else "")))

    print()
    print("== speed, complex64 forward, median ms [min ms] (lower is better) ==")
    hdr = "%-10s | %-17s %-15s %7s | %-24s | %9s %9s | %s"
    print(hdr % ("case", "ours (plan)", "wall", "gpu", "MLX mx.eval", "scipy w1", "scipy w16", "vDSP (best variant)"))
    for label, r in speed.items():
        m = r["mlx"]
        if m["status"] == "RAISES":
            ms = "RAISES"
        else:
            ms = "%.3f [%.3f]%s" % (m["ms"], m["min_ms"], "" if m["status"] == "ok" else " WRONG")
        vals = [(k, d["ms"]) for k, d in r["vdsp"].items() if "ms" in d and d["relerr"] < 1e-4]
        vbest = min(vals, key=lambda kv: kv[1]) if vals else ("-", float("nan"))
        print(hdr % (label, r["ours"]["plan"], "%.3f [%.3f]" % (r["ours"]["ms"], r["ours"]["min_ms"]),
                     "%.3f" % r["ours"]["gpu_ms"], ms,
                     "%.2f" % r["scipy_workers1"]["ms"], "%.2f" % r["scipy_workers16"]["ms"],
                     "%.2f (%s)" % (vbest[1], vbest[0])))
    fo = result["fixed_overhead_n1024"]
    print("fixed overhead (n=1024): ours %.3f [min %.3f] ms (gpu %.3f) | MLX %.3f [min %.3f] ms | "
          "empty command buffer %.3f ms | scipy %.4f ms" % (
              fo["ours_ms"], fo["ours_min_ms"], fo["ours_gpu_ms"], fo["mlx_ms"], fo["mlx_min_ms"],
              fo["empty_command_buffer_roundtrip_ms"], fo["scipy_workers1_ms"]))
    print("shipped-metallib A/B:", json.dumps(result["shipped_metallib_check"]))
    print("load average at start/end:", result["loadavg_start"], result["loadavg_end"],
          "(high values = other jobs running; timings are then noisy)")
    print()
    print("== plan alternatives at n = 2^24 (our runtime) ==")
    for r in result["plan_alternatives_2p24"]:
        print("%-34s passes=%d wall %.3f ms gpu %.3f ms per-pass gpu %s" % (
            r["plan"], r["passes"], r["ms"], r["gpu_ms"], r["per_pass_gpu_ms"]))
    print()
    print("== gate: >= 80% of MLX's speed at n = 2^24 complex64 ==")
    print(json.dumps(gate, indent=1))
    print()
    line = json.dumps(compact(result), separators=(",", ":"))
    with open(os.path.join(HERE, "build", "last_result.json"), "w") as f:
        f.write(line + "\n")
    print("RESULT " + line)


if __name__ == "__main__":
    main()
