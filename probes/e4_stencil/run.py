#!/usr/bin/env python
"""E4 probe: separable Gaussian filter, 4096 x 4096, scipy.ndimage.gaussian_filter semantics.

    ./build.sh && ../../.venv/bin/python run.py            # ~35 s
    ../../.venv/bin/python run.py --no-mlx                 # skip the optional MLX comparison
    ../../.venv/bin/python run.py --validate               # also run the self-test under MTL_DEBUG_LAYER=1
                                                           # + MTL_SHADER_VALIDATION=1 (~15 s more)

Last line printed: `RESULT {json}`.
"""
import ctypes
import gc
import json
import os
import statistics
import sys
import time

import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
u64 = ctypes.c_uint64
u32 = ctypes.c_uint32
vp = ctypes.c_void_p
dbl_p = ctypes.POINTER(ctypes.c_double)
PAGE = 16384


class Job(ctypes.Structure):
    _fields_ = [("in_addr", u64), ("out_addr", u64), ("H", u32), ("W", u32), ("radius", u32), ("in_f64", u32),
                ("out_f64", u32), ("in_mode", u32), ("out_mode", u32), ("tg_w0", u32), ("tg_h0", u32), ("tg_w1", u32),
                ("tg_h1", u32), ("span", u32), ("split", u32), ("fast_narrow", u32), ("cache_in_wrap", u32), ("spin_wait", u32)]


BD = ("wrap_in", "cpu_in", "wrap_out", "encode_commit", "sync_wait", "cpu_out", "release", "gpu_time", "native_total",
      "gpu_narrow", "gpu_pass0", "gpu_pass1", "gpu_widen", "kernel_time")

IN_MODES = {"wrap_fused": 0, "cpu_vdsp": 1, "wrap_gpu_pass": 2}
OUT_MODES = {"wrap_fused": 0, "pooled_fused": 1, "cpu_vdsp": 2, "wrap_gpu_pass": 3}

# Dispatch shape used for the headline numbers (see the sweep in the report).
# Vertical pass (axis 0): 32x32 threadgroups; horizontal pass (axis 1): 1024x1.
TG = {"tg_w0": 32, "tg_h0": 32, "tg_w1": 1024, "tg_h1": 1}


def load():
    L = ctypes.CDLL(os.path.join(HERE, "build", "libe4.dylib"))
    L.e4_init.argtypes = [ctypes.c_char_p]
    L.e4_last_error.restype = ctypes.c_char_p
    L.e4_gpu_gauss.argtypes = [ctypes.POINTER(Job), vp, dbl_p, ctypes.POINTER(vp)]
    L.e4_handle_contents.argtypes = [vp]
    L.e4_handle_contents.restype = u64
    L.e4_pool_put.argtypes = [vp]
    L.e4_pool_put.restype = None
    L.e4_pool_clear.restype = None
    L.e4_max_threads_per_group.argtypes = [ctypes.c_int]
    L.e4_pool_fresh_allocations.restype = ctypes.c_long
    L.e4_cpu_gauss.argtypes = [vp, vp, u32, u32, vp, u32, ctypes.c_int, ctypes.c_int, u32, ctypes.c_int, ctypes.c_int,
                               ctypes.c_int, dbl_p]
    rc = L.e4_init(os.path.join(HERE, "build", "kernels.metallib").encode())
    if rc != 0:
        raise RuntimeError(f"e4_init failed: {rc} {L.e4_last_error().decode()}")
    return L


L = None


def ptr(a):
    return a.__array_interface__["data"][0]


def gaussian_weights(sigma, truncate=4.0, radius=None):
    """Exactly scipy.ndimage's order-0 kernel (correlate order; symmetric)."""
    sd = float(sigma)
    lw = int(truncate * sd + 0.5) if radius is None else int(radius)
    x = np.arange(-lw, lw + 1)
    phi = np.exp(-0.5 / (sd * sd) * x ** 2)
    phi = phi / phi.sum()
    return np.ascontiguousarray(phi[::-1]), lw


class PooledBuffer:
    """Owns a pooled shared MTLBuffer; a NumPy array created from it keeps it checked out (via .base)
    and it goes back to the pool when the last array referencing it is collected."""

    def __init__(self, handle, shape, dtype):
        self._h = handle
        self.__array_interface__ = {"version": 3, "shape": tuple(shape), "typestr": np.dtype(dtype).str,
                                    "data": (L.e4_handle_contents(handle), False)}

    def __del__(self):
        h, self._h = self._h, None
        if h:
            L.e4_pool_put(h)


def gpu_gauss(a, sigma, in_mode="wrap_fused", out_mode="wrap_fused", out_dtype=None, span=1, split=False,
              fast_narrow=False, cache_in_wrap=False, spin_wait=False, tg=None, bd=None):
    """Gaussian filter on the GPU in float32 arithmetic.  a: C-contiguous 2-D float32/float64 array.
    Returns a new array of out_dtype (default: a.dtype).  Everything a caller would pay for is in here."""
    w64, r = gaussian_weights(sigma)
    w32 = w64.astype(np.float32)
    a = np.ascontiguousarray(a)  # strided / F-ordered views fall back to a copy
    out_dtype = np.dtype(out_dtype or a.dtype)
    H, W = a.shape
    j = Job()
    j.in_addr = ptr(a)
    j.H, j.W, j.radius = H, W, r
    j.in_f64 = int(a.dtype == np.float64)
    j.out_f64 = int(out_dtype == np.float64)
    j.in_mode = IN_MODES[in_mode]
    j.out_mode = OUT_MODES[out_mode]
    t = dict(TG)
    if tg:
        t.update(tg)
    j.tg_w0, j.tg_h0, j.tg_w1, j.tg_h1 = t["tg_w0"], t["tg_h0"], t["tg_w1"], t["tg_h1"]
    j.span = span
    j.split = int(split)
    j.fast_narrow = int(fast_narrow)
    j.cache_in_wrap = int(cache_in_wrap)
    j.spin_wait = int(spin_wait)
    out = None
    if out_mode != "pooled_fused":
        out = np.empty((H, W), dtype=out_dtype)
        j.out_addr = ptr(out)
    buf = (ctypes.c_double * 16)()
    h = vp()
    rc = L.e4_gpu_gauss(ctypes.byref(j), ptr(w32), buf, ctypes.byref(h))
    if rc != 0:
        raise RuntimeError(f"e4_gpu_gauss rc={rc}: {L.e4_last_error().decode()}")
    if out_mode == "pooled_fused":
        out = np.asarray(PooledBuffer(h.value, (H, W), out_dtype))
    if bd is not None:
        bd.update({k: buf[i] for i, k in enumerate(BD)})
    return out


FMA_MODES = {"off": 0, "on": 1, "scipy_build": 2}


def cpu_gauss(a, sigma, workers=0, tile=0, rows_per_item=0, fma="scipy_build", ring=True, radius=None, bd=None):
    """Strict path: same separable passes in the array's own precision on the P-cores.
    fma: how each `acc += (a + b) * w` step is evaluated (see stencil.mm); "scipy_build" reproduces the
    installed SciPy wheel bit for bit, "off"/"on" differ from it by at most a few ulp."""
    w64, r = gaussian_weights(sigma, radius=radius)
    a = np.ascontiguousarray(a)
    out = np.empty_like(a)
    buf = (ctypes.c_double * 4)()
    rc = L.e4_cpu_gauss(ptr(a), ptr(out), a.shape[0], a.shape[1], ptr(w64), r, workers, tile, rows_per_item,
                        int(a.dtype == np.float32), FMA_MODES[fma], int(ring), buf)
    if rc != 0:
        raise RuntimeError("e4_cpu_gauss failed")
    if bd is not None:
        bd.update({"pass_axis0": buf[0], "pass_axis1": buf[1], "native_total": buf[2]})
    return out


def timeit(fn, reps=5, warmup=1):
    """Median wall time of fn() over reps, after warm-up.  Result arrays are dropped between calls."""
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        r = fn()
        ts.append(time.perf_counter() - t0)
        del r
    return statistics.median(ts), min(ts)


def errors(got, ref):
    got = np.asarray(got, dtype=np.float64)
    d = np.abs(got - ref)
    m = float(d.max())
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = d / np.abs(ref)
    return {"max_abs": m, "max_rel": float(np.nanmax(rel)), "max_abs_over_max_ref": m / float(np.abs(ref).max())}


def make_image(rng, n=4096):
    """Positive, image-like float64 data in [0, 1): smooth structure + noise."""
    y, x = np.mgrid[0:n, 0:n].astype(np.float32)
    img = 0.5 + 0.25 * np.sin(x / 97.0) * np.cos(y / 61.0)
    img = img.astype(np.float64) + 0.25 * (rng.random((n, n)) - 0.5)
    return np.ascontiguousarray(img)


def med(xs):
    return statistics.median(xs)


def bench_scipy(a, sigma, reps=5):
    ts, out = [], None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = ndimage.gaussian_filter(a, sigma)
        ts.append(time.perf_counter() - t0)
    return med(ts), min(ts), out


def bench_gpu(a, sigma, reps=9, warmup=2, keep=False, **kw):
    """End-to-end wall time of gpu_gauss() (median of reps) plus the median native breakdown.
    keep=True holds every result alive, so a pooled output buffer can never be recycled."""
    for _ in range(warmup):
        gpu_gauss(a, sigma, **kw)
    walls, rows, kept = [], [], []
    for _ in range(reps):
        bd = {}
        t0 = time.perf_counter()
        out = gpu_gauss(a, sigma, bd=bd, **kw)
        wall = time.perf_counter() - t0
        if keep:
            kept.append(out)
        del out
        walls.append(wall)
        rows.append({"wall": wall, "wrap": bd["wrap_in"] + bd["wrap_out"], "cpu_convert": bd["cpu_in"] + bd["cpu_out"],
                     "gpu_kernels": bd["gpu_time"],
                     "sync_overhead": bd["encode_commit"] + bd["sync_wait"] - bd["gpu_time"],
                     "release": bd["release"], "python_overhead": wall - bd["native_total"]})
    del kept
    res = {k + "_ms": round(med([row[k] for row in rows]) * 1e3, 3) for k in rows[0]}
    res["wall_min_ms"] = round(min(walls) * 1e3, 3)
    return res


def gpu_stage_times(a, sigma, reps=5, **kw):
    """Per-stage GPU execution time, one command buffer per stage (diagnostic)."""
    acc = {"gpu_narrow": [], "gpu_pass0": [], "gpu_pass1": [], "gpu_widen": []}
    for i in range(reps + 1):
        bd = {}
        gpu_gauss(a, sigma, split=True, bd=bd, **kw)
        if i:
            for k in acc:
                acc[k].append(bd[k])
    return {k + "_ms": round(med(v) * 1e3, 3) for k, v in acc.items()}


def bench_cpu(a, sigma, reps=7, **kw):
    cpu_gauss(a, sigma, **kw)
    walls, p0, p1 = [], [], []
    for _ in range(reps):
        bd = {}
        t0 = time.perf_counter()
        out = cpu_gauss(a, sigma, bd=bd, **kw)
        walls.append(time.perf_counter() - t0)
        del out
        p0.append(bd["pass_axis0"])
        p1.append(bd["pass_axis1"])
    return {"wall_ms": round(med(walls) * 1e3, 3), "wall_min_ms": round(min(walls) * 1e3, 3),
            "pass_axis0_ms": round(med(p0) * 1e3, 3), "pass_axis1_ms": round(med(p1) * 1e3, 3)}


def dispatch_sweep(a32, sigma):
    rows = []
    shapes = ((32, 1), (256, 1), (1024, 1), (32, 32), (64, 16), (128, 8), (16, 64), (8, 128), (1, 1024))
    for w, h in shapes:
        st = gpu_stage_times(a32, sigma, reps=4, in_mode="cpu_vdsp", out_mode="pooled_fused",
                             tg={"tg_w0": w, "tg_h0": h, "tg_w1": w, "tg_h1": h})
        rows.append({"threadgroup": f"{w}x{h}", "outputs_per_thread": 1, "pass_axis0_ms": st["gpu_pass0_ms"],
                     "pass_axis1_ms": st["gpu_pass1_ms"]})
    for span in (4, 8):
        st = gpu_stage_times(a32, sigma, reps=4, in_mode="cpu_vdsp", out_mode="pooled_fused", span=span,
                             tg={"tg_w0": 32, "tg_h0": 32, "tg_w1": 256, "tg_h1": 1})
        rows.append({"threadgroup": "32x32 / 256x1", "outputs_per_thread": span, "pass_axis0_ms": st["gpu_pass0_ms"],
                     "pass_axis1_ms": st["gpu_pass1_ms"]})
    return rows


def mlx_baseline(a, a32, sigma, ref64):
    import mlx.core as mx

    w64, r = gaussian_weights(sigma)
    w = mx.array(w64.astype(np.float32))

    def shifted(x):
        H, W = x.shape
        xp = mx.concatenate([x[:r][::-1], x, x[-r:][::-1]], axis=0)
        y = w[r] * xp[r:r + H]
        for d in range(r, 0, -1):
            y = y + (xp[r - d:r - d + H] + xp[r + d:r + d + H]) * w[r - d]
        yp = mx.concatenate([y[:, :r][:, ::-1], y, y[:, -r:][:, ::-1]], axis=1)
        z = w[r] * yp[:, r:r + W]
        for d in range(r, 0, -1):
            z = z + (yp[:, r - d:r - d + W] + yp[:, r + d:r + d + W]) * w[r - d]
        return z

    def conv(x):
        xp = mx.concatenate([x[:r][::-1], x, x[-r:][::-1]], axis=0)
        y = mx.conv2d(xp[None, :, :, None], w.reshape(1, 2 * r + 1, 1, 1))[0, :, :, 0]
        yp = mx.concatenate([y[:, :r][:, ::-1], y, y[:, -r:][:, ::-1]], axis=1)
        return mx.conv2d(yp[None, :, :, None], w.reshape(1, 1, 2 * r + 1, 1))[0, :, :, 0]

    out = {}
    for name, fn in (("compiled_shifted_sum", mx.compile(shifted)), ("conv2d", conv)):
        def f32_end_to_end():
            y = fn(mx.array(a32))
            mx.eval(y)
            return np.array(y)

        def f64_end_to_end():
            y = fn(mx.array(a.astype(np.float32)))
            mx.eval(y)
            return np.array(y).astype(np.float64)

        res = f32_end_to_end()  # also compiles
        x = mx.array(a32)
        mx.eval(x)

        def eval_only():
            y = fn(x)
            mx.eval(y)
            return y

        row = {"f32_in_f32_out_ms": round(timeit(f32_end_to_end, 5)[0] * 1e3, 2),
               "f64_in_f64_out_ms": round(timeit(f64_end_to_end, 5)[0] * 1e3, 2),
               "eval_only_ms": round(timeit(eval_only, 5)[0] * 1e3, 2)}
        row.update({"err_vs_scipy_f64_" + k: v for k, v in errors(res, ref64).items()})
        out[name] = row
    return out


def selftest(rng):
    """Small-array correctness sweep of every path against SciPy (odd shapes, radius > dimension,
    offset / unaligned / strided views, all in/out modes, sliding-window kernel)."""
    worst = {"gpu_rel": 0.0, "cpu_strict_abs": 0.0, "cpu_f32_rel": 0.0}
    n_gpu = n_cpu = 0
    failures = []
    for shape in ((37, 90), (300, 257), (513, 64), (64, 1030)):
        base = rng.random((shape[0] + 7, shape[1] + 5)) + 0.5
        tall = rng.random((shape[0] + 7, shape[1])) + 0.5
        views = {"contiguous": np.ascontiguousarray(base[:shape[0], :shape[1]]),
                 "row_offset_view": tall[5:5 + shape[0]],            # C-contiguous, pointer not page aligned
                 "strided_view": base[3:3 + shape[0], 2:2 + shape[1]]}  # not contiguous -> copy fallback
        assert views["row_offset_view"].flags.c_contiguous and not views["strided_view"].flags.c_contiguous
        for sigma in (0.8, 3, 10):
            for vname, v in views.items():
                ref = ndimage.gaussian_filter(v, sigma)
                scale = float(np.abs(ref).max())
                out = cpu_gauss(v, sigma)
                n_cpu += 1
                worst["cpu_strict_abs"] = max(worst["cpu_strict_abs"], float(np.abs(out - ref).max()))
                if not np.array_equal(out, ref):
                    failures.append(("cpu_strict_not_bit_identical", shape, sigma, vname))
                out = cpu_gauss(v.astype(np.float32), sigma, fma="on")
                worst["cpu_f32_rel"] = max(worst["cpu_f32_rel"], float(np.abs(out - ref).max()) / scale)
                combos = [(i, o, 1) for i in IN_MODES for o in OUT_MODES] if vname == "contiguous" else \
                    [("wrap_fused", "wrap_fused", 1), ("cpu_vdsp", "pooled_fused", 1)]
                if gaussian_weights(sigma)[1] <= 64:
                    combos += [("wrap_fused", "wrap_fused", 4), ("cpu_vdsp", "pooled_fused", 8)]
                for im, om, span in combos:
                    for arr in (v, v.astype(np.float32)):
                        if arr.dtype == np.float32 and (im == "wrap_gpu_pass" or om == "wrap_gpu_pass"):
                            continue
                        out = gpu_gauss(arr, sigma, in_mode=im, out_mode=om, span=span)
                        n_gpu += 1
                        e = float(np.abs(out.astype(np.float64) - ref).max()) / scale
                        worst["gpu_rel"] = max(worst["gpu_rel"], e)
                        if e > 2e-6 or out.dtype != arr.dtype:
                            failures.append(("gpu", shape, sigma, vname, im, om, span, str(arr.dtype), e))
    L.e4_pool_clear()
    return {"gpu_cases": n_gpu, "cpu_cases": n_cpu, "worst_gpu_err_over_max": worst["gpu_rel"],
            "worst_cpu_strict_abs_err": worst["cpu_strict_abs"], "worst_cpu_f32_err_over_max": worst["cpu_f32_rel"],
            "failures": failures[:5], "ok": not failures}


FAST_PIPELINES = (
    # name, kwargs
    ("A cpu_vdsp_in -> pooled_fused_out", dict(in_mode="cpu_vdsp", out_mode="pooled_fused")),
    ("B cpu_vdsp_in -> cpu_vdsp_out (plain NumPy output)", dict(in_mode="cpu_vdsp", out_mode="cpu_vdsp")),
    ("C wrap_fused_in -> wrap_fused_out (fully fused zero-copy)", dict(in_mode="wrap_fused", out_mode="wrap_fused")),
    ("C' wrap_fused_in (cheap narrow) -> wrap_fused_out", dict(in_mode="wrap_fused", out_mode="wrap_fused", fast_narrow=True)),
    ("D wrap + gpu narrow pass -> wrap_fused_out", dict(in_mode="wrap_gpu_pass", out_mode="wrap_fused")),
    ("E wrap + gpu narrow pass -> pooled_fused_out", dict(in_mode="wrap_gpu_pass", out_mode="pooled_fused")),
    ("F wrap (cached) + gpu narrow pass -> pooled_fused_out", dict(in_mode="wrap_gpu_pass", out_mode="pooled_fused", cache_in_wrap=True)),
    ("G cpu_vdsp_in -> wrap_fused_out", dict(in_mode="cpu_vdsp", out_mode="wrap_fused")),
    ("A* as A but busy-poll instead of waitUntilCompleted", dict(in_mode="cpu_vdsp", out_mode="pooled_fused", spin_wait=True)),
)


def main():
    global L
    if "--selftest-only" in sys.argv:
        L = load()
        print("SELFTEST " + json.dumps(selftest(np.random.default_rng(1))))
        return
    t_start = time.perf_counter()
    L = load()
    rng = np.random.default_rng(20261118)
    n = 4096
    a = make_image(rng, n)
    a32 = a.astype(np.float32)
    result = {"probe": "e4_stencil", "shape": [n, n], "numpy": np.__version__, "perf_cores": L.e4_perf_cores(),
              "threadgroups": dict(TG)}
    result["selftest"] = selftest(rng)
    print("selftest:", json.dumps(result["selftest"]))
    if "--validate" in sys.argv:  # same self-test under the Metal API debug layer + shader validation
        import subprocess
        env = dict(os.environ, MTL_DEBUG_LAYER="1", MTL_SHADER_VALIDATION="1")
        pr = subprocess.run([sys.executable, os.path.abspath(__file__), "--selftest-only"], env=env, capture_output=True,
                            text=True, timeout=600)
        st = [json.loads(ln[9:]) for ln in pr.stdout.splitlines() if ln.startswith("SELFTEST ")]
        msgs = [ln[-300:] for ln in pr.stderr.splitlines() if ln.strip() and "Validation Enabled" not in ln]
        result["selftest_under_MTL_DEBUG_LAYER_and_SHADER_VALIDATION"] = {
            "returncode": pr.returncode, "ok": bool(st and st[0]["ok"] and pr.returncode == 0 and not msgs),
            "validation_layers_active": "Metal GPU Validation Enabled" in pr.stderr, "messages": msgs[:4]}
        print("selftest under validation layers:", json.dumps(result["selftest_under_MTL_DEBUG_LAYER_and_SHADER_VALIDATION"]))

    for sigma in (3, 10):
        key = f"sigma{sigma}"
        w64, r = gaussian_weights(sigma)
        S = {"radius": r, "taps": 2 * r + 1}
        # ---- baseline: stock SciPy
        t64, t64min, ref64 = bench_scipy(a, sigma)
        t32, t32min, ref32 = bench_scipy(a32, sigma)
        S["scipy"] = {"f64_ms": round(t64 * 1e3, 1), "f64_min_ms": round(t64min * 1e3, 1), "f32_ms": round(t32 * 1e3, 1),
                      "f32_min_ms": round(t32min * 1e3, 1)}
        sp = lambda ms: round(t64 * 1e3 / ms, 1)

        # ---- strict CPU float64
        row = bench_cpu(a, sigma)
        out = cpu_gauss(a, sigma)
        row.update(errors(out, ref64))
        row["bit_identical_to_scipy"] = bool(np.array_equal(out, ref64))
        row["speedup_vs_scipy_f64"] = sp(row["wall_ms"])
        for fma in ("off", "on"):
            row[f"fma_{fma}_max_abs"] = errors(cpu_gauss(a, sigma, fma=fma), ref64)["max_abs"]
            row[f"fma_{fma}_max_rel"] = errors(cpu_gauss(a, sigma, fma=fma), ref64)["max_rel"]
        row["no_ring_wall_ms"] = bench_cpu(a, sigma, reps=5, ring=False)["wall_ms"]
        S["strict_cpu_f64"] = row

        # ---- CPU float32 arithmetic on float32 data (context: is the GPU even needed?)
        row = bench_cpu(a32, sigma, fma="on")
        out = cpu_gauss(a32, sigma, fma="on")
        row.update({"err_vs_scipy_f64_" + k: v for k, v in errors(out, ref64).items()})
        row["speedup_vs_scipy_f32"] = round(t32 * 1e3 / row["wall_ms"], 1)
        S["cpu_f32"] = row

        # ---- GPU float32 in -> float32 out
        g32 = {}
        for name, kw in (("wrap_in -> wrap_out (zero copy)", dict(in_mode="wrap_fused", out_mode="wrap_fused")),
                         ("memcpy_in -> pooled_out", dict(in_mode="cpu_vdsp", out_mode="pooled_fused")),
                         ("wrap_in (cached) -> pooled_out", dict(in_mode="wrap_fused", out_mode="pooled_fused", cache_in_wrap=True))):
            row = bench_gpu(a32, sigma, **kw)
            out = gpu_gauss(a32, sigma, **kw)
            row.update({"err_vs_scipy_f64_" + k: v for k, v in errors(out, ref64).items()})
            row["err_vs_scipy_f32_max_abs"] = errors(out, ref32.astype(np.float64))["max_abs"]
            row["speedup_vs_scipy_f32"] = round(t32 * 1e3 / row["wall_ms"], 1)
            del out
            g32[name] = row
        S["gpu_f32_in_f32_out"] = g32

        # ---- fast mode: float64 in -> float64 out, everything included
        fast = {}
        for name, kw in FAST_PIPELINES:
            row = bench_gpu(a, sigma, reps=15, **kw)
            out = gpu_gauss(a, sigma, **kw)
            row.update(errors(out, ref64))
            row["speedup_vs_scipy_f64"] = sp(row["wall_ms"])
            del out
            fast[name] = row
        # pooled output that can never be recycled (caller keeps every result): fresh buffer per call
        L.e4_pool_clear()
        row = bench_gpu(a, sigma, reps=7, keep=True, in_mode="cpu_vdsp", out_mode="pooled_fused")
        row["speedup_vs_scipy_f64"] = sp(row["wall_ms"])
        fast["A' as A but caller keeps every result (fresh output buffer each call)"] = row
        L.e4_pool_clear()
        gc.collect()
        # cold first call (empty pool: scratch + output buffers allocated and first-touched)
        t0 = time.perf_counter()
        out = gpu_gauss(a, sigma, in_mode="cpu_vdsp", out_mode="pooled_fused")
        S["fast_cold_first_call_ms"] = round((time.perf_counter() - t0) * 1e3, 2)
        del out
        one_shot = [k for k in fast if not k.startswith(("F ", "A' ", "A* "))]
        best = min(one_shot, key=lambda k: fast[k]["wall_ms"])
        S["fast_f64_in_f64_out"] = fast
        S["fast_best"] = best
        S["fast_stage_gpu_ms"] = {
            "A (cpu narrow, fused widen)": gpu_stage_times(a, sigma, in_mode="cpu_vdsp", out_mode="pooled_fused"),
            "C (fused narrow + fused widen)": gpu_stage_times(a, sigma, in_mode="wrap_fused", out_mode="wrap_fused"),
            "D (gpu narrow pass, fused widen)": gpu_stage_times(a, sigma, in_mode="wrap_gpu_pass", out_mode="wrap_fused"),
            "f32 reference (no conversion)": gpu_stage_times(a32, sigma, in_mode="cpu_vdsp", out_mode="pooled_fused"),
        }
        S["gpu_dispatch_sweep"] = dispatch_sweep(a32, sigma)
        if "--no-mlx" not in sys.argv:
            try:
                S["mlx"] = mlx_baseline(a, a32, sigma, ref64)
            except Exception as e:  # optional context only
                S["mlx"] = {"error": repr(e)[:300]}
        result[key] = S
        del ref64, ref32
        gc.collect()

    # pooled output lifetime: a held result must not be overwritten by later calls
    r1 = gpu_gauss(a, 3, in_mode="cpu_vdsp", out_mode="pooled_fused")
    snap = r1[::97, ::89].copy()
    r2 = gpu_gauss(a, 10, in_mode="cpu_vdsp", out_mode="pooled_fused")
    result["pooled_output_lifetime_ok"] = bool(ptr(r1) != ptr(r2) and np.array_equal(r1[::97, ::89], snap)
                                               and not r1.flags.owndata and r1.flags.writeable)
    del r1, r2

    # ---- report
    for sigma in (3, 10):
        S = result[f"sigma{sigma}"]
        print(f"\n=== sigma={sigma} (radius {S['radius']}, {S['taps']} taps), 4096x4096 ===")
        print(f"stock SciPy: float64 {S['scipy']['f64_ms']:.1f} ms, float32 {S['scipy']['f32_ms']:.1f} ms")
        c = S["strict_cpu_f64"]
        print(f"strict CPU float64 : {c['wall_ms']:8.2f} ms  ({c['speedup_vs_scipy_f64']}x)  axis0 {c['pass_axis0_ms']:.2f} axis1 {c['pass_axis1_ms']:.2f}"
              f"  max_abs {c['max_abs']:.2e} max_rel {c['max_rel']:.2e} bit-identical={c['bit_identical_to_scipy']}"
              f"  [fma off: {c['fma_off_max_rel']:.1e}, fma on: {c['fma_on_max_rel']:.1e} max_rel; no ring: {c['no_ring_wall_ms']:.2f} ms]")
        c = S["cpu_f32"]
        print(f"CPU float32 (ctx)  : {c['wall_ms']:8.2f} ms  axis0 {c['pass_axis0_ms']:.2f} axis1 {c['pass_axis1_ms']:.2f}  max_abs {c['err_vs_scipy_f64_max_abs']:.2e}")
        hdr = f"{'':62s} {'wall':>7s} {'wrap':>6s} {'cpuconv':>7s} {'kernels':>7s} {'sync':>6s} {'py':>5s} {'speedup':>8s} {'max_abs':>9s} {'max_rel':>9s}"
        print("GPU float32 in -> float32 out [ms]\n" + hdr)
        for name, g in S["gpu_f32_in_f32_out"].items():
            print(f"  {name:60s} {g['wall_ms']:7.2f} {g['wrap_ms']:6.2f} {g['cpu_convert_ms']:7.2f} {g['gpu_kernels_ms']:7.2f} "
                  f"{g['sync_overhead_ms']:6.2f} {g['python_overhead_ms']:5.2f} {g['speedup_vs_scipy_f32']:7.1f}x "
                  f"{g['err_vs_scipy_f64_max_abs']:9.2e} {g['err_vs_scipy_f64_max_rel']:9.2e}")
        print("fast mode: float64 in -> float64 out, end to end [ms]\n" + hdr)
        for name, g in S["fast_f64_in_f64_out"].items():
            print(f"  {name[:60]:60s} {g['wall_ms']:7.2f} {g['wrap_ms']:6.2f} {g['cpu_convert_ms']:7.2f} {g['gpu_kernels_ms']:7.2f} "
                  f"{g['sync_overhead_ms']:6.2f} {g['python_overhead_ms']:5.2f} {g['speedup_vs_scipy_f64']:7.1f}x "
                  + (f"{g['max_abs']:9.2e} {g['max_rel']:9.2e}" if "max_abs" in g else ""))
        print(f"  best one-shot pipeline: {S['fast_best']}; cold first call (empty pool): {S['fast_cold_first_call_ms']} ms")
        print("  per-stage GPU ms (separate command buffers):", json.dumps(S["fast_stage_gpu_ms"]))
        print("  dispatch sweep, GPU ms per pass (float32):")
        for row in S["gpu_dispatch_sweep"]:
            print(f"    tg {row['threadgroup']:14s} outputs/thread {row['outputs_per_thread']}: axis0 {row['pass_axis0_ms']:6.2f}  axis1 {row['pass_axis1_ms']:6.2f}")
        if "mlx" in S:
            print("  MLX (context):", json.dumps(S["mlx"]))

    gates = {}
    for sigma in (3, 10):
        S = result[f"sigma{sigma}"]
        fast = S["fast_f64_in_f64_out"]
        gates[f"sigma{sigma}"] = {
            "fast_best_pipeline": S["fast_best"],
            "fast_best_speedup": fast[S["fast_best"]]["speedup_vs_scipy_f64"],
            "fast_plain_numpy_output_speedup": fast["B cpu_vdsp_in -> cpu_vdsp_out (plain NumPy output)"]["speedup_vs_scipy_f64"],
            "fast_fully_fused_zero_copy_speedup": fast["C wrap_fused_in -> wrap_fused_out (fully fused zero-copy)"]["speedup_vs_scipy_f64"],
            "fast_worst_pipeline_speedup": min(v["speedup_vs_scipy_f64"] for v in fast.values()),
            "fast_repeat_input_cached_wrap_speedup": fast["F wrap (cached) + gpu narrow pass -> pooled_fused_out"]["speedup_vs_scipy_f64"],
            "fast_caller_keeps_results_speedup": fast["A' as A but caller keeps every result (fresh output buffer each call)"]["speedup_vs_scipy_f64"],
            "fast_gate_ge_20x": fast[S["fast_best"]]["speedup_vs_scipy_f64"] >= 20,
            "strict_speedup": S["strict_cpu_f64"]["speedup_vs_scipy_f64"],
            "strict_gate_ge_8x": S["strict_cpu_f64"]["speedup_vs_scipy_f64"] >= 8,
            "strict_max_rel_err": S["strict_cpu_f64"]["max_rel"],
            "strict_matches_1e-12": S["strict_cpu_f64"]["max_rel"] <= 1e-12,
        }
    result["gates"] = gates
    result["wall_seconds"] = round(time.perf_counter() - t_start, 1)
    print("\ngates:", json.dumps(gates))
    with open(os.path.join(HERE, "result.json"), "w") as f:  # full result of the latest run, pretty-printed
        json.dump(result, f, indent=1)
    print("RESULT " + json.dumps(result))


if __name__ == "__main__":
    main()
