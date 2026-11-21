#!/usr/bin/env python
"""E8 probe driver: one SciMetal kernel, two encoders (standalone / MLX).

Usage (builds first, then measures, prints one final ``RESULT {json}`` line):

    /Users/tunguz/Programming/SciMetal/.venv/bin/python run.py [--no-build] [--quick]

Sections
  verify        standalone == MLX bit for bit, both ~= scipy.ndimage.correlate1d
  lazy          marginal cost of the custom op inside a lazy MLX graph
  eager         one custom op + eval vs the standalone encode+commit+wait
  commit        evidence on command-buffer commits / host syncs (swizzled counters)
  zero_copy     host memory -> mx.array and mx.array -> MTLBuffer hand-offs
  abi           MLX headers + libmlx symbols the extension is pinned to
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import glob
import json
import os
import platform
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(HERE, "build")
EXT_DIR = os.path.join(BUILD, "ext")
METALLIB = os.path.join(BUILD, "e8_kernels.metallib")
GATE_US = 30.0


def log(msg: str) -> None:
    print(f"[e8] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Standalone runner (ctypes, no libmlx)
# ---------------------------------------------------------------------------
class Runner:
    def __init__(self, np):
        self.np = np
        lib = ctypes.CDLL(os.path.join(BUILD, "libe8_runner.dylib"))
        vp, u32, u64 = ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint64
        lib.e8r_create.restype = vp
        lib.e8r_create.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
        lib.e8r_prepare.argtypes = [vp, u32, u32, u32]
        lib.e8r_host_ptr.restype = vp
        lib.e8r_host_ptr.argtypes = [vp, ctypes.c_int]
        lib.e8r_mtl_buffer.restype = vp
        lib.e8r_mtl_buffer.argtypes = [vp, ctypes.c_int]
        lib.e8r_buffer_length.restype = u64
        lib.e8r_buffer_length.argtypes = [vp, ctypes.c_int]
        lib.e8r_mtl_device.restype = vp
        lib.e8r_mtl_device.argtypes = [vp]
        lib.e8r_run.argtypes = [vp, ctypes.POINTER(u64)]
        lib.e8r_run_external.argtypes = [
            vp, vp, u64, vp, u64, vp, u64, u32, u32, u32, ctypes.POINTER(u64)]
        lib.e8r_destroy.argtypes = [vp]
        self.lib = lib
        err = ctypes.create_string_buffer(512)
        self.h = lib.e8r_create(METALLIB.encode(), err, 512)
        if not self.h:
            raise RuntimeError(f"standalone runner: {err.value.decode()}")
        self.t = (u64 * 4)()
        self.shape = None

    def _view(self, which, count):
        p = self.lib.e8r_host_ptr(self.h, which)
        return self.np.ctypeslib.as_array((ctypes.c_float * count).from_address(p))

    def prepare(self, rows, n, nw):
        if self.lib.e8r_prepare(self.h, rows, n, nw) != 0:
            raise RuntimeError("e8r_prepare failed")
        self.shape = (rows, n, nw)
        self.x = self._view(0, rows * n).reshape(rows, n)
        self.out = self._view(1, rows * n).reshape(rows, n)
        self.w = self._view(2, nw)

    def run(self):
        if self.lib.e8r_run(self.h, self.t) != 0:
            raise RuntimeError("e8r_run failed")
        return self.t

    def host_ptr(self, which):
        return self.lib.e8r_host_ptr(self.h, which)

    def mtl_buffer(self, which):
        return self.lib.e8r_mtl_buffer(self.h, which)

    def mtl_device(self):
        return self.lib.e8r_mtl_device(self.h)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def stats_us(np, ns):
    a = np.asarray(ns, dtype=np.float64) / 1e3
    return {
        "median": round(float(np.median(a)), 2),
        "p10": round(float(np.percentile(a, 10)), 2),
        "p90": round(float(np.percentile(a, 90)), 2),
    }


def paired_us(np, a, b):
    """Median of the per-round differences a - b (interleaved samples), µs."""
    d = (np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)) / 1e3
    return round(float(np.median(d)), 2)


def paired_ci95_us(np, a, b, n_boot=400):
    """Bootstrap 95% interval of the median paired difference a - b, µs."""
    d = (np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)) / 1e3
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    meds = np.median(d[idx], axis=1)
    return [round(float(np.percentile(meds, 2.5)), 2), round(float(np.percentile(meds, 97.5)), 2)]


def bench_interleaved(np, variants, rounds, warmup, post=None):
    """Round-robin the variants (rotating the start) so drift and bursts from
    other processes hit all of them alike.  Returns {name: ns array}.
    `post` (optional) is called, untimed, on each variant's return value."""
    names = list(variants)
    fns = [variants[n] for n in names]
    k = len(names)
    if post is None:
        def post(_):
            return None
    for _ in range(warmup):
        for f in fns:
            post(f())
    samples = np.empty((k, rounds), dtype=np.int64)
    clock = time.perf_counter_ns
    gc.collect()
    gc.disable()
    try:
        for r in range(rounds):
            for j in range(k):
                i = (r + j) % k
                f = fns[i]
                t0 = clock()
                res = f()
                samples[i, r] = clock() - t0
                post(res)
    finally:
        gc.enable()
    return {n: samples[i] for i, n in enumerate(names)}


def builtin_correlate(mx, t, wm):
    """The same math with MLX built-ins: reflect pad + conv1d (cross-corr)."""
    nw = wm.shape[0]
    n = t.shape[-1]
    p0 = nw // 2
    p1 = nw - 1 - p0
    parts = []
    if p0:
        parts.append(t[:, :p0][:, ::-1])
    parts.append(t)
    if p1:
        parts.append(t[:, n - p1:][:, ::-1])
    tp = mx.concatenate(parts, axis=1)
    return mx.conv1d(tp[:, :, None], wm.reshape(1, nw, 1))[:, :, 0]


# ---------------------------------------------------------------------------
# 4. verification
# ---------------------------------------------------------------------------
def section_verify(np, mx, e8, ndimage, runner):
    rng = np.random.default_rng(1234)
    cases = [(4, 64, 5), (3, 7, 6), (2, 5, 13), (1, 1, 3), (64, 1000, 8), (1024, 4096, 9)]
    rows_out = []
    all_bit = True
    all_scipy = True
    cpu_gpu_bit = True
    worst = {"standalone": 0.0, "mlx_gpu": 0.0, "mlx_cpu": 0.0}
    worst_cpu_gpu = 0.0
    for rows, n, nw in cases:
        x = rng.standard_normal((rows, n), dtype=np.float32)
        w = rng.standard_normal(nw, dtype=np.float32)
        ref = ndimage.correlate1d(x, w, axis=-1, mode="reflect")
        assert ref.dtype == np.float32

        runner.prepare(rows, n, nw)
        runner.x[:] = x
        runner.w[:] = w
        runner.run()
        o_sa = runner.out.copy()

        a, wm = mx.array(x), mx.array(w)
        o_gpu = np.array(e8.correlate1d(a, wm, "gpu"))
        o_cpu = np.array(e8.correlate1d(a, wm, "cpu"))

        bit = o_sa.tobytes() == o_gpu.tobytes()
        ok = all(
            np.allclose(o, ref, rtol=1e-5, atol=1e-5) for o in (o_sa, o_gpu, o_cpu))
        errs = {
            "standalone": float(np.abs(o_sa - ref).max()),
            "mlx_gpu": float(np.abs(o_gpu - ref).max()),
            "mlx_cpu": float(np.abs(o_cpu - ref).max()),
        }
        for k_, v in errs.items():
            worst[k_] = max(worst[k_], v)
        all_bit &= bit
        all_scipy &= ok
        cpu_gpu_bit &= o_cpu.tobytes() == o_gpu.tobytes()
        worst_cpu_gpu = max(worst_cpu_gpu, float(np.abs(o_cpu - o_gpu).max()))
        rows_out.append({
            "rows": rows, "n": n, "nw": nw,
            "mlx_gpu_eq_standalone_bitwise": bit,
            "scipy_allclose": ok,
            "max_abs_err_vs_scipy": {k_: float(f"{v:.3g}") for k_, v in errs.items()},
        })

    # MLX-only extras: N-D, non-contiguous, lazy, integer input, built-in equiv.
    w = rng.standard_normal(7, dtype=np.float32)
    wm = mx.array(w)
    x3 = rng.standard_normal((2, 3, 50), dtype=np.float32)
    x2 = rng.standard_normal((6, 50), dtype=np.float32)
    a2 = mx.array(x2)

    def close(got, want):
        return bool(np.allclose(np.array(got), want, rtol=1e-5, atol=1e-5))

    def sp(v):
        return ndimage.correlate1d(np.ascontiguousarray(v), w, axis=-1, mode="reflect")

    extras = {
        "nd_input": close(e8.correlate1d(mx.array(x3), wm), sp(x3)),
        "lazy_input": close(e8.correlate1d(a2 * 2 + 1, wm), sp(x2 * 2 + 1)),
        "transposed_input_guard": close(e8.correlate1d(a2.T, wm), sp(x2.T)),
        "strided_input_guard": close(e8.correlate1d(a2[:, ::2], wm), sp(x2[:, ::2])),
        "int32_input_cast": close(
            e8.correlate1d(mx.array(np.arange(40, dtype=np.int32).reshape(2, 20)), wm),
            sp(np.arange(40, dtype=np.float32).reshape(2, 20))),
        "builtin_conv1d_equivalent_matches": close(builtin_correlate(mx, a2, wm), sp(x2)),
        "works_inside_mx_compile": close(
            mx.compile(lambda t: e8.correlate1d(t * 2 + 1, wm))(a2), sp(x2 * 2 + 1)),
    }
    two = e8.correlate1d(a2, wm, "gpu", True, 2)
    chained = e8.correlate1d(e8.correlate1d(a2, wm, "gpu"), wm, "gpu")
    extras["two_pass_node_with_temporary_eq_two_nodes_bitwise"] = bool(
        np.array(two).tobytes() == np.array(chained).tobytes())
    extras["two_pass_node_matches_scipy"] = close(two, sp(sp(x2)))
    xl = mx.array(rng.standard_normal((1024, 4096), dtype=np.float32))
    extras["two_pass_node_eq_two_nodes_bitwise_16MB"] = bool(
        np.array(e8.correlate1d(xl, wm, "gpu", True, 2)).tobytes()
        == np.array(e8.correlate1d(e8.correlate1d(xl, wm, "gpu"), wm, "gpu")).tobytes())
    del xl
    extras["two_pass_cpu_matches_scipy"] = close(
        e8.correlate1d(a2, wm, "cpu", True, 2), sp(sp(x2)))
    try:
        mx.eval(e8.correlate1d(a2.T, wm, "default", False))
        extras["noguard_noncontiguous_raises"] = False
    except ValueError:
        extras["noguard_noncontiguous_raises"] = True
    extras["eval_ok_after_exception"] = close(e8.correlate1d(a2, wm), sp(x2))

    return {
        "cases": rows_out,
        "mlx_gpu_eq_standalone_bitwise_all": bool(all_bit),
        "scipy_allclose_all": bool(all_scipy),
        "mlx_cpu_eq_mlx_gpu_bitwise_all": bool(cpu_gpu_bit),
        "max_abs_mlx_cpu_minus_mlx_gpu": float(f"{worst_cpu_gpu:.3g}"),
        "worst_abs_err_vs_scipy": {k_: float(f"{v:.3g}") for k_, v in worst.items()},
        "tolerance": "rtol=1e-5, atol=1e-5 (float32)",
        "note_cpu_vs_gpu": "eval_cpu is built with -ffp-contract=on (fma, like the Metal "
                           "kernel); with -ffp-contract=off it differs from the GPU by 1.9e-6 max",
        "extras": extras,
    }


# ---------------------------------------------------------------------------
# 5. timings
# ---------------------------------------------------------------------------
def section_timing(np, mx, e8, runner, quick):
    rng = np.random.default_rng(7)
    nw = 9
    w = rng.standard_normal(nw, dtype=np.float32)
    w /= np.abs(w).sum()  # keeps chained correlations bounded
    wm = mx.array(w)
    mx.eval(wm)
    out = {"weights": nw}

    def graph_variants(a):
        corr = e8.correlate1d

        def noop():
            mx.eval((a * 2 + 1).sum())

        def builtin_sin():
            mx.eval(mx.sin(a * 2 + 1).sum())

        def builtin_cumsum():
            mx.eval(mx.cumsum(a * 2 + 1, axis=-1).sum())

        def custom():
            mx.eval(corr(a * 2 + 1, wm).sum())

        def custom_noguard():
            mx.eval(corr(a * 2 + 1, wm, "default", False).sum())

        def builtin_conv1d_equiv():
            mx.eval(builtin_correlate(mx, a * 2 + 1, wm).sum())

        def custom_two_pass_one_node():
            mx.eval(corr(a * 2 + 1, wm, "default", True, 2).sum())

        return {
            "noop": noop,
            "builtin_sin": builtin_sin,
            "builtin_cumsum": builtin_cumsum,
            "custom": custom,
            "custom_noguard": custom_noguard,
            "builtin_conv1d_equiv": builtin_conv1d_equiv,
            "custom_two_pass_one_node": custom_two_pass_one_node,
        }

    def host_side_variants(a):
        """Same graphs, but only the host side is timed: build the graph and
        mx.async_eval() it (encode + commit, no wait for the GPU).  The wait
        happens untimed afterwards."""
        corr = e8.correlate1d

        def noop():
            z = (a * 2 + 1).sum()
            mx.async_eval(z)
            return z

        def builtin_sin():
            z = mx.sin(a * 2 + 1).sum()
            mx.async_eval(z)
            return z

        def builtin_cumsum():
            z = mx.cumsum(a * 2 + 1, axis=-1).sum()
            mx.async_eval(z)
            return z

        def custom():
            z = corr(a * 2 + 1, wm).sum()
            mx.async_eval(z)
            return z

        def custom_noguard():
            z = corr(a * 2 + 1, wm, "default", False).sum()
            mx.async_eval(z)
            return z

        return {
            "noop": noop,
            "builtin_sin": builtin_sin,
            "builtin_cumsum": builtin_cumsum,
            "custom": custom,
            "custom_noguard": custom_noguard,
        }

    def chain_variants(a, k_small, k_big):
        corr = e8.correlate1d

        def mk(op, k):
            def f():
                t = a * 2 + 1
                for _ in range(k):
                    t = op(t)
                mx.eval(t.sum())
            return f

        ops = {
            "builtin_sin": mx.sin,                             # in-place capable
            "builtin_cumsum": lambda t: mx.cumsum(t, axis=-1),  # needs a new buffer
            "custom": lambda t: corr(t, wm),
            "custom_noguard": lambda t: corr(t, wm, "default", False),
        }
        v = {}
        for name, op in ops.items():
            v[f"{name}_x{k_small}"] = mk(op, k_small)
            v[f"{name}_x{k_big}"] = mk(op, k_big)
        return v

    def eager_variants(a):
        corr = e8.correlate1d
        run = runner.lib.e8r_run
        h, t = runner.h, runner.t

        def eager_custom():
            mx.eval(corr(a, wm))

        def eager_builtin_sin():
            mx.eval(mx.sin(a))

        def standalone():
            run(h, t)

        return {
            "mlx_custom_eager": eager_custom,
            "mlx_builtin_sin_eager": eager_builtin_sin,
            "standalone_run_pywall": standalone,
        }

    sizes = {
        "small": ((4, 64), 400 if quick else 1500, 100),
        "large": ((1024, 4096), 40 if quick else 200, 10),
    }
    lazy, eager = {}, {}
    for label, (shape, rounds, warmup) in sizes.items():
        x = rng.standard_normal(shape, dtype=np.float32)
        a = mx.array(x)
        mx.eval(a)

        # ---- lazy graph: marginal cost ------------------------------------
        e8.eval_stats(True)
        s = bench_interleaved(np, graph_variants(a), rounds, warmup)
        st = e8.eval_stats(True)
        enc = [r["encode_ns"] for r in st["recent"]]
        lz = {
            "shape": list(shape),
            "rounds": rounds,
            "graph": "z = OP(a*2+1).sum(); mx.eval(z)   (build + eval, wall µs)",
            "wall_us": {k_: stats_us(np, v) for k_, v in s.items()},
            "marginal_us_paired_median": {
                "custom_minus_noop": paired_us(np, s["custom"], s["noop"]),
                "custom_minus_builtin_sin": paired_us(np, s["custom"], s["builtin_sin"]),
                "custom_minus_builtin_cumsum": paired_us(np, s["custom"], s["builtin_cumsum"]),
                "custom_noguard_minus_noop": paired_us(np, s["custom_noguard"], s["noop"]),
                "custom_noguard_minus_builtin_sin": paired_us(np, s["custom_noguard"], s["builtin_sin"]),
                "builtin_sin_minus_noop": paired_us(np, s["builtin_sin"], s["noop"]),
                "builtin_cumsum_minus_noop": paired_us(np, s["builtin_cumsum"], s["noop"]),
                "builtin_conv1d_equiv_minus_noop": paired_us(np, s["builtin_conv1d_equiv"], s["noop"]),
                "custom_two_pass_minus_custom": paired_us(np, s["custom_two_pass_one_node"], s["custom"]),
            },
            # Difference of the 10th percentiles: less sensitive to bursts from
            # other processes than the medians.
            "marginal_us_p10_diff": {
                k_: round(float(np.percentile(s[a_], 10) - np.percentile(s[b_], 10)) / 1e3, 2)
                for k_, a_, b_ in (
                    ("custom_minus_noop", "custom", "noop"),
                    ("custom_minus_builtin_sin", "custom", "builtin_sin"),
                    ("custom_minus_builtin_cumsum", "custom", "builtin_cumsum"),
                    ("builtin_sin_minus_noop", "builtin_sin", "noop"),
                    ("builtin_cumsum_minus_noop", "builtin_cumsum", "noop"),
                )
            },
            "marginal_us_paired_median_ci95": {
                "custom_minus_noop": paired_ci95_us(np, s["custom"], s["noop"]),
                "custom_minus_builtin_sin": paired_ci95_us(np, s["custom"], s["builtin_sin"]),
                "custom_minus_builtin_cumsum": paired_ci95_us(np, s["custom"], s["builtin_cumsum"]),
            },
            # Wall time spent inside the primitive's eval_gpu (C++ clock):
            # allocate output + bind + encode the dispatch.
            "custom_eval_gpu_encode_us": dict(stats_us(np, enc), calls=st["gpu_calls"]),
        }
        lazy[label] = lz

        # ---- host side only: build + async_eval (encode + commit, no wait) --
        hs = bench_interleaved(np, host_side_variants(a), rounds, warmup, post=mx.eval)
        lz["host_side_only"] = {
            "what": "build graph + mx.async_eval(z): encode + commit, GPU wait excluded",
            "wall_us": {k_: stats_us(np, v) for k_, v in hs.items()},
            "marginal_us_paired_median": {
                "custom_minus_noop": paired_us(np, hs["custom"], hs["noop"]),
                "custom_minus_builtin_sin": paired_us(np, hs["custom"], hs["builtin_sin"]),
                "custom_minus_builtin_cumsum": paired_us(np, hs["custom"], hs["builtin_cumsum"]),
                "custom_noguard_minus_noop": paired_us(np, hs["custom_noguard"], hs["noop"]),
                "builtin_sin_minus_noop": paired_us(np, hs["builtin_sin"], hs["noop"]),
                "builtin_cumsum_minus_noop": paired_us(np, hs["builtin_cumsum"], hs["noop"]),
            },
            "custom_minus_noop_ci95": paired_ci95_us(np, hs["custom"], hs["noop"]),
        }

        # ---- eager: custom op + eval vs standalone encode+commit+wait ------
        runner.prepare(shape[0], shape[1], nw)
        runner.x[:] = x
        runner.w[:] = w
        ev = eager_variants(a)
        s = bench_interleaved(np, ev, rounds, warmup)
        # C-side breakdown of the standalone path (separate short loop).
        parts = np.empty((rounds, 4), dtype=np.int64)
        for i in range(rounds):
            t = runner.run()
            parts[i] = (t[0], t[1], t[2], t[3])
        eager[label] = {
            "shape": list(shape),
            "rounds": rounds,
            "wall_us": {k_: stats_us(np, v) for k_, v in s.items()},
            "mlx_custom_minus_standalone_us_paired_median": paired_us(
                np, s["mlx_custom_eager"], s["standalone_run_pywall"]),
            "mlx_custom_minus_mlx_builtin_sin_us_paired_median": paired_us(
                np, s["mlx_custom_eager"], s["mlx_builtin_sin_eager"]),
            "standalone_c_side_us": {
                "encode": stats_us(np, parts[:, 0]),
                "commit": stats_us(np, parts[:, 1]),
                "wait_until_completed": stats_us(np, parts[:, 2]),
                "encode_commit_wait_total": stats_us(np, parts[:, :3].sum(axis=1)),
                "gpu_execution_time": stats_us(np, parts[:, 3]),
            },
        }
        if label == "large":
            # At this size the op's cost is real GPU work, so compare the
            # marginal wall cost in the MLX graph with the kernel's own GPU
            # execution time as reported by Metal for the standalone encoder.
            lz["bridge_overhead_vs_kernel_gpu_time_us"] = round(
                lz["marginal_us_paired_median"]["custom_minus_noop"]
                - eager[label]["standalone_c_side_us"]["gpu_execution_time"]["median"], 2)

        # ---- chains: per-op slope (small only) -----------------------------
        if label == "small":
            k1, k2 = 1, 16
            cs = bench_interleaved(
                np, chain_variants(a, k1, k2), max(100, rounds // 3), 30)
            slope = {}
            for name in ("builtin_sin", "builtin_cumsum", "custom", "custom_noguard"):
                slope[name] = round(
                    paired_us(np, cs[f"{name}_x{k2}"], cs[f"{name}_x{k1}"]) / (k2 - k1), 2)
            lz["chain"] = {
                "graph": f"t = a*2+1; t = OP(t) repeated K times; mx.eval(t.sum()); K in ({k1}, {k2})",
                "wall_us": {k_: stats_us(np, v) for k_, v in cs.items()},
                "per_op_slope_us": slope,
            }

            # ---- graph construction only (no eval) -------------------------
            t_lazy = a * 2 + 1
            corr = e8.correlate1d
            n_it = 5000 if quick else 20000

            def loop(f):
                clock = time.perf_counter_ns
                for _ in range(200):
                    f()
                t0 = clock()
                for _ in range(n_it):
                    f()
                return round((clock() - t0) / n_it / 1e3, 3)

            lz["node_build_only_us"] = {
                "custom_lazy_input_guard": loop(lambda: corr(t_lazy, wm)),
                "custom_lazy_input_noguard": loop(lambda: corr(t_lazy, wm, "default", False)),
                "custom_evaluated_input": loop(lambda: corr(a, wm)),
                "builtin_sin": loop(lambda: mx.sin(t_lazy)),
                "builtin_cumsum": loop(lambda: mx.cumsum(t_lazy, axis=-1)),
                "bridge_roundtrip_only": loop(lambda: e8.roundtrip(a)),
            }
    out["lazy"] = lazy
    out["eager"] = eager
    return out


# ---------------------------------------------------------------------------
# 5b. commit / host-sync evidence
# ---------------------------------------------------------------------------
def section_commit(np, mx, e8):
    rng = np.random.default_rng(11)
    w = rng.standard_normal(9, dtype=np.float32)
    w /= np.abs(w).sum()
    wm = mx.array(w)
    corr = e8.correlate1d
    msg = e8.install_metal_counters()
    out = {"counter_install_error": msg, "per_eval": {}}
    keys = ["commit", "cb_commandBufferWithUnretainedReferences", "cb_commandBuffer",
            "cb_commandBufferWithDescriptor", "computeCommandEncoder", "endEncoding",
            "dispatchThreads", "dispatchThreadgroups", "sharedEventWait",
            "waitUntilCompleted", "waitUntilScheduled", "encodeSignalEvent",
            "addCompletedHandler"]

    def count(f, n):
        f()
        e8.metal_counters(True)
        for _ in range(n):
            f()
        c = e8.metal_counters(True)
        return {k_: round(c[k_] / n, 3) for k_ in keys if c[k_]}

    def chain(a, op, k):
        t = a * 2 + 1
        for _ in range(k):
            t = op(t)
        mx.eval(t.sum())

    verdict = True
    for label, shape, n in (("small", (4, 64), 20), ("large", (1024, 4096), 6)):
        a = mx.array(rng.standard_normal(shape, dtype=np.float32))
        mx.eval(a)
        res = {
            "noop": count(lambda: mx.eval((a * 2 + 1).sum()), n),
            "builtin_sin": count(lambda: mx.eval(mx.sin(a * 2 + 1).sum()), n),
            "builtin_cumsum": count(lambda: mx.eval(mx.cumsum(a * 2 + 1, axis=-1).sum()), n),
            "custom": count(lambda: mx.eval(corr(a * 2 + 1, wm).sum()), n),
            "custom_noguard": count(lambda: mx.eval(corr(a * 2 + 1, wm, "default", False).sum()), n),
            "custom_two_pass_one_node": count(
                lambda: mx.eval(corr(a * 2 + 1, wm, "default", True, 2).sum()), n),
        }
        if label == "small":
            for k in (45, 60, 120):
                res[f"chain{k}_builtin_sin"] = count(lambda: chain(a, mx.sin, k), 4)
                res[f"chain{k}_custom"] = count(lambda: chain(a, lambda t: corr(t, wm), k), 4)
        out["per_eval"][label] = res

        def syncs(d):
            return (d.get("sharedEventWait", 0) + d.get("waitUntilCompleted", 0)
                    + d.get("waitUntilScheduled", 0))

        def bufs(d):
            return (d.get("cb_commandBufferWithUnretainedReferences", 0)
                    + d.get("cb_commandBuffer", 0) + d.get("cb_commandBufferWithDescriptor", 0))

        b, c = res["builtin_sin"], res["custom"]
        diff = {
            "extra_commits": round(c.get("commit", 0) - b.get("commit", 0), 3),
            "extra_command_buffers": round(bufs(c) - bufs(b), 3),
            "extra_encoders": round(c.get("computeCommandEncoder", 0) - b.get("computeCommandEncoder", 0), 3),
            "extra_host_syncs": round(syncs(c) - syncs(b), 3),
            "extra_dispatches": round(
                c.get("dispatchThreads", 0) + c.get("dispatchThreadgroups", 0)
                - b.get("dispatchThreads", 0) - b.get("dispatchThreadgroups", 0), 3),
        }
        out["per_eval"][label]["custom_minus_builtin_sin"] = diff
        verdict &= diff["extra_commits"] == 0 and diff["extra_host_syncs"] == 0 \
            and diff["extra_command_buffers"] == 0
        if label == "small":
            for k in (45, 60, 120):
                verdict &= res[f"chain{k}_custom"].get("commit") == res[f"chain{k}_builtin_sin"].get("commit")

    # Pointer identity: both custom ops of one graph encode into the command
    # buffer that was already open on MLX's stream; it is replaced exactly once
    # (the single commit at the end of mx.eval).
    a = mx.array(rng.standard_normal((4, 64), dtype=np.float32))
    mx.eval(a)
    mx.synchronize()
    before = e8.stream_state()["command_buffer"]
    e8.eval_stats(True)
    z = corr(mx.sin(corr(a * 2 + 1, wm)), wm).sum()
    mx.eval(z)
    rec = e8.eval_stats(True)["recent"]
    after = e8.stream_state()["command_buffer"]
    out["pointer_identity"] = {
        "custom_ops_in_graph": len(rec),
        "all_custom_ops_encoded_into_already_open_stream_buffer": bool(
            rec and all(r["command_buffer"] == before for r in rec)),
        "needs_commit_after_any_custom_op": bool(any(r["needs_commit_after"] for r in rec)),
        "stream_buffer_replaced_after_eval": before != after,
    }
    verdict &= out["pointer_identity"]["all_custom_ops_encoded_into_already_open_stream_buffer"]
    verdict &= not out["pointer_identity"]["needs_commit_after_any_custom_op"]
    out["no_extra_commit_or_sync"] = bool(verdict)
    return out


# ---------------------------------------------------------------------------
# 6. zero-copy hand-offs
# ---------------------------------------------------------------------------
def section_zero_copy(np, mx, e8, ndimage, runner):
    rng = np.random.default_rng(21)
    w = rng.standard_normal(9, dtype=np.float32)
    wm = mx.array(w)
    page = e8.mlx_device_info()["page_size"]

    def host_view(ptr, shape):
        n = int(np.prod(shape))
        return np.ctypeslib.as_array((ctypes.c_float * n).from_address(ptr)).reshape(shape)

    def sp(v):
        return ndimage.correlate1d(v, w, axis=-1, mode="reflect")

    def wait_deleter(token, want=1, timeout=2.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            gc.collect()
            mx.synchronize()
            if e8.external_deleter_calls(token) >= want:
                break
            time.sleep(0.01)
        return e8.external_deleter_calls(token)

    def exercise(arr, info, shape):
        """GPU kernels consume the wrapped array; host writes stay visible."""
        h = host_view(info["host_ptr"], shape)
        h[:] = rng.standard_normal(shape, dtype=np.float32)
        y, s = e8.correlate1d(arr, wm), mx.sin(arr)
        mx.eval(y, s)
        ok = bool(np.allclose(np.array(y), sp(h), atol=1e-5)
                  and np.allclose(np.array(s), np.sin(h), atol=1e-6))
        h += 1.0
        y2 = e8.correlate1d(arr, wm)
        mx.eval(y2)
        vis = bool(np.allclose(np.array(y2), sp(h), atol=1e-5))
        return ok, vis

    cases = []
    keep = []  # numpy owners of "foreign" memory

    def run_case(label, kind, shape, ctor, host_ptr=0, mtl_buffer=0):
        arr, info = e8.wrap_external(kind, list(shape), ctor, host_ptr, mtl_buffer)
        rec = {
            "case": label, "ctor": ctor, "shape": list(shape), "nbytes": info["nbytes"],
            "host_ptr_page_aligned": bool(info["host_ptr_page_aligned"]),
            "nbytes_page_multiple": bool(info["nbytes_page_multiple"]),
        }
        if arr is None:
            rec["zero_copy"] = False
            rec["make_buffer_ok"] = False
            cases.append(rec)
            return
        rec["zero_copy"] = bool(info["zero_copy"])
        rec["deleter_called_during_construction"] = info["deleter_calls_after_ctor"] > 0
        if info["zero_copy"]:
            ok, vis = exercise(arr, info, shape)
            rec["gpu_kernels_consume_ok"] = ok
            rec["host_write_visible_to_gpu"] = vis
            bi = e8.buffer_info(arr)
            rec["mlx_buffer_is_foreign_mtlbuffer"] = bool(mtl_buffer) and bi["mtl_buffer"] == mtl_buffer
        tok = info["token"]
        del arr
        rec["deleter_calls_after_release"] = wait_deleter(tok)
        cases.append(rec)

    run_case("mmap page-aligned, page-multiple length", "mmap", (4, 4096), "pointer")
    run_case("mmap page-aligned, odd length", "mmap", (4, 1000), "pointer")
    run_case("mmap page-aligned, 4 bytes", "mmap", (1,), "pointer")
    run_case("malloc+64 (unaligned), 64 KiB", "malloc", (4, 4096), "pointer")
    run_case("malloc+64 (unaligned), 64 bytes", "malloc", (16,), "pointer")
    for shape in ((4, 64), (333, 777), (1024, 4096)):
        npx = np.empty(shape, dtype=np.float32)
        keep.append(npx)
        run_case(f"numpy-owned memory {shape}", "foreign", shape, "pointer", npx.ctypes.data)
    run_case("mmap odd length via allocator::make_buffer + Buffer ctor",
             "mmap", (4, 1000), "make_buffer")

    # Our own runtime's shared MTLBuffer -> MLX.
    shape = (8, 1000)
    runner.prepare(shape[0], shape[1], 9)
    run_case("standalone runner MTLBuffer.contents via pointer ctor",
             "foreign", shape, "pointer", runner.host_ptr(0), runner.mtl_buffer(0))
    run_case("standalone runner MTLBuffer adopted as allocator::Buffer",
             "foreign", shape, "adopt", runner.host_ptr(0), runner.mtl_buffer(0))

    out = {
        "page_size": page,
        "a_host_memory_to_mx_array": cases,
        "a_zero_copy_all_cases": bool(all(c["zero_copy"] for c in cases)),
        "a_copy_fallback_observed": bool(any(not c["zero_copy"] for c in cases)),
        "a_gpu_consumes_all_zero_copy_cases": bool(all(
            c.get("gpu_kernels_consume_ok", False) for c in cases if c["zero_copy"])),
        "a_deleter_called_exactly_once_all": bool(all(
            c.get("deleter_calls_after_release") == 1 for c in cases)),
        "runner_device_is_mlx_device": runner.mtl_device() == e8.mlx_device_info()["mtl_device"],
    }

    # Donation hazard: MLX reuses a uniquely referenced input buffer as the
    # output of in-place capable ops -- including memory it does not own.
    def donation(pin, hold):
        shape = (4, 4096)
        arr, info = e8.wrap_external("mmap", list(shape), "pointer", 0, 0, pin)
        h = host_view(info["host_ptr"], shape)
        h[:] = 1.0
        y = mx.sin(arr)
        held = arr if hold else None
        del arr
        gc.collect()
        mx.eval(y)
        overwritten = bool(h[0, 0] != 1.0)
        aliases = e8.buffer_info(y)["data_ptr"] == info["host_ptr"]
        del h, y, held
        if pin:
            e8.external_unpin(info["token"])
        return {"host_memory_overwritten_by_mlx": overwritten,
                "result_aliases_host_memory": bool(aliases),
                "deleter_calls_after_release": wait_deleter(info["token"])}

    out["a_donation"] = {
        "python_ref_dropped_before_eval": donation(False, False),
        "python_ref_held": donation(False, True),
        "ref_dropped_but_data_pinned": donation(True, False),
    }

    # (b) raw MTLBuffer / host pointer of evaluated arrays; page alignment.
    table = []
    for nbytes in (4, 64, 252, 256, 1024, 4096, 8192, 16380, 16384, 16388, 65536, 1 << 20, 1 << 26):
        reps = 6 if nbytes <= 65536 else 2
        consts = [mx.array(np.zeros(nbytes // 4, dtype=np.float32)) for _ in range(reps)]
        outs = [c + 1 for c in consts]
        mx.eval(consts, outs)
        ci = [e8.buffer_info(c) for c in consts]
        oi = [e8.buffer_info(o) for o in outs]
        table.append({
            "nbytes": nbytes,
            "constant_page_aligned": f"{sum(i['data_page_aligned'] for i in ci)}/{reps}",
            "op_output_page_aligned": f"{sum(i['data_page_aligned'] for i in oi)}/{reps}",
            "buffer_length": sorted({i["buffer_length"] for i in ci + oi}),
            "heap_backed": bool(any(i["heap_backed"] for i in ci + oi)),
            "storage_mode_shared": bool(all(i["storage_mode"] == 0 for i in ci + oi)),
            "offset_zero": bool(all(i["offset"] == 0 for i in ci + oi)),
            "available_after_eval": bool(all(i["is_available"] for i in ci + oi)),
        })
        del consts, outs
    out["b_evaluated_array_buffers"] = table

    def all_aligned(row):
        return all(row[k].split("/")[0] == row[k].split("/")[1]
                   for k in ("constant_page_aligned", "op_output_page_aligned"))

    out["b_smallest_nbytes_always_page_aligned"] = next(
        (r["nbytes"] for i, r in enumerate(table) if all(all_aligned(q) for q in table[i:])), None)

    # (b') an evaluated mx.array's MTLBuffer consumed by OUR queue, no copies.
    rows, n, nw = 16, 2048, 9
    x = rng.standard_normal((rows, n), dtype=np.float32)
    xm, wm2, om = mx.array(x), mx.array(w), mx.array(np.zeros((rows, n), dtype=np.float32))
    mx.eval(xm, wm2, om)
    mx.synchronize()
    bx, bw, bo = (e8.buffer_info(v) for v in (xm, wm2, om))
    t = (ctypes.c_uint64 * 4)()
    rc = runner.lib.e8r_run_external(
        runner.h,
        bx["mtl_buffer"], bx["data_ptr"] - bx["contents"],
        bw["mtl_buffer"], bw["data_ptr"] - bw["contents"],
        bo["mtl_buffer"], bo["data_ptr"] - bo["contents"],
        rows, n, nw, t)
    got = host_view(bo["data_ptr"], (rows, n))
    out["b_mlx_buffers_consumed_by_standalone_queue"] = {
        "rc": rc,
        "matches_scipy": bool(np.allclose(got, sp(x), rtol=1e-5, atol=1e-5)),
        "matches_mlx_encoder_bitwise": bool(
            got.tobytes() == np.array(e8.correlate1d(xm, wm2)).tobytes()),
    }
    return out


# ---------------------------------------------------------------------------
# 7. ABI exposure
# ---------------------------------------------------------------------------
def section_abi(mlx_dir):
    so = glob.glob(os.path.join(EXT_DIR, "_e8_mlx*.so"))[0]
    nm = subprocess.run(["nm", "-m", "-u", so], capture_output=True, text=True).stdout
    mangled = [ln.split()[-3] for ln in nm.splitlines() if "(from libmlx)" in ln]
    dem = subprocess.run(["c++filt"], input="\n".join(mangled), capture_output=True,
                         text=True).stdout.splitlines()
    mlx_syms = sorted(s for s in dem if "mlx::" in s and not s.startswith(("MTL::", "NS::")))
    metalcpp = sorted(s for s in dem if s.startswith(("MTL::", "NS::")))
    other = sorted(s for s in dem if s not in mlx_syms and s not in metalcpp)

    deps = subprocess.run(
        [os.path.join(HERE, "..", "..", ".venv", "bin", "ninja"), "-C", EXT_DIR, "-t", "deps"],
        capture_output=True, text=True).stdout
    headers, cur = set(), None
    for ln in deps.splitlines():
        if ln and not ln.startswith(" "):
            cur = ln.split(":")[0]
        elif cur and "_e8_mlx.dir" in cur and ln.strip():
            headers.add(os.path.realpath(ln.strip()))
    inc = os.path.realpath(os.path.join(mlx_dir, "include"))
    mlx_headers = sorted(os.path.relpath(h, inc) for h in headers
                         if h.startswith(os.path.join(inc, "mlx") + os.sep))
    metalcpp_headers = [h for h in headers if h.startswith(os.path.join(inc, "metal_cpp") + os.sep)]
    directly_included = [
        "mlx/array.h", "mlx/allocator.h", "mlx/primitives.h", "mlx/utils.h", "mlx/ops.h",
        "mlx/stream.h", "mlx/version.h", "mlx/backend/metal/device.h",
        "mlx/backend/cpu/encoder.h"]

    # rpath: does the module still load if its LC_RPATH to the wheel is removed?
    otool = subprocess.run(["otool", "-l", so], capture_output=True, text=True).stdout
    rpaths = [ln.split()[1] for ln in otool.splitlines() if ln.strip().startswith("path ")]
    nodir = os.path.join(BUILD, "norpath")
    shutil.rmtree(nodir, ignore_errors=True)
    os.makedirs(nodir)
    so2 = os.path.join(nodir, os.path.basename(so))
    shutil.copy2(so, so2)
    for rp in rpaths:
        subprocess.run(["install_name_tool", "-delete_rpath", rp, so2], capture_output=True)
    subprocess.run(["codesign", "-f", "-s", "-", so2], capture_output=True)

    def try_import(pre):
        code = (f"import sys; sys.path.insert(0, {nodir!r}); {pre}"
                "import _e8_mlx; print('ok' if _e8_mlx.bridge_info()['ready'] else 'bridge-not-ready')")
        p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        if p.stdout.strip():
            return p.stdout.strip()
        if "Library not loaded: @rpath/libmlx.dylib" in p.stderr:
            return "ImportError: Library not loaded: @rpath/libmlx.dylib"
        return (p.stderr.strip().splitlines() or ["?"])[-1][:120]

    return {
        "module": os.path.basename(so),
        "libmlx_symbols_total": len(dem),
        "mlx_symbols": mlx_syms,
        "metal_cpp_private_symbols_from_libmlx": {
            "count": len(metalcpp), "examples": metalcpp[:6]},
        "other_symbols_from_libmlx": other,
        "mlx_headers_directly_included": directly_included,
        "mlx_headers_transitive": mlx_headers,
        "mlx_headers_transitive_count": len(mlx_headers),
        "metal_cpp_headers_transitive_count": len(metalcpp_headers),
        "rpaths": rpaths,
        "loads_without_rpath_if_mlx_core_imported_first": try_import("import mlx.core; "),
        "loads_without_rpath_cold": try_import(""),
    }


def section_native_nanobind(mx):
    """The documented NB_DOMAIN=mlx type-sharing path, built with nanobind 3.1.0."""
    import _e8_nbnative as nn
    a = mx.array([1.0, 2.0])
    res = {}
    for name, f in (("mx_array_argument", lambda: nn.ndim(a)),
                    ("mx_array_return", lambda: nn.make()),
                    ("both", lambda: nn.twice(a))):
        try:
            f()
            res[name] = "ok"
        except Exception as ex:  # noqa: BLE001
            res[name] = f"{type(ex).__name__}: {str(ex).splitlines()[0][:90]}"
    res["works"] = all(v == "ok" for v in res.values())
    return res


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-build", action="store_true")
    ap.add_argument("--quick", action="store_true", help="fewer repetitions")
    args = ap.parse_args()

    if not args.no_build:
        subprocess.run([os.path.join(HERE, "build.sh")], check=True)

    sys.path.insert(0, EXT_DIR)
    import numpy as np
    import mlx.core as mx
    import nanobind
    import scipy
    from scipy import ndimage
    import _e8_mlx as e8

    mlx_dir = os.path.dirname(mx.__file__)
    binfo = e8.bridge_info()
    if not binfo["ready"]:
        print("RESULT " + json.dumps({"probe": "e8_mlx_primitive", "error": binfo["error"]}))
        return 1
    e8.set_metallib(BUILD, "e8_kernels")
    # One-time costs.  The standalone runner goes first, so it pays for loading
    # the metallib and building the pipeline state cold; MLX's first eval_gpu
    # (get_library + get_kernel on the same file) comes second.
    t0 = time.perf_counter_ns()
    runner = Runner(np)
    t_runner_ms = (time.perf_counter_ns() - t0) / 1e6
    mx.eval(mx.array([0.0]))  # bring up MLX's Metal device/stream first
    e8.eval_stats(True)
    mx.eval(e8.correlate1d(mx.array([[1.0, 2.0, 3.0]]), mx.array([1.0])))
    first = e8.eval_stats(True)["recent"]
    one_time = {
        "standalone_runner_create_ms": round(t_runner_ms, 2),
        "mlx_first_eval_gpu_ms": round(first[0]["encode_ns"] / 1e6, 2) if first else None,
        "order": "standalone runner first, then MLX",
    }
    dev = mx.device_info()

    result = {
        "probe": "e8_mlx_primitive",
        "env": {
            "python": platform.python_version(),
            "macos": platform.mac_ver()[0],
            "chip": dev.get("device_name"),
            "gpu_arch": dev.get("architecture"),
            "mlx": mx.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "nanobind_build": nanobind.__version__,
            "metallib": os.path.relpath(METALLIB, HERE),
            "metallib_loaded_by_mlx_from": e8.metallib_path(),
        },
        "bridge": binfo,
        "one_time_costs": one_time,
    }

    log("verify")
    result["verify"] = section_verify(np, mx, e8, ndimage, runner)
    log("nanobind native type sharing")
    result["nanobind_native_type_sharing"] = section_native_nanobind(mx)
    log("timing (no instrumentation hooks installed)")
    result.update(section_timing(np, mx, e8, runner, args.quick))
    log("commit / host-sync evidence")
    result["commit_evidence"] = section_commit(np, mx, e8)
    log("zero-copy hand-offs")
    result["zero_copy"] = section_zero_copy(np, mx, e8, ndimage, runner)
    log("abi")
    result["abi"] = section_abi(mlx_dir)

    small = result["lazy"]["small"]
    large = result["lazy"]["large"]
    m = small["marginal_us_paired_median"]
    ci = small["marginal_us_paired_median_ci95"]
    hs = small["host_side_only"]
    slopes = small["chain"]["per_op_slope_us"]
    no_extra = result["commit_evidence"]["no_extra_commit_or_sync"]
    vs_builtin = max(m["custom_minus_builtin_cumsum"], m["custom_minus_builtin_sin"])
    result["gate"] = {
        "threshold_us": GATE_US,
        "metric": "small array, z = OP(a*2+1).sum(); mx.eval(z): T(custom op) - T(comparable "
                  "built-in op in its place), paired median of interleaved runs",
        # the comparison the gate is defined on
        "custom_minus_builtin_cumsum_us": m["custom_minus_builtin_cumsum"],
        "custom_minus_builtin_cumsum_ci95_us": ci["custom_minus_builtin_cumsum"],
        "custom_minus_builtin_sin_us": m["custom_minus_builtin_sin"],
        "custom_minus_builtin_sin_ci95_us": ci["custom_minus_builtin_sin"],
        "no_extra_commit_or_sync": no_extra,
        "pass": bool(vs_builtin <= GATE_US and no_extra),
        # stricter readings, reported for completeness
        "strict": {
            "end_to_end_custom_minus_no_op_at_all_us": m["custom_minus_noop"],
            "end_to_end_custom_minus_no_op_at_all_ci95_us": ci["custom_minus_noop"],
            "end_to_end_builtin_cumsum_minus_no_op_at_all_us": m["builtin_cumsum_minus_noop"],
            "end_to_end_builtin_sin_minus_no_op_at_all_us": m["builtin_sin_minus_noop"],
            "host_side_custom_minus_no_op_at_all_us": hs["marginal_us_paired_median"]["custom_minus_noop"],
            "host_side_custom_minus_no_op_at_all_ci95_us": hs["custom_minus_noop_ci95"],
            "host_side_builtin_cumsum_minus_no_op_at_all_us": hs["marginal_us_paired_median"]["builtin_cumsum_minus_noop"],
            "chain_per_op_slope_us": slopes,
            "custom_eval_gpu_encode_median_us": small["custom_eval_gpu_encode_us"]["median"],
            "end_to_end_within_threshold": bool(m["custom_minus_noop"] <= GATE_US),
            "host_side_within_threshold": bool(
                hs["marginal_us_paired_median"]["custom_minus_noop"] <= GATE_US),
            "chain_slope_within_threshold": bool(slopes["custom"] <= GATE_US),
        },
        "large": {
            "custom_minus_noop_us": large["marginal_us_paired_median"]["custom_minus_noop"],
            "custom_minus_builtin_cumsum_us": large["marginal_us_paired_median"]["custom_minus_builtin_cumsum"],
            "custom_minus_builtin_sin_us": large["marginal_us_paired_median"]["custom_minus_builtin_sin"],
            "host_side_custom_minus_noop_us": large["host_side_only"]["marginal_us_paired_median"]["custom_minus_noop"],
            "kernel_gpu_time_standalone_us": result["eager"]["large"]["standalone_c_side_us"]["gpu_execution_time"]["median"],
            "bridge_overhead_vs_kernel_gpu_time_us": large["bridge_overhead_vs_kernel_gpu_time_us"],
        },
    }
    result["correctness_pass"] = bool(
        result["verify"]["mlx_gpu_eq_standalone_bitwise_all"]
        and result["verify"]["scipy_allclose_all"])

    with open(os.path.join(BUILD, "result.json"), "w") as f:
        json.dump(result, f, indent=1)
    print("RESULT " + json.dumps(result, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
