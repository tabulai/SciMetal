#!/usr/bin/env python
"""E6 probe: hybrid float32 LU on the GPU + float64 iterative refinement.

Run:  ./build.sh && ../../.venv/bin/python run.py
Prints a final line `RESULT {json}`.

Problem: A x = b, A float64 (NumPy C order), single right-hand side.
Pipeline timed as one call (`e6_solve`):
  narrow      A -> float32 into a shared-storage MTLBuffer (GCD-parallel), ||A||_inf
  factor      right-looking blocked LU, partial pivoting, on the buffer:
                panel (CPU, recursive, in place) | row interchanges (CPU, GCD) |
                TRSM for U12 (CPU, cblas_strsm) | trailing update (GPU, steel GEMM,
                in place, transpose identity) with look-ahead so the next panel is
                factored on the CPU while the GPU updates the rest
  solve       float32 triangular solves, blocked: NEON dot products for the
              off-diagonal part split over 8 GCD workers + cblas_strsv on the
              diagonal blocks; L is left in LINPACK form (no deferred left
              interchanges), the solve interleaves the interchanges per block
  refine      r = b - A x in float64 (cblas_dgemv on row blocks), correction with
              the float32 factors, until the normwise backward error
              ||r||_inf / (||A||_inf ||x||_inf + ||b||_inf) is <= 4u or stops
              improving (by at least 2x per step).
Test matrices: A = U diag(sigma) V^T with sigma geometric in [1/kappa, 1]
("randsvd mode 3"); U, V are products of two randomized Walsh-Hadamard
transforms (exactly orthogonal, dense, O(n^2 log n) to build).
"""
import ctypes as C
import json
import os
import sys
import time

import numpy as np
import scipy.linalg

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e6lib  # noqa: E402

lib = e6lib.load()
T_START = time.perf_counter()
NMAX = 8192
ctx = lib.e6_create(NMAX)
assert ctx
rng = np.random.default_rng(2026)
SEC = C.c_double()

DEFAULT = dict(nb=256, cpu_tail=1024, lookahead=2, panel_leaf=4, par_resid=1,
               par_trsv=1, left_swaps=0, cpu_parts=1)


def berr(A, x, b, normA):
    r = b - A @ x
    return float(np.abs(r).max() / (normA * np.abs(x).max() + np.abs(b).max()))


def run_hybrid(A, b, reps, warm=1, **kw):
    n = A.shape[0]
    o = e6lib.default_opts(**{**DEFAULT, **kw})
    st = e6lib.Stats()
    x = np.zeros(n)
    runs = []
    for r in range(warm + reps):
        info = lib.e6_solve(ctx, n, A, b, x, C.byref(o), C.byref(st))
        assert info == 0, info
        if r >= warm:
            runs.append(st.as_dict())
    runs.sort(key=lambda d: d["t_total"])
    med = runs[len(runs) // 2]
    out = {
        "total_ms": med["t_total"] * 1e3, "best_total_ms": runs[0]["t_total"] * 1e3,
        "narrow_ms": med["t_narrow"] * 1e3, "factor_ms": med["t_factor"] * 1e3,
        "solve_refine_ms": (med["t_solve0"] + med["t_refine"]) * 1e3,
        "refine_steps": med["iters"], "berr_history": med["berr"],
        "n_gpu_calls": med["n_gpu"], "n_panels": med["n_panels"], "reps": reps,
        "breakdown_ms": {
            "panel": med["t_panel"] * 1e3,
            "swaps_plus_trsm": (med["t_swap"] + med["t_trsm"] + med["t_left_swap"]) * 1e3,
            "swaps": (med["t_swap"] + med["t_left_swap"]) * 1e3, "trsm": med["t_trsm"] * 1e3,
            "lookahead_update": med["t_lookahead_gemm"] * 1e3,
            "gpu_submit": med["t_submit"] * 1e3, "sync_wait": med["t_wait"] * 1e3,
            "gpu_execution_overlapped": med["t_gpu_exec"] * 1e3,
            "cpu_tail": med["t_tail"] * 1e3,
            "residuals_f64": med["t_resid"] * 1e3, "triangular_solves_f32": med["t_trsv"] * 1e3,
        },
    }
    return out, x.copy()


def med_time(f, reps):
    ts = []
    for _ in range(reps):
        ts.append(f())
    return float(np.median(ts)) * 1e3


res = {"device": lib.msg_device_name().decode(), "config": DEFAULT, "cases": {}, "baselines": {}}

for n in (256, 512, 1024, 2048, 4096, 8192):
    reps_l = 5 if n <= 2048 else (3 if n == 4096 else 1)
    ipiv = np.zeros(n, dtype=np.int32)
    for kappa in (1e2, 1e4):
        key = f"n{n}_k{kappa:.0e}"
        A = np.empty((n, n))
        lib.e6_gen_randsvd(n, kappa, int(n + kappa), A)
        normA = float(np.abs(A).sum(axis=1).max())
        xt = rng.standard_normal(n)
        b = A @ xt
        Af = np.asfortranarray(A)                 # LAPACK's view of the same matrix
        W = np.empty_like(Af)
        base = {}

        # dsgesv (leaves A intact when its refinement converges)
        xs = np.zeros(n)
        it = C.c_int()

        def f_dsgesv():
            W[...] = Af
            info = lib.e6_dsgesv(n, W, b, xs, ipiv, C.byref(it), C.byref(SEC))
            assert info == 0
            return SEC.value

        base["dsgesv_ms"] = med_time(f_dsgesv, reps_l)
        base["dsgesv_iter"] = it.value
        base["dsgesv_berr"] = berr(A, xs, b, normA)

        # dgesv: also the float64 reference solution
        xref = np.zeros(n)

        def f_dgesv():
            W[...] = Af
            xref[:] = b
            info = lib.e6_dgesv(n, W, xref, ipiv, C.byref(SEC))
            assert info == 0
            return SEC.value

        base["dgesv_ms"] = med_time(f_dgesv, reps_l)
        base["dgesv_berr"] = berr(A, xref, b, normA)

        # explicit dgetrf + dgetrs (Accelerate's dgesv driver is slower than this)
        xg = np.zeros(n)

        def f_dgetrf():
            W[...] = Af
            xg[:] = b
            info = lib.e6_dgetrf_dgetrs(n, W, xg, ipiv, C.byref(SEC))
            assert info == 0
            return SEC.value

        base["dgetrf_dgetrs_ms"] = med_time(f_dgetrf, reps_l)
        base["dgesv_ferr_vs_true"] = float(np.abs(xref - xt).max() / np.abs(xt).max())

        if kappa == 1e2:
            Ws = np.empty((n, n), dtype=np.float32, order="F")
            Afs = Af.astype(np.float32, order="F")
            xs32 = np.zeros(n, dtype=np.float32)

            def f_sgesv():
                Ws[...] = Afs
                xs32[:] = b
                info = lib.e6_sgesv(n, Ws, xs32, ipiv, C.byref(SEC))
                assert info == 0
                return SEC.value

            base["sgesv_ms"] = med_time(f_sgesv, reps_l)
            base["sgesv_berr"] = berr(A, xs32.astype(np.float64), b, normA)

            def f_sgetrf():
                Ws[...] = Afs
                xs32[:] = b
                info = lib.e6_sgetrf_sgetrs(n, Ws, xs32, ipiv, C.byref(SEC))
                assert info == 0
                return SEC.value

            base["sgetrf_sgetrs_ms"] = med_time(f_sgetrf, reps_l)

            def f_scipy():
                t0 = time.perf_counter()
                f_scipy.x = scipy.linalg.solve(A, b)
                return time.perf_counter() - t0

            base["scipy_solve_ms"] = med_time(f_scipy, reps_l)
            base["scipy_solve_berr"] = berr(A, f_scipy.x, b, normA)
            del Ws, Afs
        del W, Af

        h, x = run_hybrid(A, b, reps=5 if n >= 1024 else 7)
        h["berr_final"] = berr(A, x, b, normA)
        h["ferr_vs_dgesv"] = float(np.abs(x - xref).max() / np.abs(xref).max())
        h["ferr_vs_true"] = float(np.abs(x - xt).max() / np.abs(xt).max())
        h["speedup_vs_dgesv"] = base["dgesv_ms"] / h["total_ms"]
        h["speedup_vs_dgetrf_dgetrs"] = base["dgetrf_dgetrs_ms"] / h["total_ms"]
        res["cases"][key] = h
        res["baselines"][key] = base
        bd = h["breakdown_ms"]
        print(f"{key:12s} hybrid {h['total_ms']:7.1f} ms (narrow {h['narrow_ms']:.1f} factor "
              f"{h['factor_ms']:.1f} solve+refine {h['solve_refine_ms']:.1f}; {h['refine_steps']} steps, "
              f"{h['n_gpu_calls']} GPU calls) berr {h['berr_final']:.1e} ferr_vs_dgesv "
              f"{h['ferr_vs_dgesv']:.1e} | dgesv {base['dgesv_ms']:.1f} dgetrf+dgetrs "
              f"{base['dgetrf_dgetrs_ms']:.1f} dsgesv {base['dsgesv_ms']:.1f} (iter {base['dsgesv_iter']})"
              + (f" sgesv {base['sgesv_ms']:.1f} sgetrf+sgetrs {base['sgetrf_sgetrs_ms']:.1f} "
                 f"scipy {base['scipy_solve_ms']:.1f}" if "sgesv_ms" in base else "")
              + f" ms | x{h['speedup_vs_dgesv']:.2f} vs dgesv, x{h['speedup_vs_dgetrf_dgetrs']:.2f} "
              f"vs dgetrf+dgetrs", flush=True)
        print(f"             factor: panel {bd['panel']:.1f} swaps {bd['swaps']:.1f} trsm {bd['trsm']:.1f} "
              f"lookahead {bd['lookahead_update']:.1f} submit {bd['gpu_submit']:.1f} sync-wait "
              f"{bd['sync_wait']:.1f} cpu-tail {bd['cpu_tail']:.1f} (GPU busy {bd['gpu_execution_overlapped']:.1f}) | "
              f"refine: residuals {bd['residuals_f64']:.1f} trsv {bd['triangular_solves_f32']:.1f} ms",
              flush=True)

        # design sweep on the largest problem, kappa = 1e2
        if n == NMAX and kappa == 1e2:
            sweep = {}
            for name, kw in (
                    ("nb128", dict(nb=128)), ("nb384", dict(nb=384)), ("nb512", dict(nb=512)),
                    ("lookahead_cpu_sgemm", dict(lookahead=1)), ("no_lookahead", dict(lookahead=0)),
                    ("panel_leaf1", dict(panel_leaf=1)), ("panel_leaf16", dict(panel_leaf=16)),
                    ("cpu_tail2048", dict(cpu_tail=2048)),
                    ("cpu_gemm_trsm_split4", dict(lookahead=1, cpu_parts=4)),
                    ("lapack_form_left_swaps", dict(left_swaps=1)),
                    ("plain_cblas_strsv", dict(par_trsv=0)),
                    ("single_dgemv_residual", dict(par_resid=0)),
                    ("cpu_only_float32_plus_refine", dict(gpu=0))):
                s, _ = run_hybrid(A, b, reps=3, warm=0, **kw)
                sweep[name] = {"total_ms": s["total_ms"], "factor_ms": s["factor_ms"],
                               "solve_refine_ms": s["solve_refine_ms"],
                               "n_gpu_calls": s["n_gpu_calls"], "breakdown_ms": s["breakdown_ms"]}
                print(f"   sweep {name:30s} total {s['total_ms']:7.1f} factor {s['factor_ms']:7.1f} "
                      f"solve+refine {s['solve_refine_ms']:6.1f} ms", flush=True)
            res["sweep_n8192_k1e2"] = sweep
        del A

lib.e6_destroy(ctx)

# ------------------------------------------------------------------- gates
c8 = [res["cases"][f"n8192_k{k:.0e}"] for k in (1e2, 1e4)]
sizes = (256, 512, 1024, 2048, 4096, 8192)
def break_even(base_key):
    """smallest tested n from which the hybrid wins at every larger tested n"""
    out = {}
    for k in (1e2, 1e4):
        wins = [n for n in sizes
                if res["cases"][f"n{n}_k{k:.0e}"]["total_ms"] <= res["baselines"][f"n{n}_k{k:.0e}"][base_key]]
        brk = None
        for n in sizes:
            if all(m in wins for m in sizes if m >= n):
                brk = n
                break
        out[f"k{k:.0e}"] = brk
    return out


be = break_even("dgesv_ms")
be2 = break_even("dgetrf_dgetrs_ms")
res["gates"] = {
    "n8192_total_ms": {"k1e+02": c8[0]["total_ms"], "k1e+04": c8[1]["total_ms"]},
    "gate_n8192_le_250ms": bool(max(c["total_ms"] for c in c8) <= 250.0),
    "max_berr_final": max(c["berr_final"] for c in res["cases"].values()),
    "gate_berr_le_1e-14": bool(max(c["berr_final"] for c in res["cases"].values()) <= 1e-14),
    "break_even_n_vs_dgesv": be,
    "break_even_n_vs_dgetrf_dgetrs": be2,
    "tested_sizes": list(sizes),
    "gate_break_even_le_2000": bool(all(v is not None and v <= 2000 for v in be.values())),
}
res["wall_seconds_total"] = time.perf_counter() - T_START
print("gates:", res["gates"])
print(f"total wall {res['wall_seconds_total']:.1f} s")
print("RESULT " + json.dumps(res))
