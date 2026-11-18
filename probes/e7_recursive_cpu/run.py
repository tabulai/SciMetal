#!/usr/bin/env python
"""E7 probe: recursive LU / Cholesky on Accelerate GEMM vs Accelerate LAPACK.

Run:  ./build.sh && ../../.venv/bin/python run.py
Prints a final line `RESULT {json}`.

Variants timed (all column-major, in place, single calling thread):
  LU   lapack     Accelerate ?getrf
       rec        Toledo column recursion: recurse left, swaps, TRSM, GEMM,
                  recurse right; unblocked leaf <= 4 columns; cblas TRSM.
                  No threads of our own anywhere.
       blk        right-looking blocked, nb = n/8, panel by the recursion above,
                  full-width rank-nb GEMM updates; row interchanges applied per
                  column under dispatch_apply (the only threading of our own,
                  never around an Accelerate call).
  Chol lapack     Accelerate ?potrf
       rec_syrk   recursive, trailing update by cblas SYRK
       rec_gemm   recursive, trailing update by full cblas GEMM
       blk_gemm   right-looking blocked, nb = n/8, GEMM update, TRSM done
                  recursively on GEMM (cblas TRSM below 256)
At n = 8192 only the headline variants are timed (and float32 Cholesky is
skipped) to keep the whole run under a minute.
Backward errors are a double-double Hutchinson estimate (see e7.cpp).
"""
import ctypes as C
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e7lib  # noqa: E402

lib = e7lib.load()
OUT = (C.c_double * 2)()
T_START = time.perf_counter()
Q_PROBE = 12

LU_LEAF, CHOL_LEAF = 4, 32


def f(name, sfx):
    return getattr(lib, f"e7_{name}_{sfx}")


def timeit(call, A0, W, reps, warm=0):
    """call() runs one factorisation of W (refreshed from A0, untimed)."""
    ts, cw, best_stats = [], [], None
    for r in range(warm + reps):
        W[...] = A0
        lib.e7_clear_stats()
        info = call()
        if r < warm:
            continue
        ts.append(OUT[0])
        cw.append(OUT[1] / OUT[0])
        if OUT[0] <= min(ts):
            best_stats = e7lib.stats(lib)
    return {"ms": float(np.median(ts)) * 1e3, "best_ms": float(np.min(ts)) * 1e3,
            "cpu_over_wall": float(np.median(cw)), "reps": reps, "info": int(info),
            "_stats": best_stats}


def breakdown(st):
    return {k[:-2] + "_ms": round(st[k] * 1e3, 2) for k in ("gemm_s", "trsm_s", "swap_s", "leaf_s", "syrk_s")}


def spd(rng, n, dt):
    R = rng.random((n, n), dtype=np.float32) * 2 - 1
    A = (R + R.T).astype(np.float64) * 0.5
    A[np.diag_indices(n)] += 2.0 * np.sqrt(n)
    return np.asfortranarray(A.astype(dt))


def set_opts(trsm_leaf=0, chol_update=0, rsyrk_leaf=256, swap_par=0):
    lib.e7_set_opts(LU_LEAF, trsm_leaf, CHOL_LEAF, chol_update, rsyrk_leaf, 1)
    lib.e7_set_swap_par(swap_par)


res = {"threads_at_start": lib.e7_thread_count(), "lu": {}, "chol": {}, "gemm": {}}
rng = np.random.default_rng(7)

for n in (2048, 4096, 8192):
    big = n == 8192
    for sfx, dt in (("d", np.float64), ("s", np.float32)):
        key = f"{sfx}{n}"
        reps_lapack = 3 if big else 5
        reps_mine = 5
        reps_extra = 3 if big else 5       # secondary variants at n = 8192
        warm = 0 if big else 1
        nb = max(256, n // 8)
        ipiv = np.zeros(n, dtype=np.int32)
        flu = (2.0 / 3.0) * n ** 3
        fch = (1.0 / 3.0) * n ** 3

        # ------------------------------------------------------------- LU
        A0 = np.asfortranarray(rng.standard_normal((n, n), dtype=np.float32).astype(dt))
        W = np.empty_like(A0)
        lu = {}
        r = timeit(lambda: f("getrf_lapack", sfx)(n, W, n, ipiv, OUT), A0, W, reps_lapack, warm)
        r["resid"] = f("lu_resid_probe", sfx)(n, A0, n, W, n, ipiv, Q_PROBE)
        lu["lapack"] = r
        if not (big and sfx == "s"):
            set_opts()
            r = timeit(lambda: f("getrf_rec", sfx)(n, n, W, n, ipiv, OUT), A0, W, reps_extra, warm)
            r["resid"] = f("lu_resid_probe", sfx)(n, A0, n, W, n, ipiv, Q_PROBE)
            lu["rec"] = r
        set_opts(swap_par=1)
        r = timeit(lambda: f("getrf_blk", sfx)(n, W, n, ipiv, nb, OUT), A0, W, reps_mine, warm)
        r["resid"] = f("lu_resid_probe", sfx)(n, A0, n, W, n, ipiv, Q_PROBE)
        r["nb"] = nb
        lu["blk"] = r
        for k, v in lu.items():
            st = v.pop("_stats")
            v["gflops"] = flu / (v["ms"] * 1e-3) / 1e9
            v["speedup_vs_lapack"] = lu["lapack"]["ms"] / v["ms"]
            if k != "lapack":
                v["breakdown"] = breakdown(st)
        res["lu"][key] = lu
        print(f"LU   {key:6s} " + " | ".join(
            f"{k} {v['ms']:8.1f} ms (x{v['speedup_vs_lapack']:.2f}, {v['gflops']:.0f} GF, "
            f"resid {v['resid']:.1e}, cpu/wall {v['cpu_over_wall']:.1f})" for k, v in lu.items()), flush=True)

        # ------------------------------------------------------- Cholesky
        if big and sfx == "s":
            continue
        A0 = spd(rng, n, dt)
        W = np.empty_like(A0)
        ch = {}
        r = timeit(lambda: f("potrf_lapack", sfx)(n, W, n, OUT), A0, W, reps_lapack, warm)
        r["resid"] = f("chol_resid_probe", sfx)(n, A0, n, W, n, Q_PROBE)
        ch["lapack"] = r
        if not big:   # SYRK-vs-GEMM comparison at n <= 4096 (SYRK is the slow one)
            set_opts(chol_update=0)
            r = timeit(lambda: f("potrf_rec", sfx)(n, W, n, OUT), A0, W, reps_mine, warm)
            r["resid"] = f("chol_resid_probe", sfx)(n, A0, n, W, n, Q_PROBE)
            ch["rec_syrk"] = r
            set_opts(chol_update=1)
            r = timeit(lambda: f("potrf_rec", sfx)(n, W, n, OUT), A0, W, reps_mine, warm)
            r["resid"] = f("chol_resid_probe", sfx)(n, A0, n, W, n, Q_PROBE)
            ch["rec_gemm"] = r
        set_opts(chol_update=1, trsm_leaf=256)
        r = timeit(lambda: f("potrf_blk", sfx)(n, W, n, nb, OUT), A0, W, reps_mine, warm)
        r["resid"] = f("chol_resid_probe", sfx)(n, A0, n, W, n, Q_PROBE)
        r["nb"] = nb
        ch["blk_gemm"] = r
        for k, v in ch.items():
            st = v.pop("_stats")
            v["gflops_n3_over_3"] = fch / (v["ms"] * 1e-3) / 1e9
            v["speedup_vs_lapack"] = ch["lapack"]["ms"] / v["ms"]
            if k != "lapack":
                v["breakdown"] = breakdown(st)
        res["chol"][key] = ch
        print(f"Chol {key:6s} " + " | ".join(
            f"{k} {v['ms']:8.1f} ms (x{v['speedup_vs_lapack']:.2f}, resid {v['resid']:.1e}, "
            f"cpu/wall {v['cpu_over_wall']:.1f})" for k, v in ch.items()), flush=True)

        # --------------------------------------------- GEMM rate (context)
        if not big:
            o3 = (C.c_double * 3)()
            f("gemm_shape", sfx)(n, n, n, 3, o3)
            res["gemm"][key] = {"gflops": 2.0 * n ** 3 / o3[0] / 1e9, "cpu_over_wall": o3[2]}
        del A0, W

# ---------------------------------------------------------------------------
# design variants at n = 4096, float64 (3 reps each, best-of)
# ---------------------------------------------------------------------------
n, sfx, dt = 4096, "d", np.float64
ipiv = np.zeros(n, dtype=np.int32)
A0 = np.asfortranarray(rng.standard_normal((n, n)))
W = np.empty_like(A0)
var = {}


def variant(name, call):
    r = timeit(call, A0, W, 2, 0)
    st = r.pop("_stats")
    var[name] = {"best_ms": r["best_ms"], "cpu_over_wall": r["cpu_over_wall"], "breakdown": breakdown(st)}


set_opts(); variant("lu_rec", lambda: f("getrf_rec", sfx)(n, n, W, n, ipiv, OUT))
set_opts(trsm_leaf=512); variant("lu_rec_recursive_trsm512", lambda: f("getrf_rec", sfx)(n, n, W, n, ipiv, OUT))
set_opts(swap_par=1); variant("lu_rec_parswap", lambda: f("getrf_rec", sfx)(n, n, W, n, ipiv, OUT))
for nb in (256, 512, 1024):
    if nb != 256:
        set_opts(); variant(f"lu_blk_nb{nb}", lambda: f("getrf_blk", sfx)(n, W, n, ipiv, nb, OUT))
    set_opts(swap_par=1); variant(f"lu_blk_nb{nb}_parswap", lambda: f("getrf_blk", sfx)(n, W, n, ipiv, nb, OUT))
lvl = None
set_opts()
W[...] = A0
lib.e7_clear_stats()
f("getrf_rec", sfx)(n, n, W, n, ipiv, OUT)
lvl = {k: {"ms": round(v["ms"], 2), "gflops": round(v["gflops"], 1)} for k, v in e7lib.gemm_levels(lib).items()}
res["lu_rec_gemm_rate_by_K_d4096"] = lvl

A0 = spd(rng, n, dt)
set_opts(chol_update=0); variant("chol_rec_syrk", lambda: f("potrf_rec", sfx)(n, W, n, OUT))
set_opts(chol_update=1); variant("chol_rec_gemm", lambda: f("potrf_rec", sfx)(n, W, n, OUT))
set_opts(chol_update=1, trsm_leaf=512); variant("chol_rec_gemm_recursive_trsm512", lambda: f("potrf_rec", sfx)(n, W, n, OUT))
set_opts(chol_update=2, rsyrk_leaf=256); variant("chol_rec_rsyrk_syrkleaf256", lambda: f("potrf_rec", sfx)(n, W, n, OUT))
set_opts(chol_update=3, rsyrk_leaf=512); variant("chol_rec_rsyrk_gemmleaf512", lambda: f("potrf_rec", sfx)(n, W, n, OUT))
set_opts(chol_update=0); variant("chol_blk_nb512_syrk", lambda: f("potrf_blk", sfx)(n, W, n, 512, OUT))
set_opts(chol_update=1); variant("chol_blk_nb512_gemm", lambda: f("potrf_blk", sfx)(n, W, n, 512, OUT))
set_opts(chol_update=1, trsm_leaf=256); variant("chol_blk_nb512_gemm_recursive_trsm256", lambda: f("potrf_blk", sfx)(n, W, n, 512, OUT))
res["variants_d4096"] = var
for k, v in var.items():
    print(f"variant d4096 {k:34s} best {v['best_ms']:7.1f} ms  cpu/wall {v['cpu_over_wall']:.1f}  {v['breakdown']}",
          flush=True)

# rank-512 SYRK vs GEMM on a 4096 x 4096 result (the shape a blocked Cholesky
# issues; this is where Accelerate's SYRK is slow)
o3 = (C.c_double * 3)()
lib.e7_rankk_d.argtypes = [C.c_int, C.c_int, C.c_int, C.c_int, C.POINTER(C.c_double)]
rk = {}
for name, use_syrk in (("dsyrk", 1), ("dgemm", 0)):
    lib.e7_rankk_d(4096, 512, use_syrk, 3, o3)
    rk[f"{name}_n4096_k512_ms"] = o3[0] * 1e3
    rk[f"{name}_n4096_k512_cpu_over_wall"] = o3[2]
rk["dsyrk_gflops_n2k"] = 4096.0 ** 2 * 512 / (rk["dsyrk_n4096_k512_ms"] * 1e-3) / 1e9
rk["dgemm_gflops_2n2k"] = 2 * 4096.0 ** 2 * 512 / (rk["dgemm_n4096_k512_ms"] * 1e-3) / 1e9
res["rank512_update_4096"] = rk
print("rank-512 update of a 4096^2 block:", {k: round(v, 1) for k, v in rk.items()}, flush=True)

# BLAS3 building blocks at n = 4096 (one n x n operand each)
o2 = (C.c_double * 2)()
blk = {}
for sfx, dt in (("d", np.float64), ("s", np.float32)):
    A = np.asfortranarray(rng.standard_normal((n, n)).astype(dt))
    L = np.asfortranarray((np.tril(A, -1) / n + np.eye(n)).astype(dt))
    B = np.asfortranarray(rng.standard_normal((n, n)).astype(dt))
    Cm = np.zeros((n, n), dtype=dt, order="F")
    best = {"trsm": 1e9, "syrk": 1e9}
    cw = {}
    for _ in range(2):
        f("trsm", sfx)(n, L, B, o2)
        if o2[0] < best["trsm"]:
            best["trsm"], cw["trsm"] = o2[0], o2[1] / o2[0]
        f("syrk", sfx)(n, A, Cm, o2)
        if o2[0] < best["syrk"]:
            best["syrk"], cw["syrk"] = o2[0], o2[1] / o2[0]
    blk[f"{sfx}trsm_4096_gflops"] = n ** 3 / best["trsm"] / 1e9
    blk[f"{sfx}syrk_4096_gflops"] = n ** 3 / best["syrk"] / 1e9
    blk[f"{sfx}trsm_cpu_over_wall"] = cw["trsm"]
    blk[f"{sfx}syrk_cpu_over_wall"] = cw["syrk"]
res["blas3_blocks"] = blk
print("blas3 blocks @4096:", {k: round(v, 1) for k, v in blk.items()}, flush=True)

res["threads_at_end"] = lib.e7_thread_count()
res["hw_ncpu"] = os.cpu_count()


def best_of(d, keys):
    return max(d[k]["speedup_vs_lapack"] for k in keys if k in d)


g = {
    "lu_d8192_speedup_rec": res["lu"]["d8192"]["rec"]["speedup_vs_lapack"],
    "lu_d8192_speedup_blk": res["lu"]["d8192"]["blk"]["speedup_vs_lapack"],
    "lu_d8192_speedup_best": best_of(res["lu"]["d8192"], ("rec", "blk")),
    "lu_d2048_speedup_rec": res["lu"]["d2048"]["rec"]["speedup_vs_lapack"],
    "lu_d2048_speedup_best": best_of(res["lu"]["d2048"], ("rec", "blk")),
    "chol_d8192_speedup_best": best_of(res["chol"]["d8192"], ("rec_syrk", "rec_gemm", "blk_gemm")),
    "chol_d2048_speedup_best": best_of(res["chol"]["d2048"], ("rec_syrk", "rec_gemm", "blk_gemm")),
}
g["gate_lu_4x_at_8192"] = bool(g["lu_d8192_speedup_best"] >= 4.0)
g["gate_chol_4x_at_8192"] = bool(g["chol_d8192_speedup_best"] >= 4.0)
g["gate_lu_1.5x_at_2048"] = bool(g["lu_d2048_speedup_best"] >= 1.5)
g["gate_chol_1.5x_at_2048"] = bool(g["chol_d2048_speedup_best"] >= 1.5)
res["gates"] = g
res["wall_seconds_total"] = time.perf_counter() - T_START
print(f"gates: {g}")
print(f"total wall {res['wall_seconds_total']:.1f} s")
print("RESULT " + json.dumps(res))
