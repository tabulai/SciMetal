#!/usr/bin/env python
"""E1 probe: MLX's steel GEMM in our own metallib + runtime.

Run:  ./build.sh && ../../.venv/bin/python run.py
Prints a final line `RESULT {json}`.
"""
import ctypes as C
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e1lib  # noqa: E402

import mlx.core as mx  # noqa: E402

lib = e1lib.load()
T4 = C.c_double * 4


def bench(f, reps=5, warm=1):
    for _ in range(warm):
        f()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), float(np.min(ts))


def gflops(n, t):
    return 2.0 * n ** 3 / t / 1e9


def relerr(x, ref):
    """max-norm relative error: max|x-ref| / max|ref|."""
    return float(np.abs(x - ref).max() / np.abs(ref).max())


def bits(x):
    return np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)


def bitdiff(x, y):
    """fraction of elements whose bit patterns differ, and max ULP distance."""
    bx, by = bits(x).astype(np.int64), bits(y).astype(np.int64)
    d = np.abs(bx - by)
    return float(np.count_nonzero(d) / d.size), int(d.max())


def ours_matmul(A, B, D, n, times=None):
    a = e1lib.args(A, 0, n, B, 0, n, D, 0, n, n, n, n)
    rc = lib.msg_gemm(C.byref(a), times)
    assert rc == 0


res = {
    "device": lib.msg_device_name().decode(),
    "arch": lib.msg_device_arch().decode(),
    "mlx_version": mx.__version__,
    "mlx_default_device": str(mx.default_device()),
}
rng = np.random.default_rng(1234)

# --------------------------------------------------------------------------
# B. fixed cost of a tiny call (64x64): encode + commit + waitUntilCompleted
#    Sampled in three batches (start / middle / end of the run) because the
#    number is dominated by scheduling latency and is sensitive to machine load.
# --------------------------------------------------------------------------
_tn = 64
_tA = e1lib.GpuBuf(lib, _tn * _tn); _tB = e1lib.GpuBuf(lib, _tn * _tn); _tD = e1lib.GpuBuf(lib, _tn * _tn)
_tA.np[:] = rng.standard_normal(_tn * _tn, dtype=np.float32)
_tB.np[:] = rng.standard_normal(_tn * _tn, dtype=np.float32)
_ta = e1lib.args(_tA, 0, _tn, _tB, 0, _tn, _tD, 0, _tn, _tn, _tn, _tn)
tiny_rows, tiny_spin, tiny_batches = [], [], {}


def tiny_batch(label, reps=300):
    t = T4()
    for _ in range(100):
        lib.msg_gemm(C.byref(_ta), t)
    rows, spin = [], []
    for _ in range(reps):
        lib.msg_gemm(C.byref(_ta), t)          # encode + commit + waitUntilCompleted
        rows.append(list(t))
        lib.msg_gemm_spin(C.byref(_ta), t)     # same, busy-polling the status
        spin.append(sum(t[:3]))
    tiny_rows.extend(rows)
    tiny_spin.extend(spin)
    tot = np.array(rows)[:, :3].sum(axis=1)
    tiny_batches[label] = {"median_ms": float(np.median(tot)) * 1e3,
                           "p10_ms": float(np.percentile(tot, 10)) * 1e3,
                           "spin_median_ms": float(np.median(spin)) * 1e3}
    print(f"tiny 64x64 [{label}]: median {tiny_batches[label]['median_ms']:.3f} ms, "
          f"p10 {tiny_batches[label]['p10_ms']:.3f} ms, busy-poll median "
          f"{tiny_batches[label]['spin_median_ms']:.3f} ms", flush=True)


def tiny_finish():
    rows = np.array(tiny_rows)
    tot = rows[:, :3].sum(axis=1)
    tiny = {
        "total_ms_median": float(np.median(tot)) * 1e3,
        "total_ms_p10": float(np.percentile(tot, 10)) * 1e3,
        "total_ms_p90": float(np.percentile(tot, 90)) * 1e3,
        "total_ms_min": float(tot.min()) * 1e3,
        "best_batch_median_ms": min(b["median_ms"] for b in tiny_batches.values()),
        "encode_ms_median": float(np.median(rows[:, 0])) * 1e3,
        "commit_ms_median": float(np.median(rows[:, 1])) * 1e3,
        "wait_ms_median": float(np.median(rows[:, 2])) * 1e3,
        "gpu_ms_median": float(np.median(rows[:, 3])) * 1e3,
        "spin_total_ms_median": float(np.median(tiny_spin)) * 1e3,
        "batches": tiny_batches,
    }
    lib.msg_gemm(C.byref(_ta), None)
    tiny["kernel"] = lib.msg_last_kernel().decode()
    # 8 tiny GEMMs: 8 command buffers pipelined, and 8 dispatches in one buffer
    arr = (e1lib.GemmArgs * 8)(*[_ta] * 8)
    ts8, ts1 = [], []
    for _ in range(100):
        t0 = time.perf_counter()
        hs = [lib.msg_gemm_submit(C.byref(_ta)) for _ in range(8)]
        for h in hs:
            lib.msg_wait(h, None)
        ts8.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        h = lib.msg_gemm_submit_many(arr, 8)
        lib.msg_wait(h, None)
        ts1.append(time.perf_counter() - t0)
    tiny["per_call_ms_8_pipelined_cmdbufs"] = float(np.median(ts8)) * 1e3 / 8
    tiny["per_call_ms_8_dispatches_one_cmdbuf"] = float(np.median(ts1)) * 1e3 / 8
    ref = _tA.np.reshape(_tn, _tn).astype(np.float64) @ _tB.np.reshape(_tn, _tn).astype(np.float64)
    tiny["relerr_vs_f64"] = relerr(_tD.np.reshape(_tn, _tn), ref)
    print(f"tiny 64x64 overall: total {tiny['total_ms_median']:.3f} ms (encode {tiny['encode_ms_median']:.3f}, "
          f"commit {tiny['commit_ms_median']:.3f}, wait {tiny['wait_ms_median']:.3f}; gpu {tiny['gpu_ms_median']:.3f}); "
          f"busy-poll {tiny['spin_total_ms_median']:.3f} ms; 8 pipelined cmdbufs "
          f"{tiny['per_call_ms_8_pipelined_cmdbufs']:.3f} ms/call; 8 dispatches in one cmdbuf "
          f"{tiny['per_call_ms_8_dispatches_one_cmdbuf']:.3f} ms/call", flush=True)
    return tiny


tiny_batch("start")

# --------------------------------------------------------------------------
# A. square float32 matmul: ours vs MLX GPU vs Accelerate sgemm / dgemm
# --------------------------------------------------------------------------
square = {}
for n in (1024, 2048, 4096, 8192):
    a = rng.standard_normal((n, n), dtype=np.float32)
    b = rng.standard_normal((n, n), dtype=np.float32)

    A = e1lib.GpuBuf(lib, n * n)
    B = e1lib.GpuBuf(lib, n * n)
    D = e1lib.GpuBuf(lib, n * n)
    A.np[:] = a.ravel()
    B.np[:] = b.ravel()
    D.np[:] = 0
    t = T4()
    gpu_ts, ours_ts, mlx_ts = [], [], []
    ma, mb = mx.array(a), mx.array(b)
    mx.eval(ma, mb)
    holder = {}

    def f_ours():
        ours_matmul(A, B, D, n, t)
        gpu_ts.append(t[3])

    # MLX GPU, measured properly: inputs evaluated first, time eval(product).
    def f_mlx():
        mc = mx.matmul(ma, mb, stream=mx.gpu)
        mx.eval(mc)
        holder["c"] = mc

    # Interleave the two so that both see the same machine load.
    for _ in range(3):
        f_ours(); f_mlx()
    gpu_ts.clear()
    for _ in range(15 if n <= 4096 else 9):
        t0 = time.perf_counter(); f_ours(); ours_ts.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); f_mlx(); mlx_ts.append(time.perf_counter() - t0)
    t_ours, t_ours_min = float(np.median(ours_ts)), float(np.min(ours_ts))
    t_mlx, t_mlx_min = float(np.median(mlx_ts)), float(np.min(mlx_ts))
    kernel = lib.msg_last_kernel().decode()
    disp = (C.c_int * 5)()
    lib.msg_last_dispatch(disp)
    d_ours = D.np.reshape(n, n).copy()
    d_mlx = np.array(holder["c"])

    # Accelerate
    c32 = np.empty((n, n), dtype=np.float32)
    t_s, t_s_min = bench(
        lambda: lib.e1_sgemm(n, n, n, 1.0, a, n, b, n, 0.0, c32, n),
        reps=5, warm=1 if n <= 4096 else 0)
    a64 = a.astype(np.float64)
    b64 = b.astype(np.float64)
    c64 = np.empty((n, n), dtype=np.float64)
    t_d, t_d_min = bench(
        lambda: lib.e1_dgemm(n, n, n, 1.0, a64, n, b64, n, 0.0, c64, n),
        reps=5, warm=1 if n <= 2048 else 0)

    square[str(n)] = {
        "ours_ms": t_ours * 1e3, "ours_gflops": gflops(n, t_ours),
        "ours_best_gflops": gflops(n, t_ours_min),
        "ours_gpu_ms_median": float(np.median(gpu_ts)) * 1e3,
        "mlx_ms": t_mlx * 1e3, "mlx_gflops": gflops(n, t_mlx),
        "mlx_best_gflops": gflops(n, t_mlx_min),
        "ratio_ours_over_mlx": t_mlx / t_ours,
        "ratio_best_ours_over_best_mlx": t_mlx_min / t_ours_min,
        "sgemm_ms": t_s * 1e3, "sgemm_gflops": gflops(n, t_s),
        "dgemm_ms": t_d * 1e3, "dgemm_gflops": gflops(n, t_d),
        "relerr_ours_vs_f64": relerr(d_ours, c64),
        "relerr_mlx_vs_f64": relerr(d_mlx, c64),
        "relerr_sgemm_vs_f64": relerr(c32, c64),
        "ours_bitwise_equal_mlx_gpu": bool(np.array_equal(bits(d_ours), bits(d_mlx))),
        "kernel": kernel, "grid": [disp[0], disp[1], 1],
        "threads_per_group": [disp[2], disp[3], disp[4]],
    }
    s = square[str(n)]
    print(f"n={n:5d}  ours {s['ours_gflops']:8.0f}  mlx {s['mlx_gflops']:8.0f}  "
          f"sgemm {s['sgemm_gflops']:7.0f}  dgemm {s['dgemm_gflops']:6.0f} GFLOPS | "
          f"ours/mlx {s['ratio_ours_over_mlx']:.2f} | err {s['relerr_ours_vs_f64']:.2e} "
          f"| bit-equal to MLX GPU: {s['ours_bitwise_equal_mlx_gpu']}", flush=True)
    A.free(); B.free(); D.free()
    del a64, b64, c64, c32, d_ours, d_mlx, ma, mb, holder
res["square"] = square

tiny_batch("after_square")

# --------------------------------------------------------------------------
# C. sub-matrix views + in-place C <- alpha*A*B + beta*C in ONE shared buffer
# --------------------------------------------------------------------------
views = []
R, LD = 1500, 1536                       # one "big matrix", row-major, ld > cols
W = e1lib.GpuBuf(lib, R * LD)
Wm = W.np.reshape(R, LD)


def view_case(name, M, N, K, r_a, c_a, r_b, c_b, r_c, c_c, alpha, beta, ta=0, tb=0, variant=0):
    Wm[:] = rng.standard_normal((R, LD), dtype=np.float32)
    before = Wm.copy()
    sa = (K, M) if ta else (M, K)
    sb = (N, K) if tb else (K, N)
    Av = before[r_a:r_a + sa[0], c_a:c_a + sa[1]].astype(np.float64)
    Bv = before[r_b:r_b + sb[0], c_b:c_b + sb[1]].astype(np.float64)
    Cv = before[r_c:r_c + M, c_c:c_c + N].astype(np.float64)
    opA = Av.T if ta else Av
    opB = Bv.T if tb else Bv
    ref = alpha * (opA @ opB) + beta * Cv
    g = e1lib.args(W, r_a * LD + c_a, LD, W, r_b * LD + c_b, LD,
                   W, r_c * LD + c_c, LD, M, N, K,
                   Cb=W, c_off=r_c * LD + c_c, ldc=LD, alpha=alpha, beta=beta,
                   ta=ta, tb=tb, variant=variant)
    rc = lib.msg_gemm(C.byref(g), None)
    got = Wm[r_c:r_c + M, c_c:c_c + N]
    scale = np.abs(alpha) * (np.abs(opA) @ np.abs(opB)).max() + np.abs(beta) * np.abs(Cv).max()
    err = float(np.abs(got - ref).max() / scale)
    after = Wm.copy()
    after[r_c:r_c + M, c_c:c_c + N] = before[r_c:r_c + M, c_c:c_c + N]
    untouched = bool(np.array_equal(bits(after), bits(before)))
    views.append({"case": name, "M": M, "N": N, "K": K, "alpha": alpha, "beta": beta,
                  "rc": rc, "relerr": err, "outside_untouched": untouched,
                  "kernel": lib.msg_last_kernel().decode()})


# LU-like layout: A = block left of C, B = block above C, C = trailing block.
view_case("lu_trailing_aligned_axpby", 1024, 1024, 256, 300, 3, 40, 261, 300, 261, -1.0, 1.0)
view_case("lu_trailing_unaligned_axpby", 1001, 1111, 131, 317, 5, 150, 139, 317, 139, -1.0, 1.0)
view_case("add_path_alpha1_beta1", 1001, 1111, 131, 317, 5, 150, 139, 317, 139, 1.0, 1.0)
view_case("general_axpby", 700, 900, 77, 500, 1, 100, 200, 500, 200, 0.5, -2.0)
view_case("small_variant_bm64_bn32_bk32", 333, 517, 200, 700, 7, 10, 300, 700, 300, -1.0, 1.0)
view_case("variant4_bm32_bn32_bk16", 333, 517, 200, 700, 7, 10, 300, 700, 300, -1.0, 1.0, variant=4)
view_case("nt_axpby (C -= A*B^T)", 640, 512, 96, 700, 3, 10, 5, 700, 400, -1.0, 1.0, tb=1)
view_case("tn_axpby (C -= A^T*B)", 640, 512, 96, 10, 3, 200, 700, 800, 5, -1.0, 1.0, ta=1)
view_case("tt_axpby", 300, 200, 150, 10, 3, 200, 700, 900, 5, -1.0, 1.0, ta=1, tb=1)

# column-major trailing update through the transpose identity, same memory:
#   column-major  A22 <- A22 - L21 * U12   (A22: m x q, L21: m x k, U12: k x q)
#   row-major     A22^T <- A22^T - U12^T * L21^T  with ld = lda
m_, q_, k_, j_ = 900, 1100, 128, 200     # panel starts at (j_, j_)
lda = LD                                  # column-major leading dim (rows) = 1536
Wf = W.np.reshape(R, LD)                  # memory = columns of length LD
Wf[:] = rng.standard_normal((R, LD), dtype=np.float32)
cm = Wf.T                                 # cm[i, j] = element (row i, col j) column-major
before = cm.copy()
L21 = before[j_ + k_:j_ + k_ + m_, j_:j_ + k_].astype(np.float64)
U12 = before[j_:j_ + k_, j_ + k_:j_ + k_ + q_].astype(np.float64)
A22 = before[j_ + k_:j_ + k_ + m_, j_ + k_:j_ + k_ + q_].astype(np.float64)
ref = A22 - L21 @ U12
off = lambda i, j: j * lda + i            # column-major element offset
g = e1lib.args(W, off(j_, j_ + k_), lda,          # kernel A = U12^T  (q x k)
               W, off(j_ + k_, j_), lda,          # kernel B = L21^T  (k x m)
               W, off(j_ + k_, j_ + k_), lda,     # kernel D = A22^T  (q x m)
               q_, m_, k_, Cb=W, c_off=off(j_ + k_, j_ + k_), ldc=lda,
               alpha=-1.0, beta=1.0)
rc = lib.msg_gemm(C.byref(g), None)
got = cm[j_ + k_:j_ + k_ + m_, j_ + k_:j_ + k_ + q_]
after = cm.copy()
after[j_ + k_:j_ + k_ + m_, j_ + k_:j_ + k_ + q_] = before[j_ + k_:j_ + k_ + m_, j_ + k_:j_ + k_ + q_]
views.append({"case": "colmajor_trailing_update_via_transpose_identity", "M": q_, "N": m_, "K": k_,
              "alpha": -1.0, "beta": 1.0, "rc": rc,
              "relerr": float(np.abs(got - ref).max() / ((np.abs(L21) @ np.abs(U12)).max() + np.abs(A22).max())),
              "outside_untouched": bool(np.array_equal(bits(after), bits(before))),
              "kernel": lib.msg_last_kernel().decode()})
W.free()
res["views_inplace"] = views
res["views_all_ok"] = all(v["rc"] == 0 and v["relerr"] < 2e-6 and v["outside_untouched"] for v in views)
for v in views:
    print(f"view {v['case']:48s} M={v['M']:5d} N={v['N']:5d} K={v['K']:4d} "
          f"err {v['relerr']:.2e} outside_untouched={v['outside_untouched']}", flush=True)

# --------------------------------------------------------------------------
# D. in-place rank-K trailing-update rate (what a blocked LU dispatches)
# --------------------------------------------------------------------------
rank = {}
n = 8192
W = e1lib.GpuBuf(lib, n * n)
W.np[:] = rng.standard_normal(n * n, dtype=np.float32) * 1e-3
for (m, K) in ((8192 - 1024, 64), (8192 - 1024, 128), (8192 - 1024, 256), (8192 - 1024, 512),
               (8192 - 1024, 1024), (4096, 256), (2048, 256), (1024, 128)):
    j0 = 0
    g = e1lib.args(W, (j0 + K) * n + j0, n, W, j0 * n + (j0 + K), n,
                   W, (j0 + K) * n + (j0 + K), n, m, m, K,
                   Cb=W, c_off=(j0 + K) * n + (j0 + K), ldc=n, alpha=-1.0, beta=1.0)
    tt, _ = bench(lambda: lib.msg_gemm(C.byref(g), None), reps=7, warm=2)
    rank[f"m{m}_k{K}"] = {"ms": tt * 1e3, "gflops": 2.0 * m * m * K / tt / 1e9}
    print(f"rank-K in-place update m={m:5d} K={K:4d}: {tt*1e3:7.2f} ms  "
          f"{rank[f'm{m}_k{K}']['gflops']:8.0f} GFLOPS", flush=True)
W.free()
res["rank_k_update"] = rank

# --------------------------------------------------------------------------
# E. the "MLX GPU matmul is bit-identical to Accelerate sgemm" oddity
# --------------------------------------------------------------------------
odd = {}
from fractions import Fraction  # noqa: E402


def round_f32(fr):
    """Round an exact rational to float32, round-to-nearest-even."""
    f = np.float32(float(fr))
    cands = [f, np.nextafter(f, np.float32(np.inf)), np.nextafter(f, np.float32(-np.inf))]
    return min(cands, key=lambda c: (abs(Fraction(float(c)) - fr), int(np.float32(c).view(np.uint32)) & 1))


def fma_chain(a, b, i, j, order):
    """acc = fma(a[i,k], b[k,j], acc) over k in `order`, exactly rounded float32."""
    acc = np.float32(0)
    for k in order:
        acc = round_f32(Fraction(float(a[i, k])) * Fraction(float(b[k, j])) + Fraction(float(acc)))
    return acc


normal = lambda m, n: rng.standard_normal((m, n), dtype=np.float32)
uniform = lambda m, n: rng.random((m, n), dtype=np.float32)
for label, (M, N, K), gen in (
        ("normal_1024_K_aligned", (1024, 1024, 1024), normal),
        ("uniform_1024_K_aligned", (1024, 1024, 1024), uniform),
        ("normal_2048_K_aligned", (2048, 2048, 2048), normal),
        ("normal_M1100_N1000_K1024_aligned", (1100, 1000, 1024), normal),
        ("normal_1000_K_unaligned", (1000, 1000, 1000), normal),
        ("uniform_1000_K_unaligned", (1000, 1000, 1000), uniform),
        ("normal_M1100_N1040_K1001_unaligned", (1100, 1040, 1001), normal),
        ("normal_512_mlx_splitk_regime", (512, 512, 512), normal),
        ("small_integers_1000", (1000, 1000, 1000),
         lambda m, n: rng.integers(-4, 5, (m, n)).astype(np.float32)),
):
    a, b = gen(M, K), gen(K, N)
    ma, mb = mx.array(a), mx.array(b)
    mx.eval(ma, mb)
    g_mlx = mx.matmul(ma, mb, stream=mx.gpu); mx.eval(g_mlx)
    c_mlx = mx.matmul(ma, mb, stream=mx.cpu); mx.eval(c_mlx)
    g_mlx, c_mlx = np.array(g_mlx), np.array(c_mlx)
    acc = np.empty((M, N), dtype=np.float32)
    lib.e1_sgemm(M, N, K, 1.0, a, K, b, N, 0.0, acc, N)
    A = e1lib.GpuBuf(lib, M * K); B = e1lib.GpuBuf(lib, K * N); D = e1lib.GpuBuf(lib, M * N)
    A.np[:] = a.ravel(); B.np[:] = b.ravel()
    g = e1lib.args(A, 0, K, B, 0, N, D, 0, N, M, N, K)
    assert lib.msg_gemm(C.byref(g), None) == 0
    ours = D.np.reshape(M, N).copy()
    ours_kernel = lib.msg_last_kernel().decode()
    A.free(); B.free(); D.free()
    f1, u1 = bitdiff(g_mlx, acc)
    f2, u2 = bitdiff(c_mlx, acc)
    f3, u3 = bitdiff(g_mlx, ours)
    f5, u5 = bitdiff(g_mlx, c_mlx)
    entry = {
        "M": M, "N": N, "K": K,
        "mlx_gpu_vs_accelerate": {"frac_bits_differ": f1, "max_ulp": u1},
        "mlx_cpu_vs_accelerate": {"frac_bits_differ": f2, "max_ulp": u2},
        "mlx_gpu_vs_our_gpu_kernel": {"frac_bits_differ": f3, "max_ulp": u3},
        "mlx_gpu_vs_mlx_cpu": {"frac_bits_differ": f5, "max_ulp": u5},
        "our_kernel": ours_kernel,
    }
    # Mechanism check on a few sampled entries: an exactly-rounded sequential
    # FMA chain in k (k = 0..K-1), and the steel order for unaligned K
    # (the K % 16 tail block first, then k = 0..).
    if label in ("normal_1024_K_aligned", "normal_1000_K_unaligned"):
        pts = [(0, 0), (1, 7), (M // 2, N // 3), (M - 1, N - 1), (513, 77), (33, 900)]
        seq = list(range(K))
        kal = (K // 16) * 16
        steel = list(range(kal, K)) + list(range(kal))
        n_seq_acc = n_seq_gpu = n_steel_gpu = 0
        for (i, j) in pts:
            v_seq = fma_chain(a, b, i, j, seq)
            v_steel = fma_chain(a, b, i, j, steel)
            n_seq_acc += int(v_seq.view(np.uint32) == acc[i, j].view(np.uint32))
            n_seq_gpu += int(v_seq.view(np.uint32) == g_mlx[i, j].view(np.uint32))
            n_steel_gpu += int(v_steel.view(np.uint32) == g_mlx[i, j].view(np.uint32))
        entry["fma_chain_check"] = {
            "entries_sampled": len(pts),
            "sequential_k_chain_equals_accelerate": n_seq_acc,
            "sequential_k_chain_equals_mlx_gpu": n_seq_gpu,
            "steel_order_chain_equals_mlx_gpu": n_steel_gpu,
        }
    odd[label] = entry
    print(f"oddity {label:36s} MLX-GPU vs Accelerate: {f1*100:6.2f}% differ (max {u1} ulp) | "
          f"MLX-CPU vs Accelerate: {f2*100:6.2f}% | MLX-GPU vs our GPU kernel: {f3*100:6.2f}%"
          + (f" | fma-chain {entry['fma_chain_check']}" if "fma_chain_check" in entry else ""), flush=True)

# timing signature: the same product on MLX's GPU stream vs its CPU stream
n = 4096
a = rng.standard_normal((n, n), dtype=np.float32)
b = rng.standard_normal((n, n), dtype=np.float32)
ma, mb = mx.array(a), mx.array(b)
mx.eval(ma, mb)


def f_g():
    mx.eval(mx.matmul(ma, mb, stream=mx.gpu))


def f_c():
    mx.eval(mx.matmul(ma, mb, stream=mx.cpu))


tg, _ = bench(f_g, reps=7, warm=3)
tc, _ = bench(f_c, reps=5, warm=1)
c32 = np.empty((n, n), dtype=np.float32)
ts_, _ = bench(lambda: lib.e1_sgemm(n, n, n, 1.0, a, n, b, n, 0.0, c32, n), reps=5, warm=1)
odd["timing_4096"] = {"mlx_gpu_stream_ms": tg * 1e3, "mlx_cpu_stream_ms": tc * 1e3,
                      "accelerate_sgemm_ms": ts_ * 1e3,
                      "mlx_gpu_gflops": gflops(n, tg), "mlx_cpu_gflops": gflops(n, tc),
                      "sgemm_gflops": gflops(n, ts_)}
print(f"timing n=4096: MLX gpu stream {tg*1e3:.1f} ms, MLX cpu stream {tc*1e3:.1f} ms, "
      f"Accelerate sgemm {ts_*1e3:.1f} ms", flush=True)
al = [k for k in odd if k.endswith("_aligned")]
un = [k for k in odd if k.endswith("_unaligned")]
odd["verdict"] = {
    "mlx_gpu_bit_identical_to_accelerate_when_K_multiple_of_16": all(
        odd[k]["mlx_gpu_vs_accelerate"]["frac_bits_differ"] == 0 for k in al),
    "mlx_gpu_bit_identical_to_accelerate_when_K_not_multiple_of_16": all(
        odd[k]["mlx_gpu_vs_accelerate"]["frac_bits_differ"] == 0 for k in un),
    "mlx_cpu_stream_bit_identical_to_accelerate_always": all(
        odd[k]["mlx_cpu_vs_accelerate"]["frac_bits_differ"] == 0 for k in al + un),
    "mlx_gpu_bit_identical_to_our_gpu_kernel_in_fused_regime": all(
        odd[k]["mlx_gpu_vs_our_gpu_kernel"]["frac_bits_differ"] == 0 for k in al + un),
    "mlx_gpu_stream_faster_than_accelerate_at_4096": odd["timing_4096"]["mlx_gpu_gflops"]
    / odd["timing_4096"]["sgemm_gflops"],
}
res["oddity"] = odd

tiny_batch("end")
tiny = tiny_finish()
res["tiny_call_64"] = tiny

# --------------------------------------------------------------------------
# gates
# --------------------------------------------------------------------------
ratios = {k: square[k]["ratio_ours_over_mlx"] for k in ("2048", "4096", "8192")}
res["gates"] = {
    "rate_ratio_ours_over_mlx": ratios,
    "rate_gate_ge_0.80": bool(all(r >= 0.80 for r in ratios.values())),
    "tiny_call_ms": tiny["total_ms_median"],
    "tiny_call_best_batch_ms": tiny["best_batch_median_ms"],
    "tiny_gate_le_0.3ms": bool(tiny["total_ms_median"] <= 0.3),
    "max_relerr_vs_f64": max(square[k]["relerr_ours_vs_f64"] for k in square),
}
print("RESULT " + json.dumps(res))
