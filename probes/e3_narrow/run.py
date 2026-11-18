#!/usr/bin/env python
"""E3 probe: float64 <-> float32 conversion cost, CPU (vDSP, 16 P-cores) vs GPU (integer-only MSL).

    ./build.sh && ../../.venv/bin/python run.py            # ~25 s
    ../../.venv/bin/python run.py --exhaustive             # + all 2^32 float32 patterns for widening (~15-25 s more)

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

HERE = os.path.dirname(os.path.abspath(__file__))
u64 = ctypes.c_uint64
u32 = ctypes.c_uint32
vp = ctypes.c_void_p
dbl_p = ctypes.POINTER(ctypes.c_double)
PAGE = 16384
NARROW, WIDEN, NARROW_F, WIDEN_F = 0, 1, 2, 3


def load():
    L = ctypes.CDLL(os.path.join(HERE, "build", "libe3.dylib"))
    L.e3_init.argtypes = [ctypes.c_char_p]
    L.e3_last_error.restype = ctypes.c_char_p
    L.e3_cpu_narrow.argtypes = [vp, vp, u64, ctypes.c_int, u64]
    L.e3_cpu_narrow.restype = None
    L.e3_cpu_widen.argtypes = [vp, vp, u64, ctypes.c_int, u64]
    L.e3_cpu_widen.restype = None
    L.e3_time_cpu.argtypes = [ctypes.c_int, vp, vp, u64, ctypes.c_int, u64, ctypes.c_int, dbl_p]
    L.e3_time_cpu.restype = None
    L.e3_wrap.argtypes = [u64, u64]
    L.e3_wrap.restype = vp
    L.e3_alloc.argtypes = [u64]
    L.e3_alloc.restype = vp
    L.e3_contents.argtypes = [vp]
    L.e3_contents.restype = u64
    L.e3_release.argtypes = [vp]
    L.e3_release.restype = None
    L.e3_gpu_convert.argtypes = [ctypes.c_int, vp, u64, vp, u64, u64, u32, u32, u32, dbl_p]
    L.e3_time_gpu.argtypes = [ctypes.c_int, vp, vp, u64, u32, u32, u32, ctypes.c_int, dbl_p]
    L.e3_time_gpu_cold.argtypes = [ctypes.c_int, u64, u64, u64, u64, u64, u32, u32, u32, ctypes.c_int, dbl_p]
    rc = L.e3_init(os.path.join(HERE, "build", "kernels.metallib").encode())
    if rc != 0:
        raise RuntimeError(f"e3_init failed: {rc} {L.e3_last_error().decode()}")
    return L


L = None
# Best GPU dispatch shape found by the sweep below (elements per thread, threadgroup w x h).
GPU_CFG = {"chunk": 1, "tg_w": 0, "tg_h": 1}


def ptr(a):
    return a.__array_interface__["data"][0]


def roundup(n):
    return (n + PAGE - 1) // PAGE * PAGE


def page_array(n, dtype):
    """NumPy array whose data pointer is page aligned (true for any large malloc on macOS; asserted)."""
    a = np.empty(max(n, 1), dtype=dtype)
    if ptr(a) % PAGE:
        raw = np.empty(n * np.dtype(dtype).itemsize + PAGE, dtype=np.uint8)
        off = (-ptr(raw)) % PAGE
        a = raw[off:off + n * np.dtype(dtype).itemsize].view(dtype)
    assert ptr(a) % PAGE == 0
    return a[:n]


def gpu_convert(which, src, dst, chunk=None, tg=None):
    """src/dst NumPy arrays (page-aligned pointers); wraps both no-copy, converts on the GPU, releases."""
    cfg = dict(GPU_CFG)
    if chunk is not None:
        cfg["chunk"] = chunk
    if tg is not None:
        cfg["tg_w"], cfg["tg_h"] = tg
    hs = L.e3_wrap(ptr(src), roundup(src.nbytes))
    hd = L.e3_wrap(ptr(dst), roundup(dst.nbytes))
    assert hs and hd, "wrap failed"
    rc = L.e3_gpu_convert(which, hs, 0, hd, 0, src.size, cfg["chunk"], cfg["tg_w"], cfg["tg_h"], None)
    L.e3_release(hs)
    L.e3_release(hd)
    if rc != 0:
        raise RuntimeError(L.e3_last_error().decode())


# ----------------------------------------------------------------------------------------
# adversarial inputs
# ----------------------------------------------------------------------------------------

def adversarial_f64(rng):
    """dict name -> float64 array (as uint64 bit patterns)."""
    sets = {}
    sets["random_bits"] = rng.integers(0, 2**64, size=1 << 22, dtype=np.uint64)
    fmax, fmin, ftrue = float(np.finfo(np.float32).max), float(np.finfo(np.float32).tiny), 2.0 ** -149
    sp = [0.0, -0.0, np.inf, -np.inf, 5e-324, -5e-324, 2.2250738585072014e-308, 2.225073858507201e-308,
          1.7976931348623157e308, -1.7976931348623157e308, 1e39, -1e39, 3.5e38, -3.5e38,
          fmax, -fmax, np.nextafter(fmax, np.inf), np.nextafter(fmax, 0),
          fmax + 2.0 ** 102, np.nextafter(fmax + 2.0 ** 102, 0), np.nextafter(fmax + 2.0 ** 102, np.inf),  # tie to inf
          -(fmax + 2.0 ** 102), 2.0 ** 128, np.nextafter(2.0 ** 128, 0),
          fmin, -fmin, np.nextafter(fmin, 0), np.nextafter(fmin, 1), fmin - 2.0 ** -150, fmin - 2.0 ** -151,
          ftrue, -ftrue, ftrue / 2, -ftrue / 2, np.nextafter(ftrue / 2, 1), np.nextafter(ftrue / 2, 0), ftrue / 4,
          1.5 * ftrue, 2.5 * ftrue, 3.5 * ftrue, np.nextafter(1.5 * ftrue, 0), np.nextafter(1.5 * ftrue, 1),
          np.nextafter(2.5 * ftrue, 0), np.nextafter(2.5 * ftrue, 1), 1.0, -1.0, 1.0 + 2.0 ** -24, 1.0 + 2.0 ** -23,
          1.0 + 3 * 2.0 ** -24, np.nextafter(1.0 + 2.0 ** -24, 2), np.nextafter(1.0 + 2.0 ** -24, 0), 0.1, 1 / 3, np.pi]
    sets["specials"] = np.array(sp, dtype=np.float64).view(np.uint64)
    # NaNs: every sign / quiet-bit combination with structured and random payloads
    frac = np.concatenate([
        np.array([1, 2, 0x10000000, 0x1FFFFFFF, 0x20000000, 0x3FFFFFFF, 0x0008000000000000, 0x0004000000000000,
                  0x0007FFFFFFFFFFFF, 0x000FFFFFFFFFFFFF, 0x0008000000000001, 0x00000000E0000000,
                  0x000FFFFFE0000000, 0x0000000020000000], dtype=np.uint64),
        np.uint64(1) << np.arange(52, dtype=np.uint64),
        rng.integers(1, 2**52, size=1 << 16, dtype=np.uint64)])
    nan_pos = np.uint64(0x7FF0000000000000) | frac
    sets["nan_payloads"] = np.concatenate([nan_pos, nan_pos | np.uint64(1 << 63)])
    # round-to-nearest-even ties: exact midpoints between adjacent float32 values, and +-1 double ulp
    f = rng.integers(0, 2**32, size=1 << 20, dtype=np.uint32)
    f = np.concatenate([f, np.arange(0, 1 << 12, dtype=np.uint32),                       # smallest subnormals
                        np.uint32(0x00800000) + np.arange(-2048, 2048).astype(np.uint32),  # around FLT_MIN
                        np.uint32(0x7F7FFFFF) - np.arange(0, 2048, dtype=np.uint32)])      # just below FLT_MAX
    a = f.view(np.float32)
    with np.errstate(all="ignore"):
        b = np.nextafter(a, np.float32(np.inf))
        ok = np.isfinite(a) & np.isfinite(b)
        mid = (a[ok].astype(np.float64) + b[ok].astype(np.float64)) / 2
    sets["ties_exact"] = mid.view(np.uint64).copy()
    sets["ties_minus_1ulp"] = np.nextafter(mid, -np.inf).view(np.uint64).copy()
    sets["ties_plus_1ulp"] = np.nextafter(mid, np.inf).view(np.uint64).copy()
    sets["exact_float32"] = a[np.isfinite(a)].astype(np.float64).view(np.uint64).copy()
    # result is subnormal / underflows: unbiased exponents -160 .. -120
    e = rng.integers(1023 - 160, 1023 - 120, size=1 << 20, dtype=np.uint64)
    m = rng.integers(0, 2**52, size=1 << 20, dtype=np.uint64)
    s = rng.integers(0, 2, size=1 << 20, dtype=np.uint64) << np.uint64(63)
    sets["subnormal_range"] = s | (e << np.uint64(52)) | m
    # a few coarse mantissas so exact subnormal ties are hit too
    m2 = rng.integers(0, 2**10, size=1 << 18, dtype=np.uint64) << np.uint64(42)
    sets["subnormal_ties_coarse"] = (rng.integers(1023 - 152, 1023 - 124, size=1 << 18, dtype=np.uint64) << np.uint64(52)) | m2
    # overflow boundary: unbiased exponents 126 .. 129
    e = rng.integers(1023 + 126, 1023 + 130, size=1 << 18, dtype=np.uint64)
    sets["overflow_range"] = (rng.integers(0, 2, size=1 << 18, dtype=np.uint64) << np.uint64(63)) | (e << np.uint64(52)) | \
        rng.integers(0, 2**52, size=1 << 18, dtype=np.uint64)
    # float64 subnormals and tiny normals
    sets["f64_subnormals"] = rng.integers(0, 2**53, size=1 << 16, dtype=np.uint64)
    return sets


def adversarial_f32(rng):
    sets = {}
    sets["random_bits"] = rng.integers(0, 2**32, size=1 << 24, dtype=np.uint32)
    e = np.arange(256, dtype=np.uint32)
    m = np.concatenate([np.array([0, 1, 2, 3, 0x400000, 0x400001, 0x3FFFFF, 0x7FFFFF, 0x200000, 0x555555, 0x2AAAAA],
                                 dtype=np.uint32), np.uint32(1) << np.arange(23, dtype=np.uint32)])
    grid = ((e[:, None] << np.uint32(23)) | m[None, :]).ravel()
    sets["all_exponents_x_mantissas"] = np.concatenate([grid, grid | np.uint32(0x80000000)])
    sets["all_subnormals_sampled"] = np.concatenate([np.arange(0, 1 << 16, dtype=np.uint32),
                                                     rng.integers(0, 1 << 23, size=1 << 18, dtype=np.uint32),
                                                     np.uint32(0x80000000) | rng.integers(0, 1 << 23, size=1 << 16, dtype=np.uint32)])
    nan = np.uint32(0x7F800000) | rng.integers(1, 1 << 23, size=1 << 16, dtype=np.uint32)
    sets["nan_payloads"] = np.concatenate([nan, nan | np.uint32(0x80000000),
                                           np.array([0x7F800001, 0x7FC00000, 0xFFC00000, 0x7FBFFFFF, 0xFF800001, 0x7FFFFFFF],
                                                    dtype=np.uint32)])
    return sets


def classify_narrow(xbits):
    """Coarse class of each float64 input, for mismatch reporting."""
    e = (xbits >> np.uint64(52)) & np.uint64(0x7FF)
    fr = xbits & np.uint64(0x000FFFFFFFFFFFFF)
    cls = np.full(xbits.shape, "normal", dtype=object)
    cls[(e >= 1023 + 128)] = "overflow_to_inf"
    cls[(e < 1023 - 126)] = "subnormal_or_zero_result"
    cls[(e == 0x7FF) & (fr == 0)] = "inf"
    cls[(e == 0x7FF) & (fr != 0)] = "nan"
    return cls


def classify_widen(xbits):
    e = (xbits >> np.uint32(23)) & np.uint32(0xFF)
    fr = xbits & np.uint32(0x7FFFFF)
    cls = np.full(xbits.shape, "normal", dtype=object)
    cls[(e == 0) & (fr != 0)] = "subnormal"
    cls[(e == 0) & (fr == 0)] = "zero"
    cls[(e == 0xFF) & (fr == 0)] = "inf"
    cls[(e == 0xFF) & (fr != 0)] = "nan"
    return cls


def mismatch_report(got_bits, ref_bits, in_bits, classify):
    bad = got_bits != ref_bits
    nbad = int(bad.sum())
    rep = {"n": int(got_bits.size), "mismatches": nbad}
    if nbad:
        cls = classify(in_bits[bad])
        kinds, counts = np.unique(cls, return_counts=True)
        rep["by_class"] = {str(k): int(c) for k, c in zip(kinds, counts)}
        i = int(np.flatnonzero(bad)[0])
        rep["first"] = {"in": hex(int(in_bits[i])), "got": hex(int(got_bits[i])), "numpy": hex(int(ref_bits[i]))}
    return rep


def validate(rng, exhaustive):
    out = {"narrow": {}, "widen": {}}
    ok_all = {"cpu_vdsp_narrow": True, "gpu_narrow": True, "cpu_vdsp_widen": True, "gpu_widen": True}
    through = {"gpu_narrow_through_float": {}, "gpu_widen_from_float": {}}
    for name, bits in adversarial_f64(rng).items():
        n = bits.size
        src = page_array(n, np.float64)
        src.view(np.uint64)[:] = bits
        with np.errstate(all="ignore"):
            ref = src.astype(np.float32).view(np.uint32)
        row = {}
        dst = page_array(n, np.float32)
        dst.view(np.uint32)[:] = 0xDEADBEEF
        L.e3_cpu_narrow(ptr(src), ptr(dst), n, 16, 4096)
        row["cpu_vdsp"] = mismatch_report(dst.view(np.uint32), ref, bits, classify_narrow)
        for chunk in (1, 16):  # both dispatch shapes must be exact
            dst.view(np.uint32)[:] = 0xDEADBEEF
            gpu_convert(NARROW, src, dst, chunk=chunk)
            row["gpu" if chunk == 1 else "gpu_chunk16"] = mismatch_report(dst.view(np.uint32), ref, bits, classify_narrow)
        dst.view(np.uint32)[:] = 0xDEADBEEF
        gpu_convert(NARROW_F, src, dst)
        through["gpu_narrow_through_float"][name] = mismatch_report(dst.view(np.uint32), ref, bits, classify_narrow)
        ok_all["cpu_vdsp_narrow"] &= row["cpu_vdsp"]["mismatches"] == 0
        ok_all["gpu_narrow"] &= row["gpu"]["mismatches"] == 0 and row["gpu_chunk16"]["mismatches"] == 0
        out["narrow"][name] = row
    for name, bits in adversarial_f32(rng).items():
        n = bits.size
        src = page_array(n, np.float32)
        src.view(np.uint32)[:] = bits
        with np.errstate(all="ignore"):
            ref = src.astype(np.float64).view(np.uint64)
        row = {}
        dst = page_array(n, np.float64)
        dst.view(np.uint64)[:] = 0xDEADBEEFDEADBEEF
        L.e3_cpu_widen(ptr(src), ptr(dst), n, 16, 4096)
        row["cpu_vdsp"] = mismatch_report(dst.view(np.uint64), ref, bits, classify_widen)
        for chunk in (1, 16):
            dst.view(np.uint64)[:] = 0xDEADBEEFDEADBEEF
            gpu_convert(WIDEN, src, dst, chunk=chunk)
            row["gpu" if chunk == 1 else "gpu_chunk16"] = mismatch_report(dst.view(np.uint64), ref, bits, classify_widen)
        dst.view(np.uint64)[:] = 0xDEADBEEFDEADBEEF
        gpu_convert(WIDEN_F, src, dst)
        through["gpu_widen_from_float"][name] = mismatch_report(dst.view(np.uint64), ref, bits, classify_widen)
        ok_all["cpu_vdsp_widen"] &= row["cpu_vdsp"]["mismatches"] == 0
        ok_all["gpu_widen"] &= row["gpu"]["mismatches"] == 0 and row["gpu_chunk16"]["mismatches"] == 0
        out["widen"][name] = row
    if exhaustive:
        # every one of the 2^32 float32 bit patterns, widening, GPU integer kernel vs numpy.astype
        t0 = time.perf_counter()
        blk = 1 << 27
        src = page_array(blk, np.float32)
        dst = page_array(blk, np.float64)
        bad = 0
        for k in range(32):
            src.view(np.uint32)[:] = np.arange(k * blk, (k + 1) * blk, dtype=np.uint64).astype(np.uint32)
            gpu_convert(WIDEN, src, dst)
            with np.errstate(all="ignore"):
                bad += int((dst.view(np.uint64) != src.astype(np.float64).view(np.uint64)).sum())
        out["widen_exhaustive_2^32"] = {"mismatches": bad, "seconds": round(time.perf_counter() - t0, 1)}
        ok_all["gpu_widen"] &= bad == 0
    out["bit_exact_vs_numpy_astype"] = ok_all
    out["through_float_datapath"] = through
    out["float_datapath_summary"] = float_datapath_summary(rng)
    return out


def float_datapath_summary(rng):
    """What the GPU's float datapath (one `x * 1.0f`) does to values the integer kernels preserve."""
    def run_narrow(bits):
        src = page_array(bits.size, np.float64)
        src.view(np.uint64)[:] = bits
        dst = page_array(bits.size, np.float32)
        gpu_convert(NARROW_F, src, dst)
        return dst.view(np.uint32).copy()

    nan = np.uint64(0x7FF0000000000000) | rng.integers(1, 2**52, size=1 << 16, dtype=np.uint64)
    nan = np.concatenate([nan, nan | np.uint64(1 << 63)])
    sub = (rng.integers(1023 - 149, 1023 - 127, size=1 << 16, dtype=np.uint64) << np.uint64(52)) | \
        rng.integers(0, 2**52, size=1 << 16, dtype=np.uint64)
    sub = np.concatenate([sub, sub | np.uint64(1 << 63)])
    norm = rng.standard_normal(1 << 16).view(np.uint64)
    with np.errstate(all="ignore"):
        norm_ok = bool(np.array_equal(run_narrow(norm), norm.view(np.float64).astype(np.float32).view(np.uint32)))
    inf = np.array([np.inf, -np.inf, 0.0, -0.0, 1e39, -1e39], dtype=np.float64).view(np.uint64)
    return {"nan_inputs_distinct_outputs": [hex(int(v)) for v in np.unique(run_narrow(nan))][:8],
            "subnormal_result_inputs_distinct_outputs": [hex(int(v)) for v in np.unique(run_narrow(sub))][:8],
            "inf_zero_overflow_outputs": [hex(int(v)) for v in run_narrow(inf)],
            "normal_values_bit_exact": norm_ok}


# ----------------------------------------------------------------------------------------
# timing
# ----------------------------------------------------------------------------------------

def med(buf, n):
    return statistics.median(buf[i] for i in range(n))


def fill_random(a, rng):
    """Fill a large float64 array quickly: tile a 4 MiB random block (values spread over float32's range)."""
    blk = 1 << 19
    block = rng.standard_normal(blk) * np.exp(rng.uniform(-20, 20, blk))
    n = a.size // blk * blk
    a[:n].reshape(-1, blk)[:] = block
    a[n:] = block[:a.size - n]


def sweep(src64, dst32, dst64, n):
    """Dispatch-shape sweep on warm wrapped buffers; returns rows and the best config."""
    rows = []
    hs = L.e3_wrap(ptr(src64), roundup(src64.nbytes))
    hd = L.e3_wrap(ptr(dst32), roundup(dst32.nbytes))
    hw = L.e3_wrap(ptr(dst64), roundup(dst64.nbytes))
    t = (ctypes.c_double * 8)()
    best = None
    for chunk in (1, 4, 16, 64):
        for tg in ((32, 1), (256, 1), (1024, 1), (32, 32)):
            L.e3_time_gpu(NARROW, hs, hd, n, chunk, tg[0], tg[1], 1, t)
            L.e3_time_gpu(NARROW, hs, hd, n, chunk, tg[0], tg[1], 5, t)
            tn = med(t, 5)
            L.e3_time_gpu(WIDEN, hd, hw, n, chunk, tg[0], tg[1], 1, t)
            L.e3_time_gpu(WIDEN, hd, hw, n, chunk, tg[0], tg[1], 5, t)
            tw = med(t, 5)
            row = {"chunk": chunk, "tg": f"{tg[0]}x{tg[1]}", "narrow_GBps": round(n * 8 / 1e9 / tn, 1),
                   "widen_GBps": round(n * 8 / 1e9 / tw, 1)}
            rows.append(row)
            score = tn + tw
            if best is None or score < best[0]:
                best = (score, chunk, tg)
    for h in (hs, hd, hw):
        L.e3_release(h)
    return rows, {"chunk": best[1], "tg_w": best[2][0], "tg_h": best[2][1]}


def timing(rng):
    out = {}
    cores = L.e3_perf_cores()
    sizes = (("256MB", 256 << 20), ("1GB", 1 << 30), ("4GB", 4 << 30))
    nmax = sizes[-1][1] // 8
    src64_all = page_array(nmax, np.float64)
    fill_random(src64_all, rng)
    dst32_all = page_array(nmax, np.float32)
    dst32_all[:] = 0
    dst64_all = page_array(nmax, np.float64)
    dst64_all[:] = 0
    t = (ctypes.c_double * 16)()

    n0 = sizes[0][1] // 8
    rows, best = sweep(src64_all[:n0], dst32_all[:n0], dst64_all[:n0], n0)
    GPU_CFG.update(best)
    out["gpu_dispatch_sweep_256MB"] = rows
    out["gpu_dispatch_best"] = dict(best)

    for label, nbytes in sizes:
        n = nbytes // 8
        src64, dst32, dst64 = src64_all[:n], dst32_all[:n], dst64_all[:n]
        reps = 7 if nbytes <= (1 << 30) else 5
        gbps = lambda sec: round(nbytes / 1e9 / sec, 2)
        row = {"float64_bytes": nbytes}

        # --- CPU vDSP, 16 workers pulling 4 MiB (of float64) chunks
        chunk_elems = 1 << 19
        for which, key, s, d in ((0, "cpu_vdsp_narrow", src64, dst32), (1, "cpu_vdsp_widen", dst32, dst64)):
            L.e3_time_cpu(which, ptr(s), ptr(d), n, cores, chunk_elems, 1, t)
            L.e3_time_cpu(which, ptr(s), ptr(d), n, cores, chunk_elems, reps, t)
            row[key + "_ms"] = round(med(t, reps) * 1e3, 3)
            row[key + "_GBps"] = gbps(med(t, reps))
        # one static chunk per worker, for comparison
        for which, key, s, d in ((0, "cpu_vdsp_narrow_static16", src64, dst32), (1, "cpu_vdsp_widen_static16", dst32, dst64)):
            L.e3_time_cpu(which, ptr(s), ptr(d), n, cores, 0, reps, t)
            row[key + "_GBps"] = gbps(med(t, reps))
        # plain C++ cast loops, same threading
        for which, key, s, d in ((2, "cpu_loop_narrow", src64, dst32), (3, "cpu_loop_widen", dst32, dst64)):
            L.e3_time_cpu(which, ptr(s), ptr(d), n, cores, chunk_elems, reps, t)
            row[key + "_GBps"] = gbps(med(t, reps))
        # single thread vDSP
        if nbytes <= (1 << 30):
            for which, key, s, d in ((0, "cpu_vdsp_narrow_1thread", src64, dst32), (1, "cpu_vdsp_widen_1thread", dst32, dst64)):
                L.e3_time_cpu(which, ptr(s), ptr(d), n, 1, 0, 3, t)
                row[key + "_GBps"] = gbps(med(t, 3))
        ref_sample = src64[::9973].astype(np.float32)
        row["cpu_sample_ok"] = bool(np.array_equal(dst32[::9973].view(np.uint32), ref_sample.view(np.uint32))
                                    and np.array_equal(dst64[::9973].view(np.uint64),
                                                       dst32[::9973].astype(np.float64).view(np.uint64)))

        # --- GPU integer kernels on warm (long-lived) no-copy wraps of the NumPy memory
        dst32[:] = 0
        dst64[:] = 0
        hs = L.e3_wrap(ptr(src64), roundup(src64.nbytes))
        hd = L.e3_wrap(ptr(dst32), roundup(dst32.nbytes))
        hw = L.e3_wrap(ptr(dst64), roundup(dst64.nbytes))
        c, tw, th = GPU_CFG["chunk"], GPU_CFG["tg_w"], GPU_CFG["tg_h"]
        for which, key, a, b in ((NARROW, "gpu_narrow", hs, hd), (WIDEN, "gpu_widen", hd, hw)):
            L.e3_time_gpu(which, a, b, n, c, tw, th, 2, t)
            rc = L.e3_time_gpu(which, a, b, n, c, tw, th, reps, t)
            assert rc == 0, L.e3_last_error()
            row[key + "_ms"] = round(med(t, reps) * 1e3, 3)
            row[key + "_GBps"] = gbps(med(t, reps))
        for h in (hs, hd, hw):
            L.e3_release(h)
        row["gpu_sample_ok"] = bool(np.array_equal(dst32[::9973].view(np.uint32), ref_sample.view(np.uint32))
                                    and np.array_equal(dst32[-4096:].view(np.uint32), src64[-4096:].astype(np.float32).view(np.uint32))
                                    and np.array_equal(dst64[::9973].view(np.uint64), dst32[::9973].astype(np.float64).view(np.uint64))
                                    and np.array_equal(dst64[-4096:].view(np.uint64), dst32[-4096:].astype(np.float64).view(np.uint64)))

        # --- GPU, cold: both NumPy arrays wrapped fresh for every call (wiring cost included)
        for which, key, s, d in ((NARROW, "gpu_narrow_cold_wraps", src64, dst32), (WIDEN, "gpu_widen_cold_wraps", dst32, dst64)):
            rc = L.e3_time_gpu_cold(which, ptr(s), roundup(s.nbytes), ptr(d), roundup(d.nbytes), n, c, tw, th, 5, t)
            assert rc == 0, L.e3_last_error()
            row[key + "_ms"] = round(med(t, 5) * 1e3, 3)
            row[key + "_GBps"] = gbps(med(t, 5))

        # --- numpy.astype for context (single thread, allocates its output)
        if nbytes <= (1 << 30):
            ts = []
            for _ in range(3):
                t0 = time.perf_counter()
                tmp = src64.astype(np.float32)
                ts.append(time.perf_counter() - t0)
            row["numpy_astype_narrow_GBps"] = gbps(statistics.median(ts))
            ts = []
            for _ in range(3):
                t0 = time.perf_counter()
                tmp2 = tmp.astype(np.float64)
                ts.append(time.perf_counter() - t0)
            row["numpy_astype_widen_GBps"] = gbps(statistics.median(ts))
            del tmp, tmp2
        out[label] = row
    del src64_all, dst32_all, dst64_all
    gc.collect()
    return out


def main():
    global L
    t_start = time.perf_counter()
    L = load()
    rng = np.random.default_rng(20261118)
    result = {"probe": "e3_narrow", "numpy": np.__version__, "perf_cores": L.e3_perf_cores()}
    result["validation"] = validate(rng, "--exhaustive" in sys.argv)
    v = result["validation"]
    print("bit-exact vs numpy.astype:", json.dumps(v["bit_exact_vs_numpy_astype"]))
    for d in ("gpu_narrow_through_float", "gpu_widen_from_float"):
        diffs = {k: r.get("by_class") for k, r in v["through_float_datapath"][d].items() if r["mismatches"]}
        print(f"{d}: classes that differ once a real float is involved: {json.dumps(diffs) if diffs else 'none'}")
    print("GPU float datapath (x * 1.0f):", json.dumps(v["float_datapath_summary"]))
    if "widen_exhaustive_2^32" in v:
        print("widen, all 2^32 float32 bit patterns, GPU integer kernel vs numpy.astype:", json.dumps(v["widen_exhaustive_2^32"]))
    result["timing"] = timing(rng)
    tm = result["timing"]
    print("\nGPU dispatch sweep at 256 MB (GB/s of float64 bytes):")
    for r in tm["gpu_dispatch_sweep_256MB"]:
        print(f"  chunk={r['chunk']:3d} tg={r['tg']:7s} narrow {r['narrow_GBps']:7.1f}  widen {r['widen_GBps']:7.1f}")
    print("best:", tm["gpu_dispatch_best"])
    print("\nGB/s of float64 bytes       256MB      1GB      4GB")
    for key in ("cpu_vdsp_narrow_GBps", "cpu_vdsp_widen_GBps", "cpu_vdsp_narrow_static16_GBps", "cpu_vdsp_widen_static16_GBps",
                "cpu_loop_narrow_GBps", "cpu_loop_widen_GBps", "cpu_vdsp_narrow_1thread_GBps", "cpu_vdsp_widen_1thread_GBps",
                "gpu_narrow_GBps", "gpu_widen_GBps", "gpu_narrow_cold_wraps_GBps", "gpu_widen_cold_wraps_GBps",
                "numpy_astype_narrow_GBps", "numpy_astype_widen_GBps"):
        vals = [tm[s].get(key) for s in ("256MB", "1GB", "4GB")]
        print(f"  {key:32s}" + "".join(f"{v:9.2f}" if v is not None else "        -" for v in vals))
    result["wall_seconds"] = round(time.perf_counter() - t_start, 1)
    with open(os.path.join(HERE, "result.json"), "w") as f:  # full result of the latest run, pretty-printed
        json.dump(result, f, indent=1)
    print("RESULT " + json.dumps(result))


if __name__ == "__main__":
    main()
