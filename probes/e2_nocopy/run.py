#!/usr/bin/env python
"""E2 probe: zero-copy between NumPy memory and Metal shared buffers.

    ./build.sh && ../../.venv/bin/python run.py

Functional cases each run in their own subprocess, once normally and once with
MTL_DEBUG_LAYER=1 (so an API-validation abort in one case cannot hide the others).
Timing runs in the parent process only.  Last line printed: `RESULT {json}`.
"""
import ctypes
import gc
import json
import mmap
import os
import statistics
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
u64 = ctypes.c_uint64
vp = ctypes.c_void_p
dbl_p = ctypes.POINTER(ctypes.c_double)
PAGE = 16384
DEALLOC_CB = ctypes.CFUNCTYPE(None, ctypes.c_void_p)


def load():
    L = ctypes.CDLL(os.path.join(HERE, "build", "libe2.dylib"))
    L.e2_init.argtypes = [ctypes.c_char_p]
    L.e2_last_error.restype = ctypes.c_char_p
    L.e2_device_name.restype = ctypes.c_char_p
    for f in ("e2_max_buffer_length", "e2_recommended_working_set", "e2_page_size"):
        getattr(L, f).restype = u64
    L.e2_dealloc_count.restype = ctypes.c_long
    L.e2_release_count.restype = ctypes.c_long
    L.e2_region.argtypes = [u64] + [ctypes.POINTER(u64)] * 2 + [ctypes.POINTER(ctypes.c_int)] * 5
    L.e2_wrappable_span.argtypes = [u64, u64, ctypes.c_int, ctypes.c_int] + [ctypes.POINTER(u64)] * 3
    L.e2_wrap.argtypes = [u64, u64, DEALLOC_CB, vp]
    L.e2_wrap.restype = vp
    L.e2_wrap_nodealloc.argtypes = [u64, u64]
    L.e2_wrap_nodealloc.restype = vp
    L.e2_alloc.argtypes = [u64]
    L.e2_alloc.restype = vp
    L.e2_release.argtypes = [vp]
    L.e2_release.restype = None
    for f in ("e2_contents", "e2_length", "e2_allocated_size"):
        getattr(L, f).argtypes = [vp]
        getattr(L, f).restype = u64
    L.e2_storage_mode.argtypes = [vp]
    L.e2_run_muladd_u32.argtypes = [vp, u64, u64, ctypes.c_uint32, ctypes.c_uint32, dbl_p]
    L.e2_run_fill_u32.argtypes = [vp, u64, ctypes.c_uint32, ctypes.c_uint32, dbl_p]
    L.e2_run_copy_muladd_u32.argtypes = [vp, u64, vp, u64, ctypes.c_uint32, ctypes.c_uint32, dbl_p]
    L.e2_run_inc_u64.argtypes = [vp, u64, dbl_p]
    L.e2_time_wrap_release.argtypes = [u64, u64, ctypes.c_int, dbl_p, dbl_p]
    L.e2_time_wrap_use_release.argtypes = [u64, u64, u64, ctypes.c_int, dbl_p, dbl_p]
    L.e2_time_alloc_release.argtypes = [u64, ctypes.c_int, dbl_p, dbl_p]
    L.e2_time_wrap_anatomy.argtypes = [ctypes.c_int, u64, u64, u64, ctypes.c_int, dbl_p, dbl_p, dbl_p, dbl_p]
    L.e2_time_fresh_output.argtypes = [ctypes.c_int, u64, ctypes.c_int, dbl_p, dbl_p, dbl_p]
    L.e2_time_memcpy.argtypes = [u64, u64, u64, ctypes.c_int, dbl_p]
    L.e2_time_memcpy.restype = None
    L.e2_time_memcpy_mt.argtypes = [u64, u64, u64, ctypes.c_int, dbl_p]
    L.e2_time_memcpy_mt.restype = None
    L.e2_make_split_mapping.argtypes = [u64, ctypes.c_int]
    L.e2_make_split_mapping.restype = u64
    L.e2_make_mapping_with_hole.argtypes = [u64]
    L.e2_make_mapping_with_hole.restype = u64
    rc = L.e2_init(os.path.join(HERE, "build", "kernels.metallib").encode())
    if rc != 0:
        raise RuntimeError(f"e2_init failed: {rc} {L.e2_last_error().decode()}")
    return L


L = None


def region(addr):
    s, z = u64(), u64()
    a = [ctypes.c_int() for _ in range(5)]
    rc = L.e2_region(addr, ctypes.byref(s), ctypes.byref(z), *[ctypes.byref(x) for x in a])
    if rc != 0:
        return None
    return {"start": s.value, "size": z.value, "prot": a[0].value, "max_prot": a[1].value,
            "share_mode": a[2].value, "user_tag": a[3].value, "is_submap": a[4].value}


def span(addr, nbytes, need_write=True, allow_multi=True):
    """Production-style check: enclosing page span fully mapped R(W) -> (start, len, offset) or None.
    allow_multi=False reproduces the stricter "one VM map entry" rule."""
    s, n, o = u64(), u64(), u64()
    k = L.e2_wrappable_span(addr, nbytes, int(need_write), int(allow_multi), ctypes.byref(s), ctypes.byref(n), ctypes.byref(o))
    return (s.value, n.value, o.value) if k else None


def span_entries(addr, nbytes, need_write=True):
    s, n, o = u64(), u64(), u64()
    return L.e2_wrappable_span(addr, nbytes, int(need_write), 1, ctypes.byref(s), ctypes.byref(n), ctypes.byref(o))


def ptr(a):
    return a.__array_interface__["data"][0]


def roundup(n):
    return (n + PAGE - 1) // PAGE * PAGE


def gpu_muladd(h, byte_offset, n, mul, add):
    rc = L.e2_run_muladd_u32(h, byte_offset, n, mul, add, None)
    if rc != 0:
        raise RuntimeError(L.e2_last_error().decode())


def roundtrip(h, byte_offset, view):
    """view: uint32 NumPy array whose memory starts byte_offset bytes into buffer h.
    Returns True when (1) the GPU sees what NumPy wrote and NumPy sees what the GPU wrote,
    (2) again after a second NumPy write (no stale snapshot on either side)."""
    n = view.size
    view[:] = np.arange(n, dtype=np.uint32)
    gpu_muladd(h, byte_offset, n, 3, 7)
    ok1 = bool(np.array_equal(view, np.arange(n, dtype=np.uint32) * np.uint32(3) + np.uint32(7)))
    view[:] = 5
    view[-1] = 11
    gpu_muladd(h, byte_offset, n, 2, 1)
    ok2 = bool(view[0] == 11 and view[n // 2] == 11 and view[-1] == 23 and int(view.sum(dtype=np.uint64)) == 11 * (n - 1) + 23)
    return ok1 and ok2


# ----------------------------------------------------------------------------------------
# functional cases (each runs in its own subprocess)
# ----------------------------------------------------------------------------------------

def case_aligned_pagemult():
    a = np.zeros((64 << 20) // 4, dtype=np.uint32)
    p = ptr(a)
    h = L.e2_wrap_nodealloc(p, a.nbytes)
    out = {"ptr_page_aligned": p % PAGE == 0, "len_page_multiple": a.nbytes % PAGE == 0, "wrapped": bool(h)}
    if h:
        out.update(contents_is_ptr=L.e2_contents(h) == p, length=L.e2_length(h), shared=L.e2_storage_mode(h) == 0,
                   roundtrip_ok=roundtrip(h, 0, a))
        L.e2_release(h)
    return out


def case_aligned_exact_nonmultiple():
    """Page-aligned pointer, length NOT a page multiple, passed exactly."""
    a = np.zeros((64 << 20) // 4 + 2, dtype=np.uint32)  # +8 bytes
    p = ptr(a)
    h = L.e2_wrap_nodealloc(p, a.nbytes)
    out = {"ptr_page_aligned": p % PAGE == 0, "len_page_multiple": a.nbytes % PAGE == 0, "wrapped": bool(h)}
    if h:
        out.update(contents_is_ptr=L.e2_contents(h) == p, length=L.e2_length(h),
                   allocated_size=L.e2_allocated_size(h), roundtrip_ok=roundtrip(h, 0, a))
        L.e2_release(h)
    return out


def case_aligned_rounded():
    """Page-aligned pointer, length rounded up to the page boundary after a vm_region check."""
    a = np.zeros((64 << 20) // 4 + 2, dtype=np.uint32)
    p = ptr(a)
    sp = span(p, a.nbytes)
    out = {"ptr_page_aligned": p % PAGE == 0, "len_page_multiple": a.nbytes % PAGE == 0,
           "span_in_one_region": sp is not None}
    if sp:
        h = L.e2_wrap_nodealloc(sp[0], sp[1])
        out.update(wrapped=bool(h), span_len=sp[1], span_offset=sp[2])
        if h:
            out.update(contents_is_ptr=L.e2_contents(h) == p, length=L.e2_length(h), roundtrip_ok=roundtrip(h, sp[2], a))
            L.e2_release(h)
    return out


def case_sizes_rounded():
    """Round-up wrap across a size sweep (arrays whose byte length is not a page multiple)."""
    rows = []
    for nbytes in (16 << 10, 64 << 10, 128 << 10, 1 << 20, 4 << 20, 8 << 20, 32 << 20, 256 << 20, 1 << 30):
        a = np.zeros(nbytes // 4 + 6, dtype=np.uint32)  # +24 bytes
        p = ptr(a)
        sp = span(p, a.nbytes)
        row = {"nbytes": a.nbytes, "ptr_page_aligned": p % PAGE == 0, "span_ok": sp is not None,
               "vm_entries": span_entries(p, a.nbytes),
               "single_entry_rule_ok": span(p, a.nbytes, allow_multi=False) is not None}
        r = region(p)
        row["vm_user_tag"] = r["user_tag"] if r else None
        if sp:
            h = L.e2_wrap_nodealloc(sp[0], sp[1])
            row["wrapped"] = bool(h)
            if h:
                row["roundtrip_ok"] = roundtrip(h, sp[2], a)
                L.e2_release(h)
        rows.append(row)
        del a
    return {"rows": rows, "all_ok": all(r.get("roundtrip_ok") for r in rows)}


def case_unaligned_raw():
    """Sliced view whose pointer is not page aligned, passed raw to Metal."""
    base = np.zeros((64 << 20) // 4, dtype=np.uint32)
    a = base[3:]  # +12 bytes
    p = ptr(a)
    nbytes = a.nbytes
    h = L.e2_wrap_nodealloc(p, nbytes)
    out = {"ptr_page_aligned": p % PAGE == 0, "wrapped": bool(h)}
    if h:
        out.update(contents=L.e2_contents(h), contents_is_ptr=L.e2_contents(h) == p, length=L.e2_length(h),
                   allocated_size=L.e2_allocated_size(h))
        out["roundtrip_ok"] = roundtrip(h, 0, a)
        out["base_prefix_untouched"] = bool((base[:3] == 0).all())
        L.e2_release(h)
    return out


def case_unaligned_span():
    """Sliced view wrapped as its enclosing page span + byte offset (the production recipe)."""
    base = np.zeros((64 << 20) // 4, dtype=np.uint32)
    a = base[1001:-7]
    p = ptr(a)
    sp = span(p, a.nbytes)
    out = {"ptr_page_aligned": p % PAGE == 0, "span_in_one_region": sp is not None}
    if sp:
        h = L.e2_wrap_nodealloc(sp[0], sp[1])
        out.update(wrapped=bool(h), span_offset=sp[2])
        if h:
            out["roundtrip_ok"] = roundtrip(h, sp[2], a)
            out["neighbours_untouched"] = bool((base[:1001] == 0).all() and (base[-7:] == 0).all())
            L.e2_release(h)
    return out


def case_page_offset_view():
    """View starting at a page boundary inside a larger allocation: raw wrap of an interior page range."""
    base = np.zeros((64 << 20) // 4, dtype=np.uint32)
    a = base[PAGE // 4 * 5: PAGE // 4 * 5 + (8 << 20) // 4]
    p = ptr(a)
    h = L.e2_wrap_nodealloc(p, a.nbytes)
    out = {"ptr_page_aligned": p % PAGE == 0, "wrapped": bool(h)}
    if h:
        out["roundtrip_ok"] = roundtrip(h, 0, a)
        out["neighbours_untouched"] = bool((base[:PAGE // 4 * 5] == 0).all() and (base[PAGE // 4 * 5 + a.size:] == 0).all())
        L.e2_release(h)
    return out


def case_small_raw():
    """Small (4 KiB) malloc-backed array passed raw (pointer usually not page aligned)."""
    keep = [np.zeros(1024, dtype=np.uint32) for _ in range(64)]
    a = next((k for k in keep if ptr(k) % PAGE), keep[0])
    p = ptr(a)
    h = L.e2_wrap_nodealloc(p, a.nbytes)
    out = {"ptr_page_aligned": p % PAGE == 0, "n_aligned_of_64": sum(ptr(k) % PAGE == 0 for k in keep), "wrapped": bool(h)}
    if h:
        out.update(contents_is_ptr=L.e2_contents(h) == p, length=L.e2_length(h), allocated_size=L.e2_allocated_size(h),
                   roundtrip_ok=roundtrip(h, 0, a))
        L.e2_release(h)
    return out


def case_small_span():
    """Small (4 KiB) array wrapped as its enclosing page(s) + offset.  Works, but shares pages with other mallocs."""
    keep = [np.zeros(1024, dtype=np.uint32) for _ in range(64)]
    a = next((k for k in keep if ptr(k) % PAGE), keep[0])
    p = ptr(a)
    sp = span(p, a.nbytes)
    out = {"ptr_page_aligned": p % PAGE == 0, "span_in_one_region": sp is not None}
    if sp:
        h = L.e2_wrap_nodealloc(sp[0], sp[1])
        out.update(wrapped=bool(h), span_len=sp[1], span_offset=sp[2])
        if h:
            out["roundtrip_ok"] = roundtrip(h, sp[2], a)
            L.e2_release(h)
    return out


def case_noncontiguous():
    """Strided view: no single contiguous byte range of the right dtype -> must copy (or pass strides to the kernel)."""
    base = np.zeros((8 << 20) // 4, dtype=np.uint32)
    a = base[::2]
    b = np.zeros((2048, 2048), dtype=np.uint32).T  # F-ordered view of C data: contiguous, but transposed
    return {"c_contiguous": bool(a.flags.c_contiguous), "needs_copy": not a.flags.c_contiguous,
            "transposed_f_contiguous": bool(b.flags.f_contiguous),
            "note": "strided -> copy fallback; F-contiguous is wrappable as-is if the kernel takes strides/axis order"}


def case_readonly_mapping():
    """PROT_READ anonymous mapping: wrap it and use it as a read-only kernel input."""
    n = (8 << 20) // 4
    m = mmap.mmap(-1, n * 4, prot=mmap.PROT_READ | mmap.PROT_WRITE)
    w = np.frombuffer(m, dtype=np.uint32)
    w[:] = np.arange(n, dtype=np.uint32)
    p = ptr(w)
    libc = ctypes.CDLL(None)
    libc.mprotect.argtypes = [vp, ctypes.c_size_t, ctypes.c_int]
    rc = libc.mprotect(p, n * 4, mmap.PROT_READ)
    r = region(p)
    out = {"mprotect_rc": rc, "region_prot": r["prot"] if r else None,
           "span_check_need_write": span(p, n * 4, True) is not None,
           "span_check_read_only": span(p, n * 4, False) is not None}
    h = L.e2_wrap_nodealloc(p, n * 4)
    out["wrapped"] = bool(h)
    if h:
        dst = L.e2_alloc(n * 4)
        rc = L.e2_run_copy_muladd_u32(h, 0, dst, n, 2, 1, None)
        d = np.ctypeslib.as_array(ctypes.cast(L.e2_contents(dst), ctypes.POINTER(ctypes.c_uint32)), shape=(n,))
        out["read_kernel_rc"] = rc
        out["read_kernel_ok"] = bool(np.array_equal(d, np.arange(n, dtype=np.uint32) * np.uint32(2) + np.uint32(1)))
        del d
        L.e2_release(dst)
        L.e2_release(h)
    return out


def case_readonly_gpu_write():
    """HAZARD CHECK: wrap a PROT_READ mapping and run a kernel that WRITES it.  Does the page protection hold?"""
    n = (1 << 20) // 4
    m = mmap.mmap(-1, n * 4, prot=mmap.PROT_READ | mmap.PROT_WRITE)
    w = np.frombuffer(m, dtype=np.uint32)
    w[:] = 1
    p = ptr(w)
    libc = ctypes.CDLL(None)
    libc.mprotect.argtypes = [vp, ctypes.c_size_t, ctypes.c_int]
    libc.mprotect(p, n * 4, mmap.PROT_READ)
    h = L.e2_wrap_nodealloc(p, n * 4)
    out = {"wrapped": bool(h)}
    if h:
        rc = L.e2_run_muladd_u32(h, 0, n, 5, 2, None)
        out["write_kernel_rc"] = rc
        if rc != 0:
            out["error"] = L.e2_last_error().decode()[:200]
        out["memory_changed_by_gpu"] = bool((w == 7).all())
        out["memory_intact"] = bool((w == 1).all())
        L.e2_release(h)
    return out


def case_memmap_file():
    """File-backed MAP_SHARED np.memmap: wrap, write from the GPU, check the file."""
    n = (8 << 20) // 4
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "mm.bin")
        mm = np.memmap(path, dtype=np.uint32, mode="w+", shape=(n,))
        p = ptr(mm)
        r = region(p)
        out = {"ptr_page_aligned": p % PAGE == 0, "share_mode": r["share_mode"] if r else None,
               "span_ok": span(p, mm.nbytes) is not None}
        h = L.e2_wrap_nodealloc(p, mm.nbytes)
        out["wrapped"] = bool(h)
        if h:
            out["roundtrip_ok"] = roundtrip(h, 0, mm)
            L.e2_release(h)
            mm.flush()
            back = np.fromfile(path, dtype=np.uint32)
            out["file_sees_gpu_writes"] = bool(back[0] == 11 and back[-1] == 23)
        del mm
    return out


def case_beyond_region_hole():
    """Length rounded past the end of the mapping into an unmapped hole: what does Metal do?"""
    base = L.e2_make_mapping_with_hole(4)
    out = {"span_check": span(base, 5 * PAGE) is not None}
    h = L.e2_wrap_nodealloc(base, 5 * PAGE)
    out["wrapped"] = bool(h)
    if h:
        # Only touch the 4 mapped pages.
        rc = L.e2_run_muladd_u32(h, 0, 4 * PAGE // 4, 1, 9, None)
        v = np.ctypeslib.as_array(ctypes.cast(base, ctypes.POINTER(ctypes.c_uint32)), shape=(4 * PAGE // 4,))
        out["kernel_rc"] = rc
        out["kernel_ok_on_mapped_part"] = bool((v == 9).all())
        L.e2_release(h)
    return out


def case_split_region():
    """Span crossing two adjacent but distinct VM map entries (both RW)."""
    pages = 64
    base = L.e2_make_split_mapping(pages, 0)
    n = 2 * pages * PAGE // 4
    r1, r2 = region(base), region(base + pages * PAGE)
    out = {"two_entries": bool(r1 and r2 and r1["start"] != r2["start"]),
           "single_entry_rule_ok": span(base, 2 * pages * PAGE, allow_multi=False) is not None,
           "multi_entry_walk_ok": span(base, 2 * pages * PAGE) is not None}
    h = L.e2_wrap_nodealloc(base, 2 * pages * PAGE)
    out["wrapped"] = bool(h)
    if h:
        v = np.ctypeslib.as_array(ctypes.cast(base, ctypes.POINTER(ctypes.c_uint32)), shape=(n,))
        out["roundtrip_ok"] = roundtrip(h, 0, v)
        L.e2_release(h)
    return out


def case_dealloc_callback():
    """Lifetime, wrap direction: the MTLBuffer deallocator block owns a reference to the NumPy array."""
    a = np.zeros((16 << 20) // 4, dtype=np.uint32)
    keep = {}
    fired = []

    def _cb(ctx):
        fired.append(ctx)
        keep.pop(ctx, None)  # drops the array reference -> NumPy memory may now be freed

    cb = DEALLOC_CB(_cb)
    p = ptr(a)
    keep[1] = a
    rc0 = sys.getrefcount(a)
    d0 = L.e2_dealloc_count()
    h = L.e2_wrap(p, a.nbytes, cb, 1)
    ok = roundtrip(h, 0, a)
    del a  # the user drops the array while the buffer is still alive
    gc.collect()
    alive = 1 in keep
    v = np.ctypeslib.as_array(ctypes.cast(p, ctypes.POINTER(ctypes.c_uint32)), shape=(4,))
    still_valid = int(v[0]) == 11
    del v
    L.e2_release(h)
    t0 = time.perf_counter()
    while not fired and time.perf_counter() - t0 < 2.0:
        time.sleep(0.001)
    return {"wrapped": bool(h), "roundtrip_ok": ok, "array_kept_alive_by_buffer": alive and still_valid,
            "deallocator_fired": bool(fired), "deallocator_count_delta": L.e2_dealloc_count() - d0,
            "deallocator_latency_ms": round((time.perf_counter() - t0) * 1e3, 3),
            "array_released_after": 1 not in keep, "refcount_before_wrap": rc0}


class MetalBuffer:
    """Owns one retained shared MTLBuffer; NumPy arrays made from it keep it alive via .base."""

    def __init__(self, nbytes, dtype, shape):
        self._h = L.e2_alloc(nbytes)
        if not self._h:
            raise MemoryError("newBufferWithLength failed")
        self.__array_interface__ = {"version": 3, "shape": tuple(shape), "typestr": np.dtype(dtype).str,
                                    "data": (L.e2_contents(self._h), False)}

    @property
    def handle(self):
        return self._h

    def __del__(self):
        h, self._h = self._h, None
        if h:
            L.e2_release(h)


def case_reverse_owned():
    """Reverse direction: shared MTLBuffer written by a kernel, exposed to NumPy with no copy."""
    n = (64 << 20) // 4
    owner = MetalBuffer(n * 4, np.uint32, (n,))
    p = L.e2_contents(owner.handle)
    rc = L.e2_run_fill_u32(owner.handle, n, 3, 5, None)
    arr = np.asarray(owner)
    out = {"kernel_rc": rc, "contents_page_aligned": p % PAGE == 0, "numpy_ptr_is_contents": ptr(arr) == p,
           "owndata": bool(arr.flags.owndata), "writeable": bool(arr.flags.writeable), "base_is_owner": arr.base is owner,
           "values_ok": bool(np.array_equal(arr, np.arange(n, dtype=np.uint32) * np.uint32(3) + np.uint32(5)))}
    # NumPy writes, GPU reads through the same buffer.
    arr[:] = 1
    L.e2_run_muladd_u32(owner.handle, 0, n, 4, 2, None)
    out["gpu_sees_numpy_writes"] = bool((arr == 6).all())
    # Lifetime: drop every reference except the array (and then only a view of it).
    r0 = L.e2_release_count()
    del owner
    gc.collect()
    out["alive_after_owner_del"] = L.e2_release_count() == r0 and int(arr[123]) == 6
    view = arr[1000:2000].reshape(10, 100)
    del arr
    gc.collect()
    out["alive_through_view"] = L.e2_release_count() == r0 and int(view[3, 3]) == 6
    del view
    gc.collect()
    out["released_with_last_array"] = L.e2_release_count() == r0 + 1
    return out


def case_mlx():
    """Second no-copy MTLBuffer over the host memory of an evaluated MLX array."""
    import mlx.core as mx

    n = (64 << 20) // 4
    x = mx.arange(n, dtype=mx.uint32) * 3 + 1
    mx.eval(x)
    v = np.asarray(x)
    p = ptr(v)
    v2 = np.asarray(x)
    out = {"mlx": mx.__version__ if hasattr(mx, "__version__") else None,
           "np_view_owndata": bool(v.flags.owndata), "np_view_writeable": bool(v.flags.writeable),
           "np_view_stable_ptr": ptr(v2) == p, "ptr_page_aligned": p % PAGE == 0,
           "len_page_multiple": v.nbytes % PAGE == 0}
    r = region(p)
    out["region"] = {"share_mode": r["share_mode"], "user_tag": r["user_tag"], "prot": r["prot"]} if r else None
    sp = span(p, v.nbytes)
    out["span_in_one_region"] = sp is not None
    if not sp:
        return out
    h = L.e2_wrap_nodealloc(sp[0], sp[1])
    out["wrapped"] = bool(h)
    if not h:
        return out
    expect = np.arange(n, dtype=np.uint32) * np.uint32(3) + np.uint32(1)
    # (1) our kernel reads what MLX's kernels wrote, through OUR buffer, into an owned buffer
    dst = L.e2_alloc(n * 4)
    L.e2_run_copy_muladd_u32(h, sp[2], dst, n, 1, 0, None)
    d = np.ctypeslib.as_array(ctypes.cast(L.e2_contents(dst), ctypes.POINTER(ctypes.c_uint32)), shape=(n,))
    out["our_kernel_reads_mlx_data"] = bool(np.array_equal(d, expect))
    # (2) our kernel writes in place; NumPy view and a fresh MLX computation both see it
    gpu_muladd(h, sp[2], n, 2, 5)
    expect2 = expect * np.uint32(2) + np.uint32(5)
    out["numpy_view_sees_our_writes"] = bool(np.array_equal(v, expect2))
    y = x + 0
    mx.eval(y)
    out["mlx_sees_our_writes"] = bool(np.array_equal(np.asarray(y), expect2))
    s = mx.sum(x.astype(mx.uint64))
    mx.eval(s)
    out["mlx_reduction_sees_our_writes"] = int(s.item()) == int(expect2.sum(dtype=np.uint64))
    # (3) writes from MLX: let MLX donate x's buffer to an in-place result, then read through our
    #     (still alive) second buffer.  x must have no other owners for donation to happen.
    del v, v2, y, s
    z = x * 7 + 2
    del x
    gc.collect()
    mx.eval(z)
    vz = np.asarray(z)
    donated = ptr(vz) == p
    out["mlx_reused_same_pages"] = donated
    expect3 = expect2 * np.uint32(7) + np.uint32(2)
    out["mlx_result_ok"] = bool(np.array_equal(vz, expect3))
    L.e2_run_copy_muladd_u32(h, sp[2], dst, n, 1, 0, None)
    out["our_buffer_sees_mlx_writes"] = bool(np.array_equal(d, expect3)) if donated else None
    del d
    L.e2_release(dst)
    L.e2_release(h)
    # MLX still healthy after our wrapper is gone
    w = z + 1
    mx.eval(w)
    out["mlx_ok_after_release"] = bool(np.array_equal(np.asarray(w), expect3 + np.uint32(1)))
    return out


CASES = {f[5:]: globals()[f] for f in list(globals()) if f.startswith("case_")}


def run_case_subprocess(name, env_extra):
    env = dict(os.environ)
    env.update(env_extra)
    t0 = time.perf_counter()
    pr = subprocess.run([sys.executable, os.path.abspath(__file__), "--case", name], env=env,
                        capture_output=True, text=True, timeout=300)
    res = {"returncode": pr.returncode, "seconds": round(time.perf_counter() - t0, 2)}
    for line in pr.stdout.splitlines():
        if line.startswith("CASE "):
            res["result"] = json.loads(line[5:])
    err = [ln for ln in pr.stderr.splitlines() if ln.strip()]
    asserts = [ln for ln in err if "Validation Enabled" not in ln
               and ("failed assertion" in ln or "validation" in ln.lower() or "error" in ln.lower())]
    res["debug_layer_banner_seen"] = any("Metal API Validation Enabled" in ln for ln in err)
    if asserts:
        res["validation_messages"] = [a[-400:] for a in asserts[:4]]
    elif pr.returncode != 0:
        res["stderr_tail"] = [e[-300:] for e in err[-4:]]
    return res


# ----------------------------------------------------------------------------------------
# timing
# ----------------------------------------------------------------------------------------

def med(buf, n):
    return statistics.median(buf[i] for i in range(n))


def timing():
    out = {}
    cores = ctypes.CDLL(os.path.join(HERE, "build", "libe2.dylib")).e2_perf_cores()
    for label, nbytes in (("1MB", 1 << 20), ("64MB", 64 << 20), ("1GB", 1 << 30), ("4GB", 4 << 30)):
        src = np.ones(nbytes // 8, dtype=np.uint64)  # faulted in
        dst = np.ones(nbytes // 8, dtype=np.uint64)
        p, q = ptr(src), ptr(dst)
        assert p % PAGE == 0 and nbytes % PAGE == 0
        reps = 9 if nbytes <= (1 << 30) else 5
        t = (ctypes.c_double * 16)()
        t2 = (ctypes.c_double * 16)()
        row = {"bytes": nbytes}
        L.e2_time_memcpy(q, p, nbytes, 1, t)  # warm-up
        L.e2_time_memcpy(q, p, nbytes, reps, t)
        row["memcpy_1thread_ms"] = med(t, reps) * 1e3
        L.e2_time_memcpy_mt(q, p, nbytes, 1, t)
        L.e2_time_memcpy_mt(q, p, nbytes, reps, t)
        row["memcpy_%dthreads_ms" % cores] = med(t, reps) * 1e3
        row["memcpy_mt_ms"] = row["memcpy_%dthreads_ms" % cores]
        # wrap + release, no use
        L.e2_time_wrap_release(p, nbytes, 2, t, t2)
        sync = L.e2_time_wrap_release(p, nbytes, reps, t, t2)
        row["wrap_ms"] = med(t, reps) * 1e3
        row["release_ms"] = med(t2, reps) * 1e3
        row["wrap_release_ms"] = statistics.median(t[i] + t2[i] for i in range(reps)) * 1e3
        row["deallocator_ran_synchronously"] = f"{sync}/{reps}"
        # first use of a freshly wrapped buffer vs steady-state use of a long-lived one
        n64 = nbytes // 8
        h = L.e2_wrap_nodealloc(p, nbytes)
        ks = ctypes.c_double()
        L.e2_run_inc_u64(h, n64, ctypes.byref(ks))
        steady = []
        for _ in range(5):
            L.e2_run_inc_u64(h, n64, ctypes.byref(ks))
            steady.append(ks.value)
        L.e2_release(h)
        row["kernel_steady_ms"] = statistics.median(steady) * 1e3
        L.e2_time_wrap_use_release(p, nbytes, n64, 1, t, t2)
        L.e2_time_wrap_use_release(p, nbytes, n64, 5, t, t2)
        row["wrap_firstuse_release_ms"] = med(t, 5) * 1e3
        row["kernel_firstuse_ms"] = med(t2, 5) * 1e3
        row["firstuse_overhead_ms"] = row["wrap_firstuse_release_ms"] - row["kernel_steady_ms"]
        # owned shared buffer allocation for context
        L.e2_time_alloc_release(nbytes, 1, t, t2)
        L.e2_time_alloc_release(nbytes, 5, t, t2)
        row["owned_alloc_ms"] = med(t, 5) * 1e3
        row["owned_release_ms"] = med(t2, 5) * 1e3
        row["wrap_release_pct_of_memcpy_1t"] = 100 * row["wrap_release_ms"] / row["memcpy_1thread_ms"]
        row["wrap_release_pct_of_memcpy_mt"] = 100 * row["wrap_release_ms"] / row["memcpy_mt_ms"]
        row["firstuse_overhead_pct_of_memcpy_1t"] = 100 * row["firstuse_overhead_ms"] / row["memcpy_1thread_ms"]
        row["firstuse_overhead_pct_of_memcpy_mt"] = 100 * row["firstuse_overhead_ms"] / row["memcpy_mt_ms"]
        row["memcpy_1thread_GBps"] = nbytes / 1e9 / (row["memcpy_1thread_ms"] / 1e3)
        row["memcpy_mt_GBps"] = nbytes / 1e9 / (row["memcpy_mt_ms"] / 1e3)
        out[label] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}
        del src, dst
        gc.collect()
    # small-size crossover: where does a wrap beat a 1-thread memcpy?  (a) wrap+release only,
    # (b) including the first-use cost of the fresh wrap (wrap+kernel+release minus steady kernel).
    cross = []
    for nbytes in (16 << 10, 64 << 10, 256 << 10, 1 << 20, 4 << 20, 16 << 20):
        src = np.ones(nbytes // 8, dtype=np.uint64)
        dst = np.ones(nbytes // 8, dtype=np.uint64)
        p, q = ptr(src), ptr(dst)
        if p % PAGE:
            continue
        t = (ctypes.c_double * 64)()
        t2 = (ctypes.c_double * 64)()
        L.e2_time_memcpy(q, p, nbytes, 31, t)
        m = med(t, 31)
        L.e2_time_wrap_release(p, nbytes, 31, t, t2)
        w = statistics.median(t[i] + t2[i] for i in range(31))
        h = L.e2_wrap_nodealloc(p, nbytes)
        ks = ctypes.c_double()
        steady = []
        for _ in range(22):
            L.e2_run_inc_u64(h, nbytes // 8, ctypes.byref(ks))
            steady.append(ks.value)
        L.e2_release(h)
        L.e2_time_wrap_use_release(p, nbytes, nbytes // 8, 21, t, t2)
        fu = med(t, 21) - statistics.median(steady[1:])
        cross.append({"bytes": nbytes, "memcpy_us": round(m * 1e6, 2), "wrap_release_us": round(w * 1e6, 2),
                      "wrap_firstuse_overhead_us": round(fu * 1e6, 1)})
    out["crossover"] = cross

    # anatomy: where the cost of a freshly wrapped / freshly allocated buffer really sits
    anat = {}
    for label, nbytes in (("128MB", 128 << 20), ("1GB", 1 << 30)):
        a = np.ones(nbytes // 8, dtype=np.uint64)
        p = ptr(a)
        row = {}
        c, k, k2, rel = ((ctypes.c_double * 8)() for _ in range(4))
        for key, kind, n64 in (("wrap_existing_tracked", 0, nbytes // 8), ("wrap_existing_untracked", 1, nbytes // 8),
                               ("wrap_existing_kernel_touches_1_1024th", 0, nbytes // 8 // 1024),
                               ("wrap_existing_two_kernels", 2, nbytes // 8)):
            L.e2_time_wrap_anatomy(kind, p, nbytes, n64, 1, c, k, k2, rel)
            L.e2_time_wrap_anatomy(kind, p, nbytes, n64, 5, c, k, k2, rel)
            row[key] = {"create_ms": round(med(c, 5) * 1e3, 3), "kernel_first_ms": round(med(k, 5) * 1e3, 3),
                        "release_ms": round(med(rel, 5) * 1e3, 3)}
            if kind == 2:
                row[key]["kernel_second_ms"] = round(med(k2, 5) * 1e3, 3)
        del a
        for key, kind in (("fresh_mmap_wrap_gpu_fill", 0), ("fresh_newBufferWithLength_gpu_fill", 1),
                          ("fresh_mmap_cpu_memset_1thread", 2), ("fresh_mmap_cpu_memset_16threads", 3),
                          ("malloc_wrap_gpu_fill", 4), ("malloc_cpu_memset_1thread", 5)):
            L.e2_time_fresh_output(kind, nbytes, 1, c, k, rel)
            L.e2_time_fresh_output(kind, nbytes, 5, c, k, rel)
            row[key] = {"create_ms": round(med(c, 5) * 1e3, 3), "fill_ms": round(med(k, 5) * 1e3, 3),
                        "release_ms": round(med(rel, 5) * 1e3, 3),
                        "total_ms": round((med(c, 5) + med(k, 5) + med(rel, 5)) * 1e3, 3)}
        h = L.e2_alloc(nbytes)
        ks = ctypes.c_double()
        ts = []
        for _ in range(7):
            L.e2_run_fill_u32(h, nbytes // 4, 1, 3, ctypes.byref(ks))
            ts.append(ks.value)
        L.e2_release(h)
        row["warm_owned_gpu_fill_ms"] = round(statistics.median(ts[1:]) * 1e3, 3)
        anat[label] = row
    out["anatomy"] = anat
    return out


def alignment_survey():
    rows = []
    for nbytes in (1 << 10, 4 << 10, 8 << 10, 16 << 10, 32 << 10, 64 << 10, 128 << 10, 1 << 20, 16 << 20):
        keep = [np.empty(nbytes // 8 + 1, dtype=np.float64) for _ in range(32)]
        al = sum(ptr(k) % PAGE == 0 for k in keep)
        sp = sum(span(ptr(k), k.nbytes) is not None for k in keep)
        r = region(ptr(keep[0]))
        rows.append({"nbytes": nbytes + 8, "page_aligned_of_32": al, "enclosing_span_in_one_region_of_32": sp,
                     "len_page_multiple": (nbytes + 8) % PAGE == 0, "vm_user_tag": r["user_tag"] if r else None})
    return rows


def main():
    global L
    if len(sys.argv) >= 3 and sys.argv[1] == "--case":
        L = load()
        res = CASES[sys.argv[2]]()
        print("CASE " + json.dumps(res))
        return
    t_start = time.perf_counter()
    L = load()
    result = {"probe": "e2_nocopy", "device": L.e2_device_name().decode(),
              "max_buffer_length_GiB": round(L.e2_max_buffer_length() / 2**30, 2), "page_size": L.e2_page_size(),
              "numpy": np.__version__}
    result["alignment_survey"] = alignment_survey()
    cases, cases_dbg, cases_sv = {}, {}, {}
    for name in CASES:
        cases[name] = run_case_subprocess(name, {})
        cases_dbg[name] = run_case_subprocess(name, {"MTL_DEBUG_LAYER": "1"})
        print(f"case {name:28s} normal rc={cases[name]['returncode']} debug rc={cases_dbg[name]['returncode']}", flush=True)
    # shader validation on the cases that the API debug layer accepted
    for name in ("aligned_rounded", "unaligned_span", "reverse_owned"):
        cases_sv[name] = run_case_subprocess(name, {"MTL_DEBUG_LAYER": "1", "MTL_SHADER_VALIDATION": "1"})
    result["cases"] = cases
    result["cases_MTL_DEBUG_LAYER"] = cases_dbg
    result["cases_MTL_DEBUG_LAYER_plus_SHADER_VALIDATION"] = cases_sv
    result["timing"] = timing()

    tm = result["timing"]
    print("\nsize   memcpy1T  memcpy16T  wrap     release  wrap+rel  first-use-ovh  kernel(steady)  [ms]")
    for k in ("1MB", "64MB", "1GB", "4GB"):
        r = tm[k]
        print(f"{k:5s} {r['memcpy_1thread_ms']:9.3f} {r['memcpy_mt_ms']:9.3f} {r['wrap_ms']:8.4f} {r['release_ms']:8.4f} "
              f"{r['wrap_release_ms']:8.4f} {r['firstuse_overhead_ms']:12.3f} {r['kernel_steady_ms']:12.3f}   "
              f"wrap+rel = {r['wrap_release_pct_of_memcpy_1t']:.3f}% of memcpy1T, {r['wrap_release_pct_of_memcpy_mt']:.3f}% of memcpy16T; "
              f"incl. first use {r['firstuse_overhead_pct_of_memcpy_1t']:.2f}% / {r['firstuse_overhead_pct_of_memcpy_mt']:.2f}%")

    an = tm["anatomy"]
    print("\nfirst-use anatomy [ms]                         128MB      1GB")
    for key, field in (("wrap_existing_tracked", "kernel_first_ms"), ("wrap_existing_untracked", "kernel_first_ms"),
                       ("wrap_existing_kernel_touches_1_1024th", "kernel_first_ms"),
                       ("fresh_mmap_wrap_gpu_fill", "total_ms"), ("fresh_newBufferWithLength_gpu_fill", "total_ms"),
                       ("malloc_wrap_gpu_fill", "total_ms"), ("fresh_mmap_cpu_memset_1thread", "total_ms"),
                       ("fresh_mmap_cpu_memset_16threads", "total_ms"), ("malloc_cpu_memset_1thread", "total_ms")):
        print(f"  {key:42s} {an['128MB'][key][field]:9.3f} {an['1GB'][key][field]:9.3f}")
    print(f"  {'wrap_existing second kernel (steady)':42s} {an['128MB']['wrap_existing_two_kernels']['kernel_second_ms']:9.3f} "
          f"{an['1GB']['wrap_existing_two_kernels']['kernel_second_ms']:9.3f}")
    print(f"  {'warm owned buffer gpu fill (steady)':42s} {an['128MB']['warm_owned_gpu_fill_ms']:9.3f} {an['1GB']['warm_owned_gpu_fill_ms']:9.3f}")

    big = ("64MB", "1GB", "4GB")
    recipe = ("aligned_pagemult", "aligned_rounded", "sizes_rounded", "unaligned_span", "page_offset_view",
              "dealloc_callback", "reverse_owned", "mlx")
    clean = lambda v: v["returncode"] == 0 and "validation_messages" not in v and v.get("debug_layer_banner_seen")
    gate = {
        "A_wrap_release_le_5pct_of_memcpy_1thread": all(tm[k]["wrap_release_pct_of_memcpy_1t"] <= 5 for k in big),
        "A_wrap_release_le_5pct_of_memcpy_16threads": all(tm[k]["wrap_release_pct_of_memcpy_mt"] <= 5 for k in big),
        "B_incl_first_use_le_5pct_of_memcpy_1thread": all(tm[k]["firstuse_overhead_pct_of_memcpy_1t"] <= 5 for k in big),
        "B_incl_first_use_le_5pct_of_memcpy_16threads": all(tm[k]["firstuse_overhead_pct_of_memcpy_mt"] <= 5 for k in big),
        "no_validation_errors_all_cases_debug_layer": all(clean(v) for v in cases_dbg.values()),
        "no_validation_errors_shader_validation": all(clean(v) for v in cases_sv.values()),
        "production_recipe_cases_ok_debug_layer": all(clean(cases_dbg[c]) for c in recipe),
        "cases_aborting_under_debug_layer": [c for c in CASES if cases_dbg[c]["returncode"] != 0],
        "worst_wrap_release_pct_of_memcpy_1t_ge64MB": max(tm[k]["wrap_release_pct_of_memcpy_1t"] for k in big),
        "worst_firstuse_overhead_pct_of_memcpy_1t_ge64MB": max(tm[k]["firstuse_overhead_pct_of_memcpy_1t"] for k in big),
    }
    result["gate"] = gate
    result["wall_seconds"] = round(time.perf_counter() - t_start, 1)
    with open(os.path.join(HERE, "result.json"), "w") as f:  # full result of the latest run, pretty-printed
        json.dump(result, f, indent=1)
    print("RESULT " + json.dumps(result))


if __name__ == "__main__":
    main()
