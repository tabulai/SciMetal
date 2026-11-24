"""ctypes binding of the ndimage family of the native core (include/sm_ndimage.h)."""
from __future__ import annotations

import ctypes
import threading

import numpy as np

from .. import _native

SM_NDIMAGE_ABI_VERSION = 1
SM_NDIMAGE_MAX_PASSES = 32
SM_NDIMAGE_MAX_WINDOW = 1 << 30
CORRELATE, UNIFORM, MINIMUM, MAXIMUM = 1, 2, 3, 4
# Boundary modes: SciPy's NI_ExtendMode codes.
NEAREST, WRAP, REFLECT, MIRROR, CONSTANT = range(5)
COMBINE_SUM, COMBINE_SQRT_SUM_SQ, COMBINE_SUB, COMBINE_MORPH_LAPLACE, COMBINE_XOR = 1, 2, 3, 4, 5


class Pass(ctypes.Structure):
    _fields_ = [("struct_size", ctypes.c_uint32), ("abi_version", ctypes.c_uint32),
                ("kind", ctypes.c_uint32), ("axis", ctypes.c_uint32),
                ("mode", ctypes.c_uint32), ("reserved0", ctypes.c_uint32),
                ("origin", ctypes.c_int64), ("size", ctypes.c_uint64),
                ("weights", ctypes.c_void_p), ("cval", ctypes.c_double),
                ("reserved", ctypes.c_uint64 * 2)]


class DenseOptions(ctypes.Structure):
    _fields_ = [("struct_size", ctypes.c_uint32), ("abi_version", ctypes.c_uint32),
                ("mode", ctypes.c_uint32), ("out_dtype", ctypes.c_uint32),
                ("wshape", ctypes.c_uint32 * _native.SM_MAX_NDIM),
                ("origins", ctypes.c_int64 * _native.SM_MAX_NDIM),
                ("weights", ctypes.c_void_p), ("cval", ctypes.c_double),
                ("reserved", ctypes.c_uint64 * 2)]


class Chain(ctypes.Structure):
    _fields_ = [("struct_size", ctypes.c_uint32), ("abi_version", ctypes.c_uint32),
                ("pass_count", ctypes.c_uint32), ("reserved0", ctypes.c_uint32),
                ("passes", ctypes.POINTER(Pass)),
                ("reserved", ctypes.c_uint64 * 2)]


class MinMaxOptions(ctypes.Structure):
    _fields_ = [("struct_size", ctypes.c_uint32), ("abi_version", ctypes.c_uint32),
                ("is_max", ctypes.c_uint32), ("mode", ctypes.c_uint32),
                ("out_dtype", ctypes.c_uint32), ("reserved0", ctypes.c_uint32),
                ("fshape", ctypes.c_uint32 * _native.SM_MAX_NDIM),
                ("origins", ctypes.c_int64 * _native.SM_MAX_NDIM),
                ("footprint", ctypes.c_void_p), ("structure", ctypes.c_void_p),
                ("cval", ctypes.c_double),
                ("reserved", ctypes.c_uint64 * 2)]


class RankOptions(ctypes.Structure):
    _fields_ = [("struct_size", ctypes.c_uint32), ("abi_version", ctypes.c_uint32),
                ("rank", ctypes.c_uint32), ("mode", ctypes.c_uint32),
                ("out_dtype", ctypes.c_uint32), ("reserved0", ctypes.c_uint32),
                ("fshape", ctypes.c_uint32 * _native.SM_MAX_NDIM),
                ("origins", ctypes.c_int64 * _native.SM_MAX_NDIM),
                ("footprint", ctypes.c_void_p),
                ("cval", ctypes.c_double),
                ("reserved", ctypes.c_uint64 * 2)]


_lock = threading.Lock()
_declared = False


def library() -> ctypes.CDLL:
    """The native library with this family's prototypes declared.

    A library built without the ndimage family (SM_FAMILIES) has no filters:
    calls then go to SciPy.
    """
    global _declared
    lib = _native.library()
    if not _declared:
        with _lock:
            if not _declared:
                if not hasattr(lib, "sm_ndimage_separable_filter"):
                    raise _native.NotAccelerated("the native library was built without the ndimage filters")
                lib.sm_ndimage_separable_filter.argtypes = [
                    ctypes.c_void_p, ctypes.POINTER(_native.Array), ctypes.POINTER(Pass), ctypes.c_uint32,
                    ctypes.POINTER(_native.RouteOptions), ctypes.POINTER(_native.ArrayResult),
                    ctypes.POINTER(_native.RouteInfo)]
                if hasattr(lib, "sm_ndimage_separable_filter_to"):
                    lib.sm_ndimage_separable_filter_to.argtypes = [
                        ctypes.c_void_p, ctypes.POINTER(_native.Array), ctypes.POINTER(Pass), ctypes.c_uint32,
                        ctypes.c_uint32, ctypes.POINTER(_native.RouteOptions), ctypes.POINTER(_native.ArrayResult),
                        ctypes.POINTER(_native.RouteInfo)]
                if hasattr(lib, "sm_ndimage_dense_filter"):
                    lib.sm_ndimage_dense_filter.argtypes = [
                        ctypes.c_void_p, ctypes.POINTER(_native.Array), ctypes.POINTER(DenseOptions),
                        ctypes.POINTER(_native.RouteOptions), ctypes.POINTER(_native.ArrayResult),
                        ctypes.POINTER(_native.RouteInfo)]
                if hasattr(lib, "sm_ndimage_composite_filter"):
                    lib.sm_ndimage_composite_filter.argtypes = [
                        ctypes.c_void_p, ctypes.POINTER(_native.Array), ctypes.POINTER(Chain), ctypes.c_uint32,
                        ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(_native.RouteOptions),
                        ctypes.POINTER(_native.ArrayResult), ctypes.POINTER(_native.RouteInfo)]
                if hasattr(lib, "sm_ndimage_minmax_filter"):
                    lib.sm_ndimage_minmax_filter.argtypes = [
                        ctypes.c_void_p, ctypes.POINTER(_native.Array), ctypes.POINTER(MinMaxOptions),
                        ctypes.POINTER(_native.RouteOptions), ctypes.POINTER(_native.ArrayResult),
                        ctypes.POINTER(_native.RouteInfo)]
                if hasattr(lib, "sm_ndimage_rank_filter"):
                    lib.sm_ndimage_rank_filter.argtypes = [
                        ctypes.c_void_p, ctypes.POINTER(_native.Array), ctypes.POINTER(RankOptions),
                        ctypes.POINTER(_native.RouteOptions), ctypes.POINTER(_native.ArrayResult),
                        ctypes.POINTER(_native.RouteInfo)]
                _declared = True
    return lib


def separable_filter(function: str, array: np.ndarray, passes, out_dtype=None) -> np.ndarray:
    """Apply 1-D passes to ``array`` in order and return a new array of ``out_dtype`` (default ``array.dtype``).

    ``passes`` is a sequence of ``(kind, axis, mode, origin, size, weights, cval)``
    with ``weights`` a float64 array of ``size`` weights for CORRELATE and None
    for UNIFORM. The route follows the current precision and device policy and
    is recorded under ``function`` for ``scimetal.explain()``.

    The callers validate their arguments the way SciPy does, so the native
    core has nothing left to reject; should it still refuse a call (or run out
    of its memory budget, which SciPy does not share), the call is declined
    rather than failed, and SciPy decides what the caller sees.
    """
    lib = library()
    if len(passes) > SM_NDIMAGE_MAX_PASSES:
        raise _native.NotAccelerated("more passes than the native filter takes")
    out_code = 0 if out_dtype is None else _native.native_dtype(out_dtype)
    if out_dtype is not None and out_code is None:
        raise _native.NotAccelerated(f"no native path for output dtype {out_dtype}")
    descriptor, held = _native.as_native(array)
    if out_code != 0 and out_code != descriptor.dtype and not hasattr(lib, "sm_ndimage_separable_filter_to"):
        raise _native.NotAccelerated("the native library was built without sm_ndimage_separable_filter_to")
    native_passes = (Pass * len(passes))()
    keep = []  # weight arrays must outlive the call
    for slot, (kind, axis, mode, origin, size, weights, cval) in zip(native_passes, passes):
        slot.struct_size = ctypes.sizeof(Pass)
        slot.abi_version = SM_NDIMAGE_ABI_VERSION
        slot.kind, slot.axis, slot.mode = kind, axis, mode
        slot.origin, slot.size, slot.cval = origin, size, cval
        if weights is not None:
            weights = np.require(weights, dtype=np.float64, requirements=("C", "A"))
            keep.append(weights)
            slot.weights = weights.ctypes.data
    options = _native.route_options()
    info = _native.new_route_info()
    result = _native.ArrayResult()
    try:
        if out_code == 0 or out_code == descriptor.dtype:
            status = lib.sm_ndimage_separable_filter(
                _native.context(), ctypes.byref(descriptor), native_passes, len(passes), ctypes.byref(options),
                ctypes.byref(result), ctypes.byref(info))
        else:
            status = lib.sm_ndimage_separable_filter_to(
                _native.context(), ctypes.byref(descriptor), native_passes, len(passes), out_code,
                ctypes.byref(options), ctypes.byref(result), ctypes.byref(info))
        _native.check(status)
    except (ValueError, MemoryError) as refusal:
        raise _native.NotAccelerated(f"the native filter refused the call: {refusal}") from None
    del held, keep
    out = _native.to_numpy(result)
    _native.record_route(function, info, input_dtype=array.dtype)
    return out


def dense_filter(function: str, array: np.ndarray, weights: np.ndarray, origins, mode: int,
                 cval: float, out_dtype=None) -> np.ndarray:
    """Apply an N-D dense correlation kernel to ``array``."""
    lib = library()
    if not hasattr(lib, "sm_ndimage_dense_filter"):
        raise _native.NotAccelerated("the native library was built without sm_ndimage_dense_filter")
    descriptor, held = _native.as_native(array)
    weights_f64 = np.require(weights, dtype=np.float64, requirements=("C", "A"))
    out_code = 0 if out_dtype is None else _native.native_dtype(out_dtype)
    if out_dtype is not None and out_code is None:
        raise _native.NotAccelerated(f"no native path for output dtype {out_dtype}")
    dense_opts = DenseOptions(struct_size=ctypes.sizeof(DenseOptions), abi_version=SM_NDIMAGE_ABI_VERSION,
                              mode=mode, out_dtype=out_code, weights=weights_f64.ctypes.data, cval=cval)
    for d in range(array.ndim):
        dense_opts.wshape[d] = weights_f64.shape[d]
        dense_opts.origins[d] = origins[d]
    options = _native.route_options()
    info = _native.new_route_info()
    result = _native.ArrayResult()
    try:
        _native.check(lib.sm_ndimage_dense_filter(
            _native.context(), ctypes.byref(descriptor), ctypes.byref(dense_opts), ctypes.byref(options),
            ctypes.byref(result), ctypes.byref(info)))
    except (ValueError, MemoryError) as refusal:
        raise _native.NotAccelerated(f"the native filter refused the call: {refusal}") from None
    del held, weights_f64
    out = _native.to_numpy(result)
    _native.record_route(function, info, input_dtype=array.dtype)
    return out


def composite_filter(function: str, array: np.ndarray, chains, combine_op: int, out_dtype=None) -> np.ndarray:
    """Apply ``len(chains)`` separable 1-D pass chains to ``array`` and combine them."""
    lib = library()
    if not hasattr(lib, "sm_ndimage_composite_filter"):
        raise _native.NotAccelerated("the native library was built without sm_ndimage_composite_filter")
    if not 1 <= len(chains) <= _native.SM_MAX_NDIM:
        raise _native.NotAccelerated("composite filter chain count out of range")
    for chain in chains:
        if len(chain) > SM_NDIMAGE_MAX_PASSES:
            raise _native.NotAccelerated("more passes in chain than the native filter takes")
    descriptor, held = _native.as_native(array)
    out_code = 0 if out_dtype is None else _native.native_dtype(out_dtype)
    if out_dtype is not None and out_code is None:
        raise _native.NotAccelerated(f"no native path for output dtype {out_dtype}")
    native_chains = (Chain * len(chains))()
    keep = []
    for chain_slot, chain in zip(native_chains, chains):
        chain_slot.struct_size = ctypes.sizeof(Chain)
        chain_slot.abi_version = SM_NDIMAGE_ABI_VERSION
        chain_slot.pass_count = len(chain)
        if chain:
            native_passes = (Pass * len(chain))()
            keep.append(native_passes)
            for slot, (kind, axis, mode, origin, size, weights, cval) in zip(native_passes, chain):
                slot.struct_size = ctypes.sizeof(Pass)
                slot.abi_version = SM_NDIMAGE_ABI_VERSION
                slot.kind, slot.axis, slot.mode = kind, axis, mode
                slot.origin, slot.size, slot.cval = origin, size, cval
                if weights is not None:
                    w = np.require(weights, dtype=np.float64, requirements=("C", "A"))
                    keep.append(w)
                    slot.weights = w.ctypes.data
            chain_slot.passes = native_passes
    options = _native.route_options()
    info = _native.new_route_info()
    result = _native.ArrayResult()
    try:
        _native.check(lib.sm_ndimage_composite_filter(
            _native.context(), ctypes.byref(descriptor), native_chains, len(chains), combine_op, out_code,
            ctypes.byref(options), ctypes.byref(result), ctypes.byref(info)))
    except (ValueError, MemoryError) as refusal:
        raise _native.NotAccelerated(f"the native filter refused the call: {refusal}") from None
    del held, keep
    out = _native.to_numpy(result)
    _native.record_route(function, info, input_dtype=array.dtype)
    return out


def minmax_filter(function: str, array: np.ndarray, is_max: bool, fshape, origins,
                  footprint: np.ndarray | None, structure: np.ndarray | None,
                  mode: int, cval: float, out_dtype=None) -> np.ndarray:
    """Apply an N-D dense minimum/maximum or grayscale erosion/dilation filter."""
    lib = library()
    if not hasattr(lib, "sm_ndimage_minmax_filter"):
        raise _native.NotAccelerated("the native library was built without sm_ndimage_minmax_filter")
    descriptor, held = _native.as_native(array)
    out_code = 0 if out_dtype is None else _native.native_dtype(out_dtype)
    if out_dtype is not None and out_code is None:
        raise _native.NotAccelerated(f"no native path for output dtype {out_dtype}")
    fp_u8 = None
    if footprint is not None:
        fp_u8 = np.require(footprint, dtype=np.uint8, requirements=("C", "A"))
    st_f64 = None
    if structure is not None:
        st_f64 = np.require(structure, dtype=np.float64, requirements=("C", "A"))
    mm_opts = MinMaxOptions(
        struct_size=ctypes.sizeof(MinMaxOptions),
        abi_version=SM_NDIMAGE_ABI_VERSION,
        is_max=1 if is_max else 0,
        mode=mode,
        out_dtype=out_code,
        footprint=fp_u8.ctypes.data if fp_u8 is not None else 0,
        structure=st_f64.ctypes.data if st_f64 is not None else 0,
        cval=cval,
    )
    for d in range(array.ndim):
        mm_opts.fshape[d] = fshape[d]
        mm_opts.origins[d] = origins[d]
    options = _native.route_options()
    info = _native.new_route_info()
    result = _native.ArrayResult()
    try:
        _native.check(lib.sm_ndimage_minmax_filter(
            _native.context(), ctypes.byref(descriptor), ctypes.byref(mm_opts),
            ctypes.byref(options), ctypes.byref(result), ctypes.byref(info)))
    except (ValueError, MemoryError) as refusal:
        raise _native.NotAccelerated(f"the native filter refused the call: {refusal}") from None
    del held, fp_u8, st_f64
    out = _native.to_numpy(result)
    _native.record_route(function, info, input_dtype=array.dtype)
    return out


def rank_filter(function: str, array: np.ndarray, rank: int, fshape, origins,
                footprint: np.ndarray | None, mode: int, cval: float,
                out_dtype=None) -> np.ndarray:
    """Apply an N-D rank / median / percentile filter."""
    lib = library()
    if not hasattr(lib, "sm_ndimage_rank_filter"):
        raise _native.NotAccelerated("the native library was built without sm_ndimage_rank_filter")
    descriptor, held = _native.as_native(array)
    out_code = 0 if out_dtype is None else _native.native_dtype(out_dtype)
    if out_dtype is not None and out_code is None:
        raise _native.NotAccelerated(f"no native path for output dtype {out_dtype}")
    fp_u8 = None
    if footprint is not None:
        fp_u8 = np.require(footprint, dtype=np.uint8, requirements=("C", "A"))
    rk_opts = RankOptions(
        struct_size=ctypes.sizeof(RankOptions),
        abi_version=SM_NDIMAGE_ABI_VERSION,
        rank=rank,
        mode=mode,
        out_dtype=out_code,
        footprint=fp_u8.ctypes.data if fp_u8 is not None else 0,
        cval=cval,
    )
    for d in range(array.ndim):
        rk_opts.fshape[d] = fshape[d]
        rk_opts.origins[d] = origins[d]
    options = _native.route_options()
    info = _native.new_route_info()
    result = _native.ArrayResult()
    try:
        _native.check(lib.sm_ndimage_rank_filter(
            _native.context(), ctypes.byref(descriptor), ctypes.byref(rk_opts),
            ctypes.byref(options), ctypes.byref(result), ctypes.byref(info)))
    except (ValueError, MemoryError) as refusal:
        raise _native.NotAccelerated(f"the native filter refused the call: {refusal}") from None
    del held, fp_u8
    out = _native.to_numpy(result)
    _native.record_route(function, info, input_dtype=array.dtype)
    return out
