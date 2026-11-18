"""ctypes binding of the ndimage family of the native core (include/ms_ndimage.h)."""
from __future__ import annotations

import ctypes
import threading

import numpy as np

from .. import _native

MS_NDIMAGE_ABI_VERSION = 1
MS_NDIMAGE_MAX_PASSES = 32
MS_NDIMAGE_MAX_WINDOW = 1 << 30
CORRELATE, UNIFORM = 1, 2
# Boundary modes: SciPy's NI_ExtendMode codes.
NEAREST, WRAP, REFLECT, MIRROR, CONSTANT = range(5)


class Pass(ctypes.Structure):
    _fields_ = [("struct_size", ctypes.c_uint32), ("abi_version", ctypes.c_uint32),
                ("kind", ctypes.c_uint32), ("axis", ctypes.c_uint32),
                ("mode", ctypes.c_uint32), ("reserved0", ctypes.c_uint32),
                ("origin", ctypes.c_int64), ("size", ctypes.c_uint64),
                ("weights", ctypes.c_void_p), ("cval", ctypes.c_double),
                ("reserved", ctypes.c_uint64 * 2)]


_lock = threading.Lock()
_declared = False


def library() -> ctypes.CDLL:
    """The native library with this family's prototypes declared.

    A library built without the ndimage family (MS_FAMILIES) has no filters:
    calls then go to SciPy.
    """
    global _declared
    lib = _native.library()
    if not _declared:
        with _lock:
            if not hasattr(lib, "ms_ndimage_separable_filter"):
                raise _native.NotAccelerated("the native library was built without the ndimage filters")
            lib.ms_ndimage_separable_filter.argtypes = [
                ctypes.c_void_p, ctypes.POINTER(_native.Array), ctypes.POINTER(Pass), ctypes.c_uint32,
                ctypes.POINTER(_native.RouteOptions), ctypes.POINTER(_native.ArrayResult),
                ctypes.POINTER(_native.RouteInfo)]
            _declared = True
    return lib


def separable_filter(function: str, array: np.ndarray, passes) -> np.ndarray:
    """Apply 1-D passes to ``array`` in order and return a new array of its dtype.

    ``passes`` is a sequence of ``(kind, axis, mode, origin, size, weights, cval)``
    with ``weights`` a float64 array of ``size`` weights for CORRELATE and None
    for UNIFORM. The route follows the current precision and device policy and
    is recorded under ``function`` for ``metal_sci.explain()``.

    The callers validate their arguments the way SciPy does, so the native
    core has nothing left to reject; should it still refuse a call (or run out
    of its memory budget, which SciPy does not share), the call is declined
    rather than failed, and SciPy decides what the caller sees.
    """
    lib = library()
    if len(passes) > MS_NDIMAGE_MAX_PASSES:
        raise _native.NotAccelerated("more passes than the native filter takes")
    descriptor, held = _native.as_native(array)
    native_passes = (Pass * len(passes))()
    keep = []  # weight arrays must outlive the call
    for slot, (kind, axis, mode, origin, size, weights, cval) in zip(native_passes, passes):
        slot.struct_size = ctypes.sizeof(Pass)
        slot.abi_version = MS_NDIMAGE_ABI_VERSION
        slot.kind, slot.axis, slot.mode = kind, axis, mode
        slot.origin, slot.size, slot.cval = origin, size, cval
        if weights is not None:
            weights = np.ascontiguousarray(weights, dtype=np.float64)
            keep.append(weights)
            slot.weights = weights.ctypes.data
    options = _native.route_options()
    info = _native.new_route_info()
    result = _native.ArrayResult()
    try:
        _native.check(lib.ms_ndimage_separable_filter(
            _native.context(), ctypes.byref(descriptor), native_passes, len(passes), ctypes.byref(options),
            ctypes.byref(result), ctypes.byref(info)))
    except (ValueError, MemoryError) as refusal:
        raise _native.NotAccelerated(f"the native filter refused the call: {refusal}") from None
    del held, keep
    _native.record_route(function, info)
    return _native.to_numpy(result)
