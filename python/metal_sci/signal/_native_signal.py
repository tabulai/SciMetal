"""ctypes prototypes and calls for the signal family (include/ms_signal.h)."""
from __future__ import annotations

import ctypes
import threading

import numpy as np

from .. import _native

MS_SIGNAL_ABI_VERSION = 1

FULL, SAME, VALID = 0, 1, 2
METHOD_AUTO, METHOD_DIRECT, METHOD_FFT = 0, 1, 2
BOUNDARY_FILL, BOUNDARY_WRAP, BOUNDARY_SYMM = 0, 1, 2
CENTER_ND, CENTER_2D_CORR = 0, 1

SIGNAL_ROUTE_FFT = 1
SIGNAL_ROUTE_NONFINITE = 2

MODES = {"full": FULL, "same": SAME, "valid": VALID}
METHODS = {"auto": METHOD_AUTO, "direct": METHOD_DIRECT, "fft": METHOD_FFT}
BOUNDARIES = {
    "fill": BOUNDARY_FILL,
    "pad": BOUNDARY_FILL,
    "wrap": BOUNDARY_WRAP,
    "circular": BOUNDARY_WRAP,
    "symm": BOUNDARY_SYMM,
    "symmetric": BOUNDARY_SYMM,
}


class ConvOptions(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("mode", ctypes.c_uint32),
        ("method", ctypes.c_uint32),
        ("correlate", ctypes.c_uint32),
        ("boundary", ctypes.c_uint32),
        ("center", ctypes.c_uint32),
        ("host_threads", ctypes.c_uint32),
        ("fillvalue_real", ctypes.c_double),
        ("fillvalue_imag", ctypes.c_double),
        ("reserved", ctypes.c_uint64 * 2),
    ]


class FftConvOptions(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("mode", ctypes.c_uint32),
        ("correlate", ctypes.c_uint32),
        ("overlap_add", ctypes.c_uint32),
        ("axis_count", ctypes.c_uint32),
        ("axes", ctypes.c_uint32 * _native.MS_MAX_NDIM),
        ("host_threads", ctypes.c_uint32),
        ("reserved0", ctypes.c_uint32),
        ("reserved", ctypes.c_uint64 * 2),
    ]


_lock = threading.Lock()
_declared = False


def library() -> ctypes.CDLL:
    """The native library with the signal prototypes declared."""
    global _declared
    lib = _native.library()
    if not _declared:
        with _lock:
            if not _declared:
                if not hasattr(lib, "ms_signal_convolve"):
                    raise _native.NotAccelerated("the native library was built without the signal family")
                lib.ms_signal_convolve.argtypes = [
                    ctypes.c_void_p,
                    ctypes.POINTER(_native.Array),
                    ctypes.POINTER(_native.Array),
                    ctypes.POINTER(ConvOptions),
                    ctypes.POINTER(_native.RouteOptions),
                    ctypes.POINTER(_native.ArrayResult),
                    ctypes.POINTER(_native.RouteInfo),
                ]
                lib.ms_signal_fftconvolve.argtypes = [
                    ctypes.c_void_p,
                    ctypes.POINTER(_native.Array),
                    ctypes.POINTER(_native.Array),
                    ctypes.POINTER(FftConvOptions),
                    ctypes.POINTER(_native.RouteOptions),
                    ctypes.POINTER(_native.ArrayResult),
                    ctypes.POINTER(_native.RouteInfo),
                ]
                lib.ms_signal_choose_conv_method.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_uint32,
                    ctypes.c_uint32,
                    ctypes.POINTER(ctypes.c_uint64),
                    ctypes.c_uint32,
                    ctypes.c_uint32,
                    ctypes.POINTER(ctypes.c_uint64),
                    ctypes.c_uint32,
                    ctypes.c_uint32,
                    ctypes.POINTER(_native.RouteOptions),
                    ctypes.POINTER(ctypes.c_uint32),
                ]
                _declared = True
    return lib


def convolve(function: str, in1: np.ndarray, in2: np.ndarray, mode: int, method: int,
             correlate: bool, boundary: int, center: int,
             fill_real: float = 0.0, fill_imag: float = 0.0, host_threads: int = 1,
             return_route_flags: bool = False) -> np.ndarray | tuple[np.ndarray, int]:
    lib = library()
    d1, h1 = _native.as_native(in1)
    d2, h2 = _native.as_native(in2)
    opts = ConvOptions(
        struct_size=ctypes.sizeof(ConvOptions),
        abi_version=MS_SIGNAL_ABI_VERSION,
        mode=mode,
        method=method,
        correlate=1 if correlate else 0,
        boundary=boundary,
        center=center,
        host_threads=host_threads,
        fillvalue_real=fill_real,
        fillvalue_imag=fill_imag,
    )
    route = _native.route_options()
    info = _native.new_route_info()
    result = _native.ArrayResult()
    try:
        _native.check(lib.ms_signal_convolve(
            _native.context(), ctypes.byref(d1), ctypes.byref(d2), ctypes.byref(opts),
            ctypes.byref(route), ctypes.byref(result), ctypes.byref(info)))
    except (ValueError, MemoryError) as refusal:
        raise _native.NotAccelerated(f"the native signal route refused the call: {refusal}") from None
    del h1, h2
    promoted = np.result_type(in1.dtype, in2.dtype)
    _native.record_route(function, info, input_dtype=promoted)
    out = _native.to_numpy(result)
    if return_route_flags:
        return out, int(info.reserved[0])
    return out


def fftconvolve(function: str, in1: np.ndarray, in2: np.ndarray, mode: int,
                correlate: bool, overlap_add: bool, axes, host_threads: int = 1) -> np.ndarray:
    lib = library()
    d1, h1 = _native.as_native(in1)
    d2, h2 = _native.as_native(in2)
    opts = FftConvOptions(
        struct_size=ctypes.sizeof(FftConvOptions),
        abi_version=MS_SIGNAL_ABI_VERSION,
        mode=mode,
        correlate=1 if correlate else 0,
        overlap_add=1 if overlap_add else 0,
        axis_count=0 if axes is None else len(axes),
        host_threads=host_threads,
    )
    if axes is not None:
        for i, ax in enumerate(axes):
            opts.axes[i] = ax
    route = _native.route_options()
    info = _native.new_route_info()
    result = _native.ArrayResult()
    try:
        _native.check(lib.ms_signal_fftconvolve(
            _native.context(), ctypes.byref(d1), ctypes.byref(d2), ctypes.byref(opts),
            ctypes.byref(route), ctypes.byref(result), ctypes.byref(info)))
    except (ValueError, MemoryError) as refusal:
        raise _native.NotAccelerated(f"the native signal fft route refused the call: {refusal}") from None
    del h1, h2
    promoted = np.result_type(in1.dtype, in2.dtype)
    _native.record_route(function, info, input_dtype=promoted)
    return _native.to_numpy(result)


def choose_conv_method(dtype1, shape1, dtype2, shape2, mode: int, host_threads: int = 1) -> int:
    lib = library()
    c1 = _native.native_dtype(dtype1)
    c2 = _native.native_dtype(dtype2)
    if c1 is None or c2 is None or len(shape1) != len(shape2) or not 1 <= len(shape1) <= _native.MS_MAX_NDIM:
        raise _native.NotAccelerated("unsupported dtype or rank for native choose_conv_method")
    s1_arr = (ctypes.c_uint64 * len(shape1))(*shape1)
    s2_arr = (ctypes.c_uint64 * len(shape2))(*shape2)
    route = _native.route_options()
    out_method = ctypes.c_uint32()
    try:
        _native.check(lib.ms_signal_choose_conv_method(
            _native.context(), c1, len(shape1), s1_arr, c2, len(shape2), s2_arr,
            mode, host_threads, ctypes.byref(route), ctypes.byref(out_method)))
    except (ValueError, MemoryError) as refusal:
        raise _native.NotAccelerated(f"native choose_conv_method refused: {refusal}") from None
    return int(out_method.value)
