"""ctypes prototypes and calls for the native FFT (include/ms_fft.h)."""
from __future__ import annotations

import ctypes

import numpy as np

from .. import _native

MS_FFT_ABI_VERSION = 1
FORWARD, INVERSE, R2C, C2R = 1, 2, 3, 4
NORMS = {None: 0, "backward": 0, "ortho": 1, "forward": 2}


class Options(ctypes.Structure):
    _fields_ = [("struct_size", ctypes.c_uint32), ("abi_version", ctypes.c_uint32),
                ("kind", ctypes.c_uint32), ("norm", ctypes.c_uint32),
                ("axis_count", ctypes.c_uint32), ("axes", ctypes.c_uint32 * _native.MS_MAX_NDIM),
                ("host_threads", ctypes.c_uint32), ("n", ctypes.c_uint64),
                ("reserved", ctypes.c_uint64 * 2)]


class PlanInfo(ctypes.Structure):
    _fields_ = [("struct_size", ctypes.c_uint32), ("abi_version", ctypes.c_uint32),
                ("status", ctypes.c_uint32), ("engine", ctypes.c_uint32),
                ("passes", ctypes.c_uint32), ("reserved", ctypes.c_uint32),
                ("device_bytes", ctypes.c_uint64), ("description", ctypes.c_char * 256)]


_lib = None
_auto_min_points = None


def library() -> ctypes.CDLL:
    """The native library with the FFT prototypes declared."""
    global _lib
    if _lib is None:
        lib = _native.library()
        lib.ms_fft_transform.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(_native.Array), ctypes.POINTER(Options),
            ctypes.POINTER(_native.RouteOptions), ctypes.POINTER(_native.ArrayResult),
            ctypes.POINTER(_native.RouteInfo)]
        lib.ms_fft_plan.argtypes = [
            ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(Options), ctypes.POINTER(_native.RouteOptions), ctypes.c_uint32,
            ctypes.POINTER(PlanInfo)]
        lib.ms_fft_auto_min_points.argtypes = []
        lib.ms_fft_auto_min_points.restype = ctypes.c_uint64
        _lib = lib
    return _lib


def auto_min_points() -> int:
    """Inputs with fewer points never reach the GPU under the 'auto' device policy."""
    global _auto_min_points
    if _auto_min_points is None:
        _auto_min_points = int(library().ms_fft_auto_min_points())
    return _auto_min_points


def options(kind: int, norm: int, axes, n: int = 0, host_threads: int = 1) -> Options:
    result = Options(struct_size=ctypes.sizeof(Options), abi_version=MS_FFT_ABI_VERSION, kind=kind,
                     norm=norm, axis_count=len(axes), host_threads=host_threads, n=n)
    for index, axis in enumerate(axes):
        result.axes[index] = axis
    return result


def transform(x: np.ndarray, fft_options: Options, function: str) -> np.ndarray:
    """Runs one native transform; raises NotAccelerated when the core declines it."""
    lib = library()
    descriptor, held = _native.as_native(x)
    route = _native.route_options()
    info = _native.new_route_info()
    result = _native.ArrayResult()
    _native.check(lib.ms_fft_transform(_native.context(), ctypes.byref(descriptor), ctypes.byref(fft_options),
                                       ctypes.byref(route), ctypes.byref(result), ctypes.byref(info)))
    del held
    _native.record_route(function, info, input_dtype=x.dtype)
    return _native.to_numpy(result)


def plan(dtype, shape, fft_options: Options, prepare: bool = False) -> PlanInfo:
    """How the core would run a transform of an array of this dtype and shape."""
    lib = library()
    code = _native.native_dtype(dtype)
    if code is None or not 1 <= len(shape) <= _native.MS_MAX_NDIM:
        raise _native.NotAccelerated(f"no native path for dtype {np.dtype(dtype)} with {len(shape)} dimensions")
    extents = (ctypes.c_uint64 * len(shape))(*shape)
    route = _native.route_options()
    info = PlanInfo(struct_size=ctypes.sizeof(PlanInfo), abi_version=MS_FFT_ABI_VERSION)
    _native.check(lib.ms_fft_plan(_native.context(), code, len(shape), extents, ctypes.byref(fft_options),
                                  ctypes.byref(route), int(prepare), ctypes.byref(info)))
    return info
