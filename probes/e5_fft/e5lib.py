"""ctypes bindings for the E5 probe runtime (build/libe5fft.dylib)."""
import ctypes
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(HERE, "build")
LIB_PATH = os.path.join(BUILD, "libe5fft.dylib")
FFT_METALLIB = os.path.join(BUILD, "mlx_fft.metallib")
EXT_METALLIB = os.path.join(BUILD, "e5_ext.metallib")

_lib = None


def lib():
    global _lib
    if _lib is None:
        L = ctypes.CDLL(LIB_PATH)
        L.e5_last_error.restype = ctypes.c_char_p
        L.e5_init.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        L.e5_init.restype = ctypes.c_int
        L.e5_set_untracked.argtypes = [ctypes.c_int]
        L.e5_device_name.argtypes = [ctypes.c_char_p, ctypes.c_int]
        L.e5_pso_build_seconds.restype = ctypes.c_double
        L.e5_pso_built.restype = ctypes.c_int
        L.e5_plan_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        L.e5_plan_create.restype = ctypes.c_void_p
        L.e5_plan_create_ms.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                        ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
        L.e5_plan_create_ms.restype = ctypes.c_void_p
        L.e5_plan_destroy.argtypes = [ctypes.c_void_p]
        L.e5_plan_in.argtypes = [ctypes.c_void_p]
        L.e5_plan_in.restype = ctypes.c_void_p
        L.e5_plan_out.argtypes = [ctypes.c_void_p]
        L.e5_plan_out.restype = ctypes.c_void_p
        L.e5_plan_num_passes.argtypes = [ctypes.c_void_p]
        L.e5_plan_num_temps.argtypes = [ctypes.c_void_p]
        L.e5_plan_set_split_cb.argtypes = [ctypes.c_void_p, ctypes.c_int]
        L.e5_plan_last_gpu_seconds.argtypes = [ctypes.c_void_p]
        L.e5_plan_last_gpu_seconds.restype = ctypes.c_double
        L.e5_plan_pass_gpu_seconds.argtypes = [ctypes.c_void_p, ctypes.c_int]
        L.e5_plan_pass_gpu_seconds.restype = ctypes.c_double
        L.e5_plan_describe.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
        L.e5_plan_exec.argtypes = [ctypes.c_void_p]
        L.e5_plan_exec.restype = ctypes.c_double
        L.e5_empty_roundtrip.restype = ctypes.c_double
        L.e5_vdsp_create.argtypes = [ctypes.c_int]
        L.e5_vdsp_create.restype = ctypes.c_void_p
        L.e5_vdsp_available.argtypes = [ctypes.c_void_p, ctypes.c_int]
        L.e5_vdsp_run.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
        L.e5_vdsp_run.restype = ctypes.c_double
        L.e5_vdsp_run_batch.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                        ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
        L.e5_vdsp_run_batch.restype = ctypes.c_double
        L.e5_vdsp_destroy.argtypes = [ctypes.c_void_p]
        _lib = L
    return _lib


def last_error():
    return lib().e5_last_error().decode()


def init(fft_metallib=FFT_METALLIB, ext_metallib=EXT_METALLIB):
    if lib().e5_init(fft_metallib.encode(), ext_metallib.encode()) != 0:
        raise RuntimeError(last_error())


def device_name():
    buf = ctypes.create_string_buffer(256)
    lib().e5_device_name(buf, 256)
    return buf.value.decode()


C2C, R2C, C2R = 0, 1, 2


class Plan:
    """A pre-planned transform over pre-allocated shared MTLBuffers.

    .x is a numpy view of the input buffer, .y of the output buffer.

    factors=None  -> faithful port of MLX 0.31.1's plan ("mlx" mode)
    factors=[..]  -> SciMetal multi-step plan, n == prod(factors),
                     `coalesce` = requested FFTs per threadgroup (MLX uses 4)."""

    def __init__(self, n, rows=1, inverse=False, kind=C2C, factors=None, coalesce=4):
        self.n, self.rows, self.inverse, self.kind = n, rows, inverse, kind
        self.factors, self.coalesce = factors, coalesce
        if factors is None:
            self.h = lib().e5_plan_create(kind, rows, n, int(inverse))
        else:
            arr = (ctypes.c_int * len(factors))(*factors)
            self.h = lib().e5_plan_create_ms(kind, rows, n, int(inverse), arr, len(factors), coalesce)
        if not self.h:
            raise RuntimeError(last_error())
        half = n // 2 + 1
        if kind == C2C:
            ishape, idt, oshape, odt = (rows, n), np.complex64, (rows, n), np.complex64
        elif kind == R2C:
            ishape, idt, oshape, odt = (rows, n), np.float32, (rows, half), np.complex64
        else:
            ishape, idt, oshape, odt = (rows, half), np.complex64, (rows, n), np.float32
        self.x = self._view(lib().e5_plan_in(self.h), ishape, idt)
        self.y = self._view(lib().e5_plan_out(self.h), oshape, odt)

    @staticmethod
    def _view(ptr, shape, dtype):
        count = int(np.prod(shape))
        nbytes = count * np.dtype(dtype).itemsize
        buf = (ctypes.c_char * nbytes).from_address(ptr)
        return np.frombuffer(buf, dtype=dtype, count=count).reshape(shape)

    def exec(self):
        t = lib().e5_plan_exec(self.h)
        if t < 0:
            raise RuntimeError(last_error())
        return t

    def gpu_seconds(self):
        return lib().e5_plan_last_gpu_seconds(self.h)

    def pass_gpu_seconds(self):
        return [lib().e5_plan_pass_gpu_seconds(self.h, i) for i in range(self.num_passes)]

    def set_split_cb(self, v):
        lib().e5_plan_set_split_cb(self.h, int(v))

    @property
    def num_passes(self):
        return lib().e5_plan_num_passes(self.h)

    @property
    def num_temps(self):
        return lib().e5_plan_num_temps(self.h)

    def describe(self):
        buf = ctypes.create_string_buffer(16384)
        lib().e5_plan_describe(self.h, buf, 16384)
        return json.loads(buf.value.decode())

    def close(self):
        if self.h:
            self.x = self.y = None
            lib().e5_plan_destroy(self.h)
            self.h = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
