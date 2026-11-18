"""ctypes bindings for the E1 probe runtime (libe1gemm.dylib)."""
import ctypes as C
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


class GemmArgs(C.Structure):
    _fields_ = [
        ("A", C.c_void_p), ("a_off", C.c_size_t), ("lda", C.c_int), ("ta", C.c_int),
        ("B", C.c_void_p), ("b_off", C.c_size_t), ("ldb", C.c_int), ("tb", C.c_int),
        ("C", C.c_void_p), ("c_off", C.c_size_t), ("ldc", C.c_int),
        ("D", C.c_void_p), ("d_off", C.c_size_t), ("ldd", C.c_int),
        ("M", C.c_int), ("N", C.c_int), ("K", C.c_int),
        ("alpha", C.c_float), ("beta", C.c_float),
        ("variant", C.c_int),
    ]


def load():
    lib = C.CDLL(os.path.join(HERE, "libe1gemm.dylib"))
    lib.msg_init.argtypes = [C.c_char_p, C.c_char_p, C.c_int]
    lib.msg_init.restype = C.c_int
    lib.msg_device_name.restype = C.c_char_p
    lib.msg_device_arch.restype = C.c_char_p
    lib.msg_last_kernel.restype = C.c_char_p
    lib.msg_last_dispatch.argtypes = [C.POINTER(C.c_int)]
    lib.msg_alloc.argtypes = [C.c_size_t]
    lib.msg_alloc.restype = C.c_void_p
    lib.msg_contents.argtypes = [C.c_void_p]
    lib.msg_contents.restype = C.c_void_p
    lib.msg_length.argtypes = [C.c_void_p]
    lib.msg_length.restype = C.c_size_t
    lib.msg_free.argtypes = [C.c_void_p]
    lib.msg_gemm.argtypes = [C.POINTER(GemmArgs), C.POINTER(C.c_double)]
    lib.msg_gemm.restype = C.c_int
    lib.msg_gemm_spin.argtypes = [C.POINTER(GemmArgs), C.POINTER(C.c_double)]
    lib.msg_gemm_spin.restype = C.c_int
    lib.msg_gemm_submit.argtypes = [C.POINTER(GemmArgs)]
    lib.msg_gemm_submit.restype = C.c_void_p
    lib.msg_gemm_submit_many.argtypes = [C.POINTER(GemmArgs), C.c_int]
    lib.msg_gemm_submit_many.restype = C.c_void_p
    lib.msg_wait.argtypes = [C.c_void_p, C.POINTER(C.c_double)]
    lib.msg_wait.restype = C.c_int
    lib.msg_done.argtypes = [C.c_void_p]
    lib.msg_done.restype = C.c_int
    lib.msg_now.restype = C.c_double
    lib.msg_set_qos.argtypes = [C.c_int]
    lib.msg_set_qos.restype = C.c_int
    fp = np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS")
    dp = np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")
    lib.e1_sgemm.argtypes = [C.c_int, C.c_int, C.c_int, C.c_float, fp, C.c_int,
                             fp, C.c_int, C.c_float, fp, C.c_int]
    lib.e1_dgemm.argtypes = [C.c_int, C.c_int, C.c_int, C.c_double, dp, C.c_int,
                             dp, C.c_int, C.c_double, dp, C.c_int]
    err = C.create_string_buffer(512)
    rc = lib.msg_init(os.path.join(HERE, "steel_gemm.metallib").encode(), err, 512)
    if rc != 0:
        raise RuntimeError(f"msg_init failed ({rc}): {err.value.decode()}")
    return lib


class GpuBuf:
    """A shared-storage MTLBuffer viewed as a float32 NumPy array (zero copy)."""

    def __init__(self, lib, nelem):
        self.lib = lib
        self.h = lib.msg_alloc(nelem * 4)
        if not self.h:
            raise MemoryError("msg_alloc failed")
        ptr = lib.msg_contents(self.h)
        self.ptr = ptr
        self.np = np.ctypeslib.as_array((C.c_float * nelem).from_address(ptr))

    def free(self):
        if self.h:
            self.np = None
            self.lib.msg_free(self.h)
            self.h = None


def args(A, a_off, lda, B, b_off, ldb, D, d_off, ldd, M, N, K,
         Cb=None, c_off=0, ldc=0, alpha=1.0, beta=0.0, ta=0, tb=0, variant=0):
    return GemmArgs(A.h, a_off, lda, ta, B.h, b_off, ldb, tb,
                    Cb.h if Cb is not None else None, c_off, ldc,
                    D.h, d_off, ldd, M, N, K, alpha, beta, variant)
