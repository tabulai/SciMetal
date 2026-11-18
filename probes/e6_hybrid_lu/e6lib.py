"""ctypes bindings for the E6 probe (libe6.dylib)."""
import ctypes as C
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


class Opts(C.Structure):
    _fields_ = [("nb", C.c_int), ("cpu_tail", C.c_int), ("lookahead", C.c_int),
                ("panel_leaf", C.c_int), ("par_resid", C.c_int), ("max_iter", C.c_int),
                ("tol", C.c_double), ("gpu", C.c_int), ("cpu_parts", C.c_int),
                ("par_trsv", C.c_int), ("left_swaps", C.c_int)]


class Stats(C.Structure):
    _fields_ = [(k, C.c_double) for k in (
        "t_narrow", "t_factor", "t_solve0", "t_refine", "t_total",
        "t_panel", "t_swap", "t_trsm", "t_lookahead_gemm", "t_submit", "t_wait", "t_gpu_exec",
        "t_tail", "t_left_swap", "t_resid", "t_trsv", "normA")] + [
        ("n_gpu", C.c_int), ("n_panels", C.c_int), ("iters", C.c_int), ("info", C.c_int),
        ("berr", C.c_double * 40), ("n_berr", C.c_int)]

    def as_dict(self):
        d = {k: getattr(self, k) for k, _ in self._fields_ if k not in ("berr", "n_berr")}
        d["berr"] = [self.berr[i] for i in range(self.n_berr)]
        return d


def load():
    lib = C.CDLL(os.path.join(HERE, "libe6.dylib"))
    lib.msg_init.argtypes = [C.c_char_p, C.c_char_p, C.c_int]
    err = C.create_string_buffer(512)
    rc = lib.msg_init(os.path.join(HERE, "steel_gemm.metallib").encode(), err, 512)
    if rc != 0:
        raise RuntimeError(f"msg_init failed ({rc}): {err.value.decode()}")
    dC = np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")
    dF = np.ctypeslib.ndpointer(np.float64, flags="F_CONTIGUOUS")
    sF = np.ctypeslib.ndpointer(np.float32, flags="F_CONTIGUOUS")
    dv = np.ctypeslib.ndpointer(np.float64, ndim=1)
    sv = np.ctypeslib.ndpointer(np.float32, ndim=1)
    iv = np.ctypeslib.ndpointer(np.int32, ndim=1)
    lib.e6_create.argtypes = [C.c_int]
    lib.e6_create.restype = C.c_void_p
    lib.e6_destroy.argtypes = [C.c_void_p]
    lib.e6_buffer.argtypes = [C.c_void_p]
    lib.e6_buffer.restype = C.POINTER(C.c_float)
    lib.e6_ipiv.argtypes = [C.c_void_p]
    lib.e6_ipiv.restype = C.POINTER(C.c_int)
    lib.e6_solve.argtypes = [C.c_void_p, C.c_int, dC, dv, dv, C.POINTER(Opts), C.POINTER(Stats)]
    lib.e6_solve.restype = C.c_int
    lib.e6_factor.argtypes = [C.c_void_p, C.c_int, C.POINTER(Opts), C.POINTER(Stats)]
    lib.e6_factor.restype = C.c_int
    lib.e6_solve_only.argtypes = [C.c_void_p, C.c_int, dv, dv, C.POINTER(Opts)]
    lib.e6_solve_only.restype = C.c_double
    lib.e6_resid_only.argtypes = [C.c_int, dC, dv, dv, dv, C.c_int]
    lib.e6_resid_only.restype = C.c_double
    lib.e6_dgesv.argtypes = [C.c_int, dF, dv, iv, C.POINTER(C.c_double)]
    lib.e6_sgesv.argtypes = [C.c_int, sF, sv, iv, C.POINTER(C.c_double)]
    lib.e6_dgetrf_dgetrs.argtypes = [C.c_int, dF, dv, iv, C.POINTER(C.c_double)]
    lib.e6_sgetrf_sgetrs.argtypes = [C.c_int, sF, sv, iv, C.POINTER(C.c_double)]
    lib.e6_dsgesv.argtypes = [C.c_int, dF, dv, dv, iv, C.POINTER(C.c_int), C.POINTER(C.c_double)]
    lib.e6_gen_randsvd.argtypes = [C.c_int, C.c_double, C.c_ulonglong, dC]
    lib.msg_device_name.restype = C.c_char_p
    return lib


def default_opts(nb=256, cpu_tail=1024, lookahead=1, panel_leaf=8, par_resid=1,
                 max_iter=30, tol=4 * 1.1102230246251565e-16, gpu=1, cpu_parts=1,
                 par_trsv=1, left_swaps=0):
    return Opts(nb, cpu_tail, lookahead, panel_leaf, par_resid, max_iter, tol, gpu, cpu_parts,
                par_trsv, left_swaps)
