"""ctypes bindings for the E7 probe (libe7.dylib)."""
import ctypes as C
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load():
    lib = C.CDLL(os.path.join(HERE, "libe7.dylib"))
    ip = np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS")
    out = C.POINTER(C.c_double)
    for sfx, dt in (("d", np.float64), ("s", np.float32)):
        m = np.ctypeslib.ndpointer(dt, flags="F_CONTIGUOUS")
        getattr(lib, f"e7_getrf_lapack_{sfx}").argtypes = [C.c_int, m, C.c_int, ip, out]
        getattr(lib, f"e7_getrf_rec_{sfx}").argtypes = [C.c_int, C.c_int, m, C.c_int, ip, out]
        getattr(lib, f"e7_getrf_blk_{sfx}").argtypes = [C.c_int, m, C.c_int, ip, C.c_int, out]
        getattr(lib, f"e7_potrf_lapack_{sfx}").argtypes = [C.c_int, m, C.c_int, out]
        getattr(lib, f"e7_potrf_rec_{sfx}").argtypes = [C.c_int, m, C.c_int, out]
        getattr(lib, f"e7_potrf_blk_{sfx}").argtypes = [C.c_int, m, C.c_int, C.c_int, out]
        f = getattr(lib, f"e7_lu_resid_{sfx}")
        f.argtypes = [C.c_int, m, C.c_int, m, C.c_int, ip]
        f.restype = C.c_double
        f = getattr(lib, f"e7_chol_resid_{sfx}")
        f.argtypes = [C.c_int, m, C.c_int, m, C.c_int]
        f.restype = C.c_double
        f = getattr(lib, f"e7_lu_resid_probe_{sfx}")
        f.argtypes = [C.c_int, m, C.c_int, m, C.c_int, ip, C.c_int]
        f.restype = C.c_double
        f = getattr(lib, f"e7_chol_resid_probe_{sfx}")
        f.argtypes = [C.c_int, m, C.c_int, m, C.c_int, C.c_int]
        f.restype = C.c_double
        getattr(lib, f"e7_gemm_{sfx}").argtypes = [C.c_int, m, m, m, out]
        getattr(lib, f"e7_trsm_{sfx}").argtypes = [C.c_int, m, m, out]
        getattr(lib, f"e7_syrk_{sfx}").argtypes = [C.c_int, m, m, out]
        getattr(lib, f"e7_gemm_shape_{sfx}").argtypes = [C.c_int, C.c_int, C.c_int, C.c_int, out]
    lib.e7_set_opts.argtypes = [C.c_int] * 6
    lib.e7_set_swap_par.argtypes = [C.c_int]
    lib.e7_get_stats.argtypes = [out]
    lib.e7_get_gemm_levels.argtypes = [out]
    lib.e7_thread_count.restype = C.c_int
    return lib


def stats(lib):
    s = (C.c_double * 10)()
    lib.e7_get_stats(s)
    return {"gemm_s": s[0], "trsm_s": s[1], "swap_s": s[2], "leaf_s": s[3], "syrk_s": s[4],
            "n_gemm": int(s[5]), "n_trsm": int(s[6]), "n_swap": int(s[7]),
            "n_leaf": int(s[8]), "n_syrk": int(s[9])}


def gemm_levels(lib):
    """GEMM time and achieved GFLOPS bucketed by floor(log2(K))."""
    s = (C.c_double * 40)()
    lib.e7_get_gemm_levels(s)
    return {f"K>={1 << i}": {"ms": s[i] * 1e3, "gflops": (s[20 + i] / s[i] / 1e9) if s[i] > 0 else 0.0,
                             "flops": s[20 + i]}
            for i in range(20) if s[20 + i] > 0}
