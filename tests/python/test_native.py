"""The native bridge: loading, array hand-off both ways, buffer reuse, errors."""
import gc
import os

import numpy as np
import pytest

import metal_sci
from metal_sci import _native


@pytest.mark.parametrize("source, target", [(np.float64, np.float32), (np.float32, np.float64),
                                            (np.complex128, np.complex64), (np.complex64, np.complex128),
                                            (np.float64, np.float64)])
def test_astype_matches_numpy_bit_for_bit(source, target):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((257, 1031)).astype(np.float64)
    x.flat[:6] = [0.0, -0.0, np.inf, -np.inf, 1e-320, 3.5e38]
    if np.issubdtype(source, np.complexfloating):
        x = x + 1j * rng.standard_normal(x.shape)
    x = x.astype(source)
    with np.errstate(over="ignore"):
        expected = x.astype(target)
    out = _native.astype(x, target)
    assert out.dtype == expected.dtype and out.shape == expected.shape
    assert out.tobytes() == expected.tobytes()


def test_results_view_native_memory_and_stay_alive_through_views():
    x = np.arange(1 << 20, dtype=np.float64)
    out = _native.astype(x, np.float32)
    assert not out.flags.owndata and out.flags.c_contiguous and out.flags.writeable
    view = out[::2]
    del out
    gc.collect()
    np.testing.assert_array_equal(view, x[::2].astype(np.float32))
    view[0] = 7.0  # results are ordinary writable arrays
    assert view[0] == 7.0


def test_released_results_are_reused_from_the_buffer_cache():
    x = np.ones(1 << 22, dtype=np.float64)
    first = _native.astype(x, np.float32)
    address = first.ctypes.data
    del first
    gc.collect()
    second = _native.astype(x, np.float32)
    assert second.ctypes.data == address
    np.testing.assert_array_equal(second, 1.0)


def test_non_contiguous_and_empty_inputs():
    x = np.arange(64, dtype=np.float64).reshape(8, 8)
    np.testing.assert_array_equal(_native.astype(x[:, ::2], np.float32), x[:, ::2].astype(np.float32))
    np.testing.assert_array_equal(_native.astype(x.T, np.float32), x.T.astype(np.float32))
    empty = _native.astype(np.empty((0, 3)), np.float32)
    assert empty.shape == (0, 3) and empty.dtype == np.float32


def test_unsupported_inputs_decline_or_raise_cleanly():
    with pytest.raises(metal_sci.NotAccelerated):
        _native.astype(np.zeros(10, dtype=np.float16), np.float32)  # float16 input
    with pytest.raises(metal_sci.NotAccelerated):
        _native.astype(np.zeros((1,) * 9), np.float32)             # rank 9
    with pytest.raises(ValueError, match="complex"):
        _native.astype(np.zeros(4), np.complex64)                   # real -> complex


_ALL_REAL_DTYPES = (
    np.bool_, np.int8, np.int16, np.int32, np.int64,
    np.uint8, np.uint16, np.uint32, np.uint64,
    np.float32, np.float64,
)


@pytest.mark.parametrize("source", _ALL_REAL_DTYPES)
@pytest.mark.parametrize("target", _ALL_REAL_DTYPES)
def test_real_astype_matches_numpy_bit_for_bit(source, target):
    if source is np.bool_:
        arr = np.array(([0, 1, 2, 255] * 64), dtype=np.uint8).view(np.bool_)
    elif np.issubdtype(source, np.floating):
        arr = np.array([-3.7, -1.5, -0.0, 0.0, 0.25, 1.5, 42.9, 127.0, 255.0], dtype=source)
    else:
        info = np.iinfo(source)
        lo = max(info.min, -1000)
        hi = min(info.max, 1000)
        vals = [0, 1, lo, hi, -1 if info.min < 0 else 255]
        if info.bits >= 32:
            vals += [(1 << 24) - 1, 1 << 24, (1 << 24) + 1]
        if info.bits == 64:
            if source is np.int64:
                vals += [(1 << 62) + (1 << 38) + 1, -(1 << 60) - 7, info.min, info.max]
            else:
                vals += [(1 << 63) + (1 << 39) + 1, (1 << 60) + 7, info.max]
        arr = np.array(vals, dtype=source)

    expected = arr.astype(target)
    out = _native.astype(arr, target)
    assert out.dtype == expected.dtype and out.shape == expected.shape
    assert out.tobytes() == expected.tobytes()


def test_integer_astype_corner_cases():
    # Signed int8(-1) -> uint64 sign-extends to UINT64_MAX.
    neg1 = np.array([-1], dtype=np.int8)
    assert _native.astype(neg1, np.uint64).tobytes() == neg1.astype(np.uint64).tobytes()

    # Single rounding for int64 -> float32 (not double-rounded via float64).
    i64_tie_plus = np.array([(1 << 62) + (1 << 38) + 1], dtype=np.int64)
    got_f32 = _native.astype(i64_tie_plus, np.float32)
    exp_f32 = i64_tie_plus.astype(np.float32)
    assert got_f32.tobytes() == exp_f32.tobytes()
    assert got_f32[0] != np.float32(np.float64(i64_tie_plus[0]))


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs fork")
def test_forked_child_falls_through_instead_of_touching_metal():
    _native.astype(np.zeros(8), np.float32)  # make sure the parent context exists
    read, write = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(read)
        try:
            _native.astype(np.zeros(8), np.float32)
            os.write(write, b"ran")
        except metal_sci.NotAccelerated:
            os.write(write, b"declined")
        except BaseException as error:  # pragma: no cover
            os.write(write, repr(error).encode())
        finally:
            os.close(write)
            os._exit(0)
    os.close(write)
    try:
        data = os.read(read, 64)
    finally:
        os.close(read)
    _, status = os.waitpid(pid, 0)
    assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0
    assert data == b"declined"


def test_as_native_rejects_non_ndarray_and_handles_unaligned_and_longdouble():
    class FakeArray:
        dtype = np.dtype(np.float64)
        shape = (4,)

    with pytest.raises(metal_sci.NotAccelerated):
        _native.as_native(FakeArray())

    assert _native.native_dtype(np.longdouble) is None
    assert _native.native_dtype(np.clongdouble) is None
    with pytest.raises(metal_sci.NotAccelerated):
        _native.astype(np.ones(4, dtype=np.longdouble), np.float32)

    # Unaligned float64 view must be repacked cleanly and match NumPy bit-for-bit.
    raw = bytearray(40)
    unaligned = np.frombuffer(raw, dtype=np.float64, count=4, offset=1)
    assert not unaligned.flags.aligned
    unaligned[:] = [1.5, -2.25, 3.75, 4.0]
    out = _native.astype(unaligned, np.float32)
    assert out.tobytes() == unaligned.astype(np.float32).tobytes()


def test_precision_warning_as_error_does_not_leak_native_buffers():
    import ctypes
    import warnings

    if not _native.has_metal():
        pytest.skip("requires Metal GPU build")

    class MemoryInfo(ctypes.Structure):
        _fields_ = [
            ("struct_size", ctypes.c_uint32),
            ("abi_version", ctypes.c_uint32),
            ("limit_bytes", ctypes.c_uint64),
            ("owned_bytes", ctypes.c_uint64),
            ("borrowed_bytes", ctypes.c_uint64),
            ("reserved_bytes", ctypes.c_uint64),
            ("cached_bytes", ctypes.c_uint64),
            ("scratch_bytes", ctypes.c_uint64),
            ("external_bytes", ctypes.c_uint64),
            ("total_bytes", ctypes.c_uint64),
            ("peak_bytes", ctypes.c_uint64),
            ("allocation_count", ctypes.c_uint64),
            ("external_accounting_complete", ctypes.c_uint32),
            ("reserved", ctypes.c_uint32),
        ]

    lib = _native.library()
    lib.ms_context_memory_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(MemoryInfo)]
    lib.ms_context_memory_info.restype = ctypes.c_int32

    def owned_count():
        gc.collect()
        info = MemoryInfo(struct_size=ctypes.sizeof(MemoryInfo), abi_version=_native.MS_RUNTIME_ABI_VERSION)
        assert lib.ms_context_memory_info(_native.context(), ctypes.byref(info)) == _native.MS_OK
        return info.owned_bytes, info.allocation_count

    import metal_sci.fft as mfft
    import metal_sci.ndimage as mn
    import metal_sci.signal as msig

    # Warm up context and confirm clean baseline.
    _native.astype(np.ones(4, dtype=np.float64), np.float32)
    before = owned_count()

    prev_small = metal_sci._config.small_to_scipy()
    metal_sci.set_small_inputs("native")
    try:
        with metal_sci.precision("fast"), metal_sci.device("gpu"):
            with warnings.catch_warnings():
                warnings.simplefilter("error", metal_sci.PrecisionWarning)
                with pytest.raises(metal_sci.PrecisionWarning):
                    mn.gaussian_filter(np.ones((16, 16), dtype=np.float64), 1.0)
                with pytest.raises(metal_sci.PrecisionWarning):
                    mfft.fft(np.ones(64, dtype=np.complex128))
                with pytest.raises(metal_sci.PrecisionWarning):
                    msig.fftconvolve(np.ones(64, dtype=np.float64), np.ones(16, dtype=np.float64))
    finally:
        metal_sci.set_small_inputs("scipy" if prev_small else "native")

    assert owned_count() == before


def test_interpreter_shutdown_with_live_native_buffer_and_invalid_env():
    import pathlib
    import subprocess
    import sys

    root = pathlib.Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    if not os.environ.get("METAL_SCI_TEST_INSTALLED"):
        pp = str(root / "python")
        env["PYTHONPATH"] = os.pathsep.join([pp, env["PYTHONPATH"]]) if env.get("PYTHONPATH") else pp

    # Keep a live native buffer until interpreter teardown; stderr must stay clean.
    proc = subprocess.run(
        [sys.executable, "-c",
         "import numpy as np; from metal_sci import _native; live = _native.astype(np.ones(128), np.float32)"],
        cwd=str(root), env=env, capture_output=True, text=True,
    )
    assert proc.returncode == 0 and proc.stderr == "", proc.stdout + proc.stderr

    # Invalid METAL_SCI_SMALL_INPUTS raises ValueError at import.
    bad_small = dict(env, METAL_SCI_SMALL_INPUTS="bogus")
    p_small = subprocess.run(
        [sys.executable, "-c", "import metal_sci"],
        cwd=str(root), env=bad_small, capture_output=True, text=True,
    )
    assert p_small.returncode != 0 and "ValueError" in p_small.stderr

    # Invalid METAL_SCI_MEMORY_LIMIT_GB raises ValueError on context creation.
    bad_mem = dict(env, METAL_SCI_MEMORY_LIMIT_GB="-5")
    p_mem = subprocess.run(
        [sys.executable, "-c", "import numpy as np; from metal_sci import _native; _native.astype(np.ones(4), np.float32)"],
        cwd=str(root), env=bad_mem, capture_output=True, text=True,
    )
    assert p_mem.returncode != 0 and "ValueError" in p_mem.stderr
