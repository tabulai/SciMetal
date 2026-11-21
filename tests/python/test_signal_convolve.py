"""Tests for scimetal.signal convolution and correlation routes."""
from __future__ import annotations

import inspect
import os
import pathlib
import subprocess
import sys
import threading
import warnings

import numpy as np
import pytest
import scipy.signal as ss
from hypothesis import given, settings
from hypothesis import strategies as st

import scimetal
import scimetal.signal as msig
from scimetal.signal._signaltools import _verified_scipy_fir

FUNCTIONS = (
    "convolve",
    "correlate",
    "fftconvolve",
    "oaconvolve",
    "convolve2d",
    "correlate2d",
    "choose_conv_method",
)

# Skip if native library lacks signal family.
with scimetal.precision("strict"), scimetal.device("cpu"):
    msig.convolve(np.ones(4), np.ones(2), method="direct")
    if scimetal.last_route().engine != "cpu":
        pytest.skip(f"no native signal routes: {scimetal.last_route().reason}", allow_module_level=True)


@pytest.fixture(scope="module")
def gpu():
    with scimetal.device("gpu"), scimetal.precision("fast"):
        msig.fftconvolve(np.ones(64, dtype=np.float32), np.ones(17, dtype=np.float32))
        if scimetal.last_route().engine != "gpu":
            pytest.skip(f"no GPU engine: {scimetal.last_route().reason}")


def rand_arr(shape, dtype, seed=0):
    rng = np.random.default_rng(seed)
    dt = np.dtype(dtype)
    real_dt = np.float32 if dt in (np.float32, np.complex64) else np.float64
    x = rng.standard_normal(shape).astype(real_dt)
    if dt.kind == "c":
        x = (x + 1j * rng.standard_normal(shape).astype(real_dt)).astype(dt)
    return np.ascontiguousarray(x)


def assert_close_direct(ours, ref, in1, in2, engine, cval=0.0):
    assert ours.dtype == ref.dtype and ours.shape == ref.shape
    if engine == "cpu":
        tol = 1e-11 if ref.dtype in (np.float64, np.complex128) else 1e-5
        scale = max(1.0, float(np.abs(ref).max()))
        assert np.abs(ours - ref).max() <= tol * scale
    else:
        u = 2.0 ** -24
        m = max(float(np.abs(in1).max()), float(abs(cval)))
        k = float(np.abs(in2).max())
        taps = min(in1.size, in2.size) if in1.size != in2.size else in2.size
        bound = 4.0 * u * max(in1.size, in2.size, taps) * m * k + 1e-5
        # Tighter bound: u * taps * m * k (with small factor for rounding/complex ops).
        tight = 4.0 * u * in2.size * m * k + 1e-5
        assert np.abs(ours.astype(np.complex128) - ref.astype(np.complex128)).max() <= max(tight, 1e-5 * float(np.abs(ref).max()))


def assert_close_fft(ours, ref, in1, in2):
    assert ours.dtype == ref.dtype and ours.shape == ref.shape
    u = 2.0 ** -24
    scale = max(float(np.abs(ref).max()), float(np.linalg.norm(in1) * np.linalg.norm(in2)), 1.0)
    err = float(np.abs(ours.astype(np.complex128) - ref.astype(np.complex128)).max())
    assert err <= 128.0 * u * scale + 1e-5


def test_wrappers_keep_scipys_signatures_and_registry():
    for name in FUNCTIONS:
        ours, theirs = getattr(msig, name), getattr(ss, name)
        assert ours is not theirs and callable(ours.__scimetal_native__)
        assert inspect.signature(ours) == inspect.signature(theirs)
        assert ours.__doc__ == theirs.__doc__ and ours.__name__ == name
        assert name in dir(msig)
    assert msig.lfilter is ss.lfilter


@pytest.mark.parametrize("dtype1,dtype2", [
    (np.float64, np.float64),
    (np.float32, np.float32),
    (np.complex128, np.complex128),
    (np.complex64, np.complex64),
    (np.float32, np.float64),
    (np.float32, np.complex64),
    (np.float64, np.complex64),
])
@pytest.mark.parametrize("mode", ["full", "same", "valid"])
@pytest.mark.parametrize("func", ["convolve", "correlate"])
def test_direct_cpu_modes_and_dtypes(func, mode, dtype1, dtype2):
    a = rand_arr((13, 17), dtype1, seed=1)
    b = rand_arr((5, 6), dtype2, seed=2)
    with scimetal.precision("strict"), scimetal.device("cpu"):
        ours = getattr(msig, func)(a, b, mode=mode, method="direct")
        route = scimetal.last_route()
        assert route.engine == "cpu"
    ref = getattr(ss, func)(a, b, mode=mode, method="direct")
    assert_close_direct(ours, ref, a, b, "cpu")
    if mode == "valid":
        # Swapped valid mode.
        with scimetal.precision("strict"), scimetal.device("cpu"):
            ours_sw = getattr(msig, func)(b, a, mode="valid", method="direct")
            assert scimetal.last_route().engine == "cpu"
        ref_sw = getattr(ss, func)(b, a, mode="valid", method="direct")
        assert_close_direct(ours_sw, ref_sw, b, a, "cpu")


def test_convolve2d_correlate2d_even_center_and_boundaries(gpu):
    a = rand_arr((24, 30), np.float64, seed=3)
    b = rand_arr((4, 6), np.float64, seed=4)
    for boundary in ("fill", "pad", "wrap", "circular", "symm", "symmetric"):
        for mode in ("full", "same", "valid"):
            fill = -1.25 if boundary in ("fill", "pad") else 123.0
            for func in ("convolve2d", "correlate2d"):
                with scimetal.precision("fast"), scimetal.device("cpu"):
                    ours = getattr(msig, func)(a, b, mode=mode, boundary=boundary, fillvalue=fill)
                    assert scimetal.last_route().engine == "cpu"
                ref = getattr(ss, func)(a, b, mode=mode, boundary=boundary, fillvalue=fill)
                np.testing.assert_allclose(ours, ref, rtol=1e-12, atol=1e-12)

                with scimetal.precision("fast"), scimetal.device("gpu"):
                    ours_g = getattr(msig, func)(a, b, mode=mode, boundary=boundary, fillvalue=fill)
                    assert scimetal.last_route().engine == "gpu"
                assert_close_direct(ours_g, ref, a, b, "gpu", cval=fill if boundary in ("fill", "pad") else 0.0)


@pytest.mark.parametrize("device", ["auto", "cpu", "gpu"])
def test_strict_convolve2d_cancellation(device):
    image = np.ones((128, 128), dtype=np.float64)
    kernel = np.array([[1e16, -1e16, 1.0]])
    with scimetal.precision("strict"), scimetal.device(device):
        got = msig.convolve2d(image, kernel, mode="valid")
        engine = "cpu" if _verified_scipy_fir() and device != "gpu" else "scipy"
        assert scimetal.last_route().engine == engine
    expected = ss.convolve2d(image, kernel, mode="valid")
    assert got.tobytes() == expected.tobytes()


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
@pytest.mark.parametrize("mode,boundary", [("full", "symm"), ("same", "fill"), ("valid", "wrap")])
def test_strict_convolve2d_matches_scipy_bits(dtype, mode, boundary):
    image = rand_arr((37, 31), dtype, seed=112)[:, ::-1]
    kernel = rand_arr((4, 7), dtype, seed=24)
    with scimetal.precision("strict"):
        got = msig.convolve2d(image, kernel, mode=mode, boundary=boundary, fillvalue=-1.75)
    expected = ss.convolve2d(image, kernel, mode=mode, boundary=boundary, fillvalue=-1.75)
    assert (got.shape, got.dtype) == (expected.shape, expected.dtype)
    assert got.tobytes() == expected.tobytes()


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("width", [1, 2, 3, 4, 5, 7, 8, 9, 12])
def test_strict_native_convolve2d_groups_reset_per_row(dtype, width):
    if not _verified_scipy_fir():
        pytest.skip("requires the verified SciPy FIR build")
    overhang = width == 12
    image = rand_arr((2, 7) if overhang else (21, 33), dtype, seed=112)[:, ::-1]
    kernel = rand_arr((3, width), dtype, seed=24)[::-1]
    with scimetal.precision("strict"), scimetal.device("auto"):
        for mode in ("full", "same", "valid"):
            for boundary in ("fill", "wrap", "symm"):
                # Exercise valid mode's operand swap. The other modes retain
                # input order even when the kernel overhangs the image.
                a, b = (kernel, image) if mode == "valid" and not overhang else (image, kernel)
                got = msig.convolve2d(a, b, mode=mode, boundary=boundary, fillvalue=-1.75)
                assert scimetal.last_route().engine == "cpu"
                expected = ss.convolve2d(a, b, mode=mode, boundary=boundary, fillvalue=-1.75)
                assert got.shape == expected.shape and got.dtype == expected.dtype
                assert got.tobytes() == expected.tobytes()


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_strict_native_convolve2d_signed_zeros_and_subnormals(dtype):
    if not _verified_scipy_fir():
        pytest.skip("requires the verified SciPy FIR build")
    tiny = np.nextafter(dtype(0), dtype(1))
    image = np.tile(np.array([tiny, -tiny, -0.0, 0.0], dtype=dtype), (12, 21))
    for weights in ([[0.1]], [[0.1, -0.0, 1.5, -0.5, 0.2]]):
        kernel = np.array(weights, dtype=dtype)
        with scimetal.precision("strict"):
            got = msig.convolve2d(image, kernel, mode="same", fillvalue=-0.0)
        assert scimetal.last_route().engine == "cpu"
        expected = ss.convolve2d(image, kernel, mode="same", fillvalue=-0.0)
        assert got.tobytes() == expected.tobytes()


@pytest.mark.parametrize("image_value,kernel_value", [
    (1e308, 2.0),
    (np.array(0xFFF8000000001234, dtype=np.uint64).view(np.float64).item(), 1.0),
])
def test_strict_convolve2d_falls_back_for_nonfinite_results(image_value, kernel_value):
    image = np.full((64, 65), image_value)
    kernel = np.full((3, 5), kernel_value)
    with scimetal.precision("strict"):
        got = msig.convolve2d(image, kernel, mode="same")
    route = scimetal.last_route()
    assert route.engine == "scipy"
    if _verified_scipy_fir():
        assert "nonfinite" in route.reason
    expected = ss.convolve2d(image, kernel, mode="same")
    assert got.tobytes() == expected.tobytes()


@pytest.mark.parametrize("extension_exists", [True, False])
def test_strict_convolve2d_falls_back_for_unknown_scipy_fir(monkeypatch, tmp_path, extension_exists):
    from scipy.signal import _sigtools

    extension = tmp_path / "_sigtools.so"
    if extension_exists:
        extension.write_bytes(b"an unverified _sigtools build")
    image = rand_arr((63, 65), np.float64)
    kernel = rand_arr((5, 5), np.float64, seed=4)
    _verified_scipy_fir.cache_clear()
    try:
        with monkeypatch.context() as patch, scimetal.precision("strict"), scimetal.device("auto"):
            patch.setattr(_sigtools, "__file__", str(extension))
            got = msig.convolve2d(image, kernel, mode="same")
    finally:
        _verified_scipy_fir.cache_clear()
    assert scimetal.last_route().engine == "scipy"
    assert "verified SciPy FIR binary" in scimetal.last_route().reason
    assert got.tobytes() == ss.convolve2d(image, kernel, mode="same").tobytes()


def test_correlate2d_complex_swapped_valid_parity():
    small = rand_arr((3, 4), np.complex128, seed=5)
    big = rand_arr((7, 8), np.complex128, seed=6)
    with scimetal.precision("strict"):
        ours = msig.correlate2d(small, big, mode="valid")
        ref = ss.correlate2d(small, big, mode="valid")
        np.testing.assert_array_equal(ours, ref)


def test_prune_zeros_false_preserves_nan():
    a = np.array([1.0, np.nan, 2.0, 3.0], dtype=np.float64)
    b = np.array([1.0, 0.0], dtype=np.float64)
    with scimetal.precision("strict"):
        ours = msig.convolve(a, b, mode="same", method="direct")
        ref = ss.convolve(a, b, mode="same", method="direct")
    np.testing.assert_array_equal(np.isnan(ours), np.isnan(ref))


@pytest.mark.parametrize("func", ["fftconvolve", "oaconvolve"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
@pytest.mark.parametrize("mode", ["full", "same", "valid"])
def test_freq_convolves_gpu(gpu, func, dtype, mode):
    s1 = (256, 33) if func == "oaconvolve" else (37, 45)
    s2 = (11, 5) if func == "oaconvolve" else (12, 15)
    a = rand_arr(s1, dtype, seed=10)
    b = rand_arr(s2, dtype, seed=11)
    with scimetal.precision("fast"), scimetal.device("gpu"):
        ours = getattr(msig, func)(a, b, mode=mode)
        route = scimetal.last_route()
        assert route.engine == "gpu"
    ref = getattr(ss, func)(a, b, mode=mode)
    assert_close_fft(ours, ref, a, b)


def test_freq_convolves_broadcast_and_off_last_axes(gpu):
    a = rand_arr((3, 1, 64, 17), np.float64, seed=20)
    b = rand_arr((1, 4, 19, 17), np.float64, seed=21)
    with scimetal.precision("fast"), scimetal.device("gpu"):
        ours = msig.fftconvolve(a, b, mode="same", axes=(2,))
        assert scimetal.last_route().engine == "gpu"
    ref = ss.fftconvolve(a, b, mode="same", axes=(2,))
    assert_close_fft(ours, ref, a, b)

    # Multi-axis OA with non-contiguous axes and broadcasting.
    a2 = rand_arr((2, 120, 1, 90), np.complex128, seed=22)
    b2 = rand_arr((2, 11, 5, 9), np.complex128, seed=23)
    with scimetal.precision("fast"), scimetal.device("gpu"):
        ours2 = msig.oaconvolve(a2, b2, mode="full", axes=(-1, 1))
        assert scimetal.last_route().engine == "gpu"
    ref2 = ss.oaconvolve(a2, b2, mode="full", axes=(-1, 1))
    assert_close_fft(ours2, ref2, a2, b2)


def test_convolve_method_fft_nonfinite_warning(gpu):
    a = rand_arr(1000, np.float32, seed=30)
    a[100] = np.nan
    b = rand_arr(128, np.float32, seed=31)
    with scimetal.precision("fast"), scimetal.device("gpu"):
        with pytest.warns(RuntimeWarning, match="NAN or inf"):
            out = msig.convolve(a, b, mode="same", method="fft")
            assert scimetal.last_route().engine == "gpu"
            assert np.isnan(out.flat[0])

    # Float64 NaN declines to SciPy and also warns.
    a64 = a.astype(np.float64)
    b64 = b.astype(np.float64)
    with scimetal.precision("fast"), scimetal.device("auto"):
        with pytest.warns(RuntimeWarning, match="NAN or inf"):
            out64 = msig.convolve(a64, b64, mode="same", method="fft")
            assert scimetal.last_route().engine == "scipy"
            assert np.isnan(out64.flat[0])


def test_a5_overflow_guard_rails_and_explain(gpu):
    # Direct path overflow: re-routes to CPU with reason.
    a = np.ones((2048, 2048), dtype=np.float64)
    a[5, 7] = 1e39
    b = np.ones((7, 7), dtype=np.float64)
    with scimetal.precision("fast"), scimetal.device("auto"):
        out = msig.convolve(a, b, mode="same", method="direct")
        route = scimetal.last_route()
        assert route.engine == "cpu" and route.compute_dtype == "float64"
        assert "overflow" in route.reason or "single-precision" in route.reason
        assert np.all(np.isfinite(out))
        assert out[5, 7] >= 1e39

    # FFT path magnitude overflow and product overflow: declines to SciPy cleanly.
    x1 = np.ones(1 << 17, dtype=np.float64) * 1e20
    x2 = np.ones(1025, dtype=np.float64) * 1e20
    with scimetal.precision("fast"), scimetal.device("auto"):
        out_prod = msig.fftconvolve(x1, x2, mode="valid")
        route = scimetal.last_route()
        assert route.engine == "scipy" and np.all(np.isfinite(out_prod))
        assert out_prod[0] == pytest.approx(1025.0 * 1e40, rel=1e-6)

    x_huge = np.ones(1 << 16, dtype=np.float64) * 1e30
    x_one = np.ones(128, dtype=np.float64)
    with scimetal.precision("fast"), scimetal.device("auto"):
        out_h = msig.oaconvolve(x_huge, x_one, mode="same")
        assert scimetal.last_route().engine == "scipy"
        assert np.all(np.isfinite(out_h))


def test_precision_warning_emitted_once_for_signal_gpu(gpu):
    code = (
        "import warnings, numpy as np, scimetal, scimetal.signal as msig\n"
        "scimetal.set_small_inputs('native')\n"
        "scimetal.set_precision('fast')\n"
        "scimetal.set_device('gpu')\n"
        "a = np.ones(256, dtype=np.float64)\n"
        "b = np.ones(32, dtype=np.float64)\n"
        "with warnings.catch_warnings(record=True) as w:\n"
        "    for _ in range(2):\n"
        "        msig.fftconvolve(a, b)\n"
        "    msig.fftconvolve(a, b)\n"
        "    pw = [x for x in w if issubclass(x.category, scimetal.PrecisionWarning)]\n"
        "    assert len(pw) == 1, len(pw)\n"
        "    msg = str(pw[0].message)\n"
        "    assert 'set_precision' in msg and 'strict' in msg\n"
    )
    root = pathlib.Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    if not os.environ.get("SCIMETAL_TEST_INSTALLED"):
        python_dir = str(root / "python")
        existing_pp = env.get("PYTHONPATH")
        env["PYTHONPATH"] = os.pathsep.join([python_dir, existing_pp]) if existing_pp else python_dir
    proc = subprocess.run([sys.executable, "-c", code], cwd=str(root), env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_threaded_fft_fallback_context(monkeypatch):
    import scipy.fft as sfft

    seen_workers = []
    orig_set_workers = sfft.set_workers

    def spy_set_workers(workers):
        seen_workers.append(workers)
        return orig_set_workers(workers)

    monkeypatch.setattr(sfft, "set_workers", spy_set_workers)
    a = np.ones(1 << 17, dtype=np.float64)
    b = np.ones(64, dtype=np.float64)

    monkeypatch.delenv("SCIMETAL_FFT_THREADED_FALLBACK", raising=False)
    with scimetal.precision("strict"):
        out1 = msig.fftconvolve(a, b, mode="valid")
        assert scimetal.last_route().engine == "scipy"
    assert seen_workers == [scimetal._native.performance_cores()]
    assert out1.shape == (len(a) - len(b) + 1,)

    seen_workers.clear()
    monkeypatch.setenv("SCIMETAL_FFT_THREADED_FALLBACK", "0")
    with scimetal.precision("strict"):
        msig.fftconvolve(a, b, mode="valid")
        assert scimetal.last_route().engine == "scipy"
    assert seen_workers == []


def assert_declined_like_scipy(func, *args, **kwargs):
    before = list(scimetal._dispatch.counters[f"signal.{func}"])
    try:
        expected = getattr(ss, func)(*args, **kwargs)
        err = None
    except Exception as exc:
        expected = None
        err = exc
    if err is None:
        got = getattr(msig, func)(*args, **kwargs)
        np.testing.assert_array_equal(got, expected)
    else:
        with pytest.raises(type(err)):
            getattr(msig, func)(*args, **kwargs)
    assert scimetal.last_route().engine == "scipy"
    after = scimetal._dispatch.counters[f"signal.{func}"]
    assert after == [before[0], before[1] + 1]


def test_decline_paths():
    a = rand_arr((8, 10), np.float64, seed=40)
    b = rand_arr((3, 4), np.float64, seed=41)
    for bad in (
        a.astype(np.int32),
        a.astype(np.uint8),
        a > 0,
        a.astype(np.float16),
        a.astype(np.longdouble),
        a.astype(">f8"),
        a.tolist(),
        np.ma.masked_array(a),
        np.empty((0, 4)),
        np.float64(2.0),
        np.zeros((2,) * 9),
    ):
        for func in ("convolve", "correlate", "fftconvolve", "oaconvolve"):
            assert_declined_like_scipy(func, bad, b if np.asarray(bad).ndim == 2 else np.ones(3))
    unaligned = np.frombuffer(b"\0" + a.tobytes(), dtype=np.float64, offset=1).reshape(a.shape)
    assert not unaligned.flags.aligned
    assert_declined_like_scipy("convolve", unaligned, b)

    # Invalid arguments decline to SciPy so SciPy raises its own errors.
    assert_declined_like_scipy("convolve", a, b, mode="bogus")
    assert_declined_like_scipy("convolve", a, b, method="bogus")
    assert_declined_like_scipy("fftconvolve", a, b, axes=[])
    assert_declined_like_scipy("fftconvolve", a, b, axes=[0, 0])
    assert_declined_like_scipy("convolve2d", a, b, boundary="reflect")
    assert_declined_like_scipy("convolve2d", a, b, fillvalue=1j)
    assert_declined_like_scipy("convolve2d", a, b, fillvalue=[1, 2])
    assert_declined_like_scipy("convolve2d", a, b, fillvalue=[])
    assert_declined_like_scipy("choose_conv_method", a.astype(np.int64), b.astype(np.int64))


def test_choose_conv_method(gpu):
    small1 = rand_arr((12, 12), np.float64, seed=50)
    small2 = rand_arr((4, 4), np.float64, seed=51)
    big1 = rand_arr((1024, 1024), np.float64, seed=52)
    big2 = rand_arr((256, 256), np.float64, seed=53)
    with scimetal.precision("fast"):
        assert msig.choose_conv_method(small1, small2) == "direct"
        assert msig.choose_conv_method(big1, big2) == "fft"
        m, times = msig.choose_conv_method(small1, small2, measure=True)
        assert m in ("fft", "direct") and set(times) == {"fft", "direct"}


@st.composite
def random_signal_case(draw):
    ndim = draw(st.integers(1, 3))
    s1 = tuple(draw(st.lists(st.integers(2, 18), min_size=ndim, max_size=ndim)))
    s2 = tuple(draw(st.lists(st.integers(2, 12), min_size=ndim, max_size=ndim)))
    mode = draw(st.sampled_from(["full", "same", "valid"]))
    if mode == "valid":
        # Ensure one dominates.
        if draw(st.booleans()):
            s1 = tuple(max(a, b) for a, b in zip(s1, s2))
        else:
            s2 = tuple(max(a, b) for a, b in zip(s1, s2))
    dtype = draw(st.sampled_from([np.float32, np.float64, np.complex64, np.complex128]))
    return s1, s2, mode, dtype


@settings(max_examples=80, deadline=None, database=None)
@given(random_signal_case(), st.integers(0, 10000))
def test_property_strict_and_gpu(gpu, drawn, seed):
    s1, s2, mode, dtype = drawn
    a = rand_arr(s1, dtype, seed=seed)
    b = rand_arr(s2, dtype, seed=seed + 1)

    with scimetal.precision("strict"), scimetal.device("cpu"):
        for func in ("convolve", "correlate"):
            ours = getattr(msig, func)(a, b, mode=mode, method="direct")
            assert scimetal.last_route().engine == "cpu"
            ref = getattr(ss, func)(a, b, mode=mode, method="direct")
            assert_close_direct(ours, ref, a, b, "cpu")

    with scimetal.precision("fast"), scimetal.device("gpu"):
        for func in ("fftconvolve", "oaconvolve"):
            ours = getattr(msig, func)(a, b, mode=mode)
            assert scimetal.last_route().engine == "gpu"
            ref = getattr(ss, func)(a, b, mode=mode)
            assert_close_fft(ours, ref, a, b)


def test_thread_safety(gpu):
    a = rand_arr((256, 256), np.float64, seed=60)
    b = rand_arr((15, 15), np.float64, seed=61)
    ref = ss.fftconvolve(a, b, mode="same")
    failures = []

    def worker(dev):
        try:
            with scimetal.precision("fast"), scimetal.device(dev):
                for _ in range(10):
                    if dev == "gpu":
                        out = msig.fftconvolve(a, b, mode="same")
                        assert scimetal.last_route().engine == "gpu"
                        assert_close_fft(out, ref, a, b)
                    else:
                        out = msig.convolve2d(a, b, mode="same")
                        assert scimetal.last_route().engine == "cpu"
                        np.testing.assert_allclose(out, ref, rtol=1e-11, atol=1e-11)
        except BaseException as exc:
            failures.append(exc)

    threads = [threading.Thread(target=worker, args=("gpu" if i % 2 == 0 else "cpu",)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not failures, failures


def test_mixed_precision_fillvalue_overflow_guard_rails(gpu):
    # Bug 1: single-precision input + double-precision kernel + large fillvalue.
    z_f32 = np.zeros((2048, 2048), dtype=np.float32)
    k_f64 = np.ones((7, 7), dtype=np.float64) * 1e5
    with scimetal.precision("fast"), scimetal.device("auto"):
        out = msig.convolve2d(z_f32, k_f64, mode="same", boundary="fill", fillvalue=1e38)
        route = scimetal.last_route()
        assert route.engine == "cpu" and route.compute_dtype == "float64"
        assert route.reason
        assert np.all(np.isfinite(out)) and out.dtype == np.float64
        assert out[0, 0] == pytest.approx(3.3e44, rel=1e-12)
        assert out[0, 10] == pytest.approx(2.1e44, rel=1e-12)

    z_small = np.zeros((32, 32), dtype=np.float32)
    with scimetal.precision("fast"), scimetal.device("gpu"):
        out_g = msig.convolve2d(z_small, k_f64, mode="same", boundary="fill", fillvalue=1e38)
        assert scimetal.last_route().engine == "scipy"
        assert np.all(np.isfinite(out_g))
        np.testing.assert_allclose(out_g, ss.convolve2d(z_small, k_f64, mode="same", boundary="fill", fillvalue=1e38))

    # Complex single-precision input + complex128 kernel + fillvalue=1e38j / 1e38.
    z_c64 = np.zeros((2500, 2500), dtype=np.complex64)
    k_c128 = np.ones((5, 5), dtype=np.complex128) * 1e5
    for fv in (1e38, 1e38j):
        with scimetal.precision("fast"), scimetal.device("auto"):
            out_c = msig.convolve2d(z_c64, k_c128, mode="same", boundary="fill", fillvalue=fv)
            route = scimetal.last_route()
            assert route.engine == "cpu" and route.compute_dtype == "complex128"
            assert route.reason and np.all(np.isfinite(out_c))

        with scimetal.precision("fast"), scimetal.device("gpu"):
            out_cg = msig.convolve2d(z_c64[:32, :32], k_c128, mode="same", boundary="fill", fillvalue=fv)
            assert scimetal.last_route().engine == "scipy" and np.all(np.isfinite(out_cg))

    # Non-zero f32 input + 1e20 f64 kernel + 1e20 fillvalue under forced gpu declines cleanly.
    o_f32 = np.ones((32, 32), dtype=np.float32)
    k_1e20 = np.ones((5, 5), dtype=np.float64) * 1e20
    with scimetal.precision("fast"), scimetal.device("gpu"):
        out_o = msig.convolve2d(o_f32, k_1e20, mode="same", boundary="fill", fillvalue=1e20)
        assert scimetal.last_route().engine == "scipy" and np.all(np.isfinite(out_o))

    # Intentional Inf/NaN fillvalue still runs on GPU and propagates as expected.
    k_small = np.ones((3, 3), dtype=np.float64)
    with scimetal.precision("fast"), scimetal.device("gpu"), np.errstate(invalid="ignore"):
        out_inf = msig.convolve2d(o_f32, k_small, mode="same", boundary="fill", fillvalue=np.inf)
        assert scimetal.last_route().engine == "gpu"
        assert np.isinf(out_inf[0, 0]) and np.isfinite(out_inf[2, 2])


def test_real_in_complex_out_zero_kernel_fillvalue_overflow(gpu):
    # Bug 2: real_in_complex_out with zero kernel and fillvalue=1e39/1e39j must not produce NaN.
    k_zero = np.zeros((3, 3), dtype=np.complex128)
    r64_small = np.ones((8, 8), dtype=np.float64)
    with scimetal.precision("fast"), scimetal.device("gpu"):
        out = msig.convolve2d(r64_small, k_zero, mode="same", boundary="fill", fillvalue=1e39)
        assert scimetal.last_route().engine == "scipy"
        np.testing.assert_array_equal(out, np.zeros((8, 8), dtype=np.complex128))

    r64_big = np.ones((2048, 2048), dtype=np.float64)
    with scimetal.precision("fast"), scimetal.device("auto"):
        out2 = msig.convolve2d(r64_big, k_zero, mode="same", boundary="fill", fillvalue=1e39)
        assert scimetal.last_route().engine == "cpu"
        assert not np.isnan(out2).any() and np.all(out2 == 0)

    r32_small = np.ones((16, 16), dtype=np.float32)
    for fv in (1e39, 1e39j):
        with scimetal.precision("fast"), scimetal.device("gpu"):
            out3 = msig.convolve2d(r32_small, k_zero, mode="same", boundary="fill", fillvalue=fv)
            assert scimetal.last_route().engine == "scipy"
            np.testing.assert_array_equal(out3, np.zeros((16, 16), dtype=np.complex128))

    r32_big = np.ones((2048, 2048), dtype=np.float32)
    with scimetal.precision("fast"), scimetal.device("auto"):
        out4 = msig.convolve2d(r32_big, k_zero, mode="same", boundary="fill", fillvalue=1e39j)
        assert scimetal.last_route().engine == "cpu"
        assert not np.isnan(out4).any() and np.all(out4 == 0)

    # Large enough work (>= 128 MiB) to enter GPU decision and A5-reroute to CPU with reason.
    k_zero7 = np.zeros((7, 7), dtype=np.complex128)
    with scimetal.precision("fast"), scimetal.device("auto"):
        out4b = msig.convolve2d(r64_big, k_zero7, mode="same", boundary="fill", fillvalue=1e39)
        r4b = scimetal.last_route()
        assert r4b.engine == "cpu" and r4b.reason and not np.isnan(out4b).any() and np.all(out4b == 0)

    # rev_conj_out path (swapped real valid correlate2d) with zero kernel and fillvalue=1e39.
    with scimetal.precision("fast"), scimetal.device("gpu"):
        out5 = msig.correlate2d(np.ones((4, 4), np.float64), np.zeros((8, 8), np.float64),
                                mode="valid", boundary="fill", fillvalue=1e39)
        assert scimetal.last_route().engine == "scipy"
        np.testing.assert_array_equal(out5, np.zeros((5, 5), dtype=np.float64))


def test_convolve_valid_method_fft_uses_full_est(gpu, monkeypatch):
    # Bug 3: convolve(..., mode='valid', method='fft') uses full_est for break-even check.
    monkeypatch.setattr(scimetal._config, "_small_to_scipy", True)
    a = rand_arr(50000, np.float32, seed=70)
    b = rand_arr(50000, np.float32, seed=71)
    with scimetal.precision("fast"), scimetal.device("auto"):
        out = msig.convolve(a, b, mode="valid", method="fft")
        assert scimetal.last_route().engine == "gpu"
        ref = ss.convolve(a, b, mode="valid", method="fft")
        assert_close_fft(out, ref, a, b)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
@pytest.mark.parametrize("mode", ["full", "same"])
def test_1d_direct_nonfinite_kernel_omits_oob_matches_scipy(dtype, mode):
    """Bug B: 1-D signal.convolve/correlate omits out-of-bounds taps matching SciPy/NumPy."""
    a = np.ones((12,), dtype=dtype)
    b_inf = np.array([1.0, 1.0, 1.0, 1.0, np.inf], dtype=dtype)
    a0 = np.zeros((12,), dtype=dtype)
    b_nan = np.array([0.0, 0.0, 0.0, 0.0, np.nan], dtype=dtype)

    for policy, dev in (("strict", "cpu"), ("fast", "cpu"), ("fast", "auto"), ("fast", "gpu")):
        with scimetal.precision(policy), scimetal.device(dev):
            for fn_name in ("convolve", "correlate"):
                got_inf = getattr(msig, fn_name)(a, b_inf, mode=mode, method="direct")
                ref_inf = getattr(ss, fn_name)(a, b_inf, mode=mode, method="direct")
                np.testing.assert_array_equal(np.isnan(got_inf), np.isnan(ref_inf))
                np.testing.assert_array_equal(np.isinf(got_inf), np.isinf(ref_inf))
                fin = np.isfinite(ref_inf)
                np.testing.assert_allclose(got_inf[fin], ref_inf[fin])

                got_nan = getattr(msig, fn_name)(a0, b_nan, mode=mode, method="direct")
                ref_nan = getattr(ss, fn_name)(a0, b_nan, mode=mode, method="direct")
                np.testing.assert_array_equal(np.isnan(got_nan), np.isnan(ref_nan))

    # Swapped 1-D full (len(in1) < len(in2)) and 1-D same (len(in1) < len(in2)) also match SciPy.
    a_sm = np.ones((3,), dtype=dtype)
    for fn_name in ("convolve", "correlate"):
        for m in ("full", "same"):
            with scimetal.precision("strict"), scimetal.device("cpu"):
                got_sm = getattr(msig, fn_name)(a_sm, b_inf, mode=m, method="direct")
                ref_sm = getattr(ss, fn_name)(a_sm, b_inf, mode=m, method="direct")
                np.testing.assert_array_equal(np.isnan(got_sm), np.isnan(ref_sm))
                np.testing.assert_array_equal(np.isinf(got_sm), np.isinf(ref_sm))
                fin = np.isfinite(ref_sm)
                np.testing.assert_allclose(got_sm[fin], ref_sm[fin])

    # In-bounds 0.0 * NaN = NaN is preserved when NaN is in in1, in valid mode, and in convolve2d.
    a_in1_nan = np.array([1.0, np.nan, 0.0, 0.0, 0.0, 0.0], dtype=dtype)
    b0 = np.zeros((4,), dtype=dtype)
    with scimetal.precision("strict"), scimetal.device("cpu"):
        for fn_name in ("convolve", "correlate"):
            for m in ("full", "same", "valid"):
                got_i1 = getattr(msig, fn_name)(a_in1_nan, b0, mode=m, method="direct")
                ref_i1 = getattr(ss, fn_name)(a_in1_nan, b0, mode=m, method="direct")
                np.testing.assert_array_equal(np.isnan(got_i1), np.isnan(ref_i1))

        a2 = np.ones((4, 4), dtype=dtype)
        b2_inf = np.array([[1.0, np.inf]], dtype=dtype)
        got2d = msig.convolve2d(a2, b2_inf, mode="full", boundary="fill", fillvalue=0.0)
        ref2d = ss.convolve2d(a2, b2_inf, mode="full", boundary="fill", fillvalue=0.0)
        np.testing.assert_array_equal(np.isnan(got2d), np.isnan(ref2d))
        np.testing.assert_array_equal(np.isinf(got2d), np.isinf(ref2d))


def test_1d_direct_nonfinite_kernel_large_gpu_reroute(gpu):
    """Large 1-D signal.convolve with non-finite kernel re-routes AUTO to CPU and declines forced GPU."""
    a_big = np.ones((200000,), dtype=np.float32)
    b_inf = np.ones((1000,), dtype=np.float32)
    b_inf[-1] = np.inf
    ref = ss.convolve(a_big, b_inf, mode="full", method="direct")
    with scimetal.precision("fast"), scimetal.device("auto"):
        got = msig.convolve(a_big, b_inf, mode="full", method="direct")
        route = scimetal.last_route()
        assert route.engine == "cpu" and route.reason
        np.testing.assert_array_equal(np.isnan(got), np.isnan(ref))
        np.testing.assert_array_equal(np.isinf(got), np.isinf(ref))
        np.testing.assert_allclose(got[:999], ref[:999])
    with scimetal.precision("fast"), scimetal.device("gpu"):
        got_g = msig.convolve(a_big, b_inf, mode="full", method="direct")
        assert scimetal.last_route().engine == "scipy"
        np.testing.assert_array_equal(np.isnan(got_g), np.isnan(ref))
        np.testing.assert_array_equal(np.isinf(got_g), np.isinf(ref))
        np.testing.assert_allclose(got_g[:999], ref[:999])


def _assert_complex_nan_inf_parity(got, ref):
    assert got.dtype == ref.dtype and got.shape == ref.shape
    np.testing.assert_array_equal(np.isnan(got.real), np.isnan(ref.real))
    np.testing.assert_array_equal(np.isnan(got.imag), np.isnan(ref.imag))
    np.testing.assert_array_equal(np.isinf(got.real), np.isinf(ref.real))
    np.testing.assert_array_equal(np.isinf(got.imag), np.isinf(ref.imag))
    fin_re = np.isfinite(ref.real)
    fin_im = np.isfinite(ref.imag)
    if np.any(fin_re):
        np.testing.assert_allclose(got.real[fin_re], ref.real[fin_re], rtol=1e-5, atol=1e-5)
    if np.any(fin_im):
        np.testing.assert_allclose(got.imag[fin_im], ref.imag[fin_im], rtol=1e-5, atol=1e-5)


def _run_mixed_real_complex_nan_matrix(policy, dev, big_n, side2d):
    dtype_pairs = [(np.float32, np.complex128), (np.float64, np.complex64), (np.float32, np.complex64)]
    nf_complex = [complex(0.0, np.nan), complex(np.inf, 0.0), complex(1.0, np.inf), complex(np.inf, np.inf)]
    ck_kernels = [0 + 1j, 1 + 0j, 1 + 1j]

    with scimetal.precision(policy), scimetal.device(dev):
        for dt_r, dt_c in dtype_pairs:
            for nf in nf_complex:
                a = np.ones((5,), dtype=dt_r)
                b = np.ones((big_n,), dtype=dt_c)
                b[0] = nf
                for mode in ("full", "same", "valid"):
                    for fn_name in ("convolve", "correlate"):
                        # Swapped real + complex (len(a) < len(b) triggers swap for full & valid).
                        g_sw = getattr(msig, fn_name)(a, b, mode=mode, method="direct")
                        if dev != "auto":
                            assert scimetal.last_route().engine == dev
                        r_sw = getattr(ss, fn_name)(a, b, mode=mode, method="direct")
                        _assert_complex_nan_inf_parity(g_sw, r_sw)

                        # Unswapped complex + real.
                        g_un = getattr(msig, fn_name)(b, a, mode=mode, method="direct")
                        if dev != "auto":
                            assert scimetal.last_route().engine == dev
                        r_un = getattr(ss, fn_name)(b, a, mode=mode, method="direct")
                        _assert_complex_nan_inf_parity(g_un, r_un)

            for ck in ck_kernels:
                a_r = np.ones((big_n,), dtype=dt_r)
                a_r[0] = np.inf
                b_c = np.ones((5,), dtype=dt_c)
                b_c[0] = ck
                for mode in ("full", "same", "valid"):
                    for fn_name in ("convolve", "correlate"):
                        g_rc = getattr(msig, fn_name)(a_r, b_c, mode=mode, method="direct")
                        if dev != "auto":
                            assert scimetal.last_route().engine == dev
                        r_rc = getattr(ss, fn_name)(a_r, b_c, mode=mode, method="direct")
                        _assert_complex_nan_inf_parity(g_rc, r_rc)

        # 2-D convolve / correlate (all 4 non-finite patterns) & convolve2d / correlate2d.
        b2r = np.ones((3, 3), dtype=np.float32)
        for nf in nf_complex:
            a2c = np.ones((side2d, side2d), dtype=np.complex64)
            a2c[1, 1] = nf
            for mode in ("full", "same", "valid"):
                for fn_name in ("convolve", "correlate"):
                    g2 = getattr(msig, fn_name)(a2c, b2r, mode=mode, method="direct")
                    if dev != "auto":
                        assert scimetal.last_route().engine == dev
                    r2 = getattr(ss, fn_name)(a2c, b2r, mode=mode, method="direct")
                    _assert_complex_nan_inf_parity(g2, r2)
                # Note: SciPy's C _firfilter.cc uses GCC __complex__ (__mulsc3) which recovers
                # Inf+Infj on (inf+infj)*(1+0j), whereas NumPy/SciPy signal.convolve yields NaN+NaNj.
                if nf != complex(np.inf, np.inf):
                    for fn2d in ("convolve2d", "correlate2d"):
                        g2d = getattr(msig, fn2d)(a2c, b2r, mode=mode, boundary="fill", fillvalue=0.0)
                        if policy == "strict" and fn2d == "convolve2d":
                            assert scimetal.last_route().engine == "scipy"
                        elif dev != "auto":
                            assert scimetal.last_route().engine == dev
                        r2d = getattr(ss, fn2d)(a2c, b2r, mode=mode, boundary="fill", fillvalue=0.0)
                        _assert_complex_nan_inf_parity(g2d, r2d)

        for ck in ck_kernels:
            a2r = np.ones((side2d, side2d), dtype=np.float32)
            a2r[1, 1] = np.inf
            b2c = np.ones((3, 3), dtype=np.complex64)
            b2c[0, 0] = ck
            for mode in ("full", "same", "valid"):
                for fn2d in ("convolve2d", "correlate2d"):
                    g2d = getattr(msig, fn2d)(a2r, b2c, mode=mode, boundary="fill", fillvalue=0.0)
                    if policy == "strict" and fn2d == "convolve2d":
                        assert scimetal.last_route().engine == "scipy"
                    elif dev != "auto":
                        assert scimetal.last_route().engine == dev
                    r2d = getattr(ss, fn2d)(a2r, b2c, mode=mode, boundary="fill", fillvalue=0.0)
                    _assert_complex_nan_inf_parity(g2d, r2d)


def test_mixed_real_complex_nan_semantics_cpu():
    """Mixed real+complex 0.0 cross-term NaN parity on strict CPU (1-D and 2-D)."""
    _run_mixed_real_complex_nan_matrix("strict", "cpu", big_n=12, side2d=8)


def test_mixed_real_complex_nan_semantics_gpu(gpu):
    """Mixed real+complex 0.0 cross-term NaN parity on fast GPU and auto (150k 1-D and 2-D)."""
    _run_mixed_real_complex_nan_matrix("fast", "gpu", big_n=150000, side2d=128)
    _run_mixed_real_complex_nan_matrix("fast", "auto", big_n=150000, side2d=128)


@pytest.mark.parametrize("dtype1,dtype2", [
    (np.float32, np.float32),
    (np.float64, np.float64),
    (np.complex64, np.complex64),
    (np.complex128, np.complex128),
    (np.float32, np.complex64),
])
def test_convolve2d_correlate2d_full_no_swap_nan_inf_parity(gpu, dtype1, dtype2):
    """Bug 1: convolve2d/correlate2d mode='full' never swaps s2 > s1 (matching SciPy _firfilter.cc)."""
    routes = [("strict", "cpu", (3, 3), (8, 8)), ("fast", "gpu", (3, 3), (70, 70)), ("fast", "auto", (3, 3), (70, 70))]
    shape_pairs = [
        None,  # uses (3, 3) x (8, 8) or (3, 3) x (70, 70)
        ((3, 10), (10, 3)),
        ((3, 8), (9, 4)),
    ]
    for policy, dev, s1_def, s2_def in routes:
        for sp_pair in shape_pairs:
            s1, s2 = (s1_def, s2_def) if sp_pair is None else sp_pair
            for fill in (0.0, 2.5):
                for nf_in_in1 in (True, False):
                    for nf_val in (np.nan, np.inf):
                        a = np.ones(s1, dtype=dtype1)
                        b = np.ones(s2, dtype=dtype2)
                        if nf_in_in1:
                            a[s1[0] // 2, s1[1] // 2] = nf_val
                        else:
                            b[s2[0] // 2, s2[1] // 2] = nf_val
                        with scimetal.precision(policy), scimetal.device(dev):
                            for fn_name in ("convolve2d", "correlate2d"):
                                got = getattr(msig, fn_name)(a, b, mode="full", boundary="fill", fillvalue=fill)
                                if policy == "strict" and fn_name == "convolve2d":
                                    assert scimetal.last_route().engine == "scipy"
                                elif dev != "auto":
                                    assert scimetal.last_route().engine == dev
                                ref = getattr(ss, fn_name)(a, b, mode="full", boundary="fill", fillvalue=fill)
                                _assert_complex_nan_inf_parity(got, ref) if got.dtype.kind == "c" else (
                                    np.testing.assert_array_equal(np.isnan(got), np.isnan(ref)),
                                    np.testing.assert_array_equal(np.isinf(got), np.isinf(ref)),
                                    np.testing.assert_allclose(got[np.isfinite(ref)], ref[np.isfinite(ref)], rtol=1e-4, atol=1e-4),
                                )

    # Complex fillvalue with complex inputs when s2 > s1.
    ac = np.ones((3, 3), dtype=np.complex64)
    ac[1, 1] = np.nan
    bc = np.ones((70, 70), dtype=np.complex64)
    with scimetal.precision("fast"), scimetal.device("gpu"):
        for fn_name in ("convolve2d", "correlate2d"):
            got = getattr(msig, fn_name)(ac, bc, mode="full", boundary="fill", fillvalue=1.5 + 2.5j)
            assert scimetal.last_route().engine == "gpu"
            ref = getattr(ss, fn_name)(ac, bc, mode="full", boundary="fill", fillvalue=1.5 + 2.5j)
            _assert_complex_nan_inf_parity(got, ref)

    # Confirm 2-D convolve/correlate(..., method='direct') STILL swaps when s2 > s1 (matching SciPy).
    a_nd = np.ones((3, 3), dtype=np.float32)
    a_nd[1, 1] = np.nan
    b_nd = np.ones((70, 70), dtype=np.float32)
    with scimetal.precision("fast"), scimetal.device("gpu"):
        for fn_name in ("convolve", "correlate"):
            got = getattr(msig, fn_name)(a_nd, b_nd, mode="full", method="direct")
            assert scimetal.last_route().engine == "gpu"
            ref = getattr(ss, fn_name)(a_nd, b_nd, mode="full", method="direct")
            np.testing.assert_array_equal(np.isnan(got), np.isnan(ref))


@pytest.mark.parametrize("dtype", [np.float32, np.complex64])
@pytest.mark.parametrize("nf_val", [np.nan, np.inf])
def test_convolve_method_fft_nonfinite_past_block0_and_oa_parity(gpu, dtype, nf_val):
    """Bug 2: method='fft' disables OA (propagating NaN/Inf globally) and warns on any non-finite output."""
    x = np.ones(40000, dtype=dtype)
    x[30000] = nf_val
    k = np.ones(100, dtype=dtype)

    for fn_name in ("convolve", "correlate"):
        with scimetal.precision("fast"), scimetal.device("gpu"):
            with pytest.warns(RuntimeWarning, match="NAN or inf"):
                got = getattr(msig, fn_name)(x, k, mode="full", method="fft")
            assert scimetal.last_route().engine == "gpu"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            ref = getattr(ss, fn_name)(x, k, mode="full", method="fft")
        np.testing.assert_array_equal(np.isnan(got), np.isnan(ref))
        np.testing.assert_array_equal(np.isfinite(got), np.isfinite(ref))
        assert not np.isfinite(got).any()

    # oaconvolve keeps OA block isolation (block boundaries depend on backend next_fast_len) without RuntimeWarning.
    with scimetal.precision("fast"), scimetal.device("gpu"):
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            got_oa = msig.oaconvolve(x, k, mode="full")
        assert not any(issubclass(w.category, RuntimeWarning) for w in rec)
        assert scimetal.last_route().engine == "gpu"
    ref_oa = ss.oaconvolve(x, k, mode="full")
    assert np.isfinite(got_oa[0]) and np.isfinite(ref_oa[0])
    assert np.isnan(got_oa[30000]) and np.isnan(ref_oa[30000])
    assert np.allclose(got_oa[:1000], ref_oa[:1000], rtol=1e-4, atol=1e-4)
    assert np.allclose(got_oa[-1000:], ref_oa[-1000:], rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape1,shape2,dtype", [
    ((40000,), (1000,), np.float32),
    ((50000,), (2000,), np.float32),
    ((100000,), (1000,), np.float32),
    ((100000,), (1000,), np.complex64),
    ((256, 256), (16, 16), np.complex64),
    ((256, 256), (32, 32), np.float32),
])
@pytest.mark.parametrize("nf_val", [np.nan, np.inf])
@pytest.mark.parametrize("pos_kind", ["start", "middle", "end"])
def test_convolve_correlate_method_auto_nonfinite_warning_and_full_propagation(gpu, shape1, shape2, dtype, nf_val, pos_kind):
    """Iter 6: method='auto' routing to GPU FFT warns and propagates NaN/Inf globally (1-D & 2-D)."""
    x = np.ones(shape1, dtype=dtype)
    k = np.ones(shape2, dtype=dtype)
    if pos_kind == "start":
        x.flat[0] = nf_val
    elif pos_kind == "middle":
        x.flat[x.size // 2] = nf_val
    else:
        x.flat[-1] = nf_val

    for dev in ("gpu", "auto"):
        for fn_name in ("convolve", "correlate"):
            with scimetal.precision("fast"), scimetal.device(dev):
                assert msig.choose_conv_method(x, k, mode="full") == "fft"
                with pytest.warns(RuntimeWarning, match="NAN or inf"):
                    got = getattr(msig, fn_name)(x, k, mode="full", method="auto")
                assert scimetal.last_route().engine == "gpu"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                ref = getattr(ss, fn_name)(x, k, mode="full", method="auto")
            assert not np.isfinite(got).any()
            np.testing.assert_array_equal(np.isnan(got), np.isnan(ref))
            np.testing.assert_array_equal(np.isfinite(got), np.isfinite(ref))


def test_convolve_correlate_method_auto_direct_and_finite_no_warning(gpu):
    """method='auto' routed to direct (with NaN) or to FFT (with finite inputs) must not warn."""
    # Direct route with NaN (1-D 2000 x 5 and 2-D float32 256x256 x 16x16)
    prev_small = "scipy" if scimetal._config.small_to_scipy() else "native"
    scimetal.set_small_inputs("native")
    try:
        for x_shape, k_shape in [((2000,), (5,)), ((256, 256), (16, 16))]:
            x = np.ones(x_shape, dtype=np.float32)
            x.flat[x.size // 2] = np.nan
            k = np.ones(k_shape, dtype=np.float32)
            with scimetal.precision("fast"), scimetal.device("gpu"):
                assert msig.choose_conv_method(x, k, mode="full") == "direct"
                for fn_name in ("convolve", "correlate"):
                    for method in ("auto", "direct"):
                        with warnings.catch_warnings(record=True) as rec:
                            warnings.simplefilter("always")
                            got = getattr(msig, fn_name)(x, k, mode="full", method=method)
                        assert not any(issubclass(w.category, RuntimeWarning) for w in rec)
                        ref = getattr(ss, fn_name)(x, k, mode="full", method="direct")
                        np.testing.assert_array_equal(np.isnan(got), np.isnan(ref))
    finally:
        scimetal.set_small_inputs(prev_small)

    # Finite FFT route via method='auto' must not warn (including OA shapes)
    xf = np.ones(40000, dtype=np.float32)
    kf = np.ones(1000, dtype=np.float32)
    with scimetal.precision("fast"), scimetal.device("gpu"):
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            got_f = msig.convolve(xf, kf, mode="full", method="auto")
        assert not any(issubclass(w.category, RuntimeWarning) for w in rec)
        assert scimetal.last_route().engine == "gpu"
    ref_f = ss.convolve(xf, kf, mode="full", method="auto")
    scale = max(1.0, float(np.abs(ref_f).max()))
    np.testing.assert_allclose(got_f, ref_f, rtol=1e-5, atol=2e-5 * scale)

    # fftconvolve & oaconvolve continue emitting 0 RuntimeWarnings on non-finite inputs
    xnf = np.ones(40000, dtype=np.float32)
    xnf[30000] = np.nan
    with scimetal.precision("fast"), scimetal.device("gpu"):
        for fn_name in ("fftconvolve", "oaconvolve"):
            with warnings.catch_warnings(record=True) as rec:
                warnings.simplefilter("always")
                getattr(msig, fn_name)(xnf, kf, mode="full")
            assert not any(issubclass(w.category, RuntimeWarning) for w in rec)


@pytest.mark.parametrize("shape1,shape2,dtype,val", [
    ((32769,), (32768,), np.float32, 1e17),
    ((32769,), (32768,), np.float32, 1e19),
    ((32806,), (32805,), np.complex64, 1e19),
])
def test_convolve_correlate_direct_publish_inverse_fft_overflow_warning(gpu, shape1, shape2, dtype, val):
    """Iter 7: direct_publish inverse-FFT overflow at index > 0 must warn for convolve/correlate and stay silent for fftconvolve."""
    x = np.zeros(shape1, dtype=dtype)
    k = np.zeros(shape2, dtype=dtype)
    x[101] = val
    k[100] = val

    for dev in ("gpu", "auto"):
        for fn_name in ("convolve", "correlate"):
            for method in ("auto", "fft"):
                with scimetal.precision("fast"), scimetal.device(dev):
                    with warnings.catch_warnings(record=True) as rec:
                        warnings.simplefilter("always")
                        got = getattr(msig, fn_name)(x, k, mode="full", method=method)
                    assert scimetal.last_route().engine == "gpu"
                assert np.any(~np.isfinite(got))
                assert any(issubclass(w.category, RuntimeWarning) and "NAN or inf" in str(w.message) for w in rec), (
                    dev, fn_name, method, got[0], rec
                )

    # fftconvolve on the exact same overflowing direct_publish inputs must match SciPy's non-warning behavior.
    with scimetal.precision("fast"), scimetal.device("gpu"):
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            got_fc = msig.fftconvolve(x, k, mode="full")
        assert scimetal.last_route().engine == "gpu"
        assert not any(issubclass(w.category, RuntimeWarning) for w in rec)
    assert np.any(~np.isfinite(got_fc))

    # Finite direct_publish must not emit RuntimeWarning.
    xf = np.zeros(shape1, dtype=dtype)
    kf = np.zeros(shape2, dtype=dtype)
    xf[101] = 1.0
    kf[100] = 1.0
    with scimetal.precision("fast"), scimetal.device("gpu"):
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            got_fin = msig.convolve(xf, kf, mode="full", method="auto")
        assert scimetal.last_route().engine == "gpu"
        assert not any(issubclass(w.category, RuntimeWarning) for w in rec)
    assert np.isfinite(got_fin).all()


def test_review_regressions(gpu):
    # 1. 0-D float64 ndarray fillvalue on float32 convolve2d raises ValueError like SciPy.
    a32 = np.ones((64, 64), dtype=np.float32)
    b32 = np.ones((5, 5), dtype=np.float32)
    fv64 = np.array(1.5, dtype=np.float64)
    with pytest.raises(ValueError):
        ss.convolve2d(a32, b32, fillvalue=fv64)
    prev_small = scimetal._config.small_to_scipy()
    scimetal.set_small_inputs("native")
    try:
        with scimetal.device("cpu"):
            with pytest.raises(ValueError):
                msig.convolve2d(a32, b32, fillvalue=fv64)
            assert scimetal.last_route().engine == "scipy"

        # 2. RuntimeWarning from FFT convolve on NaN points at caller frame (this file).
        xnf = np.ones(40000, dtype=np.float32)
        xnf[100] = np.nan
        knf = np.ones(100, dtype=np.float32)
        with scimetal.precision("fast"), scimetal.device("gpu"):
            with warnings.catch_warnings(record=True) as rec:
                warnings.simplefilter("always")
                msig.convolve(xnf, knf, mode="full", method="fft")
        rw = [w for w in rec if issubclass(w.category, RuntimeWarning)]
        assert rw and rw[0].filename == __file__

        # 3. correlate with unit-length axis reversal on FFT path matches SciPy.
        rng = np.random.default_rng(77)
        u1 = rng.standard_normal((1, 200)).astype(np.float32)
        u2 = rng.standard_normal((5, 200)).astype(np.float32)
        with scimetal.precision("fast"), scimetal.device("gpu"):
            for mode in ("full", "same", "valid"):
                got = msig.correlate(u1, u2, mode=mode, method="fft")
                ref = ss.correlate(u1, u2, mode=mode, method="fft")
                np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-4)

        # 4. float64 spectral overflow (1e15*1e15 len 65536) and underflow (1e-25*1e-25) fall back cleanly.
        with scimetal.precision("fast"), scimetal.device("gpu"):
            big1 = np.full(65536, 1e15, dtype=np.float64)
            big2 = np.full(65536, 1e15, dtype=np.float64)
            got_big = msig.fftconvolve(big1, big2)
            assert scimetal.last_route().engine == "scipy"
            assert np.isfinite(got_big).all() and got_big[65535] > 6e34

            tiny1 = np.full(256, 1e-25, dtype=np.float64)
            tiny2 = np.full(64, 1e-25, dtype=np.float64)
            got_tiny = msig.fftconvolve(tiny1, tiny2)
            assert scimetal.last_route().engine == "scipy"
            np.testing.assert_allclose(got_tiny, ss.fftconvolve(tiny1, tiny2))
    finally:
        scimetal.set_small_inputs("scipy" if prev_small else "native")
