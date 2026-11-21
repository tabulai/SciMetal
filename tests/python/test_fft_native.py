"""scimetal.fft against scipy.fft: results, dtypes, shapes, routing and fall-through.

Accuracy bound. The GPU engine works in single precision (unit roundoff
u = 2^-24, about 6e-8). A length-n transform is log2(n)/log2(radix) butterfly
stages, each adding O(u) relative error in the 2-norm, and MLX's kernels form
their twiddles as cos/sin of a single-precision angle, good to about 1e-6
near the end of the circle; Rader and Bluestein sizes run two transforms and
two table multiplications per result. Over every length from 2 to 4096 and
all four transforms the measured relative 2-norm error against SciPy's
complex128 result is at most 2.6e-6 (Rader, n = 2029); powers of two stay
below 1.2e-6 up to 2^24. The tests allow 5e-6 per transformed axis, about
twice the worst case measured.
"""
import gc
import multiprocessing
import os
import pathlib
import pickle
import subprocess
import sys
import threading

import numpy as np
import pytest
import scipy.fft

import scimetal
import scimetal.fft as mfft
from scimetal import _native
from scimetal._dispatch import counters
from scimetal.fft import _native_fft

TOLERANCE = 5e-6
REAL = {np.complex64: np.float32, np.complex128: np.float64}


@pytest.fixture
def gpu():
    if not _native.has_metal():
        pytest.skip("requires Metal GPU build")
    with scimetal.device("gpu"), scimetal.precision("fast"):
        yield


def random(shape, dtype, seed=0):
    rng = np.random.default_rng(seed)
    dtype = np.dtype(dtype)
    real = np.float32 if dtype.itemsize // (2 if dtype.kind == "c" else 1) == 4 else np.float64
    x = rng.standard_normal(shape, dtype=real)
    if dtype.kind == "c":
        x = x + 1j * rng.standard_normal(shape, dtype=real)
    return x.astype(dtype)


def error(result, reference):
    assert result.shape == reference.shape
    reference = np.asarray(reference)
    return float(np.linalg.norm(np.asarray(result, reference.dtype) - reference) / np.linalg.norm(reference))


def reference(function, x, *args, **kwargs):
    """SciPy's result for the double-precision version of x."""
    wide = x.astype(np.complex128 if x.dtype.kind == "c" else np.float64)
    return getattr(scipy.fft, function)(wide, *args, **kwargs)


def check(function, x, *args, transformed=1, **kwargs):
    """Runs natively and matches SciPy in dtype, shape and (within the bound) value.

    ``transformed`` is the number of axes the call transforms.
    """
    result = getattr(mfft, function)(x, *args, **kwargs)
    route = scimetal.last_route()
    assert route.function == f"fft.{function}" and route.engine == "gpu", str(route)
    assert route.compute_dtype in ("complex64", "float32")
    expected = getattr(scipy.fft, function)(x, *args, **kwargs)
    assert result.dtype == expected.dtype and result.shape == expected.shape
    assert error(result, reference(function, x, *args, **kwargs)) < TOLERANCE * transformed
    return result


def declined(function, x, *args, **kwargs):
    """Falls through: the result is SciPy's own.

    A batch handed to SciPy with worker threads is bit for bit SciPy's result
    for that worker count, which can differ from the single-worker one in the
    last place.
    """
    result = getattr(mfft, function)(x, *args, **kwargs)
    route = scimetal.last_route()
    assert route.function == f"fft.{function}" and route.engine == "scipy", str(route)
    expected = getattr(scipy.fft, function)(x, *args, **kwargs)
    assert result.dtype == expected.dtype and result.shape == expected.shape
    if "SciPy with workers=" in route.reason:
        threads = int(route.reason.rsplit("=", 1)[1])
        assert threads == _native.performance_cores() and "workers" not in kwargs
        assert result.tobytes() == getattr(scipy.fft, function)(x, *args, **kwargs, workers=threads).tobytes()
        np.testing.assert_allclose(result, expected, rtol=0, atol=1e-13 * np.abs(expected).max())
    else:
        assert result.tobytes() == expected.tobytes()
    return route


# Every class of plan: Stockham (all radices), Rader, Bluestein, MLX's four-step,
# our strided chains, and the multi-pass powers of two.
LENGTHS = [2, 3, 4, 5, 7, 8, 11, 13, 16, 17, 26, 30, 47, 64, 100, 128, 243, 289, 500, 1000, 1009, 1021,
           1024, 1982, 2003, 2038, 2048, 2187, 3003, 3645, 4095, 4096, 5000, 6000, 8192, 19683, 30030,
           65536, 100000, 131072]


@pytest.mark.parametrize("n", LENGTHS)
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_every_plan_class_matches_scipy(gpu, n, dtype):
    x = random((3, n), dtype, seed=n)
    r = random((3, n), REAL[dtype], seed=n + 1)
    h = random((3, n // 2 + 1), dtype, seed=n + 2)
    check("fft", x)
    check("ifft", x)
    check("rfft", r)
    check("irfft", h, n)  # general input: SciPy ignores imag(x[0]) and imag(x[n/2])
    check("fft", r)       # a real input to the complex transform


@pytest.mark.parametrize("log2n", [20, 21, 22, 23, 24])
def test_long_powers_of_two(gpu, log2n):
    # MLX 0.31.1's own GPU FFT raises at 2^21 and 2^22 and is wrong at 2^23 and 2^24.
    n = 1 << log2n
    x = random(n, np.complex64, seed=log2n)
    assert error(mfft.fft(x), reference("fft", x)) < TOLERANCE
    assert scimetal.last_route().engine == "gpu"
    assert error(mfft.ifft(x), reference("ifft", x)) < TOLERANCE
    r = random(n, np.float64, seed=log2n)
    spectrum = mfft.rfft(r)
    assert spectrum.dtype == np.complex128 and error(spectrum, scipy.fft.rfft(r)) < TOLERANCE
    assert error(mfft.irfft(spectrum, n), r) < 2 * TOLERANCE


def test_long_non_power_of_two(gpu):
    for n in (10 ** 6, 3 * 2 ** 20, 5 ** 9):
        x = random(n, np.complex128, seed=n)
        check("fft", x)
        check("ifft", x)
        r = random(n, np.float64, seed=n)
        check("rfft", r)
        check("irfft", x[: n // 2 + 1], n)


@pytest.mark.parametrize("n", [1, 2049, 2051, 3001, 4094, 4099, 8194, 100003])
def test_lengths_without_a_plan_fall_through(gpu, n):
    x = random((2, n), np.complex64, seed=n)
    route = declined("fft", x)
    declined("ifft", x)
    declined("rfft", random((2, n), np.float32))
    declined("irfft", random((2, 40), np.complex64), n)
    if n > 1:
        assert "no GPU plan" in route.reason


@pytest.mark.parametrize("shape", [(1, 96), (2, 96), (5, 96), (64, 96), (257, 1000), (1000, 47), (3, 4, 5, 96),
                                   (2, 1, 3, 2, 2, 2, 2, 48), (33, 8192), (300, 4096), (9, 30030)])
def test_batches(gpu, shape):
    # Odd and even row counts (the real kernels transform rows in pairs), rows
    # sharing a threadgroup, and leading axes folded into the batch.
    x = random(shape, np.complex128, seed=7)
    r = random(shape, np.float32, seed=8)
    check("fft", x)
    check("ifft", x.astype(np.complex64))
    check("rfft", r)
    check("irfft", x, shape[-1])
    if shape[-1] <= 1000:  # every length up to 2048 has a plan: the natural and the odd output lengths
        check("irfft", x)
        check("irfft", x, 2 * shape[-1] - 1)


@pytest.mark.parametrize("norm", [None, "backward", "ortho", "forward"])
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_norms_and_lengths(gpu, norm, dtype):
    x = random((5, 7, 96), dtype)
    r = random((5, 7, 96), REAL[dtype])
    for n in (None, 1, 2, 50, 96, 97, 200, 1021, 6000):
        if n == 1:  # one point: SciPy's
            declined("fft", x, n, norm=norm)
            continue
        check("fft", x, n, norm=norm)
        check("ifft", x, n, norm=norm)
        check("rfft", r, n, norm=norm)
        check("irfft", x, n, norm=norm)
        check("fft", r, n, norm=norm)
        check("irfft", r, n, norm=norm)   # SciPy takes a real input as complex
    check("fft", x, n=np.int64(64), norm=norm)
    check("fft", x, 64, -1, norm)         # positional arguments


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_any_axis(gpu, dtype):
    x = random((6, 10, 36), dtype)
    for axis in (0, 1, 2, -1, -2, -3):
        for norm in (None, "ortho"):
            check("fft", x, axis=axis, norm=norm)
            check("ifft", x, axis=axis, norm=norm)
            check("fft", x.real.copy(), axis=axis, norm=norm)
    # Real transforms and resizing run natively along the last axis only.
    declined("rfft", x.real.copy(), axis=0)
    declined("irfft", x, axis=1)
    declined("fft", x, 20, axis=0)
    check("fft", x, 10, axis=1)           # n equal to the axis length is no resizing
    # An axis whose length has a prime factor above 13 has no plan off the last axis.
    declined("fft", random((17, 64), dtype), axis=0)
    check("fft", random((17, 64), dtype), axis=1)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_n_dimensional(gpu, dtype):
    x = random((6, 10, 36), dtype)
    r = random((6, 10, 36), REAL[dtype])
    for norm in (None, "ortho", "forward"):
        for axes in (None, (0, 1), (1, 2), (0, 2), (2, 0), (-1,), (0,), (0, 1, 2), (2, 1, 0), 2, [-2, -1]):
            count = 3 if axes is None else np.size(axes)
            check("fftn", x, axes=axes, norm=norm, transformed=count)
            check("ifftn", x, axes=axes, norm=norm, transformed=count)
            check("fftn", r, axes=axes, norm=norm, transformed=count)
            if axes is None or np.atleast_1d(axes)[-1] in (2, -1):
                check("rfftn", r, axes=axes, norm=norm, transformed=count)
                check("irfftn", x, axes=axes, norm=norm, transformed=count)
            else:  # the half-spectrum axis must be the last
                declined("rfftn", r, axes=axes, norm=norm)
                declined("irfftn", x, axes=axes, norm=norm)
        check("fft2", x, norm=norm, transformed=2)
        check("ifft2", x, norm=norm, transformed=2)
        check("rfft2", r, norm=norm, transformed=2)
        check("irfft2", x, norm=norm, transformed=2)
        check("fft2", x, axes=(0, 2), norm=norm, transformed=2)
        # s resizes the last axis; -1 keeps an axis as it is.
        check("fft2", x, s=(10, 64), norm=norm, transformed=2)
        check("fftn", x, s=(6, 10, 50), norm=norm, transformed=3)
        check("fftn", x, s=(-1, 40), norm=norm, transformed=2)
        check("rfft2", r, s=(10, 130), norm=norm, transformed=2)
        check("irfft2", x, s=(10, 101), norm=norm, transformed=2)
        check("irfft2", x, s=(10, 36), norm=norm, transformed=2)   # output length equal to the input's
        check("irfftn", x, s=(6, 10, 70), norm=norm, transformed=3)
        check("rfftn", r, s=(50,), norm=norm)
        check("fftn", x, (6, 10, 36), None, norm)                  # positional arguments
        # Resizing any other axis is SciPy's.
        declined("fft2", x, s=(12, 36), norm=norm)
        declined("irfft2", x, s=(8, 70), norm=norm)
        declined("rfftn", r, s=(4, 10, 36), norm=norm)


def test_two_dimensional_images(gpu):
    # Tails the threadgroup width does and does not divide, small and large.
    for shape in ((256, 258), (143, 77), (1024, 1024), (2048, 2100), (300, 5000), (4096, 2100)):
        r = random(shape, np.float32, seed=sum(shape))
        spectrum = check("rfft2", r, transformed=2)
        check("irfft2", spectrum, s=shape, transformed=2)
        x = random(shape, np.complex64, seed=sum(shape))
        check("fft2", x, transformed=2)
        check("ifft2", x, transformed=2)
    volume = random((64, 60, 100), np.float64)
    check("rfftn", volume, transformed=3)
    check("fftn", volume, transformed=3)


def test_dtypes_follow_scipy(gpu):
    x = np.arange(4 * 96).reshape(4, 96)
    cases = [(np.float32, np.complex64), (np.float64, np.complex128), (np.complex64, np.complex64),
             (np.complex128, np.complex128), (np.float16, np.complex64), (np.int16, np.complex128),
             (np.uint8, np.complex128), (np.int64, np.complex128), (np.bool_, np.complex128)]
    for source, target in cases:
        a = (x % 7).astype(source)
        for function in ("fft", "ifft", "fft2", "fftn"):
            assert check(function, a, transformed=2).dtype == target
        if np.dtype(source).kind != "c":
            assert check("rfft", a).dtype == target
            assert check("rfftn", a, transformed=2).dtype == target
        real = np.float32 if target == np.complex64 else np.float64
        assert check("irfft", a).dtype == real
        assert check("irfft2", a, transformed=2).dtype == real
    # No native path: SciPy's result or error, whatever it is.
    if np.dtype(np.longdouble) != np.dtype(np.float64):
        declined("fft", x.astype(np.longdouble))
    declined("fft", x.astype(object))


@pytest.mark.parametrize("dtype,value", [
    (np.int32, -(1 << 24)), (np.uint32, 1 << 24),
    (np.int64, np.iinfo(np.int64).min), (np.uint64, np.iinfo(np.uint64).max),
])
def test_integer_fft_range_boundaries(gpu, dtype, value):
    x = np.zeros(32, dtype=dtype)
    # Impulses test the exact cutoffs in both directions. Values just inside
    # the guard retain their native path.
    x[0] = -(2**24 - 1) if value < 0 else 2**24 - 1
    check("fft", x)
    x[0] = value
    assert "2^24" in declined("fft", x).reason


def test_double_inputs_outside_single_precision_range_fall_through(gpu):
    x = random((4, 1000), np.complex128)
    for scale in (1e-25, 1e-9, 1.0, 1e9, 1e25):          # within range: any scale gives the same relative accuracy
        check("fft", x * scale)
        check("irfft", x * scale)
    assert not mfft.fft(np.zeros((4, 1000))).any() and scimetal.last_route().engine == "gpu"   # all zeros
    mixed = x.copy()
    mixed[0, :10] *= 1e-200                                # a few tiny values among ordinary ones are fine
    check("fft", mixed)
    for scale in (1e30, 1e200, 1e-32, 1e-200):
        route = declined("fft", x * scale)                 # would overflow or vanish in single precision
        assert "single precision" in route.reason
        declined("rfft2", x.real * scale)
    for value in (np.inf, -np.inf, np.nan):
        special = x.copy()
        special[2, 17] = value
        result = mfft.ifft(special)                        # NaN results cannot be compared bit for bit
        assert scimetal.last_route().engine == "scipy" and "single precision" in scimetal.last_route().reason
        with np.errstate(invalid="ignore"):
            np.testing.assert_array_equal(np.isfinite(result), np.isfinite(scipy.fft.ifft(special)))
    # Beyond the truncation n the values are not read.
    tail = x.copy()
    tail[:, 600:] = 1e300
    check("fft", tail, 512)
    declined("fft", tail, 700)
    # Single-precision inputs are transformed in their own precision, as by SciPy.
    check("fft", (x * 1e30).astype(np.complex64))


def test_memory_layouts(gpu):
    base = random((12, 200), np.complex128)
    for view in (base[::2], base[:, ::2], base[::-1, ::-1], base.T, np.asfortranarray(base),
                 base.astype(">c16"), base.astype(np.complex64).astype(">c8")):
        check("fft", view)
        check("ifft", view, axis=0)
    unaligned = np.frombuffer(b"\0" + base.tobytes(), dtype=np.complex128, offset=1).reshape(base.shape)
    assert not unaligned.flags.aligned
    check("fft", unaligned)
    # The matrix subclass comes back as a plain array, as from SciPy.
    with pytest.warns(PendingDeprecationWarning):
        matrix = np.matrix(base)
    assert type(mfft.fft(matrix)) is np.ndarray


def test_results_are_ordinary_independent_arrays(gpu):
    x = random((64, 1000), np.complex128)
    before = x.copy()
    first = mfft.fft(x, overwrite_x=True)
    assert x.tobytes() == before.tobytes()        # overwrite_x is permission, never used
    assert first.flags.c_contiguous and first.flags.writeable and first.flags.aligned
    kept = first.copy()
    second = mfft.fft(x[::-1].copy())             # must not reuse the buffer `first` still holds
    np.testing.assert_array_equal(first, kept)
    assert not np.shares_memory(first, second)
    first[0, 0] = 5.0
    view = first[3:5]
    del first, second
    gc.collect()
    np.testing.assert_array_equal(view, kept[3:5])
    single = mfft.fft(x.astype(np.complex64))     # a single-precision result is the device buffer itself
    assert single.flags.c_contiguous and single.flags.writeable
    np.testing.assert_allclose(single, kept, rtol=0, atol=TOLERANCE * np.abs(kept).max() * 30)


def test_arguments_scipy_rejects_raise_scipys_errors(gpu):
    x = random((4, 96), np.complex128)
    cases = [("fft", (x, 0), {}), ("fft", (x, -3), {}), ("fft", (x,), {"norm": "unitary"}),
             ("fft", (x,), {"axis": 2}), ("fft", (x,), {"axis": -3}), ("rfft", (x,), {}),
             ("fft", (x,), {"plan": object()}), ("fft", (x,), {"workers": 0}), ("fft", (x,), {"workers": -10 ** 6}),
             ("irfft", (x[:, :1],), {}), ("fftn", (x,), {"axes": (0, 0)}), ("fftn", (x,), {"axes": (0, 2)}),
             ("fftn", (x,), {"s": (4, 96, 3)}), ("fftn", (x,), {"s": (4,), "axes": (0, 1)}),
             ("fftn", (x,), {"s": (0, 96)}), ("rfftn", (x,), {}), ("rfftn", (x.real,), {"axes": ()}),
             ("fft", (np.float64(3.0),), {}), ("fft", (np.zeros((3, 0)),), {})]
    for function, args, kwargs in cases:
        with pytest.raises(Exception) as ours:
            getattr(mfft, function)(*args, **kwargs)
        with pytest.raises(Exception) as theirs:
            getattr(scipy.fft, function)(*args, **kwargs)
        assert type(ours.value) is type(theirs.value), (function, kwargs, ours.value, theirs.value)
        assert str(ours.value) == str(theirs.value)
    # What SciPy accepts and SciMetal does not take is SciPy's result.
    declined("fft", [[1.0, 2.0, 3.0, 4.0]])
    declined("fftn", x, axes=())
    declined("fft", x, 96.0) if _accepts(scipy.fft.fft, x, 96.0) else None


def _accepts(function, *args):
    try:
        function(*args)
    except Exception:
        return False
    return True


def test_precision_policies():
    if not _native.has_metal():
        pytest.skip("requires Metal GPU build")
    x = random((64, 4096), np.complex128)
    single = x.astype(np.complex64)
    with scimetal.device("gpu"):
        with scimetal.precision("strict"):
            route = declined("fft", x)                     # exact: SciPy
            assert "strict" in route.reason
            declined("rfft2", x.real.copy())
            check("fft", single)                           # single precision is exact on the GPU engine too
            check("irfft", single)
        for policy in ("fast", "refined"):
            with scimetal.precision(policy):
                check("fft", x)
                assert scimetal.last_route().policy == policy
    with scimetal.device("auto"), scimetal.precision("strict"):
        declined("fft", x)
        check("fft", single)


def test_device_policies_and_the_crossover():
    if not _native.has_metal():
        pytest.skip("requires Metal GPU build")
    floor = _native_fft.auto_min_points()
    assert 1 << 12 < floor < 1 << 19
    x = random((1024, 4096), np.complex64)
    with scimetal.precision("fast"):
        with scimetal.device("cpu"):
            assert "cpu" in declined("fft", x).reason
        with scimetal.device("auto"):
            check("fft", x)                                    # far above the crossover
            check("fft2", x[:256], transformed=2)
            small = random((4, 512), np.complex64)
            assert "crossover" in declined("fft", small).reason
            assert "crossover" in declined("fftn", small).reason
            # Either side of the crossover the result is right and the route is reported.
            for points in (floor // 2, floor - 1, floor, 2 * floor, 8 * floor):
                for shape in ((points,), (max(points // 1024, 1), 1024), (max(points // 16, 1), 16)):
                    y = random(shape, np.complex64, seed=points)
                    result = mfft.fft(y)
                    route = scimetal.last_route()
                    assert error(result, reference("fft", y)) < (TOLERANCE if route.engine == "gpu" else 1e-6)
                    plan = _native_fft.plan(y.dtype, y.shape, _native_fft.options(_native_fft.FORWARD, 0, [y.ndim - 1]))
                    assert (route.engine == "gpu") == (plan.status == 0)
                    if y.size < floor:
                        assert route.engine == "scipy"
            # One long transform crosses over with fewer points than many short rows do.
            long_plan = _native_fft.plan(np.complex64, (1 << 18,), _native_fft.options(_native_fft.FORWARD, 0, [0]))
            short_plan = _native_fft.plan(np.complex64, (1 << 14, 16), _native_fft.options(_native_fft.FORWARD, 0, [1]))
            assert long_plan.status == 0 and short_plan.status != 0
            # SciPy threaded over a batch is a stronger alternative: the crossover moves up.
            batch = random((128, 4096), np.complex64)
            check("fft", batch)
            assert "crossover" in declined("fft", batch, workers=16).reason
        with scimetal.device("gpu"):
            check("fft", random((2, 8), np.complex64))         # forced: any size


def test_small_inputs_go_to_scipy_by_default():
    small = random((4, 512), np.complex64)
    with scimetal.device("auto"), scimetal.precision("fast"):
        try:
            scimetal.set_small_inputs("scipy")
            assert "elements" in declined("fft", small).reason
            assert "elements" in declined("fftn", small).reason
        finally:
            scimetal.set_small_inputs("native")
        if _native.has_metal():
            assert "crossover" in declined("fft", small).reason
        else:
            assert declined("fft", small).engine == "scipy"


def test_declined_batches_use_scipys_worker_threads():
    cores = _native.performance_cores()
    assert cores >= 1
    x = random((64, 4096), np.complex128)     # 2^18 points: enough for SciPy's threads to pay
    expected = scipy.fft.fft(x)
    long = np.tile(x[0], 40)
    tally = counters["fft.fft"]
    native, fell = tally
    # Under strict, declined calls stay on single-worker SciPy so they remain bit-identical.
    with scimetal.device("auto"), scimetal.precision("strict"):
        for fn, arg, ref in ((mfft.fft, x, expected),
                             (mfft.fft, long, scipy.fft.fft(long)),
                             (mfft.fft2, x, scipy.fft.fft2(x))):
            result = fn(arg)
            route = scimetal.last_route()
            assert route.engine == "scipy" and "strict" in route.reason
            assert "workers=" not in route.reason
            assert result.tobytes() == ref.tobytes()
    # Under fast/refined, large declined calls (e.g. device="cpu") use SciPy's worker threads.
    with scimetal.device("cpu"), scimetal.precision("fast"):
        before_cpu_fell = tally[1]
        result = mfft.fft(x)
        route = scimetal.last_route()
        assert route.engine == "scipy" and "cpu" in route.reason
        assert tally == [native, before_cpu_fell + 1]
        if cores > 1:
            assert f"workers={cores}" in route.reason
            assert result.tobytes() == scipy.fft.fft(x, workers=cores).tobytes()
        np.testing.assert_allclose(result, expected, rtol=0, atol=1e-13 * np.abs(expected).max())
        # Positional workers=None must also receive the override without TypeError.
        result_pos = mfft.fft(x, None, -1, None, False, None)
        if cores > 1:
            assert f"workers={cores}" in scimetal.last_route().reason
            assert result_pos.tobytes() == scipy.fft.fft(x, workers=cores).tobytes()
        # The caller's own choice of workers is left alone.
        for workers in (1, 2, -1):
            mfft.fft(x, workers=workers)
            assert "workers=" not in scimetal.last_route().reason
        with scipy.fft.set_workers(2):
            mfft.fft(x)
            assert "workers=" not in scimetal.last_route().reason
        # Too little data to thread, or an axis holding a single transform of a
        # multi-dimensional array (where SciPy's threads cost time): as made.
        for unthreaded in (x[0], x[:2, :64], x[:16], x[:1].copy(), np.tile(x[0], 40)[None, :]):
            mfft.fft(unthreaded)
            assert "workers=" not in scimetal.last_route().reason, unthreaded.shape
        mfft.fft2(np.tile(x[0], 40)[None, :])
        assert "workers=" not in scimetal.last_route().reason
        # One long transform of a 1-D array threads inside SciPy.
        result = mfft.fft(long)
        if cores > 1:
            assert f"workers={cores}" in scimetal.last_route().reason
            assert result.tobytes() == scipy.fft.fft(long, workers=cores).tobytes()
        np.testing.assert_allclose(result, scipy.fft.fft(long), rtol=0, atol=1e-13 * np.abs(result).max())
        # N-D transforms thread too, and SciPy's errors still come from SciPy.
        result = mfft.fft2(x)
        if cores > 1:
            assert f"workers={cores}" in scimetal.last_route().reason
            assert result.tobytes() == scipy.fft.fft2(x, workers=cores).tobytes()
        np.testing.assert_allclose(result, scipy.fft.fft2(x), rtol=0, atol=1e-13 * np.abs(result).max())
        with pytest.raises(ValueError):
            mfft.fft(x, norm="nope")
    assert tally[0] == native


def test_counters_and_explain(gpu):
    tally = counters["fft.rfft"]
    native, fell = tally
    mfft.rfft(random((8, 1000), np.float32))
    assert tally == [native + 1, fell]
    text = scimetal.explain()
    assert "fft.rfft" in text and "engine=gpu" in text and "float32" in text
    mfft.rfft(random((8, 3001), np.float32))
    assert tally == [native + 1, fell + 1]
    assert "engine=scipy" in scimetal.explain() and "3001" in scimetal.explain()
    assert set(scimetal.accelerated()["fft"]) >= {"fft", "ifft", "rfft", "irfft", "fft2", "ifft2", "fftn",
                                                    "ifftn", "rfft2", "irfft2", "rfftn", "irfftn"}
    # Everything else in the namespace is SciPy's own object.
    assert mfft.dct is scipy.fft.dct and mfft.next_fast_len is scipy.fft.next_fast_len
    assert mfft.fft.__doc__ == scipy.fft.fft.__doc__


def test_functions_pickle_and_work_in_worker_processes():
    for name in ("fft", "ifft", "rfft", "irfft", "fft2", "rfftn"):
        function = getattr(mfft, name)
        assert pickle.loads(pickle.dumps(function)) is function
        assert function.__module__ == "scimetal.fft" and function.__name__ == name
    x = [np.ones(100) + index for index in range(4)]
    with multiprocessing.get_context("spawn").Pool(2) as pool:
        results = pool.map(mfft.rfft, x)
    for result, argument in zip(results, x):
        np.testing.assert_array_equal(result, scipy.fft.rfft(argument))


def test_plan_description(gpu):
    options = _native_fft.options(_native_fft.R2C, 0, [0, 1])
    plan = _native_fft.plan(np.float32, (64, 96), options, prepare=True)
    assert plan.status == 0 and plan.passes == 2
    assert plan.description == b"axis 1: stockham 96; axis 0: strided 64 tail 49"
    plan = _native_fft.plan(np.complex64, (2, 3001), _native_fft.options(_native_fft.FORWARD, 0, [1]))
    assert plan.status != 0 and b"3001" in plan.description


def test_threads_share_the_context(gpu):
    x = random((32, 4096), np.complex64)
    expected = reference("fft", x)
    errors = []

    def worker(seed):
        with scimetal.device("gpu"):
            for _ in range(20):
                y = random((32, 4096), np.complex64, seed=seed)
                if error(mfft.fft(y), reference("fft", y)) > TOLERANCE or error(mfft.fft(x), expected) > TOLERANCE:
                    errors.append(seed)

    threads = [threading.Thread(target=worker, args=(seed,)) for seed in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []


def test_prepare_supported_kinds_and_dtypes(gpu):
    # Sentinel route and counter snapshot: prepare must not mutate either.
    mfft.fft(random((4, 32), np.complex64))
    route_before = scimetal.last_route()
    counters_before = {k: list(v) for k, v in counters.items()}

    for kind in ("fft", "ifft", "fft2", "ifft2", "fftn", "ifftn"):
        assert mfft.prepare((8, 16, 32), np.complex64, kind=kind) is True
        assert mfft.prepare((8, 16, 32), np.complex128, kind=kind) is True
    for kind in ("rfft", "rfft2", "rfftn"):
        assert mfft.prepare((8, 16, 32), np.float32, kind=kind) is True
        assert mfft.prepare((8, 16, 32), np.float64, kind=kind) is True
    for kind in ("irfft", "irfft2", "irfftn"):
        assert mfft.prepare((8, 16, 17), np.complex64, kind=kind) is True
        assert mfft.prepare((8, 16, 17), np.complex128, kind=kind) is True

    # Shape as int, axes as int / negative / non-last, promoted dtypes, all norms.
    assert mfft.prepare(64, np.complex64) is True
    assert mfft.prepare((16, 32), np.float16, kind="rfft") is True
    assert mfft.prepare((16, 32), np.int64, kind="fft") is True
    assert mfft.prepare((16, 32), np.bool_, kind="rfft") is True
    assert mfft.prepare((16, 32), np.complex64, kind="fft", axes=0) is True
    assert mfft.prepare((16, 32), np.complex64, kind="fft", axes=-2) is True
    assert mfft.prepare((8, 16, 32), np.complex64, kind="fft2", axes=(-3, -2)) is True
    for norm in (None, "backward", "ortho", "forward"):
        assert mfft.prepare((8, 32), np.complex64, kind="fft", norm=norm) is True

    # n / s resizing on last axis, -1 in s, and irfft* even/odd/equal output lengths.
    assert mfft.prepare((8, 32), np.complex64, kind="fft", n=64) is True
    assert mfft.prepare((8, 32), np.complex64, kind="fft2", s=(-1, 64)) is True
    assert mfft.prepare((8, 32), np.complex64, kind="fftn", s=(64,), axes=(-1,)) is True
    assert mfft.prepare((8, 17), np.complex64, kind="irfft", n=32) is True
    assert mfft.prepare((8, 17), np.complex64, kind="irfft", n=33) is True
    assert mfft.prepare((8, 17), np.complex64, kind="irfft", n=17) is True
    assert mfft.prepare((8, 17), np.complex64, kind="irfft2", s=(8, 33)) is True
    assert mfft.prepare((8, 17), np.complex64, kind="irfftn", s=(8, 32)) is True

    assert scimetal.last_route() == route_before
    assert {k: list(v) for k, v in counters.items()} == counters_before


def test_prepare_policies_and_declines_return_false():
    if not _native.has_metal():
        pytest.skip("requires Metal GPU build")
    with scimetal.device("cpu"), scimetal.precision("fast"):
        assert mfft.prepare((1024, 4096), np.complex64) is False

    for dev in ("auto", "gpu"):
        with scimetal.device(dev), scimetal.precision("strict"):
            assert mfft.prepare((1 << 18,), np.complex128) is False
            assert mfft.prepare((1 << 18,), np.float64, kind="rfft") is False
            assert mfft.prepare((1 << 18,), np.complex64) is True
            assert mfft.prepare((1 << 18,), np.float32, kind="rfft") is True

    with scimetal.device("auto"), scimetal.precision("fast"):
        assert mfft.prepare((2, 64), np.complex64) is False
        assert mfft.prepare((1 << 14, 16), np.complex64) is False
        assert mfft.prepare((1 << 18,), np.complex64) is True
        assert mfft.prepare((1024, 4096), np.complex64) is True
        assert mfft.prepare((128, 4096), np.complex64) is True
        with scipy.fft.set_workers(16):
            assert mfft.prepare((128, 4096), np.complex64) is False

    with scimetal.device("gpu"), scimetal.precision("fast"):
        assert mfft.prepare((4, 3001), np.complex64) is False
        assert mfft.prepare((4, 1), np.complex64) is False
        assert mfft.prepare((17, 64), np.complex64, kind="fft", axes=0) is False
        assert mfft.prepare((8, 16), np.float32, kind="rfft", axes=0) is False
        assert mfft.prepare((8, 16), np.complex64, kind="irfft", axes=0) is False
        assert mfft.prepare((8, 16), np.float32, kind="rfftn", axes=(1, 0)) is False
        assert mfft.prepare((8, 16), np.complex64, kind="irfftn", axes=(1, 0)) is False
        assert mfft.prepare((6, 10, 36), np.complex64, kind="fft2", s=(12, 36)) is False
        assert mfft.prepare((0, 8), np.complex64, kind="fft") is False
        assert mfft.prepare((0, 8), np.complex64, kind="fft2") is False
        assert mfft.prepare((4, 16), np.complex64, kind="fft", n=1 << 31) is False
        assert mfft.prepare((4, 16), np.complex64, kind="fft2", s=(4, 1 << 31)) is False


def test_prepare_invalid_arguments_raise_value_error(gpu):
    invalid_calls = [
        ((8, 16), {"kind": "dct"}),
        ((8, 16), {"kind": ["fft"]}),
        ((8, 16), {"norm": "unitary"}),
        ((8, 16), {"norm": ["ortho"]}),
        ((8, 16), {"kind": "rfft"}),  # default dtype is complex128
        ((8, 16), {"dtype": np.complex64, "kind": "rfft"}),
        ((8, 16), {"dtype": np.complex128, "kind": "rfft2"}),
        ((8, 16), {"dtype": np.complex64, "kind": "rfftn"}),
        ((8, 16), {"dtype": object}),
        ((), {}),
        ((-4, 8), {}),
        ((2,) * 9, {}),
        ((8, 16), {"kind": "fft", "n": 16, "s": (16,)}),
        ((8, 16), {"kind": "fft", "s": (16,)}),
        ((8, 16), {"kind": "fft2", "n": 16}),
        ((8, 16), {"kind": "fft", "n": 0}),
        ((8, 16), {"kind": "fft", "n": -4}),
        ((8, 16), {"kind": "fft2", "s": (0, 16)}),
        ((8, 16), {"kind": "fft2", "s": (-2, 16)}),
        ((8, 16), {"kind": "fftn", "s": (8, 16), "axes": (1,)}),
        ((8, 16), {"kind": "fftn", "s": (2, 8, 16)}),
        ((8, 16), {"kind": "fft", "axes": (0, 1)}),
        ((8, 16), {"kind": "fft", "axes": 2}),
        ((8, 16), {"kind": "fft", "axes": -3}),
        ((8, 16), {"kind": "fft2", "axes": (0, 0)}),
        ((8, 16), {"kind": "fft2", "axes": (0, 2)}),
        ((16,), {"kind": "fft2"}),
        ((8, 16), {"kind": "fftn", "axes": ()}),
    ]
    for shape, kwargs in invalid_calls:
        with pytest.raises(ValueError):
            mfft.prepare(shape, **kwargs)


def test_prepare_warms_pipeline_for_first_call():
    if not _native.has_metal():
        pytest.skip("requires Metal GPU build")
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    if not os.environ.get("SCIMETAL_TEST_INSTALLED"):
        existing_pp = env.get("PYTHONPATH")
        python_dir = str(repo_root / "python")
        env["PYTHONPATH"] = os.pathsep.join([python_dir, existing_pp]) if existing_pp else python_dir

    common = (
        "import time, numpy as np, scipy.fft, scimetal, scimetal.fft as mfft\n"
        "x = np.ones((4, 1409), dtype=np.complex64)\n"
        "with scimetal.device('gpu'):\n"
        "    mfft.fft(np.ones(8, dtype=np.complex64))\n"
    )
    cold_script = common + (
        "    t0 = time.perf_counter()\n"
        "    y = mfft.fft(x)\n"
        "    dt_ms = (time.perf_counter() - t0) * 1e3\n"
        "    err = float(np.linalg.norm(y - scipy.fft.fft(x)) / np.linalg.norm(scipy.fft.fft(x)))\n"
        "    print(f'{dt_ms:.6f} {err:.6e}')\n"
    )
    prepared_script = common + (
        "    assert mfft.prepare((4, 1409), np.complex64) is True\n"
        "    t0 = time.perf_counter()\n"
        "    y = mfft.fft(x)\n"
        "    dt_ms = (time.perf_counter() - t0) * 1e3\n"
        "    err = float(np.linalg.norm(y - scipy.fft.fft(x)) / np.linalg.norm(scipy.fft.fft(x)))\n"
        "    print(f'{dt_ms:.6f} {err:.6e}')\n"
    )
    cold_res = subprocess.run([sys.executable, "-c", cold_script], env=env, capture_output=True, text=True, check=True)
    prep_res = subprocess.run([sys.executable, "-c", prepared_script], env=env, capture_output=True, text=True, check=True)
    cold_ms, cold_err = map(float, cold_res.stdout.strip().split())
    prep_ms, prep_err = map(float, prep_res.stdout.strip().split())
    assert cold_err < TOLERANCE and prep_err < TOLERANCE
    assert prep_ms < 0.5 * cold_ms, f"expected prepared ({prep_ms:.3f} ms) < 0.5 * cold ({cold_ms:.3f} ms)"


def test_prepare_exports_and_backend():
    assert "prepare" in mfft.__all__
    assert "prepare" in dir(mfft)


def test_review_regressions():
    from scimetal._dispatch import counters

    # 1. Declined direct mfft.fft() under register() does not re-enter uarray or double-count.
    was_reg = mfft.registered()
    mfft.register()
    try:
        x_big = np.ones((4, 1 << 15), dtype=np.complex128)
        with scimetal.device("cpu"), scimetal.precision("fast"):
            before_fall = counters["fft.fft"][1]
            out = mfft.fft(x_big)
            after_fall = counters["fft.fft"][1]
            route = scimetal.last_route()
        assert after_fall == before_fall + 1
        assert route.engine == "scipy"
        if _native.performance_cores() > 1:
            assert "SciPy with workers=" in route.reason
        np.testing.assert_allclose(out, scipy.fft.fft(x_big))
    finally:
        if not was_reg:
            mfft.unregister()

    # 2. overwrite_x=True on read-only complex array matches SciPy's ValueError.
    ro = np.ones(64, dtype=np.complex64)
    ro.setflags(write=False)
    prev_small = scimetal._config.small_to_scipy()
    scimetal.set_small_inputs("native")
    try:
        with scimetal.device("gpu"):
            with pytest.raises(ValueError, match="writeable|read-only"):
                mfft.fft(ro, overwrite_x=True)
    finally:
        scimetal.set_small_inputs("scipy" if prev_small else "native")

    # 3. Large workers value doesn't truncate to 0 in uint32 options().
    opts = _native_fft.options(_native_fft.FORWARD, 0, [0], 0, 1 << 32)
    assert opts.host_threads == (1 << 32) - 1

    # 4. Huge shape >= 2^64 returns False from prepare() without uint64 truncation.
    assert mfft.prepare((1 << 65,), np.complex64) is False
    assert "prepare" not in scimetal.accelerated()["fft"]
    assert not mfft.registered()
    mfft.register()
    try:
        assert mfft.registered()
    finally:
        mfft.unregister()
