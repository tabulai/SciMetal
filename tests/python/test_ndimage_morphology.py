"""Tests for scipy.ndimage morphology, min/max, and rank/median native routes."""
from __future__ import annotations

import concurrent.futures
import inspect
import warnings

import hypothesis
from hypothesis import strategies as st
import numpy as np
import pytest
import scipy.ndimage as sp_nd

import scimetal
import scimetal._native as _native
import scimetal.ndimage as sm_nd

MODES = ("nearest", "wrap", "reflect", "mirror", "constant")
REAL_DTYPES = (
    np.bool_, np.int8, np.int16, np.int32, np.int64,
    np.uint8, np.uint16, np.uint32, np.uint64, np.float32, np.float64,
)
GPU_EXACT_DTYPES = (np.bool_, np.int8, np.int16, np.uint8, np.uint16, np.float32)

NEW_FUNCTIONS = (
    "minimum_filter1d", "maximum_filter1d", "minimum_filter", "maximum_filter",
    "rank_filter", "median_filter", "percentile_filter",
    "grey_erosion", "grey_dilation", "grey_opening", "grey_closing",
    "morphological_gradient", "morphological_laplace", "white_tophat", "black_tophat",
)


def _make_array(shape, dtype, seed=0):
    rng = np.random.default_rng(seed)
    dt = np.dtype(dtype)
    if dt == np.bool_:
        return rng.integers(0, 2, size=shape).astype(np.bool_)
    if dt.kind == "i":
        lo = max(np.iinfo(dt).min // 4, -1000)
        hi = min(np.iinfo(dt).max // 4, 1000)
        return rng.integers(lo, hi, size=shape, dtype=dt)
    if dt.kind == "u":
        hi = min(np.iinfo(dt).max // 4, 1000)
        return rng.integers(0, hi, size=shape, dtype=dt)
    return (rng.standard_normal(shape) * 10.0).astype(dt)


def test_accelerated_registry_includes_all_morphology_and_rank_functions():
    acc = scimetal.accelerated()["ndimage"]
    for name in NEW_FUNCTIONS:
        assert name in acc
        ours, theirs = getattr(sm_nd, name), getattr(sp_nd, name)
        assert ours is not theirs and callable(ours.__scimetal_native__)
        assert inspect.signature(ours) == inspect.signature(theirs)
        assert ours.__doc__ == theirs.__doc__ and ours.__name__ == name
        assert name in dir(sm_nd)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
@pytest.mark.parametrize("mode", MODES)
def test_minmax_1d_strict_and_gpu(dtype, mode):
    x = _make_array((17, 19), dtype, seed=7)
    for fn_name in ("minimum_filter1d", "maximum_filter1d"):
        sp_fn = getattr(sp_nd, fn_name)
        sm_fn = getattr(sm_nd, fn_name)
        for axis in (0, 1):
            for size, origin in ((1, 0), (3, 0), (4, -1), (7, 2)):
                ref = sp_fn(x, size, axis=axis, mode=mode, cval=3.5, origin=origin)
                with scimetal.precision("strict"), scimetal.device("cpu"):
                    got = sm_fn(x, size, axis=axis, mode=mode, cval=3.5, origin=origin)
                    assert scimetal.last_route().engine == "cpu"
                np.testing.assert_array_equal(got, ref)
                if _native.has_metal():
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", scimetal.PrecisionWarning)
                        with scimetal.precision("fast"), scimetal.device("gpu"):
                            got_gpu = sm_fn(x, size, axis=axis, mode=mode, cval=3.5, origin=origin)
                            assert scimetal.last_route().engine == "gpu"
                    if dtype == np.float64:
                        np.testing.assert_allclose(got_gpu, ref, rtol=1e-6, atol=1e-5)
                    else:
                        np.testing.assert_array_equal(got_gpu, ref)


def test_minmax_1d_large_window_vanherk_gpu():
    """Windows > 64 exercise the 2-pass GPU van Herk / Gil-Werman kernels."""
    if not _native.has_metal():
        pytest.skip("no GPU in CPU-only build")
    x = _make_array((256, 256), np.float32, seed=11)
    for fn_name in ("minimum_filter1d", "maximum_filter1d"):
        sp_fn = getattr(sp_nd, fn_name)
        sm_fn = getattr(sm_nd, fn_name)
        for axis in (0, 1):
            for mode in ("reflect", "constant", "nearest", "wrap", "mirror"):
                ref = sp_fn(x, 85, axis=axis, mode=mode, cval=-2.0, origin=7)
                with scimetal.precision("fast"), scimetal.device("gpu"):
                    got = sm_fn(x, 85, axis=axis, mode=mode, cval=-2.0, origin=7)
                    assert scimetal.last_route().engine == "gpu"
                np.testing.assert_array_equal(got, ref)


@pytest.mark.parametrize("size", (3, 5, 7, 9, 11))
@pytest.mark.parametrize("mode", MODES)
def test_median_2d_tiled_and_medium_gpu_exact_float32(size, mode):
    x = _make_array((35, 41), np.float32, seed=size)
    ref = sp_nd.median_filter(x, size=size, mode=mode, cval=-5.25)
    with scimetal.precision("strict"), scimetal.device("cpu"):
        got_cpu = sm_nd.median_filter(x, size=size, mode=mode, cval=-5.25)
        assert scimetal.last_route().engine == "cpu"
    np.testing.assert_array_equal(got_cpu, ref)
    if _native.has_metal():
        with scimetal.precision("fast"), scimetal.device("gpu"):
            got_gpu = sm_nd.median_filter(x, size=size, mode=mode, cval=-5.25)
            assert scimetal.last_route().engine == "gpu"
        np.testing.assert_array_equal(got_gpu, ref)


def test_rank_and_percentile_3d_and_arbitrary_footprint():
    x3 = _make_array((9, 11, 13), np.float32, seed=21)
    fp3 = sp_nd.generate_binary_structure(3, 1)  # 7-tap 3D cross
    for rank in (0, 1, 3, 5, 6):
        ref = sp_nd.rank_filter(x3, rank, footprint=fp3, mode="reflect")
        with scimetal.precision("strict"), scimetal.device("cpu"):
            got_cpu = sm_nd.rank_filter(x3, rank, footprint=fp3, mode="reflect")
            assert scimetal.last_route().engine == "cpu"
        np.testing.assert_array_equal(got_cpu, ref)
        if _native.has_metal():
            with scimetal.precision("fast"), scimetal.device("gpu"):
                got_gpu = sm_nd.rank_filter(x3, rank, footprint=fp3, mode="reflect")
                assert scimetal.last_route().engine == "gpu"
            np.testing.assert_array_equal(got_gpu, ref)

    # 3x3x3 cube median (27 taps)
    ref_med = sp_nd.median_filter(x3, size=3, mode="constant", cval=0.5)
    if _native.has_metal():
        with scimetal.precision("fast"), scimetal.device("gpu"):
            got_med = sm_nd.median_filter(x3, size=3, mode="constant", cval=0.5)
            assert scimetal.last_route().engine == "gpu"
        np.testing.assert_array_equal(got_med, ref_med)

    # Percentile filter (negative, 0, 50, 100)
    for p in (-25.0, 0.0, 20.0, 50.0, 100.0):
        ref_p = sp_nd.percentile_filter(x3, p, size=(3, 2, 3))
        with scimetal.precision("strict"), scimetal.device("cpu"):
            got_p = sm_nd.percentile_filter(x3, p, size=(3, 2, 3))
            assert scimetal.last_route().engine == "cpu"
        np.testing.assert_array_equal(got_p, ref_p)


@pytest.mark.parametrize("fn_name", (
    "grey_erosion", "grey_dilation", "grey_opening", "grey_closing",
    "morphological_gradient", "morphological_laplace", "white_tophat", "black_tophat",
))
@pytest.mark.parametrize("kind", ("size", "footprint", "structure"))
def test_grey_morphology_all_ops_strict_and_gpu(fn_name, kind):
    x = _make_array((29, 31), np.float64, seed=31)
    sp_fn = getattr(sp_nd, fn_name)
    sm_fn = getattr(sm_nd, fn_name)
    if kind == "size":
        kw = dict(size=(5, 4), origin=(1, -1), mode="reflect")
    elif kind == "footprint":
        fp = np.array([[0, 1, 1], [1, 1, 0], [0, 1, 0]], dtype=bool)
        kw = dict(footprint=fp, mode="nearest")
    else:
        fp = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
        st_arr = np.array([[0.0, 0.5, 0.0], [0.5, 1.0, 0.5], [0.0, 0.5, 0.0]])
        kw = dict(footprint=fp, structure=st_arr, mode="constant", cval=1.25)

    ref = sp_fn(x, **kw)
    with scimetal.precision("strict"), scimetal.device("cpu"):
        got_cpu = sm_fn(x, **kw)
        assert scimetal.last_route().engine == "cpu"
    np.testing.assert_array_equal(got_cpu, ref)

    if _native.has_metal():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", scimetal.PrecisionWarning)
            with scimetal.precision("fast"), scimetal.device("gpu"):
                got_gpu = sm_fn(x, **kw)
                assert scimetal.last_route().engine == "gpu"
        np.testing.assert_allclose(got_gpu, ref, rtol=1e-5, atol=1e-4)


def test_minmax_1d_nan_bit_identical_strict():
    x = np.array([1.0, np.nan, 3.0, 2.0, -5.0, np.nan, 4.0, 0.0, 9.0, 2.0])
    for fn_name in ("minimum_filter1d", "maximum_filter1d"):
        ref = getattr(sp_nd, fn_name)(x, 4, mode="reflect")
        with scimetal.precision("strict"), scimetal.device("cpu"):
            got = getattr(sm_nd, fn_name)(x, 4, mode="reflect")
            assert scimetal.last_route().engine == "cpu"
        np.testing.assert_array_equal(got, ref)
    # 2D strided axis with NaN
    x2 = np.arange(36.0).reshape(6, 6)
    x2[2, 3] = np.nan
    for fn_name in ("minimum_filter1d", "maximum_filter1d"):
        ref = getattr(sp_nd, fn_name)(x2, 3, axis=0, mode="constant", cval=1.0)
        with scimetal.precision("strict"), scimetal.device("cpu"):
            got = getattr(sm_nd, fn_name)(x2, 3, axis=0, mode="constant", cval=1.0)
            assert scimetal.last_route().engine == "cpu"
        np.testing.assert_array_equal(got, ref)


def test_guardrails_overflow_subnormal_and_large_int():
    if not _native.has_metal():
        pytest.skip("no GPU in CPU-only build")
    # Large float64 overflows float32 -> forced GPU declines to SciPy, AUTO falls back to CPU.
    huge = np.full((64, 64), 1e300, dtype=np.float64)
    with scimetal.precision("fast"), scimetal.device("gpu"):
        got = sm_nd.median_filter(huge, size=3)
        assert scimetal.last_route().engine == "scipy"
    np.testing.assert_array_equal(got, sp_nd.median_filter(huge, size=3))

    # Subnormal float64 -> forced GPU declines to SciPy, AUTO falls back to CPU.
    tiny = np.full((64, 64), 1e-45, dtype=np.float64)
    with scimetal.precision("fast"), scimetal.device("gpu"):
        got_tiny = sm_nd.maximum_filter(tiny, size=3)
        assert scimetal.last_route().engine == "scipy"
    np.testing.assert_array_equal(got_tiny, sp_nd.maximum_filter(tiny, size=3))

    # int64 with |x| >= 2^24 -> forced GPU declines to SciPy, AUTO stays exact on CPU.
    big_int = np.full((64, 64), (1 << 30) + 7, dtype=np.int64)
    with scimetal.precision("fast"), scimetal.device("gpu"):
        got_int = sm_nd.median_filter(big_int, size=3)
        assert scimetal.last_route().engine == "scipy"
    np.testing.assert_array_equal(got_int, sp_nd.median_filter(big_int, size=3))


def test_inplace_and_output_dtypes():
    x = _make_array((19, 23), np.float32, seed=44)
    for fn_name, kw in (
        ("minimum_filter", dict(size=3)),
        ("maximum_filter", dict(size=(3, 5))),
        ("median_filter", dict(size=3)),
        ("rank_filter", dict(rank=2, size=3)),
        ("grey_erosion", dict(size=3)),
        ("grey_dilation", dict(size=3)),
        ("grey_opening", dict(size=3)),
        ("grey_closing", dict(size=3)),
    ):
        a = x.copy()
        b = x.copy()
        getattr(sp_nd, fn_name)(a, output=a, **kw)
        with scimetal.precision("strict"):
            ret = getattr(sm_nd, fn_name)(b, output=b, **kw)
        assert ret is b
        np.testing.assert_array_equal(b, a)

        # Cross-dtype output=np.float64
        ref_d = getattr(sp_nd, fn_name)(x, output=np.float64, **kw)
        with scimetal.precision("strict"):
            got_d = getattr(sm_nd, fn_name)(x, output=np.float64, **kw)
        assert got_d.dtype == np.float64
        np.testing.assert_array_equal(got_d, ref_d)


def test_declines_preserve_scipy_exceptions():
    x = np.arange(16.0).reshape(4, 4)
    # All-zero footprint -> ValueError
    with pytest.raises(ValueError):
        sm_nd.minimum_filter(x, footprint=np.zeros((3, 3), dtype=bool))
    assert scimetal.last_route().engine == "scipy"
    # Invalid origin -> ValueError
    with pytest.raises(ValueError):
        sm_nd.median_filter(x, size=3, origin=2)
    assert scimetal.last_route().engine == "scipy"
    # Non-integer rank -> TypeError
    with pytest.raises(TypeError):
        sm_nd.rank_filter(x, 0.5, size=3)
    assert scimetal.last_route().engine == "scipy"
    # Complex input -> TypeError
    with pytest.raises(TypeError):
        sm_nd.maximum_filter(x.astype(np.complex64), size=3)
    assert scimetal.last_route().engine == "scipy"
    # Sequence of modes on non-separable -> RuntimeError
    with pytest.raises(RuntimeError):
        sm_nd.median_filter(x, size=3, mode=["reflect", "nearest"])
    assert scimetal.last_route().engine == "scipy"


def test_multithreaded_morphology_and_rank():
    x = _make_array((128, 128), np.float32, seed=99)
    ref_med = sp_nd.median_filter(x, size=3)
    ref_top = sp_nd.white_tophat(x, size=5)

    def worker(i):
        if i % 2 == 0:
            out = sm_nd.median_filter(x, size=3)
            np.testing.assert_array_equal(out, ref_med)
        else:
            out = sm_nd.white_tophat(x, size=5)
            np.testing.assert_array_equal(out, ref_top)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(worker, range(32)))


@hypothesis.settings(max_examples=40, deadline=None)
@hypothesis.given(
    h=st.integers(min_value=1, max_value=24),
    w=st.integers(min_value=1, max_value=24),
    k0=st.integers(min_value=1, max_value=7),
    k1=st.integers(min_value=1, max_value=7),
    mode=st.sampled_from(MODES),
    dtype=st.sampled_from(GPU_EXACT_DTYPES),
    seed=st.integers(min_value=0, max_value=10000),
)
def test_hypothesis_rank_and_minmax_2d(h, w, k0, k1, mode, dtype, seed):
    x = _make_array((h, w), dtype, seed=seed)
    cval = 2.0 if dtype != np.bool_ else 1.0
    o0 = -(k0 // 2) if (seed & 1) else 0
    o1 = (k1 - 1) // 2 if (seed & 2) else 0
    rank = (seed // 4) % (k0 * k1)

    ref_r = sp_nd.rank_filter(x, rank, size=(k0, k1), mode=mode, cval=cval, origin=(o0, o1))
    with scimetal.precision("strict"), scimetal.device("cpu"):
        got_cpu = sm_nd.rank_filter(x, rank, size=(k0, k1), mode=mode, cval=cval, origin=(o0, o1))
        assert scimetal.last_route().engine == "cpu"
    np.testing.assert_array_equal(got_cpu, ref_r)

    if _native.has_metal():
        with scimetal.precision("fast"), scimetal.device("gpu"):
            got_gpu = sm_nd.rank_filter(x, rank, size=(k0, k1), mode=mode, cval=cval, origin=(o0, o1))
            expected_engine = "gpu" if (k0 > 1 or k1 > 1) else "cpu"
            assert scimetal.last_route().engine == expected_engine
        np.testing.assert_array_equal(got_gpu, ref_r)


@hypothesis.settings(max_examples=30, deadline=None)
@hypothesis.given(
    h=st.integers(min_value=2, max_value=20),
    w=st.integers(min_value=2, max_value=20),
    k0=st.integers(min_value=1, max_value=6),
    k1=st.integers(min_value=1, max_value=6),
    mode=st.sampled_from(MODES),
    op=st.sampled_from(("grey_opening", "grey_closing", "morphological_gradient", "morphological_laplace",
                        "white_tophat", "black_tophat")),
    seed=st.integers(min_value=0, max_value=10000),
)
def test_hypothesis_grey_morphology_2d(h, w, k0, k1, mode, op, seed):
    x = _make_array((h, w), np.float32, seed=seed)
    ref = getattr(sp_nd, op)(x, size=(k0, k1), mode=mode, cval=-1.5)
    with scimetal.precision("strict"), scimetal.device("cpu"):
        got_cpu = getattr(sm_nd, op)(x, size=(k0, k1), mode=mode, cval=-1.5)
        assert scimetal.last_route().engine == "cpu"
    np.testing.assert_array_equal(got_cpu, ref)

    if _native.has_metal():
        with scimetal.precision("fast"), scimetal.device("gpu"):
            got_gpu = getattr(sm_nd, op)(x, size=(k0, k1), mode=mode, cval=-1.5)
            is_noop = (k0 == 1 and k1 == 1 and op in ("grey_opening", "grey_closing"))
            assert scimetal.last_route().engine == ("cpu" if is_noop else "gpu")
        np.testing.assert_allclose(got_gpu, ref, rtol=1e-6, atol=1e-5)
