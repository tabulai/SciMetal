import numpy as np
import pytest
import scipy.ndimage as sp_nd

import scimetal
import scimetal.ndimage as sm_nd


@pytest.mark.parametrize("name,args", [
    ("median_filter", ()), ("percentile_filter", (25,)), ("rank_filter", (1,)),
])
def test_wide_boolean_rank_uses_scipy_sliding_window(name, args):
    x = np.random.default_rng(12).integers(0, 2, size=65536).astype(bool)
    kwargs = dict(size=1001, mode="wrap", origin=13)
    with scimetal.device("auto"), scimetal.precision("fast"):
        result = getattr(sm_nd, name)(x, *args, **kwargs)
        assert scimetal.last_route().engine == "scipy"
    np.testing.assert_array_equal(result, getattr(sp_nd, name)(x, *args, **kwargs))


def test_sliding_rank_preserves_integer_padding_and_int64_order():
    with scimetal.device("auto"), scimetal.precision("strict"):
        x = np.resize(np.array([1, 2, 3], np.uint8), 16384)
        kwargs = dict(size=31, mode="constant", cval=-1, origin=15)
        actual = sm_nd.median_filter(x, **kwargs)
        assert scimetal.last_route().engine == "scipy"
        np.testing.assert_array_equal(actual, sp_nd.median_filter(x, **kwargs))

        x = np.full(16384, 2**53 + 1, dtype=np.int64)
        actual = sm_nd.median_filter(x, size=3)
        assert scimetal.last_route().engine == "scipy"
        np.testing.assert_array_equal(actual, x)


@pytest.mark.parametrize("special", [np.nan, -0.0])
def test_sliding_rank_keeps_scipy_float_tie_order(special):
    x = np.zeros(16384, dtype=np.float32)
    x[10] = special
    with scimetal.precision("strict"), scimetal.device("auto"):
        result = sm_nd.rank_filter(x, 14, size=32)
        assert scimetal.last_route().engine == "scipy"
    expected = sp_nd.rank_filter(x, 14, size=32)
    np.testing.assert_array_equal(result.view(np.uint32), expected.view(np.uint32))


def test_fast_1d_median_remains_native():
    x = np.random.default_rng(6).standard_normal(65536).astype(np.float32)
    with scimetal.precision("fast"), scimetal.device("auto"):
        result = sm_nd.median_filter(x, size=31)
        assert scimetal.last_route().engine in {"gpu", "cpu"}
    np.testing.assert_array_equal(result, sp_nd.median_filter(x, size=31))


@pytest.mark.parametrize("policy", ["strict", "fast"])
@pytest.mark.parametrize("dtype,width,rank,origin,mode", [
    (np.float32, 127, 63, -17, "reflect"),
    (np.int16, 128, 5, 63, "constant"),
    (np.float64, 1023, 511, 34, "wrap"),
    (np.uint32, 1023, 1021, -511, "mirror"),
])
def test_native_sliding_rank_strided_arrays(policy, dtype, width, rank, origin, mode):
    # Repeated values span chunk boundaries, including a partial final chunk.
    # Strides require copying in and scattering back out.
    n = 32783
    trend = (np.arange(2 * n) // 131) % 93
    x = (np.random.default_rng(17).integers(0, 21, size=2 * n) + trend).astype(dtype)[1::2]
    storage = np.full(2 * n, 1000, dtype=dtype)
    out = storage[::2]
    kwargs = dict(size=width, origin=origin, mode=mode, cval=71)
    with scimetal.device("auto"), scimetal.precision(policy):
        result = sm_nd.rank_filter(x, rank, output=out, **kwargs)
        assert scimetal.last_route().engine == "cpu"
    assert result is out
    np.testing.assert_array_equal(out, sp_nd.rank_filter(x, rank, **kwargs))
    np.testing.assert_array_equal(storage[1::2], 1000)


def test_constant_rank_copies_and_preserves_output_contract():
    x = np.full(16384, 2, np.float32)
    with scimetal.device("auto"):
        y = sm_nd.median_filter(x, 127)
        assert scimetal.last_route().reason == "constant 1-D rank input: copied"
        assert not np.shares_memory(y, x)
        np.testing.assert_array_equal(y, x)

        with pytest.warns(UserWarning, match="ignoring size") as seen:
            result = sm_nd.rank_filter(x, 5, size=31, footprint=np.ones(21, bool), output=x)
        assert len(seen) == 1
        assert result is x
        np.testing.assert_array_equal(result, 2)

        for pad in (0.0, -0.0):
            kwargs = dict(size=127, origin=63, mode="constant", cval=pad)
            actual = sm_nd.median_filter(x, **kwargs)
            assert scimetal.last_route().reason != "constant 1-D rank input: copied"
            expected = sp_nd.median_filter(x, **kwargs)
            np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))

        x[20:25] = 3  # The three sentinel values still match.
        actual = sm_nd.rank_filter(x, 5, size=7)
        np.testing.assert_array_equal(actual, sp_nd.rank_filter(x, 5, size=7))


@pytest.mark.parametrize("dtype,value,pad,expect_copy", [
    (np.float32, 0.0, 0.0, True),
    (np.float32, -0.0, -0.0, True),
    (np.float32, 1.0, 1.0 + 2**-26, True),
    (np.uint8, 2, 2.75, True),
    (np.float32, -0.0, 0.0, False),
    (np.uint8, 2, 258, False),
    (np.int16, 2, -1, False),
])
def test_constant_rank_padding_uses_effective_ordering_dtype(dtype, value, pad, expect_copy):
    x = np.full(65536, value, dtype=dtype)
    kwargs = dict(size=31, mode="constant", cval=pad, origin=15)
    with scimetal.device("auto"):
        actual = sm_nd.median_filter(x, **kwargs)
        if expect_copy:
            assert scimetal.last_route().reason == "constant 1-D rank input: copied"
        else:
            assert scimetal.last_route().engine == "scipy"
    expected = sp_nd.median_filter(x, **kwargs)
    np.testing.assert_array_equal(actual.view(np.uint8), expected.view(np.uint8))
