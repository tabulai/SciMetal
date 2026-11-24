"""CPU min/max paths for degenerate and oversized windows."""
import numpy as np
import pytest
import scipy.ndimage as scipy_ndimage

import scimetal
import scimetal.ndimage as ndimage


@pytest.mark.parametrize("name", ["minimum_filter1d", "maximum_filter1d"])
@pytest.mark.parametrize("mode", ["reflect", "mirror", "wrap", "constant", "nearest"])
def test_window_covering_whole_line(name, mode):
    # The last two rows require tap-order handling for signed zeros and NaNs.
    x = np.array([[2., 8., 3., 1.], [0., -0., 0., -0.], [3., np.nan, -5., 2.]])
    with scimetal.precision("strict"), scimetal.device("cpu"):
        for origin in (-4, 0, 4):
            options = dict(size=17, origin=origin, mode=mode, cval=-0.0)
            got = getattr(ndimage, name)(x, **options)
            expected = getattr(scipy_ndimage, name)(x, **options)
            assert scimetal.last_route().engine == "cpu"
            assert got.tobytes() == expected.tobytes()


@pytest.mark.parametrize("name", ["minimum_filter1d", "maximum_filter1d"])
def test_oversized_one_sided_windows_preserve_edge_values(name):
    x = np.arange(12., dtype=np.float32).reshape(3, 4)
    with scimetal.precision("strict"), scimetal.device("cpu"):
        for mode in ("nearest", "constant"):
            for origin in (-8, 8):
                options = dict(size=17, mode=mode, origin=origin, cval=7.)
                np.testing.assert_array_equal(getattr(ndimage, name)(x, **options),
                                              getattr(scipy_ndimage, name)(x, **options))
                assert scimetal.last_route().engine == "cpu"


@pytest.mark.parametrize("name", ["minimum_filter1d", "maximum_filter1d"])
@pytest.mark.parametrize("size", [1, 127])
def test_singleton_axis_preserves_scipy_line_conversion(name, size):
    # Integer comparisons go through a double line buffer even for one tap.
    a = np.array([[2**53 + 1], [2**53 + 3]], dtype=np.int64)
    snans = np.array([[0x7f800001], [0xff800023]], dtype=np.uint32).view(np.float32)
    with scimetal.precision("strict"), scimetal.device("cpu"):
        for x in (a, snans):
            for out_dtype in (x.dtype, np.float64):
                options = dict(size=size, output=out_dtype)
                got = getattr(ndimage, name)(x, **options)
                expected = getattr(scipy_ndimage, name)(x, **options)
                assert scimetal.last_route().engine == "cpu"
                assert got.tobytes() == expected.tobytes()
