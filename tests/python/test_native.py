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
        _native.astype(np.arange(10), np.float32)            # integer input
    with pytest.raises(metal_sci.NotAccelerated):
        _native.astype(np.zeros((1,) * 9), np.float32)       # rank 9
    with pytest.raises(ValueError, match="complex"):
        _native.astype(np.zeros(4), np.complex64)             # real -> complex


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs fork")
def test_forked_child_falls_through_instead_of_touching_metal():
    _native.astype(np.zeros(8), np.float32)  # make sure the parent context exists
    read, write = os.pipe()
    pid = os.fork()
    if pid == 0:
        try:
            _native.astype(np.zeros(8), np.float32)
            os.write(write, b"ran")
        except metal_sci.NotAccelerated:
            os.write(write, b"declined")
        except BaseException as error:  # pragma: no cover
            os.write(write, repr(error).encode())
        finally:
            os._exit(0)
    os.waitpid(pid, 0)
    assert os.read(read, 64) == b"declined"
