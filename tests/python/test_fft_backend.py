"""metal_sci.fft as a scipy.fft backend: plain scipy.fft calls run natively."""
import numpy as np
import pytest
import scipy.fft
import scipy.signal
from scipy._lib.uarray import BackendNotImplementedError

import metal_sci
import metal_sci.fft as mfft
from metal_sci._dispatch import counters
from metal_sci.fft import _basic

TOLERANCE = 5e-6
NAMES = ["fft", "ifft", "rfft", "irfft", "fft2", "ifft2", "fftn", "ifftn", "rfft2", "irfft2", "rfftn", "irfftn"]


@pytest.fixture
def registered():
    assert not mfft.registered()
    mfft.register()
    try:
        with metal_sci.device("gpu"), metal_sci.precision("fast"):
            yield
    finally:
        mfft.unregister()
        assert not mfft.registered()


def arrays(seed=0):
    rng = np.random.default_rng(seed)
    real = rng.standard_normal((48, 1000))
    return real + 1j * rng.standard_normal((48, 1000)), real


def error(result, reference):
    return float(np.linalg.norm(result - reference) / np.linalg.norm(reference))


def tallies():
    return {name: tuple(counters[f"fft.{name}"]) for name in NAMES}


def test_registered_backend_runs_scipy_fft_natively(registered):
    x, r = arrays()
    for name in NAMES:
        argument = r if name.startswith("r") else x
        mfft.unregister()
        exact = getattr(scipy.fft, name)(argument)
        mfft.register()
        before = tallies()
        result = getattr(scipy.fft, name)(argument)
        route = metal_sci.last_route()
        assert route.function == f"fft.{name}" and route.engine == "gpu", str(route)
        after = tallies()
        assert after[name] == (before[name][0] + 1, before[name][1])
        assert {key: value for key, value in after.items() if key != name} == \
               {key: value for key, value in before.items() if key != name}
        assert result.dtype == exact.dtype and result.shape == exact.shape
        assert error(result, exact) < 2 * TOLERANCE
        assert result.tobytes() != exact.tobytes()   # single precision on the GPU, not SciPy's arithmetic


def test_arguments_reach_the_native_route(registered):
    x, r = arrays(1)
    cases = [("fft", (x, 512), {"norm": "ortho"}), ("fft", (x,), {"axis": 0}), ("ifft", (x, None, -1, "forward"), {}),
             ("rfft", (r,), {"n": 2048, "overwrite_x": True}), ("irfft", (x, 1999), {}),
             ("fft2", (x,), {"s": (48, 1024)}), ("fftn", (x,), {"axes": (1,), "workers": 1}),
             ("irfftn", (x,), {"s": (48, 1998)}), ("rfft2", (r,), {"norm": "forward"}), ("fft", (), {"x": x})]
    for name, args, kwargs in cases:
        mfft.unregister()
        exact = getattr(scipy.fft, name)(*args, **kwargs)
        mfft.register()
        result = getattr(scipy.fft, name)(*args, **kwargs)
        assert metal_sci.last_route().engine == "gpu", (name, kwargs, str(metal_sci.last_route()))
        assert result.dtype == exact.dtype and result.shape == exact.shape and error(result, exact) < 2 * TOLERANCE


def test_declined_calls_continue_to_scipy(registered):
    x, r = arrays(2)
    cases = [("fft", (np.tile(x, (1, 4))[:4, :3001].copy(),), {}),       # no plan for 3001 points
             ("fft", ([1.0, 2.0, 3.0],), {}),                             # not an array
             ("rfft", (r,), {"axis": 0}),                                 # real transform off the last axis
             ("fft2", (x,), {"s": (50, 1000)}),                           # resizing off the last axis
             ("fft", (x[:17, :64].copy(),), {"axis": 0})]                 # 17 points off the last axis
    for name, args, kwargs in cases:
        mfft.unregister()
        exact = getattr(scipy.fft, name)(*args, **kwargs)
        mfft.register()
        before = tallies()
        result = getattr(scipy.fft, name)(*args, **kwargs)
        route = metal_sci.last_route()
        assert route.function == f"fft.{name}" and route.engine == "scipy", str(route)
        assert tallies()[name] == (before[name][0], before[name][1] + 1)
        if "workers=" in route.reason:
            np.testing.assert_allclose(result, exact, rtol=0, atol=1e-13 * np.abs(exact).max())
        else:
            assert result.tobytes() == exact.tobytes()
    # Policies decline through the backend exactly as through metal_sci.fft.
    with metal_sci.device("auto"):
        scipy.fft.fft(x[:2, :64].copy())
        assert metal_sci.last_route().engine == "scipy" and "crossover" in metal_sci.last_route().reason
    with metal_sci.precision("strict"):
        batch = np.tile(x, (4, 1))   # 192 000 points: enough for SciPy's threads to pay
        result = scipy.fft.fft(batch)
        route = metal_sci.last_route()
        assert route.engine == "scipy" and "strict" in route.reason
        cores = _basic.performance_cores()
        if cores > 1:
            assert f"workers={cores}" in route.reason
            assert result.tobytes() == scipy.fft.fft(batch, workers=cores).tobytes()
        scipy.fft.fft(batch, workers=2)
        assert "workers=" not in metal_sci.last_route().reason
        scipy.fft.fft(x)             # too little data to thread: as the caller made it
        assert "workers=" not in metal_sci.last_route().reason


def test_errors_and_other_functions_are_scipys(registered):
    x, r = arrays(3)
    with pytest.raises(ValueError, match="invalid number of data points"):
        scipy.fft.fft(x, 0)
    with pytest.raises(TypeError, match="real sequence"):
        scipy.fft.rfft(x)
    with pytest.raises(NotImplementedError):
        scipy.fft.fft(x, plan=object())
    with pytest.raises(TypeError):
        scipy.fft.fft(x, 8, -1, None, False, None, None)   # too many positional arguments
    # Functions without a native route are untouched.
    mfft.unregister()
    expected = (scipy.fft.dct(r), scipy.fft.hfft(x), scipy.fft.ihfft(r), scipy.fft.fftshift(r))
    mfft.register()
    before = tallies()
    for result, exact in zip((scipy.fft.dct(r), scipy.fft.hfft(x), scipy.fft.ihfft(r), scipy.fft.fftshift(r)), expected):
        assert result.tobytes() == exact.tobytes()
    assert tallies() == before
    assert scipy.fft.next_fast_len(1000) == 1000


def test_unregister_restores_scipys_backend():
    x, r = arrays(4)
    with metal_sci.device("gpu"):
        exact = {name: getattr(scipy.fft, name)(r if name.startswith("r") else x) for name in NAMES}
        mfft.register()
        mfft.register()   # idempotent
        assert mfft.registered()
        assert scipy.fft.fft(x).tobytes() != exact["fft"].tobytes()
        mfft.unregister()
        assert not mfft.registered()
        before = tallies()
        for name in NAMES:
            assert getattr(scipy.fft, name)(r if name.startswith("r") else x).tobytes() == exact[name].tobytes()
        assert tallies() == before
        mfft.unregister()  # harmless twice
        mfft.register()
        assert error(scipy.fft.fft(x), exact["fft"]) < TOLERANCE and metal_sci.last_route().engine == "gpu"
        mfft.unregister()
        assert scipy.fft.fft(x).tobytes() == exact["fft"].tobytes()


def test_backend_object_with_scipys_context_manager():
    x, r = arrays(5)
    exact = scipy.fft.fft(x)
    exact_inner = scipy.fft.fft(x[:17], axis=0)
    assert not mfft.registered()
    with metal_sci.device("gpu"):
        with scipy.fft.set_backend(mfft.backend):
            result = scipy.fft.fft(x)
            assert metal_sci.last_route().engine == "gpu"
            assert result.tobytes() != exact.tobytes() and error(result, exact) < TOLERANCE
            # A declined call falls back to SciPy unless the caller asked for this backend only.
            assert scipy.fft.fft(x[:17], axis=0).tobytes() == exact_inner.tobytes()
            assert metal_sci.last_route().engine == "scipy"
            with scipy.fft.skip_backend(mfft.backend):
                assert scipy.fft.fft(x).tobytes() == exact.tobytes()
        with scipy.fft.set_backend(mfft.backend, only=True):
            assert error(scipy.fft.fft(x), exact) < TOLERANCE
            with pytest.raises(BackendNotImplementedError):
                scipy.fft.fft(x[:, :3], axis=0, n=3001)
        assert scipy.fft.fft(x).tobytes() == exact.tobytes()


def test_scipy_signal_uses_the_backend(registered):
    # fftconvolve calls scipy.fft.rfftn / irfftn internally.
    rng = np.random.default_rng(6)
    a, b = rng.standard_normal(1 << 19), rng.standard_normal(4097)
    mfft.unregister()
    exact = scipy.signal.fftconvolve(a, b)
    mfft.register()
    before = tallies()
    result = scipy.signal.fftconvolve(a, b)
    after = tallies()
    assert after["rfftn"][0] == before["rfftn"][0] + 2 and after["irfftn"][0] == before["irfftn"][0] + 1
    assert result.dtype == exact.dtype and error(result, exact) < 3 * TOLERANCE
