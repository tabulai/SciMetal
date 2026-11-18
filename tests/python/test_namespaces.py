"""The package is a drop-in for SciPy: names fall through, policies behave."""
import importlib
import threading

import pytest
import scipy.ndimage
import scipy.optimize
import scipy.signal
import scipy.sparse.csgraph
import scipy.sparse.linalg
import scipy.stats

import metal_sci


def test_unaccelerated_names_are_scipys_own_objects():
    import metal_sci.ndimage as ndimage
    import metal_sci.sparse.linalg as spla

    assert ndimage.label is scipy.ndimage.label
    assert spla.splu is scipy.sparse.linalg.splu
    assert metal_sci.sparse.csr_array is scipy.sparse.csr_array


def test_namespaces_without_a_module_alias_scipy():
    assert metal_sci.stats.norm is scipy.stats.norm
    from metal_sci import optimize
    assert optimize.minimize is scipy.optimize.minimize


def test_nested_scipy_modules_import_through_the_alias_finder():
    windows = importlib.import_module("metal_sci.signal.windows")
    assert windows.hann is scipy.signal.windows.hann
    from metal_sci.sparse.csgraph import dijkstra
    assert dijkstra is scipy.sparse.csgraph.dijkstra


def test_missing_names_raise_attribute_error():
    import metal_sci.ndimage as ndimage

    with pytest.raises(AttributeError, match="no_such_function"):
        ndimage.no_such_function
    with pytest.raises(AttributeError):
        metal_sci.no_such_namespace
    with pytest.raises(ImportError):
        importlib.import_module("metal_sci.no_such_namespace")


def test_dir_lists_scipy_names():
    import metal_sci.signal as signal

    assert {"convolve", "fftconvolve", "windows"} <= set(dir(signal))


def test_precision_policy_default_override_and_validation():
    assert metal_sci.get_precision() == "fast"
    with metal_sci.precision("strict"):
        assert metal_sci.get_precision() == "strict"
        with metal_sci.precision("refined"):
            assert metal_sci.get_precision() == "refined"
        assert metal_sci.get_precision() == "strict"
    assert metal_sci.get_precision() == "fast"
    with pytest.raises(ValueError, match="precision"):
        metal_sci.set_precision("exact")
    with pytest.raises(ValueError, match="device"):
        metal_sci.set_device("npu")


def test_context_override_does_not_leak_across_threads():
    seen = []
    with metal_sci.precision("strict"):
        worker = threading.Thread(target=lambda: seen.append(metal_sci.get_precision()))
        worker.start()
        worker.join()
    assert seen == ["fast"]


def test_fall_through_is_recorded_and_returns_scipys_result():
    import numpy as np
    from metal_sci._dispatch import NotAccelerated, accelerated, counters, registry

    # A function no family accelerates, so the probe route shadows nothing.
    assert "white_tophat" not in registry.get("ndimage", ())

    @accelerated("ndimage.white_tophat")
    def white_tophat(input, size=None, *args, **kwargs):
        raise NotAccelerated("probe route always declines")

    try:
        x = np.arange(10.0)
        out = white_tophat(x, 3)
        np.testing.assert_array_equal(out, scipy.ndimage.white_tophat(x, 3))
        route = metal_sci.last_route()
        assert route.engine == "scipy" and "declines" in route.reason
        assert "white_tophat" in metal_sci.explain()
        assert counters["ndimage.white_tophat"] == [0, 1]
    finally:
        registry["ndimage"].discard("white_tophat")
        counters.pop("ndimage.white_tophat", None)


def test_aliased_modules_are_never_loaded_twice():
    import sys

    import scipy.stats.distributions

    aliased = importlib.import_module("metal_sci.stats.distributions")
    assert aliased is scipy.stats.distributions
    assert importlib.import_module("metal_sci.signal.windows") is scipy.signal.windows
    duplicates = [name for name, module in sys.modules.items()
                  if name.startswith("metal_sci.") and getattr(module, "__name__", "").startswith("metal_sci.")
                  and "site-packages/scipy" in (getattr(module, "__file__", None) or "")]
    assert duplicates == []


def test_accelerated_functions_pickle_as_themselves():
    import pickle

    import metal_sci.fft
    import metal_sci.ndimage

    for function in (metal_sci.ndimage.gaussian_filter, metal_sci.fft.fft):
        assert pickle.loads(pickle.dumps(function)) is function
