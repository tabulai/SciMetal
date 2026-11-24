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

import scimetal


def test_unaccelerated_names_are_scipys_own_objects():
    import scimetal.ndimage as ndimage
    import scimetal.sparse.linalg as spla

    assert ndimage.label is scipy.ndimage.label
    assert spla.splu is scipy.sparse.linalg.splu
    assert scimetal.sparse.csr_array is scipy.sparse.csr_array


def test_namespaces_without_a_module_alias_scipy():
    assert scimetal.stats.norm is scipy.stats.norm
    from scimetal import optimize
    assert optimize.minimize is scipy.optimize.minimize


def test_nested_scipy_modules_import_through_the_alias_finder():
    windows = importlib.import_module("scimetal.signal.windows")
    assert windows.hann is scipy.signal.windows.hann
    from scimetal.sparse.csgraph import dijkstra
    assert dijkstra is scipy.sparse.csgraph.dijkstra


def test_missing_names_raise_attribute_error():
    import scimetal.ndimage as ndimage

    with pytest.raises(AttributeError, match="no_such_function"):
        ndimage.no_such_function
    with pytest.raises(AttributeError):
        scimetal.no_such_namespace
    with pytest.raises(ImportError):
        importlib.import_module("scimetal.no_such_namespace")


def test_dir_lists_scipy_names():
    import scimetal.signal as signal

    assert {"convolve", "fftconvolve", "windows"} <= set(dir(signal))


def test_precision_policy_default_override_and_validation():
    assert scimetal.get_precision() == "fast"
    with scimetal.precision("strict"):
        assert scimetal.get_precision() == "strict"
        with scimetal.precision("refined"):
            assert scimetal.get_precision() == "refined"
        assert scimetal.get_precision() == "strict"
    assert scimetal.get_precision() == "fast"
    with pytest.raises(ValueError, match="precision"):
        scimetal.set_precision("exact")
    with pytest.raises(ValueError, match="device"):
        scimetal.set_device("npu")


def test_context_override_does_not_leak_across_threads():
    seen = []
    with scimetal.precision("strict"):
        worker = threading.Thread(target=lambda: seen.append(scimetal.get_precision()))
        worker.start()
        worker.join()
    assert seen == ["fast"]


def test_fall_through_is_recorded_and_returns_scipys_result():
    import numpy as np
    from scimetal._dispatch import NotAccelerated, accelerated, counters, registry

    # A function no family accelerates, so the probe route shadows nothing.
    assert "distance_transform_bf" not in registry.get("ndimage", ())

    @accelerated("ndimage.distance_transform_bf")
    def distance_transform_bf(input, *args, **kwargs):
        raise NotAccelerated("probe route always declines")

    try:
        x = np.array([0, 1, 1, 0, 1])
        out = distance_transform_bf(x)
        np.testing.assert_array_equal(out, scipy.ndimage.distance_transform_bf(x))
        route = scimetal.last_route()
        assert route.engine == "scipy" and "declines" in route.reason
        assert "distance_transform_bf" in scimetal.explain()
        assert counters["ndimage.distance_transform_bf"] == [0, 1]
    finally:
        registry["ndimage"].discard("distance_transform_bf")
        counters.pop("ndimage.distance_transform_bf", None)


def test_aliased_modules_are_never_loaded_twice():
    import sys

    import scipy.stats.distributions

    aliased = importlib.import_module("scimetal.stats.distributions")
    assert aliased is scipy.stats.distributions
    assert importlib.import_module("scimetal.signal.windows") is scipy.signal.windows
    duplicates = [name for name, module in sys.modules.items()
                  if name.startswith("scimetal.") and getattr(module, "__name__", "").startswith("scimetal.")
                  and "site-packages/scipy" in (getattr(module, "__file__", None) or "")]
    assert duplicates == []


def test_accelerated_functions_pickle_as_themselves():
    import pickle

    import scimetal.fft
    import scimetal.ndimage

    for function in (scimetal.ndimage.gaussian_filter, scimetal.fft.fft):
        assert pickle.loads(pickle.dumps(function)) is function


def test_alias_preserves_scipy_metadata_and_handles_private_submodules_and_all():
    import importlib.resources

    orig_spec = scipy.stats.__spec__
    orig_loader = scipy.stats.__loader__
    orig_pkg = scipy.stats.__package__

    mstats = importlib.import_module("scimetal.stats")
    assert mstats is scipy.stats
    assert scipy.stats.__spec__ is orig_spec
    assert scipy.stats.__loader__ is orig_loader
    assert scipy.stats.__package__ is orig_pkg
    assert importlib.resources.files(scipy.stats) is not None

    import scipy.stats._constants as sp_consts
    m_consts = importlib.import_module("scimetal.stats._constants")
    assert m_consts is sp_consts

    import scimetal.ndimage as mnd
    import scimetal.signal as msig

    assert "butter" in msig.__all__
    assert "label" in mnd.__all__
    assert {"stats", "optimize", "ndimage", "fft", "signal"} <= set(dir(scimetal))
