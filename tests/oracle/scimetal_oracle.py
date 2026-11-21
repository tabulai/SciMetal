"""pytest plugin: run SciPy's own test suite against SciMetal's native routes.

    PYTHONPATH=python:tests/oracle python -m pytest -p scimetal_oracle \
        --pyargs scipy.ndimage.tests.test_filters

Before collection, every public SciPy function that has a native route is
replaced in its SciPy namespace by SciMetal's wrapper, so the tests exercise
the native code with SciPy's own inputs, edge cases and tolerances. Calls a
route declines still reach SciPy's original function. The end-of-run summary
counts native calls against fall-throughs per function: a suite that passes
only because everything fell through shows up there.

Environment:
    SCIMETAL_ORACLE_PRECISION   strict (default), refined or fast
    SCIMETAL_ORACLE_DEVICE      auto (default), cpu or gpu
"""
import importlib
import os


def pytest_configure(config):
    import scimetal

    scimetal.set_precision(os.environ.get("SCIMETAL_ORACLE_PRECISION", "strict"))
    scimetal.set_device(os.environ.get("SCIMETAL_ORACLE_DEVICE", "auto"))
    # SciPy's tests use small arrays: keep them on the native routes.
    scimetal.set_small_inputs("native")
    # SciPy's test modules look the functions they exercise up in its array-API
    # capabilities table, keyed by function object, while they are imported:
    # give each wrapper the entry of the SciPy function it stands in for.
    from scipy._lib import _array_api
    table = getattr(_array_api, "xp_capabilities_table", None)
    patched = []
    for namespace, names in scimetal.accelerated().items():
        scipy_module = importlib.import_module(f"scipy.{namespace}")
        ours = importlib.import_module(f"scimetal.{namespace}")
        for name in names:
            wrapper = getattr(ours, name)
            original = getattr(wrapper, "__wrapped__", None)  # functools.wraps: SciPy's function
            if table is not None and original in table:
                table[wrapper] = table[original]
            setattr(scipy_module, name, wrapper)
            patched.append(f"{namespace}.{name}")
    config._scimetal_patched = patched


def pytest_runtest_setup(item):
    # Some SciPy test suites (e.g. scipy.fft.tests.test_basic) draw unseeded
    # arrays from NumPy's legacy global RNG; pin it per test so oracle runs
    # are deterministic.
    import numpy as np

    np.random.seed(0)


def pytest_report_header(config):
    import scimetal

    patched = getattr(config, "_scimetal_patched", [])
    return (f"SciMetal oracle: precision={scimetal.get_precision()}, device={scimetal.get_device()}, "
            f"{len(patched)} functions patched: {', '.join(patched) or 'none'}")


def pytest_terminal_summary(terminalreporter):
    from scimetal._dispatch import counters

    terminalreporter.section("SciMetal native routes")
    if not counters:
        terminalreporter.write_line("no accelerated functions are registered")
    for name, (native, fell_through) in sorted(counters.items()):
        terminalreporter.write_line(f"{name}: {native} native, {fell_through} fell through to SciPy")
