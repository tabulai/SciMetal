"""SciPy's own filter tests, run against the native routes (the oracle).

The oracle (tests/oracle/metal_sci_oracle.py) runs in a subprocess under both
configurations, so the ordinary test suite fails if SciPy's suite stops passing
in strict mode, if the native routes stop being exercised, or if the GPU route
starts failing tests other than the ones float32 arithmetic is known to fail.
"""
import os
import pathlib
import re
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
FUNCTIONS = ("convolve", "convolve1d", "correlate", "correlate1d", "gaussian_filter", "gaussian_filter1d",
             "gaussian_gradient_magnitude", "gaussian_laplace", "generic_gradient_magnitude", "generic_laplace",
             "laplace", "prewitt", "sobel", "uniform_filter", "uniform_filter1d")


def run_oracle(**settings):
    """Run SciPy's filter tests with the native routes patched in.

    Returns pytest's exit code, its output and {function: (native calls, calls handed to SciPy)}.
    """
    env = dict(os.environ, **settings)
    env["PYTHONPATH"] = os.pathsep.join(str(ROOT / part) for part in ("python", "tests/oracle"))
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "metal_sci_oracle", "-p", "no:cacheprovider", "--pyargs",
         "scipy.ndimage.tests.test_filters", "-q"],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=1800)
    output = result.stdout + result.stderr
    counters = {name: (int(native), int(declined)) for name, native, declined in
                re.findall(r"^ndimage\.(\w+): (\d+) native, (\d+) fell through to SciPy$", output, re.M)}
    return result.returncode, output, counters


def summary(output):
    return "\n".join(output.splitlines()[-40:])


@pytest.fixture(scope="module")
def native_filters():
    """Nothing to check where no native filter can run (see test_ndimage_filters)."""
    import numpy as np

    import metal_sci
    import metal_sci.ndimage as mn

    with metal_sci.precision("strict"), metal_sci.device("cpu"):
        mn.correlate1d(np.zeros(4), [1.0])
        if metal_sci.last_route().engine != "cpu":
            pytest.skip(f"no native ndimage filters: {metal_sci.last_route().reason}")


def test_scipys_filter_tests_pass_on_the_native_routes_in_strict_mode(native_filters):
    code, output, counters = run_oracle(METAL_SCI_ORACLE_PRECISION="strict", METAL_SCI_ORACLE_DEVICE="auto")
    assert code == 0 and re.search(r"\b\d+ passed", output), summary(output)
    # The suite really ran native code: every function has native calls.
    assert set(counters) == set(FUNCTIONS), summary(output)
    for name, (native, _) in counters.items():
        assert native > 0, f"{name} never ran natively:\n{summary(output)}"


# What float32 arithmetic cannot pass. test_filter_axes* compare two float64
# results that differ only in the order of the passes to 6e-8 relative (four
# times the square root of the float64 epsilon); the gaussian_laplace tests
# compare sums of second-derivative filters of data up to 900 to 1.5e-6
# absolute. Float32 is good for about 1e-7 of the data scale per operation.
FLOAT32_ACCURACY = re.compile(r"test_filter_axes(_kwargs)?\[numpy-axes\d+-(gaussian|uniform)_filter\]"
                              r"|test_derivative_filter_axes\[numpy-gaussian_gradient_magnitude\]"
                              r"|test_gaussian_laplace0[12]\[numpy-float(32|64)\]")


def test_gpu_route_fails_only_scipy_tests_that_need_more_than_float32(native_filters):
    code, output, counters = run_oracle(METAL_SCI_ORACLE_PRECISION="fast", METAL_SCI_ORACLE_DEVICE="gpu")
    failed = re.findall(r"^FAILED \S+::(\S+)", output, re.M)
    unexpected = [name for name in failed if not FLOAT32_ACCURACY.fullmatch(name)]
    assert not unexpected, f"not float32 accuracy: {unexpected}\n{summary(output)}"
    assert code in (0, 1), summary(output)  # 1: some tests failed; anything else is a broken run
    assert set(counters) == set(FUNCTIONS), summary(output)
    # Failures are differences of rounding, never wrong values: every reported
    # mismatch is below 1e-3 absolute on data of scale 100 to 1000. (A build
    # without a GPU engine hands every forced-GPU call to SciPy and fails nothing.)
    differences = [float(x) for x in re.findall(r"Max absolute difference among violations: (\S+)", output)]
    assert len(differences) == len(failed) and all(d < 1e-3 for d in differences), summary(output)
