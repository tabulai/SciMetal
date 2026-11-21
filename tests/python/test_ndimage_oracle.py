"""SciPy's own filter tests, run against the native routes (the oracle).

The oracle (tests/oracle/scimetal_oracle.py) runs in a subprocess under both
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
    parts = [str(ROOT / "tests/oracle")]
    if not os.environ.get("SCIMETAL_TEST_INSTALLED"):
        parts.insert(0, str(ROOT / "python"))
    if env.get("PYTHONPATH"):
        parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(parts)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "scimetal_oracle", "-p", "no:cacheprovider", "--pyargs",
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

    import scimetal
    import scimetal.ndimage as mn

    with scimetal.precision("strict"), scimetal.device("cpu"):
        mn.correlate1d(np.zeros(4), [1.0])
        if scimetal.last_route().engine != "cpu":
            pytest.skip(f"no native ndimage filters: {scimetal.last_route().reason}")


def test_scipys_filter_tests_pass_on_the_native_routes_in_strict_mode(native_filters):
    code, output, counters = run_oracle(SCIMETAL_ORACLE_PRECISION="strict", SCIMETAL_ORACLE_DEVICE="auto")
    assert code == 0 and re.search(r"\b\d+ passed", output), summary(output)
    # The suite really ran native code: every function has native calls, including integer/bool dtypes.
    assert set(counters) == set(FUNCTIONS), summary(output)
    for name, (native, _) in counters.items():
        assert native > 0, f"{name} never ran natively:\n{summary(output)}"
    assert counters["correlate1d"][0] >= 1000, summary(output)
    assert counters["convolve1d"][0] >= 800, summary(output)
    assert counters["correlate"][0] >= 700, summary(output)
    assert counters["convolve"][0] >= 650, summary(output)
    assert counters["uniform_filter"][0] >= 150, summary(output)
    assert counters["prewitt"][0] >= 50, summary(output)
    assert counters["sobel"][0] >= 50, summary(output)


# What float32 arithmetic cannot pass. test_filter_axes* compare two float64
# results that differ only in the order of the passes to 6e-8 relative (four
# times the square root of the float64 epsilon); the gaussian_laplace tests
# compare sums of second-derivative filters of data up to 900 to 1.5e-6
# absolute. Float32 is good for about 1e-7 of the data scale per operation.
# Separately, forced-GPU multi-pass integer filters compute intermediates in
# float32 and quantize once on output, whereas SciPy's composite references
# quantize each intermediate derivative/correlation pass into uint32/uint64/int32.
FLOAT32_ACCURACY = re.compile(r"test_filter_axes(_kwargs)?\[numpy-axes\d+-(gaussian|uniform)_filter\]"
                              r"|test_derivative_filter_axes\[numpy-gaussian_gradient_magnitude\]"
                              r"|test_gaussian_laplace0[12]\[numpy-float(32|64)\]")
INT_INTERMEDIATE_QUANTIZATION = re.compile(
    r"test_(prewitt|sobel)0[123]\[numpy-uint(32|64)\]"
    r"|test_gaussian_(laplace|gradient_magnitude)0[12]\[numpy-int32\]"
)


def test_gpu_route_fails_only_scipy_tests_that_need_more_than_float32(native_filters):
    import scimetal._native as _native

    if not _native.has_metal():
        pytest.skip("no GPU engine in CPU-only build")
    code, output, counters = run_oracle(SCIMETAL_ORACLE_PRECISION="fast", SCIMETAL_ORACLE_DEVICE="gpu")
    failed = re.findall(r"^FAILED \S*::(.+?)(?: - .*)?$", output, re.M)
    unexpected = [name for name in failed if not (FLOAT32_ACCURACY.fullmatch(name) or INT_INTERMEDIATE_QUANTIZATION.fullmatch(name))]
    assert not unexpected, f"not float32/int-intermediate accuracy: {unexpected}\n{summary(output)}"
    assert code in (0, 1), summary(output)  # 1: some tests failed; anything else is a broken run
    assert set(counters) == set(FUNCTIONS), summary(output)
    for name, (native, _) in counters.items():
        assert native > 0, f"{name} never ran natively in GPU mode:\n{summary(output)}"
    # Differences are either tiny rounding (below 1e-3 for FLOAT32_ACCURACY) or
    # intermediate pass clamping/truncation (<= 20 for INT_INTERMEDIATE_QUANTIZATION).
    differences = [float(x) for x in re.findall(r"Max absolute difference among violations: (\S+)", output)]
    assert len(differences) == len(failed), summary(output)
    for name, d in zip(failed, differences):
        if INT_INTERMEDIATE_QUANTIZATION.fullmatch(name):
            assert d <= 20, f"{name}: unexpected int quantization diff {d}\n{summary(output)}"
        else:
            assert d < 1e-3, f"{name}: unexpected float32 accuracy diff {d}\n{summary(output)}"
