"""SciPy's own scipy.fft test-suite, run against metal-sci's routes (the oracle).

Two roles:

- As a test module it runs ``scipy.fft.tests.test_basic`` in a subprocess with
  the oracle plugin (tests/oracle/metal_sci_oracle.py) in both configurations
  and checks the outcome: everything passes in the default strict mode, and
  with the GPU forced under the fast policy the only failures are SciPy tests
  that hold single-precision results to double-precision tolerances.

- As a pytest plugin (``-p test_fft_oracle``, with tests/python on the path)
  it makes those runs possible: SciPy's tests look every function they
  exercise up in ``scipy._lib._array_api.xp_capabilities_table`` by identity,
  so the wrappers the oracle patches into ``scipy.fft`` must be entered there
  with the capabilities of the functions they replace. (The oracle plugin is
  the place for this; until it does it for every namespace, this hook does it
  for scipy.fft.)

    METAL_SCI_LIBRARY=... PYTHONPATH=python:tests/oracle:tests/python python -m pytest \
        -p metal_sci_oracle -p test_fft_oracle --pyargs scipy.fft.tests.test_basic -q
"""
import os
import pathlib
import re
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
NAMES = ["fft", "fft2", "fftn", "ifft", "ifft2", "ifftn", "irfft", "irfft2", "irfftn", "rfft", "rfft2", "rfftn"]


@pytest.hookimpl(trylast=True)
def pytest_configure(config):
    if not config.pluginmanager.hasplugin("metal_sci_oracle"):
        return
    import scipy.fft
    from scipy._lib._array_api import xp_capabilities_table

    for name in NAMES:
        patched = getattr(scipy.fft, name)
        original = getattr(patched, "__wrapped__", None)
        if original in xp_capabilities_table:
            xp_capabilities_table[patched] = xp_capabilities_table[original]


def run_oracle(**environment):
    env = dict(os.environ)
    env.pop("METAL_SCI_PRECISION", None)
    env.pop("METAL_SCI_DEVICE", None)
    env.update(environment)
    paths = [str(ROOT / "python"), str(ROOT / "tests" / "oracle"), str(ROOT / "tests" / "python")]
    env["PYTHONPATH"] = os.pathsep.join(paths + [env["PYTHONPATH"]] if env.get("PYTHONPATH") else paths)
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "metal_sci_oracle", "-p", "test_fft_oracle", "--pyargs",
         "scipy.fft.tests.test_basic", "-q", "-rf", "-p", "no:cacheprovider"],
        capture_output=True, text=True, env=env, cwd=str(ROOT), timeout=1800)
    output = completed.stdout + completed.stderr
    summary = re.search(r"^(?:(\d+) failed, )?(\d+) passed.*$", output, re.MULTILINE)
    assert summary, output[-3000:]
    counters = {name: (int(native), int(fell)) for name, native, fell in
                re.findall(r"^fft\.(\w+): (\d+) native, (\d+) fell through to SciPy$", output, re.MULTILINE)}
    assert set(counters) == set(NAMES), output[-3000:]
    failed = sorted(set(re.findall(r"^FAILED \S+::(\S+?)(?: - .*)?$", output, re.MULTILINE)))
    return int(summary.group(1) or 0), int(summary.group(2)), counters, failed, output


def test_scipy_suite_passes_under_the_default_strict_policy():
    failed_count, passed, counters, failed, output = run_oracle()
    assert failed_count == 0 and passed > 100, output[-3000:]
    # Strict keeps double precision on SciPy; the suite's inputs are small, so
    # single-precision ones stay below the crossover too. Every call must still
    # have gone through the routes.
    assert all(native + fell > 0 for native, fell in counters.values()), counters
    assert sum(fell for _, fell in counters.values()) > 1000, counters


def test_scipy_suite_with_the_gpu_forced_fails_only_on_double_precision_tolerances():
    # With the GPU forced under the fast policy most calls run natively, in
    # single precision. SciPy's tests compare the results with double-precision
    # references at relative tolerances of 6e-8 to 1e-7 with no absolute term
    # (xp_assert_close's default for 64-bit dtypes), which single-precision
    # arithmetic cannot meet, so some two dozen of them fail (which ones varies
    # a little: the suite draws unseeded random inputs). What this test pins
    # down is that nothing fails any other way: a wrong dtype, shape or
    # exception would not be a tolerance failure.
    failed_count, passed, counters, failed, output = run_oracle(METAL_SCI_ORACLE_DEVICE="gpu",
                                                               METAL_SCI_ORACLE_PRECISION="fast")
    assert passed > 120 and failed_count < 40, output[-3000:]
    assert sum(native for native, _ in counters.values()) > 500, counters
    blocks = re.findall(r"^_{5,} (\S+) _{5,}$\n(.*?)(?=^_{5,} \S+ _{5,}$|^=+ )", output, re.MULTILINE | re.DOTALL)
    assert len(blocks) == failed_count, (len(blocks), failed_count)
    for name, text in blocks:
        assert "AssertionError" in text and "Not equal to tolerance" in text, \
            f"{name} failed on something other than a numerical tolerance:\n{text[-2000:]}"
        tolerance = float(re.search(r"Not equal to tolerance rtol=([0-9.e+-]+)", text).group(1))
        assert tolerance <= 1e-6, f"{name}: single precision should meet rtol={tolerance}\n{text[-2000:]}"
        # Single-precision accuracy: errors stay around 1e-6 of the data's scale
        # (the suite's arrays are O(1) to O(1e3) in magnitude).
        worst = float(re.search(r"Max absolute difference among violations: ([0-9.e+-]+)", text).group(1))
        assert worst < 2e-3, f"{name}: absolute difference {worst}\n{text[-2000:]}"
