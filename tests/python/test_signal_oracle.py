"""SciPy's own signal convolution/correlation test-suite, run against SciMetal (the oracle)."""
import os
import pathlib
import re
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
FUNCTIONS = (
    "choose_conv_method",
    "convolve",
    "convolve2d",
    "correlate",
    "correlate2d",
    "fftconvolve",
    "oaconvolve",
)


def run_oracle(*, extra_args=(), **settings):
    env = dict(os.environ)
    env.pop("SCIMETAL_PRECISION", None)
    env.pop("SCIMETAL_DEVICE", None)
    env.update(settings)
    parts = [str(ROOT / "tests/oracle")]
    if not os.environ.get("SCIMETAL_TEST_INSTALLED"):
        parts.insert(0, str(ROOT / "python"))
    if env.get("PYTHONPATH"):
        parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(parts)
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-p",
        "scimetal_oracle",
        "-p",
        "no:cacheprovider",
        "--pyargs",
        "scipy.signal.tests.test_signaltools",
        "-k",
        "Convolve or Correlate or choose_conv_method",
        "-q",
        "-rf",
        *extra_args,
    ]
    completed = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=1800)
    output = completed.stdout + completed.stderr
    counters = {
        name: (int(native), int(declined))
        for name, native, declined in re.findall(
            r"^signal\.(\w+): (\d+) native, (\d+) fell through to SciPy$", output, re.M
        )
    }
    return completed.returncode, output, counters


def summary(output):
    return "\n".join(output.splitlines()[-40:])


@pytest.fixture(scope="module")
def native_signal():
    import numpy as np

    import scimetal
    import scimetal.signal as msig

    with scimetal.precision("strict"), scimetal.device("cpu"):
        msig.convolve(np.ones(4), np.ones(2), method="direct")
        if scimetal.last_route().engine != "cpu":
            pytest.skip(f"no native signal routes: {scimetal.last_route().reason}")


def test_scipys_signal_tests_pass_in_strict_mode(native_signal):
    # Note: TestOAConvolve non-slow subset is run here for speed; full suite runs in integration gate.
    code, output, counters = run_oracle(
        extra_args=["-m", "not slow"],
        SCIMETAL_ORACLE_PRECISION="strict",
        SCIMETAL_ORACLE_DEVICE="auto",
    )
    assert code == 0 and re.search(r"\b\d+ passed", output), summary(output)
    assert set(counters) == set(FUNCTIONS), summary(output)
    # CPU-capable routes run natively in strict; GPU-only FFT routes decline double-precision tests cleanly.
    for name in ("convolve", "correlate", "convolve2d", "correlate2d", "choose_conv_method"):
        assert counters[name][0] > 0, f"{name} never ran natively:\n{summary(output)}"
    for name in ("fftconvolve", "oaconvolve"):
        assert sum(counters[name]) > 0, f"{name} never invoked:\n{summary(output)}"


def test_gpu_route_fails_only_scipy_tests_that_need_more_than_float32(native_signal):
    import scimetal._native as _native

    if not _native.has_metal():
        pytest.skip("no GPU engine in CPU-only build")
    code, output, counters = run_oracle(
        extra_args=["-m", "not slow"],
        SCIMETAL_ORACLE_PRECISION="fast",
        SCIMETAL_ORACLE_DEVICE="gpu",
    )
    assert code in (0, 1), summary(output)
    assert set(counters) == set(FUNCTIONS), summary(output)
    for name, (native, _) in counters.items():
        assert native > 0, f"{name} never ran natively in GPU mode:\n{summary(output)}"

    # Every failure under forced GPU fast mode must be a float32 vs float64 numerical tolerance failure.
    assert not re.search(r"\b\d+ error", output), summary(output)
    m_passed = re.search(r"\b(\d+) passed\b", output)
    assert m_passed and int(m_passed.group(1)) > 350, summary(output)
    m_failed = re.search(r"\b(\d+) failed\b", output)
    expected_failed = int(m_failed.group(1)) if m_failed else 0
    failed = re.findall(r"^FAILED \S*::(.+?)(?: - .*)?$", output, re.M)
    assert len(failed) == expected_failed and expected_failed < 1000, (len(failed), expected_failed, summary(output))
    blocks = re.findall(
        r"^_{5,} (.+?) _{5,}$\n(.*?)(?=^_{5,} .+? _{5,}$|^=+ )",
        output,
        re.M | re.DOTALL,
    )
    assert len(blocks) == expected_failed, (len(blocks), expected_failed, summary(output))
    for name, text in blocks:
        assert "AssertionError" in text and (
            "Not equal to tolerance" in text or "Arrays are not almost equal" in text
        ), f"{name} failed on something other than numerical tolerance:\n{text[-2000:]}"
        m_abs = re.search(r"Max absolute difference(?: among violations)?: ([0-9.eE+-]+)", text)
        assert m_abs is not None, f"{name} missing max absolute difference:\n{text[-2000:]}"
        assert float(m_abs.group(1)) < 5e-3, f"{name}: unexpected max abs diff {m_abs.group(1)}\n{text[-2000:]}"
