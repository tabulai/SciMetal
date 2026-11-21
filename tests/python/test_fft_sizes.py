"""Every transform length the planner accepts computes the right transform.

The kernels are specialised per length, and MLX's thread-layout heuristics
have special cases for individual sizes, so nothing short of trying a length
shows that its plan is right. The default run tries a spread of lengths from
every class; ``METAL_SCI_FFT_EXHAUSTIVE=1`` tries all of them (every length
from 2 to 4096, every longer length up to 2^17 that factors into the radices,
and every such length along an inner axis), which builds some 20 000
pipelines and takes about half an hour the first time.
"""
import os

import numpy as np
import pytest
import scipy.fft

import metal_sci
import metal_sci.fft as mfft
from metal_sci import _native

TOLERANCE = 5e-6
EXHAUSTIVE = os.environ.get("METAL_SCI_FFT_EXHAUSTIVE", "0") == "1"


def smooth(n):
    for prime in (2, 3, 5, 7, 11, 13):
        while n % prime == 0:
            n //= prime
    return n == 1


def has_plan(n):
    """The planner's rule: radix factors at any length, anything up to 2048."""
    return n >= 2 and (smooth(n) or n <= 2048)


def spread(candidates, count):
    candidates = list(candidates)
    return candidates if EXHAUSTIVE or len(candidates) <= count else candidates[:: max(len(candidates) // count, 1)]


@pytest.fixture(autouse=True)
def gpu():
    if not _native.has_metal():
        pytest.skip("requires Metal GPU build")
    with metal_sci.device("gpu"), metal_sci.precision("fast"):
        yield


def error(result, reference):
    return float(np.linalg.norm(result - reference) / np.linalg.norm(reference))


def transforms(n, rows, seed):
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal((rows, n)) + 1j * rng.standard_normal((rows, n))).astype(np.complex64)
    r = rng.standard_normal((rows, n)).astype(np.float32)
    h = (rng.standard_normal((rows, n // 2 + 1)) + 1j * rng.standard_normal((rows, n // 2 + 1))).astype(np.complex64)
    return [("fft", lambda: mfft.fft(x), lambda: scipy.fft.fft(x.astype(np.complex128))),
            ("ifft", lambda: mfft.ifft(x), lambda: scipy.fft.ifft(x.astype(np.complex128))),
            ("rfft", lambda: mfft.rfft(r), lambda: scipy.fft.rfft(r.astype(np.float64))),
            ("irfft", lambda: mfft.irfft(h, n), lambda: scipy.fft.irfft(h.astype(np.complex128), n))]


def check_lengths(lengths, rows):
    failures, worst = [], 0.0
    for n in lengths:
        for name, ours, reference in transforms(n, rows, seed=n):
            result = ours()
            route = metal_sci.last_route()
            if has_plan(n):
                if route.engine != "gpu":
                    failures.append((name, n, route.reason))
                    continue
                value = error(result, reference())
                worst = max(worst, value)
                if not value < TOLERANCE:
                    failures.append((name, n, value))
            elif route.engine != "scipy":
                failures.append((name, n, "ran without a plan"))
    assert not failures, failures[:20]
    return worst


def test_lengths_up_to_4096():
    # Odd and even batches: the real kernels transform two rows at a time.
    lengths = spread(range(2, 4097), 150)
    worst = max(check_lengths(lengths, 3), check_lengths(lengths if EXHAUSTIVE else lengths[::5], 4))
    print(f"lengths 2..4096: {len(lengths)} tried, worst relative error {worst:.2e}")


def test_longer_lengths_with_radix_factors():
    lengths = spread((n for n in range(4097, (1 << 17) + 1) if smooth(n)), 60)
    lengths += [3 * 2 ** 18, 10 ** 6, 5 ** 9, 7 * 2 ** 20, 2 ** 21, 13 ** 5, 11 * 13 * 2 ** 12]
    worst = check_lengths(lengths, 2)
    print(f"longer lengths: {len(lengths)} tried, worst relative error {worst:.2e}")
    # Everything else above 4096 declines.
    declined = [4097, 4099, 5003, 8194, 17 * 4096, 100003, 2 ** 20 + 2]
    assert not any(has_plan(n) for n in declined)
    check_lengths(declined, 2)


def test_inner_axis_lengths():
    lengths = spread((n for n in range(2, 4097) if smooth(n)), 60)
    failures, worst = [], 0.0
    for n in lengths:
        for tail in (3, 16) if n > 256 else (1, 3, 16, 37):
            rng = np.random.default_rng(n + tail)
            x = (rng.standard_normal((2, n, tail)) + 1j * rng.standard_normal((2, n, tail))).astype(np.complex64)
            for name, function in (("fft", mfft.fft), ("ifft", mfft.ifft)):
                result = function(x, axis=1)
                if metal_sci.last_route().engine != "gpu":
                    failures.append((name, n, tail, metal_sci.last_route().reason))
                    continue
                value = error(result, getattr(scipy.fft, name)(x.astype(np.complex128), axis=1))
                worst = max(worst, value)
                if not value < TOLERANCE:
                    failures.append((name, n, tail, value))
    assert not failures, failures[:20]
    print(f"inner axis lengths: {len(lengths)} tried, worst relative error {worst:.2e}")
