"""Properties every DFT has, checked on the GPU route with Hypothesis.

Lengths are drawn from a fixed pool that covers every plan class (so the run
reuses a bounded set of pipelines); shapes, dtypes, norms and data are free.
Tolerances are multiples of the single-precision bound of test_fft_native.
"""
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import scimetal
import scimetal.fft as mfft
from scimetal import _native

TOLERANCE = 5e-6
# Stockham, Rader, Bluestein, four-step and strided-chain lengths.
LENGTHS = [2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 16, 17, 30, 47, 64, 96, 100, 243, 256, 289, 1000, 1009, 1021, 1024,
           2003, 2048, 3003, 4096, 5000, 6000, 8192, 30030, 65536]
INNER_LENGTHS = [2, 3, 5, 8, 12, 13, 16, 30, 64, 77, 96, 100, 256]
DTYPES = [np.complex64, np.complex128]
NORMS = [None, "backward", "ortho", "forward"]

cases = settings(max_examples=60, deadline=None)


@pytest.fixture(autouse=True)
def gpu():
    if not _native.has_metal():
        pytest.skip("requires Metal GPU build")
    with scimetal.device("gpu"), scimetal.precision("fast"):
        yield


def draw(seed, shape, dtype):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    return x.astype(dtype)


def close(a, b, factor=1.0):
    scale = max(float(np.linalg.norm(b)), 1e-30)
    assert float(np.linalg.norm(a - b)) <= factor * TOLERANCE * scale, float(np.linalg.norm(a - b)) / scale


def on_gpu():
    assert scimetal.last_route().engine == "gpu", str(scimetal.last_route())


shapes = st.tuples(st.integers(1, 6), st.sampled_from(LENGTHS))
seeds = st.integers(0, 2 ** 32 - 1)


@cases
@given(shapes, st.sampled_from(DTYPES), st.sampled_from(NORMS), seeds)
def test_inverse_round_trip(shape, dtype, norm, seed):
    x = draw(seed, shape, dtype)
    spectrum = mfft.fft(x, norm=norm)
    on_gpu()
    back = mfft.ifft(spectrum, norm=norm)
    on_gpu()
    assert back.dtype == dtype
    close(back, x, 2)
    real = x.real.copy()
    restored = mfft.irfft(mfft.rfft(real, norm=norm), shape[-1], norm=norm)
    on_gpu()
    assert restored.dtype == real.dtype
    close(restored, real, 2)


@cases
@given(shapes, st.sampled_from(DTYPES), seeds, st.floats(-3, 3), st.floats(-3, 3))
def test_linearity(shape, dtype, seed, a, b):
    x, y = draw(seed, shape, dtype), draw(seed + 1, shape, dtype)
    a, b = dtype(a + 0.5j), dtype(b - 0.25j)
    combined = mfft.fft(a * x + b * y)
    on_gpu()
    expected = a * mfft.fft(x) + b * mfft.fft(y)
    scale = np.linalg.norm(mfft.fft(x)) * abs(a) + np.linalg.norm(mfft.fft(y)) * abs(b)
    assert np.linalg.norm(combined - expected) <= 3 * TOLERANCE * max(scale, 1e-30)


@cases
@given(shapes, st.sampled_from(DTYPES), st.sampled_from(NORMS), seeds)
def test_parseval(shape, dtype, norm, seed):
    x = draw(seed, shape, dtype)
    n = shape[-1]
    spectrum = mfft.fft(x, norm=norm).astype(np.complex128)
    on_gpu()
    # sum |X|^2 = n sum |x|^2 for the unnormalised transform, per row.
    factor = {None: n, "backward": n, "ortho": 1.0, "forward": 1.0 / n}[norm]
    energy = np.sum(np.abs(x.astype(np.complex128)) ** 2, axis=-1)
    np.testing.assert_allclose(np.sum(np.abs(spectrum) ** 2, axis=-1), factor * energy, rtol=4 * TOLERANCE)
    # The zero-frequency term is the sum of the input.
    scale = {None: 1.0, "backward": 1.0, "ortho": n ** -0.5, "forward": 1.0 / n}[norm]
    np.testing.assert_allclose(spectrum[..., 0], scale * x.astype(np.complex128).sum(axis=-1), rtol=0,
                               atol=4 * TOLERANCE * np.sqrt(n * energy.max()) * scale)


@cases
@given(shapes, st.sampled_from([np.float32, np.float64]), st.sampled_from(NORMS), seeds)
def test_real_input_symmetry(shape, dtype, norm, seed):
    x = draw(seed, shape, np.complex128).real.astype(dtype)
    n = shape[-1]
    full = mfft.fft(x, norm=norm)
    on_gpu()
    half = mfft.rfft(x, norm=norm)
    on_gpu()
    assert half.shape[-1] == n // 2 + 1 and half.dtype == full.dtype
    close(half, full[..., : n // 2 + 1], 2)
    # A real signal has a Hermitian spectrum: X[n - k] = conj(X[k]).
    close(full[..., 1:][..., ::-1], np.conj(full[..., 1:]), 2)
    # irfft reads only the half spectrum and ignores imag(X[0]).
    perturbed = half.copy()
    perturbed[..., 0] += 1j
    close(mfft.irfft(perturbed, n, norm=norm), mfft.irfft(half, n, norm=norm), 2)


@cases
@given(shapes, st.sampled_from(DTYPES), seeds, st.integers(0, 10 ** 6))
def test_shift_theorem(shape, dtype, seed, shift):
    x = draw(seed, shape, dtype)
    n = shape[-1]
    shift %= n
    k = np.arange(n)
    ramp = np.exp(-2j * np.pi * ((k * shift) % n) / n)
    shifted = mfft.fft(np.roll(x, shift, axis=-1))
    on_gpu()
    close(shifted, mfft.fft(x).astype(np.complex128) * ramp, 2)


@cases
@given(st.sampled_from(INNER_LENGTHS), st.sampled_from(INNER_LENGTHS), st.integers(1, 3), st.sampled_from(DTYPES),
       st.sampled_from(NORMS), seeds)
def test_separability(rows, columns, depth, dtype, norm, seed):
    x = draw(seed, (depth, rows, columns), dtype)
    both = mfft.fft2(x, norm=norm)
    on_gpu()
    close(both, mfft.fft(mfft.fft(x, axis=-1, norm=norm), axis=-2, norm=norm), 2)
    close(mfft.ifft2(both, norm=norm), x, 3)
    real = x.real.copy()
    half = mfft.rfft2(real, norm=norm)
    on_gpu()
    close(half, mfft.fft2(real, norm=norm)[..., : columns // 2 + 1], 2)
    close(mfft.irfft2(half, s=(rows, columns), norm=norm), real, 3)
    # fftn over all axes equals fft2 followed by the remaining axis.
    close(mfft.fftn(x, norm=norm), mfft.fft(both, axis=0, norm=norm), 3)


@cases
@given(shapes, st.sampled_from(DTYPES), seeds, st.integers(1, 3))
def test_padding_and_truncation(shape, dtype, seed, extra):
    # n pads with zeros or truncates before the same transform of the same
    # numbers: the results are identical, not merely close.
    x = draw(seed, shape, dtype)
    n = shape[-1]
    longer = [m for m in LENGTHS if m > n][:extra]
    if longer:
        m = longer[-1]
        padded = np.zeros(shape[:-1] + (m,), dtype)
        padded[..., :n] = x
        np.testing.assert_array_equal(mfft.fft(x, m), mfft.fft(padded))
        on_gpu()
    shorter = [m for m in LENGTHS if m < n]
    if shorter:
        m = shorter[-1]
        np.testing.assert_array_equal(mfft.ifft(x, m), mfft.ifft(np.ascontiguousarray(x[..., :m])))
        on_gpu()
