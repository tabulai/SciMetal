"""Work Package A5 fast-mode guard rails: overflow re-routing, explain() reasons, and PrecisionWarning."""
from __future__ import annotations

import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest
import scipy.fft
import scipy.ndimage as sn

import scimetal
import scimetal.fft as mfft
import scimetal.ndimage as mn
from scimetal import _native

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
PYTHON_DIR = str(REPO_ROOT / "python")


@pytest.fixture
def gpu():
    if not _native.has_metal():
        pytest.skip("requires Metal GPU build")
    with scimetal.device("gpu"), scimetal.precision("fast"):
        yield


def _make_1e39_array(shape, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(shape).astype(np.float64)
    flat = a.ravel()
    flat[0] = 1e39
    flat[-1] = -1e39
    return a


# ---- 1. ndimage fast-mode overflow guard rails (all 8 routes) ---------------------------------

NDIMAGE_AUTO_GPU_CASES = [
    ("correlate1d", (3072, 3072), (np.ones(9, dtype=np.float64) / 9.0,), {"axis": 0}),
    ("convolve1d", (3072, 3072), (np.ones(9, dtype=np.float64) / 9.0,), {"axis": 0}),
    ("gaussian_filter1d", (2048, 2048), (5.0,), {"axis": 0}),
    ("gaussian_filter", (2048, 2048), (3.0,), {}),
    ("uniform_filter1d", (2048, 2048), (9,), {"axis": 0}),
    ("uniform_filter", (2048, 2048), (5,), {}),
    ("prewitt", (3072, 3072), (), {}),
    ("sobel", (3072, 3072), (), {}),
    ("correlate", (2896, 2896), (np.full((5, 5), 0.04, dtype=np.float64) + np.eye(5) * 0.01,), {}),
    ("convolve", (2896, 2896), (np.full((5, 5), 0.04, dtype=np.float64) + np.eye(5) * 0.01,), {}),
    ("laplace", (2896, 2896), (), {}),
    ("gaussian_laplace", (2048, 2048), (3.0,), {}),
    ("gaussian_gradient_magnitude", (2048, 2048), (3.0,), {}),
    ("generic_laplace", (2048, 2048),
     (lambda inp, ax, out, mode, cval: mn.gaussian_filter(inp, 3.0, [2 if i == ax else 0 for i in range(inp.ndim)], out, mode, cval),),
     {}),
    ("generic_gradient_magnitude", (2048, 2048),
     (lambda inp, ax, out, mode, cval: mn.gaussian_filter(inp, 3.0, [1 if i == ax else 0 for i in range(inp.ndim)], out, mode, cval),),
     {}),
]


@pytest.mark.parametrize("name, shape, args, kwargs", NDIMAGE_AUTO_GPU_CASES)
def test_ndimage_auto_1e39_reroutes_to_cpu(name, shape, args, kwargs):
    if not _native.has_metal():
        pytest.skip("requires Metal GPU build")
    ours_fn = getattr(mn, name)
    theirs_fn = getattr(sn, name)

    # Confirm the shape/params genuinely target the GPU for normal data under AUTO.
    normal = np.ones(shape, dtype=np.float64)
    with scimetal.device("auto"), scimetal.precision("fast"):
        ours_fn(normal, *args, **kwargs)
        normal_route = scimetal.last_route()
        assert normal_route.engine == "gpu", f"expected GPU for control {name}, got {normal_route}"
        assert normal_route.reason == ""

    # With 1e39, AUTO must re-route to CPU in float64 with non-empty reason and return finite exact values.
    a = _make_1e39_array(shape, seed=1)
    with scimetal.device("auto"), scimetal.precision("fast"):
        got = ours_fn(a, *args, **kwargs)
        route = scimetal.last_route()
        explanation = scimetal.explain()
    ref = theirs_fn(a, *args, **kwargs)

    assert route.engine == "cpu"
    assert route.compute_dtype == "float64"
    assert route.reason != ""
    assert route.reason in explanation
    assert np.all(np.isfinite(got))
    assert got.tobytes() == ref.tobytes()


@pytest.mark.parametrize("name, args, kwargs", [
    ("correlate1d", (np.ones(5, dtype=np.float64) / 5.0,), {}),
    ("convolve1d", (np.ones(5, dtype=np.float64) / 5.0,), {}),
    ("gaussian_filter1d", (2.0,), {}),
    ("gaussian_filter", (2.0,), {}),
    ("uniform_filter1d", (5,), {}),
    ("uniform_filter", (5,), {}),
    ("prewitt", (), {}),
    ("sobel", (), {}),
    ("correlate", (np.ones((3, 3), dtype=np.float64) / 9.0,), {}),
    ("convolve", (np.ones((3, 3), dtype=np.float64) / 9.0,), {}),
    ("laplace", (), {}),
    ("gaussian_laplace", (2.0,), {}),
    ("gaussian_gradient_magnitude", (2.0,), {}),
])
def test_ndimage_forced_gpu_1e39_declines_to_scipy(gpu, name, args, kwargs):
    ours_fn = getattr(mn, name)
    theirs_fn = getattr(sn, name)
    a = _make_1e39_array((128, 128), seed=2)

    got = ours_fn(a, *args, **kwargs)
    route = scimetal.last_route()
    explanation = scimetal.explain()
    ref = theirs_fn(a, *args, **kwargs)

    assert route.engine == "scipy"
    assert route.reason != ""
    assert route.reason in explanation
    assert np.all(np.isfinite(got))
    np.testing.assert_array_equal(got, ref)


def test_ndimage_gain_amplified_1e38_and_cval_guard(gpu):
    # Sobel has gain 8 (2 * 4 in 2-D): 1e38 fits float32 alone, but 8e38 overflows float32.
    a = np.full((3072, 3072), 1e38, dtype=np.float64)
    with scimetal.device("auto"), scimetal.precision("fast"):
        got = mn.sobel(a)
        route = scimetal.last_route()
    assert route.engine == "cpu" and route.compute_dtype == "float64" and route.reason != ""
    assert np.all(np.isfinite(got))
    assert got.tobytes() == sn.sobel(a).tobytes()

    small = np.full((128, 128), 1e38, dtype=np.float64)
    got_gpu = mn.sobel(small)
    assert scimetal.last_route().engine == "scipy" and scimetal.last_route().reason != ""
    np.testing.assert_array_equal(got_gpu, sn.sobel(small))

    # Constant cval = 1e39 on unit data triggers guard too.
    unit = np.ones((128, 128), dtype=np.float64)
    got_cval = mn.gaussian_filter(unit, 1.5, mode="constant", cval=1e39)
    assert scimetal.last_route().engine == "scipy" and scimetal.last_route().reason != ""
    assert np.all(np.isfinite(got_cval))
    np.testing.assert_array_equal(got_cval, sn.gaussian_filter(unit, 1.5, mode="constant", cval=1e39))

    # Negative constant cval = -1e39 triggers guard under AUTO and forced GPU.
    big_unit = np.ones((2048, 2048), dtype=np.float64)
    with scimetal.device("auto"), scimetal.precision("fast"):
        got_neg_auto = mn.gaussian_filter(big_unit, 3.0, mode="constant", cval=-1e39)
        route_neg = scimetal.last_route()
    assert route_neg.engine == "cpu" and route_neg.compute_dtype == "float64" and route_neg.reason != ""
    assert np.all(np.isfinite(got_neg_auto))
    assert got_neg_auto.tobytes() == sn.gaussian_filter(big_unit, 3.0, mode="constant", cval=-1e39).tobytes()

    got_neg_gpu = mn.gaussian_filter(unit, 1.5, mode="constant", cval=-1e39)
    assert scimetal.last_route().engine == "scipy" and scimetal.last_route().reason != ""
    np.testing.assert_array_equal(got_neg_gpu, sn.gaussian_filter(unit, 1.5, mode="constant", cval=-1e39))

    # Non-constant mode (reflect) with cval = 1e39 ignores cval and stays on GPU.
    with scimetal.device("auto"), scimetal.precision("fast"):
        mn.gaussian_filter(big_unit, 3.0, mode="reflect", cval=1e39)
        route_reflect = scimetal.last_route()
    assert route_reflect.engine == "gpu" and route_reflect.compute_dtype == "float32" and route_reflect.reason == ""


def test_ndimage_symmetric_2e38_and_intermediate_overflow(gpu):
    # 1. Symmetric/antisymmetric (below +- above) pre-multiply overflow on 2e38 across filters.
    cases = [
        ("uniform_filter", (2048, 2048), np.full((2048, 2048), 2e38, dtype=np.float64), (5,), {}),
        ("uniform_filter1d", (2048, 2048), np.full((2048, 2048), 2e38, dtype=np.float64), (9,), {"axis": 0}),
        ("gaussian_filter", (2048, 2048), np.full((2048, 2048), 2e38, dtype=np.float64), (3.0,), {}),
        ("gaussian_filter1d", (2048, 2048), np.full((2048, 2048), 2e38, dtype=np.float64), (3.0,), {"axis": 0}),
        ("correlate1d", (3072, 3072), np.full((3072, 3072), 2e38, dtype=np.float64),
         (np.array([0.2, 0.6, 0.2], dtype=np.float64),), {"axis": 0}),
        ("convolve1d", (3072, 3072), np.full((3072, 3072), 2e38, dtype=np.float64),
         (np.array([0.2, 0.6, 0.2], dtype=np.float64),), {"axis": 0}),
        ("correlate1d", (3072, 3072), np.full((3072, 3072), 2e38, dtype=np.float64),
         (np.array([0.0, 1.0, 0.0], dtype=np.float64),), {"axis": 0}),
    ]
    alt = np.empty((3072, 3072), dtype=np.float64)
    alt[0::2, :] = -2e38
    alt[1::2, :] = 2e38
    cases.extend([
        ("correlate1d", (3072, 3072), alt, (np.array([-0.1, 0.0, 0.1], dtype=np.float64),), {"axis": 0}),
        ("convolve1d", (3072, 3072), alt, (np.array([-0.1, 0.0, 0.1], dtype=np.float64),), {"axis": 0}),
    ])

    for name, shape, arr, args, kwargs in cases:
        ours_fn = getattr(mn, name)
        theirs_fn = getattr(sn, name)
        with scimetal.device("auto"), scimetal.precision("fast"):
            ours_fn(np.ones(shape, dtype=np.float64), *args, **kwargs)
            assert scimetal.last_route().engine == "gpu", f"control failed for {name}"
            got = ours_fn(arr, *args, **kwargs)
            route = scimetal.last_route()
        assert route.engine == "cpu" and route.compute_dtype == "float64" and route.reason != ""
        assert np.all(np.isfinite(got))
        assert got.tobytes() == theirs_fn(arr, *args, **kwargs).tobytes()

        small = arr[:128, :128].copy()
        got_gpu = ours_fn(small, *args, **kwargs)
        assert scimetal.last_route().engine == "scipy" and scimetal.last_route().reason != ""
        assert np.all(np.isfinite(got_gpu))
        np.testing.assert_array_equal(got_gpu, theirs_fn(small, *args, **kwargs))

    # 2. Multi-pass grow-then-shrink intermediate-pass overflow.
    a_2e38 = np.full((2048, 2048), 2e38, dtype=np.float64)
    a_2e38[0, 0] = -2e38
    a_1e38 = np.full((2048, 2048), 1e38, dtype=np.float64)
    for arr, sigma in [(a_2e38, (0.8, 10.0)), (a_1e38, (0.4, 10.0))]:
        with scimetal.device("auto"), scimetal.precision("fast"):
            mn.gaussian_filter(np.ones(arr.shape, dtype=np.float64), sigma=sigma, order=(2, 1))
            assert scimetal.last_route().engine == "gpu"
            got = mn.gaussian_filter(arr, sigma=sigma, order=(2, 1))
            route = scimetal.last_route()
        assert route.engine == "cpu" and route.compute_dtype == "float64" and route.reason != ""
        assert np.all(np.isfinite(got))
        assert got.tobytes() == sn.gaussian_filter(arr, sigma=sigma, order=(2, 1)).tobytes()

        small = arr[:128, :128].copy()
        got_gpu = mn.gaussian_filter(small, sigma=sigma, order=(2, 1))
        assert scimetal.last_route().engine == "scipy" and scimetal.last_route().reason != ""
        assert np.all(np.isfinite(got_gpu))
        np.testing.assert_array_equal(got_gpu, sn.gaussian_filter(small, sigma=sigma, order=(2, 1)))


def test_ndimage_flt_max_rounding_guard(gpu):
    flt_max = float(np.finfo(np.float32).max)
    w_8 = np.array([0.1] * 7 + [0.3], dtype=np.float64)
    w_6 = np.array([0.1] * 5 + [0.5], dtype=np.float64)
    cases = [
        ("uniform_filter", (2048, 2048), (10,), {}),
        ("uniform_filter", (2048, 2048), (20,), {}),
        ("uniform_filter", (2048, 2048), (24,), {}),
        ("uniform_filter1d", (2048, 2048), (10,), {"axis": 0}),
        ("correlate1d", (3072, 3072), (w_8,), {"axis": 0}),
        ("correlate1d", (3072, 3072), (w_6,), {"axis": 0}),
    ]
    for name, shape, args, kwargs in cases:
        ours_fn = getattr(mn, name)
        theirs_fn = getattr(sn, name)
        normal = np.ones(shape, dtype=np.float64)
        with scimetal.device("auto"), scimetal.precision("fast"):
            ours_fn(normal, *args, **kwargs)
            assert scimetal.last_route().engine == "gpu", f"control failed for {name} {args}"

        for val in (flt_max, 3.402823e38):
            arr = np.full(shape, val, dtype=np.float64)
            with scimetal.device("auto"), scimetal.precision("fast"):
                got = ours_fn(arr, *args, **kwargs)
                route = scimetal.last_route()
            assert route.engine == "cpu" and route.compute_dtype == "float64" and route.reason != ""
            assert np.all(np.isfinite(got))
            assert got.tobytes() == theirs_fn(arr, *args, **kwargs).tobytes()

            small = arr[:128, :128].copy()
            got_gpu = ours_fn(small, *args, **kwargs)
            assert scimetal.last_route().engine == "scipy" and scimetal.last_route().reason != ""
            assert np.all(np.isfinite(got_gpu))
            np.testing.assert_array_equal(got_gpu, theirs_fn(small, *args, **kwargs))

    # Constant cval = FLT_MAX on unit input with even-sized uniform_filter(10).
    big_unit = np.ones((2048, 2048), dtype=np.float64)
    with scimetal.device("auto"), scimetal.precision("fast"):
        mn.uniform_filter(big_unit, 10, mode="constant", cval=0.0)
        assert scimetal.last_route().engine == "gpu"
        got_cval = mn.uniform_filter(big_unit, 10, mode="constant", cval=flt_max)
        route_cval = scimetal.last_route()
    assert route_cval.engine == "cpu" and route_cval.compute_dtype == "float64" and route_cval.reason != ""
    assert np.all(np.isfinite(got_cval))
    assert got_cval.tobytes() == sn.uniform_filter(big_unit, 10, mode="constant", cval=flt_max).tobytes()

    small_unit = np.ones((128, 128), dtype=np.float64)
    got_cval_gpu = mn.uniform_filter(small_unit, 10, mode="constant", cval=flt_max)
    assert scimetal.last_route().engine == "scipy" and scimetal.last_route().reason != ""
    assert np.all(np.isfinite(got_cval_gpu))
    np.testing.assert_array_equal(got_cval_gpu, sn.uniform_filter(small_unit, 10, mode="constant", cval=flt_max))


def test_refined_precision_guard_rails(gpu):
    normal = np.ones((2048, 2048), dtype=np.float64)
    over = _make_1e39_array((2048, 2048), seed=7)
    two_e38 = np.full((2048, 2048), 2e38, dtype=np.float64)
    with scimetal.device("auto"), scimetal.precision("refined"):
        mn.gaussian_filter(normal, 3.0)
        assert scimetal.last_route().engine == "gpu"
        assert scimetal.last_route().policy == "refined"

        got_over = mn.gaussian_filter(over, 3.0)
        r_over = scimetal.last_route()
        assert r_over.engine == "cpu" and r_over.compute_dtype == "float64"
        assert r_over.policy == "refined" and r_over.reason != ""
        assert np.all(np.isfinite(got_over))
        assert got_over.tobytes() == sn.gaussian_filter(over, 3.0).tobytes()

        got_sym = mn.uniform_filter(two_e38, 5)
        r_sym = scimetal.last_route()
        assert r_sym.engine == "cpu" and r_sym.compute_dtype == "float64"
        assert r_sym.policy == "refined" and r_sym.reason != ""
        assert np.all(np.isfinite(got_sym))
        assert got_sym.tobytes() == sn.uniform_filter(two_e38, 5).tobytes()

    with scimetal.device("gpu"), scimetal.precision("refined"):
        got_gpu = mn.gaussian_filter(over[:128, :128], 2.0)
        r_gpu = scimetal.last_route()
        assert r_gpu.engine == "scipy" and r_gpu.policy == "refined" and r_gpu.reason != ""
        np.testing.assert_array_equal(got_gpu, sn.gaussian_filter(over[:128, :128], 2.0))


def test_ndimage_gradient_magnitude_1e20_sqsum_guard(gpu):
    # 1e20 fits in float32, yet squaring inside gradient magnitude overflows float32 (~1e40 > FLT_MAX).
    a = np.arange(2048 * 2048, dtype=np.float64).reshape(2048, 2048) * 0.0
    a[::2, :] = 1e20
    a[1::2, :] = -1e20
    with scimetal.device("auto"), scimetal.precision("fast"):
        got = mn.gaussian_gradient_magnitude(a, 1.5)
        route = scimetal.last_route()
    assert route.engine == "cpu" and route.compute_dtype == "float64" and route.reason != ""
    assert np.all(np.isfinite(got))
    assert got.tobytes() == sn.gaussian_gradient_magnitude(a, 1.5).tobytes()

    small = a[:64, :64].copy()
    got_gpu = mn.gaussian_gradient_magnitude(small, 1.5)
    assert scimetal.last_route().engine == "scipy" and scimetal.last_route().reason != ""
    np.testing.assert_array_equal(got_gpu, sn.gaussian_gradient_magnitude(small, 1.5))


# ---- 2. fft & scipy.fft backend fast-mode overflow guard rails (all 12 routes) -----------------

@pytest.mark.parametrize("name, complex_input, kwargs", [
    ("fft", False, {}),
    ("fft", True, {}),
    ("ifft", False, {}),
    ("ifft", True, {}),
    ("rfft", False, {}),
    ("irfft", False, {}),
    ("fft2", False, {}),
    ("fft2", True, {}),
    ("ifft2", False, {}),
    ("ifft2", True, {}),
    ("fftn", False, {}),
    ("fftn", True, {}),
    ("ifftn", False, {}),
    ("ifftn", True, {}),
    ("rfft2", False, {}),
    ("irfft2", False, {}),
    ("rfftn", False, {}),
    ("irfftn", False, {}),
])
def test_fft_1e39_declines_to_scipy(name, complex_input, kwargs):
    # Use GPU-eligible sizes for every route (1D fft/ifft/rfft need >= 2^18 real points to cross over).
    shape = (262144,) if name in ("fft", "ifft", "rfft") else (256, 1024)
    ours_fn = getattr(mfft, name)
    theirs_fn = getattr(scipy.fft, name)

    if _native.has_metal():
        control = np.ones(shape, dtype=np.complex128 if complex_input else np.float64)
        with scimetal.device("auto"), scimetal.precision("fast"):
            ours_fn(control, **kwargs)
            assert scimetal.last_route().engine == "gpu", f"expected GPU control for {name} {shape}"

    a = _make_1e39_array(shape, seed=3)
    if complex_input:
        a = a + 1j * _make_1e39_array(shape, seed=4) * 0.5

    with scimetal.device("auto"), scimetal.precision("fast"):
        got = ours_fn(a, **kwargs)
        route = scimetal.last_route()
        explanation = scimetal.explain()
    ref = theirs_fn(a, **kwargs)

    assert route.engine == "scipy"
    assert route.reason != ""
    assert route.reason in explanation
    assert np.all(np.isfinite(got))
    # Declined large FFTs run threaded pocketfft (workers=performance_cores), which may
    # differ from single-worker SciPy by ~1 double ULP of the 1e39 data scale at cancelled bins.
    scale = float(np.abs(ref).max())
    np.testing.assert_allclose(got, ref, rtol=1e-13, atol=1e-13 * scale)


def test_scipy_fft_backend_1e39_declines_cleanly():
    shape = (256, 1024)
    a = _make_1e39_array(shape, seed=5)
    with scimetal.device("auto"), scimetal.precision("fast"):
        for name in ("fft", "rfft", "irfft", "fft2", "rfftn"):
            with scipy.fft.set_backend(mfft.backend):
                got = getattr(scipy.fft, name)(a)
                route = scimetal.last_route()
                assert route.engine == "scipy" and route.reason != ""
                assert route.reason in scimetal.explain()
                assert np.all(np.isfinite(got))
            ref = getattr(scipy.fft, name)(a)
            scale = float(np.abs(ref).max())
            np.testing.assert_allclose(got, ref, rtol=1e-13, atol=1e-13 * scale)

    # Also test register() / unregister()
    try:
        mfft.register()
        with scimetal.device("auto"), scimetal.precision("fast"):
            got = scipy.fft.fft(a)
            assert scimetal.last_route().engine == "scipy" and scimetal.last_route().reason != ""
            assert np.all(np.isfinite(got))
    finally:
        mfft.unregister()


# ---- 3. PrecisionWarning subprocess tests -----------------------------------------------------

def _run_py(code: str, extra_args: list[str] | None = None) -> str:
    env = os.environ.copy()
    if not os.environ.get("SCIMETAL_TEST_INSTALLED"):
        env["PYTHONPATH"] = PYTHON_DIR + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    cmd = [sys.executable, *(extra_args or []), "-c", code]
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, check=True)
    return proc.stdout


def test_precision_warning_emitted_once_by_default_ndimage_and_fft():
    if not _native.has_metal():
        pytest.skip("requires Metal GPU build")
    code = """
import warnings
import numpy as np
import scimetal
import scimetal.ndimage as mn
import scimetal.fft as mfft

scimetal.set_small_inputs("native")
a = np.ones((64, 64), dtype=np.float64)
z = np.ones((256, 1024), dtype=np.complex128)
with warnings.catch_warnings(record=True) as caught:
    with scimetal.device("gpu"), scimetal.precision("fast"):
        mn.gaussian_filter(a, 1.0)
        mn.sobel(a)
        mfft.fft(z)
assert len(caught) == 1, f"expected 1 warning, got {len(caught)}: {[str(w.message) for w in caught]}"
w = caught[0]
assert issubclass(w.category, scimetal.PrecisionWarning)
msg = str(w.message)
assert "set_precision('strict')" in msg
assert "PrecisionWarning" in msg
print("OK")
"""
    assert _run_py(code).strip() == "OK"


def test_precision_warning_emitted_once_on_complex128_fft_and_backend():
    if not _native.has_metal():
        pytest.skip("requires Metal GPU build")
    code = """
import warnings
import numpy as np
import scipy.fft
import scimetal
import scimetal.fft as mfft

z = np.ones((256, 1024), dtype=np.complex128)
with warnings.catch_warnings(record=True) as caught:
    with scimetal.device("auto"), scimetal.precision("fast"):
        mfft.fft(z)
        with scipy.fft.set_backend(mfft.backend):
            scipy.fft.fft2(z)
assert len(caught) == 1, f"expected 1 warning, got {len(caught)}"
assert issubclass(caught[0].category, scimetal.PrecisionWarning)
print("OK")
"""
    assert _run_py(code).strip() == "OK"


def test_precision_warning_can_be_silenced():
    if not _native.has_metal():
        pytest.skip("requires Metal GPU build")
    # Silenced after importing scimetal
    code_after = """
import warnings
import numpy as np
import scimetal
import scimetal.ndimage as mn

scimetal.set_small_inputs("native")
warnings.filterwarnings("ignore", category=scimetal.PrecisionWarning)
a = np.ones((64, 64), dtype=np.float64)
with warnings.catch_warnings(record=True) as caught:
    with scimetal.device("gpu"), scimetal.precision("fast"):
        mn.gaussian_filter(a, 1.0)
assert len(caught) == 0
print("OK")
"""
    assert _run_py(code_after).strip() == "OK"

    # Also verify true stderr output with default filters vs silenced via -W
    code_stderr = """
import numpy as np
import scimetal
import scimetal.ndimage as mn

scimetal.set_small_inputs("native")
a = np.ones((64, 64), dtype=np.float64)
with scimetal.device("gpu"), scimetal.precision("fast"):
    mn.gaussian_filter(a, 1.0)
    mn.sobel(a)
print("OK")
"""
    env = os.environ.copy()
    if not os.environ.get("SCIMETAL_TEST_INSTALLED"):
        env["PYTHONPATH"] = PYTHON_DIR + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    proc_default = subprocess.run([sys.executable, "-c", code_stderr],
                                  cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, check=True)
    assert proc_default.stdout.strip() == "OK"
    assert proc_default.stderr.count("PrecisionWarning:") == 1
    assert "set_precision('strict')" in proc_default.stderr

    proc_silenced = subprocess.run([sys.executable, "-W", "error", "-W", "ignore::UserWarning", "-c", code_stderr],
                                   cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, check=True)
    assert proc_silenced.stdout.strip() == "OK"
    assert proc_silenced.stderr == ""

    # Silenced via warnings.filterwarnings even when -W error is active globally
    code_werror = """
import warnings
import numpy as np
import scimetal
import scimetal.ndimage as mn

scimetal.set_small_inputs("native")
warnings.filterwarnings("ignore", category=scimetal.PrecisionWarning)
a = np.ones((64, 64), dtype=np.float64)
with scimetal.device("gpu"), scimetal.precision("fast"):
    mn.gaussian_filter(a, 1.0)
print("OK")
"""
    proc_werror = subprocess.run([sys.executable, "-W", "error", "-c", code_werror],
                                 cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, check=True)
    assert proc_werror.stdout.strip() == "OK"
    assert proc_werror.stderr == ""


def test_precision_warning_not_emitted_for_strict_float32_or_overflow_reroute():
    if not _native.has_metal():
        pytest.skip("requires Metal GPU build")
    code = """
import warnings
import numpy as np
import scimetal
import scimetal.ndimage as mn
import scimetal.fft as mfft

scimetal.set_small_inputs("native")
a64 = np.ones((64, 64), dtype=np.float64)
a32 = np.ones((64, 64), dtype=np.float32)
c64 = np.ones((256, 1024), dtype=np.complex64)
over = np.full((2048, 2048), 1e39, dtype=np.float64)
over_small = np.full((64, 64), 1e39, dtype=np.float64)
over_fft = np.full((256, 1024), 1e39, dtype=np.complex128)

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always", scimetal.PrecisionWarning)
    # 1. strict mode (float64 stays float64 on CPU / SciPy)
    with scimetal.precision("strict"):
        mn.gaussian_filter(a64, 1.0)
        mfft.fft(np.ones((256, 1024), dtype=np.complex128))
    # 2. single-precision inputs in fast mode
    with scimetal.device("gpu"), scimetal.precision("fast"):
        mn.gaussian_filter(a32, 1.0)
        mfft.fft(c64)
    # 3. 1e39 re-routed to CPU under AUTO
    with scimetal.device("auto"), scimetal.precision("fast"):
        mn.gaussian_filter(over, 3.0)
        assert scimetal.last_route().engine == "cpu"
    # 4. 1e39 declined to SciPy under forced GPU / FFT
    with scimetal.device("gpu"), scimetal.precision("fast"):
        mn.gaussian_filter(over_small, 1.0)
        assert scimetal.last_route().engine == "scipy"
        mfft.fft(over_fft)
        assert scimetal.last_route().engine == "scipy"

assert len(caught) == 0, f"unexpected warnings: {[str(w.message) for w in caught]}"
print("OK")
"""
    assert _run_py(code).strip() == "OK"


def test_caller_stacklevel_handles_none_module_name():
    from scimetal import _dispatch

    ns = {"_caller_stacklevel": _dispatch._caller_stacklevel, "__name__": None}
    exec("level = _caller_stacklevel()", ns)
    assert isinstance(ns["level"], int) and ns["level"] >= 1
