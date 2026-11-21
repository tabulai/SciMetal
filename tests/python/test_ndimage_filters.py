"""scimetal.ndimage's separable filters against scipy.ndimage.

What "agrees with SciPy" means depends on the engine that ran, which every
check reads from ``scimetal.last_route()``:

CPU engine (strict precision; any policy on a forced-CPU device; small arrays):
    bit-identical to this SciPy wheel (1.18.1, arm64) for float32 and float64,
    compared as raw bytes. That is stronger than the 1e-12 the strict policy
    promises, and it holds because the engine performs SciPy's operations in
    SciPy's order, including which multiply-adds its binary fuses. Against a
    SciPy build that fuses differently the results differ by an ulp or two.

GPU engine (fast and refined precision above the routing threshold, or a
forced-GPU device): float32 arithmetic, so the agreement is bounded relative
to the data scale. With u = 2^-24 and, for pass i, K_i taps and S_i = sum|weights|,
a bound m on the magnitude of the data and a bound e on its error are carried
through the passes:

    m_0 = max|input|          e_0 = u * m_0
    m'  = max(m_{i-1}, |cval|)    e' = max(e_{i-1}, u * |cval|)    (constant boundaries only)
    e_i = S_i * e' + (K_i + 2) * u * S_i * m'
    m_i = S_i * m'

because: narrowing the float64 input to float32 perturbs it by at most u * m_0;
a constant boundary feeds cval, itself narrowed, into every pass at full
size, whatever the earlier passes did to the data; a pass computes a K-term
dot product with weights rounded to float32, where each term goes through at
most K + 2 rounded operations (weight, pair sum or product, and the running
sum), so it adds at most (K + 2) * u * S * max|its input| (Higham, Accuracy and
Stability of Numerical Algorithms, section 3.1) and scales the incoming error
by at most S. Widening the result to float64 is exact. A uniform pass is a dot
product with weights 1/size (S = 1). For unit gains and no constant boundary
the bound is u * max|input| * (1 + sum_i(K_i + 2)).

SciPy's own result is exact to ~1e-16 for float64 arrays; for float32 arrays
it is rounded to float32 once per pass, one more u * S * m' in each e_i.
`tolerance` below is e_n with 0.1% for the second-order terms, plus the
underflow term every float32 error analysis carries: an operation whose result
is subnormal (below 1.2e-38, which the GPU flushes to zero) errs by up to that
much instead of relatively. Nothing else is added. Observed errors are
typically 5 to 30 times smaller (they grow like sqrt(K), not K).
"""
import inspect
import itertools
import threading

import numpy as np
import pytest
import scipy.ndimage as sn
from hypothesis import given, settings
from hypothesis import strategies as st

import scimetal
import scimetal.ndimage as mn

# Nothing to test where no native filter can run: a library built without the
# ndimage family, or one whose context cannot allocate (the route then says why).
with scimetal.precision("strict"), scimetal.device("cpu"):
    mn.correlate1d(np.zeros(4), [1.0])
    if scimetal.last_route().engine != "cpu":
        pytest.skip(f"no native ndimage filters: {scimetal.last_route().reason}", allow_module_level=True)

UNIT = 2.0 ** -24
FUNCTIONS = ("correlate1d", "convolve1d", "gaussian_filter1d", "gaussian_filter", "uniform_filter1d",
             "uniform_filter", "prewitt", "sobel")
MODES = ("reflect", "constant", "nearest", "mirror", "wrap", "grid-mirror", "grid-constant", "grid-wrap")
CONSTANT_MODES = ("constant", "grid-constant")
SHAPES = {1: (40,), 2: (23, 31), 3: (9, 14, 11), 4: (5, 6, 7, 8)}
CVAL = -2.25


def gaussian_weights(sigma, order=0, truncate=4.0, radius=None):
    """The Gaussian window SciPy uses, from the closed forms of its derivatives (orders 0 to 3)."""
    reach = int(truncate * float(sigma) + 0.5) if radius is None else radius
    x = np.arange(-reach, reach + 1, dtype=np.float64)
    phi = np.exp(-0.5 * x * x / sigma ** 2)
    phi /= phi.sum()
    v = float(sigma) ** 2
    polynomial = {0: np.ones_like(x), 1: -x / v, 2: x ** 2 / v ** 2 - 1 / v,
                  3: -x ** 3 / v ** 3 + 3 * x / v ** 2}[order]
    return polynomial * phi


def uniform_weights(size):
    return np.full(int(size), 1.0 / int(size))


class Call:
    """One filter call and what its error bound needs: the windows of its passes."""

    def __init__(self, function, args=(), kwargs=None, windows=(), constant=False):
        self.function, self.args, self.kwargs = function, args, dict(kwargs or {})
        self.windows, self.constant = [np.asarray(w, dtype=np.float64) for w in windows], constant

    def __repr__(self):
        return f"{self.function}{self.args!r} {self.kwargs!r}"

    def scipy(self, a):
        return getattr(sn, self.function)(a, *self.args, **self.kwargs)

    def ours(self, a):
        return getattr(mn, self.function)(a, *self.args, **self.kwargs)

    def tolerance(self, a):
        """The float32 error bound of the module docstring for this call on array `a`."""
        # When any pass has a constant boundary, treat every pass as having one: an upper bound.
        cval = abs(self.kwargs.get("cval", 0.0)) if self.constant else 0.0
        tiny = float(np.finfo(np.float32).tiny)
        rounded_reference = 1 if a.dtype == np.float32 else 0  # SciPy stores float32 results
        magnitude = float(np.abs(a).max())
        error = UNIT * magnitude
        for window in self.windows:
            gain = float(np.abs(window).sum())
            magnitude = max(magnitude, cval)
            error = max(error, UNIT * cval)
            error = gain * error + (len(window) + 2 + rounded_reference) * (UNIT * gain * magnitude + tiny)
            magnitude *= gain
        return 1.001 * error


def calls_for(ndim, function):
    """A grid of calls of `function` for an array of rank `ndim`: every mode and alias, every
    axis, extreme and central origins, odd and even windows, per-axis sequences."""
    rng = np.random.default_rng(ndim * 1000 + FUNCTIONS.index(function))
    general = {n: rng.standard_normal(n) for n in (1, 2, 4, 5, 9, 24)}
    symmetric = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
    antisymmetric = np.array([-0.5, -1.0, 0.0, 1.0, 0.5])
    long_symmetric = gaussian_weights(6.0)  # 24 pairs: SciPy's vectorized prefix and scalar tail
    calls = []
    for index, mode in enumerate(MODES):
        constant = mode in CONSTANT_MODES
        common = dict(mode=mode, cval=CVAL)
        axis = index % ndim
        negative_axis = axis - ndim
        if function in ("correlate1d", "convolve1d"):
            for weights in (*general.values(), symmetric, antisymmetric, long_symmetric):
                size = len(weights)
                for origin in sorted({-(size // 2), 0, (size - 1) // 2}):
                    calls.append(Call(function, (weights,), dict(common, axis=axis, origin=origin), [weights], constant))
            calls.append(Call(function, ([1, 2, 3], negative_axis), common, [[1, 2, 3]], constant))  # list weights
        elif function == "gaussian_filter1d":
            for sigma, order in itertools.product((0.6, 2.5, 4), (0, 1, 2, 3)):
                calls.append(Call(function, (sigma,), dict(common, axis=axis, order=order),
                                  [gaussian_weights(sigma, order)], constant))
            calls.append(Call(function, (1.5, negative_axis), dict(common, truncate=2.5),
                              [gaussian_weights(1.5, truncate=2.5)], constant))
            calls.append(Call(function, (1.5,), dict(common, radius=7), [gaussian_weights(1.5, radius=7)], constant))
            calls.append(Call(function, (np.float64(0.9),), dict(common, radius=0), [gaussian_weights(0.9, radius=0)], constant))
        elif function == "gaussian_filter":
            for sigma, order in itertools.product((0.6, 2.5), (0, 1, 2)):
                calls.append(Call(function, (sigma,), dict(common, order=order),
                                  [gaussian_weights(sigma, order)] * ndim, constant))
            sigmas = [0.5 + 0.75 * i for i in range(ndim)]
            orders = [i % 3 for i in range(ndim)]
            modes = [MODES[(index + i) % len(MODES)] for i in range(ndim)]
            calls.append(Call(function, (sigmas,), dict(order=orders, mode=modes, cval=CVAL),
                              [gaussian_weights(s, o) for s, o in zip(sigmas, orders)],
                              any(m in CONSTANT_MODES for m in modes)))
            calls.append(Call(function, (tuple(sigmas),), dict(common, radius=list(range(1, ndim + 1)), truncate=3.0),
                              [gaussian_weights(s, radius=r) for s, r in zip(sigmas, range(1, ndim + 1))], constant))
            # A zero sigma skips its axis; all zero is a copy.
            skipping = [0.0 if i == 0 else 1.2 for i in range(ndim)]
            calls.append(Call(function, (skipping,), common, [gaussian_weights(1.2)] * (ndim - 1), constant))
            calls.append(Call(function, (0,), common, [], constant))
            if ndim > 1:
                calls.append(Call(function, (1.7,), dict(common, axes=(ndim - 1, 0)), [gaussian_weights(1.7)] * 2, constant))
                calls.append(Call(function, ((1.1, 2.2),), dict(common, axes=[-1, 0], order=(1, 0)),
                                  [gaussian_weights(1.1, 1), gaussian_weights(2.2)], constant))
                calls.append(Call(function, (2.0,), dict(common, axes=-1), [gaussian_weights(2.0)], constant))
        elif function == "uniform_filter1d":
            for size in (1, 2, 3, 6, 17):
                for origin in sorted({-(size // 2), 0, (size - 1) // 2}):
                    calls.append(Call(function, (size,), dict(common, axis=axis, origin=origin),
                                      [uniform_weights(size)], constant))
            calls.append(Call(function, (np.int64(4), negative_axis), common, [uniform_weights(4)], constant))
        elif function == "uniform_filter":
            for size in (2, 3, 8):
                calls.append(Call(function, (size,), common, [uniform_weights(size)] * ndim, constant))
            calls.append(Call(function, (), common, [uniform_weights(3)] * ndim, constant))  # default size
            sizes = [2 + i for i in range(ndim)]
            origins = [(s - 1) // 2 if i % 2 else -(s // 2) for i, s in enumerate(sizes)]  # the extremes
            modes = [MODES[(index + 2 * i) % len(MODES)] for i in range(ndim)]
            calls.append(Call(function, (sizes,), dict(origin=origins, mode=modes, cval=CVAL),
                              [uniform_weights(s) for s in sizes], any(m in CONSTANT_MODES for m in modes)))
            # Sizes of at most 1 skip their axis; non-integers truncate.
            calls.append(Call(function, ([1] + [4.9] * (ndim - 1),), common, [uniform_weights(4)] * (ndim - 1), constant))
            calls.append(Call(function, (1,), common, [], constant))
            if ndim > 1:
                calls.append(Call(function, (5,), dict(common, axes=(0, ndim - 1), origin=(-2, 2)),
                                  [uniform_weights(5)] * 2, constant))
        else:  # prewitt, sobel
            smoothing = [1, 1, 1] if function == "prewitt" else [1, 2, 1]
            for derivative_axis in range(ndim):
                calls.append(Call(function, (derivative_axis,), common, [[-1, 0, 1]] + [smoothing] * (ndim - 1), constant))
            modes = [MODES[(index + 3 * i) % len(MODES)] for i in range(ndim)]
            calls.append(Call(function, (), dict(mode=modes, cval=CVAL), [[-1, 0, 1]] + [smoothing] * (ndim - 1),
                              any(m in CONSTANT_MODES for m in modes)))
    return calls


def array(shape, dtype, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(shape) * 3 + 0.5).astype(dtype)


def assert_identical(ours, reference, context=""):
    assert ours.dtype == reference.dtype and ours.shape == reference.shape, context
    if ours.tobytes() != reference.tobytes():
        diff = (f"mismatched bytes {int(np.count_nonzero(ours.view(np.uint8) != reference.view(np.uint8)))}"
                if ours.dtype == np.bool_
                else f"max difference {np.abs(ours.astype(np.float64) - reference.astype(np.float64)).max():.3g}")
        pytest.fail(f"{context}: not bit-identical, {diff}")


def check(call, a, expect_engine):
    """Run `call` both ways and compare according to the engine that ran. Returns error / bound."""
    reference = call.scipy(a)
    ours = call.ours(a)
    route = scimetal.last_route()
    assert route.function == f"ndimage.{call.function}"
    assert route.engine == expect_engine, f"{call}: ran on {route}"
    if route.engine == "gpu":
        assert route.compute_dtype == "float32"
        assert ours.dtype == reference.dtype and ours.shape == reference.shape
        if not call.windows:  # a copy is exact on any engine
            assert_identical(ours, reference, repr(call))
            return 0.0
        error = float(np.abs(ours.astype(np.float64) - reference.astype(np.float64)).max())
        bound = call.tolerance(a)
        assert error <= bound, f"{call}: error {error:.3g} exceeds the float32 bound {bound:.3g}"
        return error / bound if bound else 0.0
    assert_identical(ours, reference, repr(call))
    return 0.0


@pytest.fixture(scope="module")
def gpu():
    """Skip GPU-engine tests when the native library was built without Metal."""
    with scimetal.device("gpu"), scimetal.precision("fast"):
        mn.correlate1d(np.zeros(8, dtype=np.float32), [1.0])
        if scimetal.last_route().engine != "gpu":
            pytest.skip(f"no GPU engine: {scimetal.last_route().reason}")


# ---- the CPU engine is SciPy, bit for bit ---------------------------------------------------

@pytest.mark.parametrize("function", FUNCTIONS)
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("policy, device", [("strict", "auto"), ("strict", "cpu"), ("fast", "cpu"),
                                            ("refined", "cpu"), ("fast", "auto"), ("refined", "auto")])
def test_cpu_engine_is_bit_identical_to_scipy(function, dtype, policy, device):
    # Under 'auto' these arrays are far below the GPU crossover, whatever the policy.
    with scimetal.precision(policy), scimetal.device(device):
        for ndim, shape in SHAPES.items():
            a = array(shape, dtype, seed=ndim)
            for call in calls_for(ndim, function):
                check(call, a, "cpu")
                route = scimetal.last_route()
                assert route.policy == policy
                assert route.compute_dtype == ("float64" if call.windows else dtype.__name__)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_cpu_engine_awkward_shapes(dtype):
    """Length-1 axes, windows longer than the axis, lines and tiles, several lanes."""
    shapes = [(1,), (2,), (1, 1), (1, 9), (9, 1), (3, 2, 1, 4), (2, 3, 300), (300, 3, 2), (700, 520), (5, 64, 70),
              (100003,), (4, 30011)]
    with scimetal.precision("strict"):
        for seed, shape in enumerate(shapes):
            a = array(shape, dtype, seed)
            for axis in range(a.ndim):
                for mode in MODES[:5]:
                    for sigma in (0.8, 3, 12):  # 3, 12 and 48 pairs
                        for order in (0, 1):
                            window = [gaussian_weights(sigma, order)]
                            check(Call("gaussian_filter1d", (sigma, axis), dict(order=order, mode=mode, cval=CVAL), window), a, "cpu")
                    weights = np.linspace(-1.0, 2.0, 31)  # longer than the short axes, no symmetry
                    for origin in (-15, 0, 15):
                        check(Call("correlate1d", (weights, axis), dict(mode=mode, cval=CVAL, origin=origin), [weights]), a, "cpu")
                    for origin in (-15, 0, 14):  # an even window reaches one less to the right
                        check(Call("convolve1d", (weights[:30], axis), dict(mode=mode, cval=CVAL, origin=origin), [weights[:30]]), a, "cpu")
                    for size, origin in ((1, 0), (2, -1), (7, 3), (40, -20), (40, 19)):
                        check(Call("uniform_filter1d", (size, axis), dict(mode=mode, cval=CVAL, origin=origin),
                                   [uniform_weights(size)]), a, "cpu")
                    check(Call("sobel", (axis,), dict(mode=mode, cval=CVAL), [[1]]), a, "cpu")


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_every_window_length_matches_scipys_summation(dtype):
    """Which taps SciPy's binary fuses depends on the number of pairs (ten or more: groups of
    eight unfused, the remainder fused), so sweep the lengths across those boundaries."""
    a = array((64, 37), dtype, seed=3)
    rng = np.random.default_rng(5)
    with scimetal.precision("strict"):
        for pairs in range(0, 70):
            half = rng.standard_normal(pairs + 1)
            symmetric = np.concatenate([half[:0:-1], half])
            antisymmetric = np.concatenate([-half[:0:-1], [0.0], half[1:]])
            for weights in (symmetric, antisymmetric, rng.standard_normal(2 * pairs + 1), rng.standard_normal(2 * pairs + 2)):
                for axis in (0, 1):
                    check(Call("correlate1d", (weights, axis), dict(mode="mirror"), [weights]), a, "cpu")


def test_weights_symmetric_only_to_rounding_follow_scipy():
    """SciPy treats halves that agree within DBL_EPSILON as symmetric and then uses the lower half."""
    a = array((50, 41), np.float64, seed=4)
    base = gaussian_weights(2.0)
    for scale in (0.25, 1.0, 4.0):
        nearly = base.copy()
        nearly[-3] += scale * np.finfo(float).eps  # 0.25 eps rounds away; 1 eps is "symmetric"; 4 eps is not
        nearly_anti = np.concatenate([-base[:8], [0.0], base[:8][::-1]])
        nearly_anti[-2] += scale * np.finfo(float).eps
        with scimetal.precision("strict"):
            for weights in (nearly, nearly_anti):
                for axis in (0, 1):
                    check(Call("correlate1d", (weights, axis), {}, [weights]), a, "cpu")
                    check(Call("convolve1d", (weights, axis), {}, [weights]), a, "cpu")


def test_non_contiguous_inputs_are_filtered_natively():
    base = array((40, 30, 12), np.float64, seed=6)
    views = {"strided": base[::2, 1::3], "fortran": np.asfortranarray(base), "transposed": base.T,
             "reversed": base[::-1, :, ::-1], "offset": base[3:, 2:, 1:]}
    with scimetal.precision("strict"):
        for name, view in views.items():
            assert name == "offset" or not view.flags.c_contiguous
            for call in (Call("gaussian_filter", (1.5,), {}), Call("uniform_filter", (4,), dict(mode="wrap")),
                         Call("sobel", (1,), {}), Call("correlate1d", ([1.0, -2.0, 0.5, 3.0], 0), dict(origin=-2))):
                check(call, view, "cpu")
                assert call.ours(view).flags.c_contiguous


def test_special_values_propagate_like_scipy():
    a = array((60, 50), np.float64, seed=7)
    a[5, 7], a[20, 3], a[33, 40], a[40, 10] = np.inf, -np.inf, np.nan, -0.0
    with scimetal.precision("strict"), np.errstate(invalid="ignore"):
        for call in (Call("gaussian_filter", (2.0,), {}), Call("uniform_filter", (5,), dict(mode="constant", cval=np.inf)),
                     Call("correlate1d", ([0.0, 1.0, 0.0],), {}), Call("uniform_filter1d", (1,), {}),
                     Call("gaussian_filter", (1.0,), dict(mode="constant", cval=np.nan))):
            reference, ours = call.scipy(a), call.ours(a)
            assert scimetal.last_route().engine == "cpu"
            np.testing.assert_array_equal(np.isnan(ours), np.isnan(reference))
            finite = ~np.isnan(reference)
            assert ours[finite].tobytes() == reference[finite].tobytes()  # includes the signs of zeros


# ---- the GPU engine is float32 arithmetic within its bound ------------------------------------

@pytest.mark.parametrize("function", FUNCTIONS)
@pytest.mark.parametrize("dtype, policy", [(np.float64, "fast"), (np.float64, "refined"), (np.float32, "fast"),
                                           (np.float32, "refined"), (np.float32, "strict")])
def test_gpu_engine_stays_within_the_float32_bound(gpu, function, dtype, policy):
    worst = 0.0
    with scimetal.precision(policy), scimetal.device("gpu"):
        for ndim, shape in SHAPES.items():
            a = array(shape, dtype, seed=10 + ndim)
            for call in calls_for(ndim, function):
                if not call.windows:
                    continue  # a copy never reaches an engine
                worst = max(worst, check(call, a, "gpu"))
    assert worst > 1e-4, "errors this small mean the GPU engine did not compute in float32"


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_gpu_engine_awkward_shapes(gpu, dtype):
    """Shapes that are not multiples of any threadgroup, length-1 axes, windows longer than the axis."""
    shapes = [(1,), (2,), (1, 1), (1, 9), (9, 1), (3, 2, 1, 4), (2, 3, 300), (300, 3, 2), (257, 1031), (33, 65, 17),
              (100003,), (4, 30011)]
    with scimetal.precision("fast"), scimetal.device("gpu"):
        for seed, shape in enumerate(shapes):
            a = array(shape, dtype, seed)
            for axis in range(a.ndim):
                for index, mode in enumerate(MODES[:5]):
                    constant = mode == "constant"
                    sigma, order = (0.8, 3, 12)[(index + axis) % 3], (index + seed) % 2
                    check(Call("gaussian_filter1d", (sigma, axis), dict(order=order, mode=mode, cval=CVAL),
                               [gaussian_weights(sigma, order)], constant), a, "gpu")
                    weights = np.linspace(-1.0, 2.0, 31)
                    for origin in (-15, 0, 15):
                        check(Call("correlate1d", (weights, axis), dict(mode=mode, cval=CVAL, origin=origin), [weights], constant), a, "gpu")
                    for size, origin in ((2, -1), (7, 3), (40, 19)):
                        check(Call("uniform_filter1d", (size, axis), dict(mode=mode, cval=CVAL, origin=origin),
                                   [uniform_weights(size)], constant), a, "gpu")
            for mode in ("reflect", "constant", "wrap"):
                check(Call("gaussian_filter", (1.5,), dict(mode=mode, cval=CVAL), [gaussian_weights(1.5)] * a.ndim,
                           mode == "constant"), a, "gpu")
                check(Call("sobel", (0,), dict(mode=mode, cval=CVAL), [[-1, 0, 1]] + [[1, 2, 1]] * (a.ndim - 1),
                           mode == "constant"), a, "gpu")


def test_gpu_float64_results_are_exactly_widened_float32_results(gpu):
    """The last pass stores float64 directly: same value as the float32 result, another representation."""
    a32 = array((123, 211), np.float32, seed=8)
    a64 = a32.astype(np.float64)
    with scimetal.precision("fast"), scimetal.device("gpu"):
        for call in (Call("gaussian_filter", (2.0,), {}), Call("gaussian_filter1d", (3.0, 0), dict(order=1)),
                     Call("uniform_filter", (6,), dict(mode="constant", cval=0.5)), Call("sobel", (), {})):
            single, double = call.ours(a32), call.ours(a64)
            assert single.dtype == np.float32 and double.dtype == np.float64
            assert double.tobytes() == single.astype(np.float64).tobytes()


def test_gpu_special_values(gpu):
    """Infinities and NaNs reach the same outputs as in SciPy for correlations. (Not for uniform
    filters: SciPy's running sum smears a non-finite value over the rest of its line, which its
    documentation calls undefined behaviour; the CPU engine reproduces the smear, the GPU
    engine's direct sum confines it to the windows that contain the value.)"""
    a = array((60, 50), np.float64, seed=7)
    a[5, 7], a[20, 3], a[33, 40] = np.inf, -np.inf, np.nan
    asymmetric = [0.5, 0.25, 0.125, 1.0]
    with scimetal.precision("fast"), scimetal.device("gpu"), np.errstate(invalid="ignore"):
        for call in (Call("gaussian_filter", (2.0,), {}, [gaussian_weights(2.0)] * 2),
                     Call("correlate1d", (asymmetric,), dict(mode="constant", cval=np.inf), [asymmetric]),
                     Call("sobel", (0,), dict(mode="wrap"), [[-1, 0, 1], [1, 2, 1]])):
            reference, ours = call.scipy(a), call.ours(a)
            assert scimetal.last_route().engine == "gpu"
            np.testing.assert_array_equal(np.isnan(ours), np.isnan(reference))
            np.testing.assert_array_equal(np.isinf(ours), np.isinf(reference))
            finite = np.isfinite(reference)
            scale = float(np.abs(a[np.isfinite(a)]).max())
            # Finite outputs come from finite inputs only, so the bound of the finite data applies.
            bound = call.tolerance(np.array([scale]))
            assert np.abs(ours[finite] - reference[finite]).max() <= bound


# ---- routing ---------------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("policy", ["fast", "refined", "strict"])
@pytest.mark.parametrize("device", ["auto", "cpu", "gpu"])
def test_policy_and_device_matrix(gpu, dtype, policy, device):
    a = array((96, 120), dtype, seed=9)
    call = Call("gaussian_filter", (2.0,), dict(mode="mirror"), [gaussian_weights(2.0)] * 2)
    with scimetal.precision(policy), scimetal.device(device):
        if device == "gpu" and policy == "strict" and dtype == np.float64:
            # Strict keeps float64 off the GPU and a forced GPU never falls back to the CPU
            # engine: the call is SciPy's, with the reason on record.
            assert_identical(call.ours(a), call.scipy(a))
            route = scimetal.last_route()
            assert route.engine == "scipy" and "strict" in route.reason
        elif device == "gpu":
            check(call, a, "gpu")
        else:
            check(call, a, "cpu")  # small: 'auto' keeps it on the CPU
        assert scimetal.last_route().policy == policy
        assert "gaussian_filter" in scimetal.explain()


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_sizes_straddling_the_routing_threshold(gpu, dtype):
    """Under 'auto' small arrays run on the CPU and large ones on the GPU, with one switch in
    between; both sides agree with SciPy to their engine's standard."""
    sides = (128, 256, 512, 724, 1024, 1448, 2048, 2896)
    call = Call("gaussian_filter", (3.0,), {}, [gaussian_weights(3.0)] * 2)
    engines = []
    for policy in ("fast", "refined"):
        engines.clear()
        with scimetal.precision(policy):
            for side in sides:
                a = array((side, side), dtype, seed=side)
                call.ours(a)
                engines.append(scimetal.last_route().engine)
                check(call, a, engines[-1])
        assert engines[0] == "cpu" and engines[-1] == "gpu", engines
        switch = engines.index("gpu")
        assert engines == ["cpu"] * switch + ["gpu"] * (len(sides) - switch), engines
    # Strict stays on the CPU, bit-identical, at every size and for both dtypes.
    with scimetal.precision("strict"):
        for side in (sides[0], sides[-1]):
            check(call, array((side, side), dtype, seed=side), "cpu")


def test_long_uniform_windows_stay_on_the_cpu(gpu):
    """The CPU's running sum costs the same for any window; the GPU's direct sum does not."""
    a = array((2048, 2048), np.float64, seed=11)
    with scimetal.precision("fast"):
        check(Call("uniform_filter", (9,), {}, [uniform_weights(9)] * 2), a, "gpu")
        check(Call("uniform_filter", (301,), {}, [uniform_weights(301)] * 2), a, "cpu")


def test_explain_reports_the_route(gpu):
    a = array((64, 64), np.float64, seed=12)
    with scimetal.precision("fast"), scimetal.device("gpu"):
        mn.gaussian_filter(a, 1.0)
        assert scimetal.explain() == "ndimage.gaussian_filter: engine=gpu, computed in float32, policy=fast"
    with scimetal.precision("strict"):
        mn.uniform_filter1d(a, 3)
        assert scimetal.explain() == "ndimage.uniform_filter1d: engine=cpu, computed in float64, policy=strict"
        mn.gaussian_filter(a, 0)
        assert "copied" in scimetal.explain() and scimetal.last_route().engine == "cpu"
        mn.sobel(a.astype(np.int32))
        route = scimetal.last_route()
        assert route.engine == "cpu" and route.compute_dtype == "float64"
        mn.sobel(a.astype(np.complex128))
        route = scimetal.last_route()
        assert route.engine == "scipy" and "complex128" in route.reason


# ---- output argument --------------------------------------------------------------------------

@pytest.mark.parametrize("function, args", [("correlate1d", ([1.0, 2.0, -1.0],)), ("convolve1d", ([1.0, 2.0],)),
                                            ("gaussian_filter1d", (1.5,)), ("gaussian_filter", (1.5,)),
                                            ("uniform_filter1d", (4,)), ("uniform_filter", (4,)),
                                            ("prewitt", ()), ("sobel", ())])
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_output_argument(function, args, dtype):
    a = array((30, 40), dtype, seed=13)
    ours, theirs = getattr(mn, function), getattr(sn, function)
    reference = theirs(a, *args)
    with scimetal.precision("strict"):
        # A dtype equal to the input's, however it is spelled: a new array.
        for spelling in (dtype, np.dtype(dtype), np.dtype(dtype).name, np.dtype(dtype).char):
            assert_identical(ours(a, *args, output=spelling), reference)
            assert scimetal.last_route().engine == "cpu"
        # An array of the input's dtype is filled and returned.
        target = np.full_like(a, 7.0)
        returned = ours(a, *args, output=target)
        assert returned is target and scimetal.last_route().engine == "cpu"
        assert_identical(target, reference)
        # A non-contiguous output array.
        wide = np.zeros((30, 80), dtype)
        returned = ours(a, *args, output=wide[:, ::2])
        assert scimetal.last_route().engine == "cpu"
        assert_identical(np.ascontiguousarray(returned), reference)
        # In place.
        work = a.copy()
        returned = ours(work, *args, output=work)
        assert returned is work and scimetal.last_route().engine == "cpu"
        assert_identical(work, reference)
        # The other float dtype (float32 <-> float64) runs natively too, bit-identical in strict!
        other = np.float32 if dtype == np.float64 else np.float64
        ref_other = theirs(a, *args, output=other)
        spellings = [other, np.dtype(other), np.dtype(other).name, np.dtype(other).char]
        spellings.append("f4" if other == np.float32 else "f8")
        if other == np.float64:
            spellings.append(float)
        for spelling in spellings:
            assert_identical(ours(a, *args, output=spelling), ref_other)
            assert scimetal.last_route().engine == "cpu"
        target_other = np.full_like(a, 5.0, dtype=other)
        returned = ours(a, *args, output=target_other)
        assert returned is target_other and scimetal.last_route().engine == "cpu"
        assert_identical(target_other, ref_other)
        wide_other = np.zeros((30, 80), other)
        returned = ours(a, *args, output=wide_other[:, ::2])
        assert scimetal.last_route().engine == "cpu"
        assert_identical(np.ascontiguousarray(returned), ref_other)
        # Integer and boolean output dtypes run natively on CPU in strict mode, bit-identical to SciPy.
        for int_dtype in (np.int32, np.uint8, bool):
            ref_int = theirs(a, *args, output=int_dtype)
            assert_identical(ours(a, *args, output=int_dtype), ref_int)
            assert scimetal.last_route().engine == "cpu"
            target_int = np.zeros(a.shape, int_dtype)
            returned_int = ours(a, *args, output=target_int)
            assert returned_int is target_int and scimetal.last_route().engine == "cpu"
            assert_identical(target_int, ref_int)
        # Non-real output dtypes, another shape, and partial/cross-dtype overlap decline to SciPy.
        for bad_dtype in (np.float16, np.longdouble, np.complex64, np.dtype(">f4")):
            assert_declined_like_scipy(function, a, *args, output=bad_dtype)
            assert_declined_like_scipy(function, a, *args, output=np.zeros(a.shape, bad_dtype))
        raw = np.zeros((30, 80), dtype=np.float32)
        view_f64 = raw[:, :80].view(np.float64)
        view_f32 = raw[:, :40]
        ours(view_f64 if dtype == np.float64 else view_f32, *args,
             output=view_f32 if dtype == np.float64 else view_f64)
        assert scimetal.last_route().engine == "scipy" and "overlap" in scimetal.last_route().reason
        buffer = np.zeros((31, 40), dtype)
        buffer[:30] = a
        ours(buffer[:30], *args, output=buffer[1:])
        assert scimetal.last_route().engine == "scipy" and "overlap" in scimetal.last_route().reason
        with pytest.raises(RuntimeError, match="output shape not correct"):
            ours(a, *args, output=np.zeros((30, 41), dtype))


# ---- everything else is SciPy's ----------------------------------------------------------------

def scipy_outcome(function, *args, **kwargs):
    try:
        return getattr(sn, function)(*args, **kwargs), None
    except Exception as error:  # noqa: BLE001 - whatever SciPy raises is the expectation
        return None, error


def assert_declined_like_scipy(function, *args, **kwargs):
    """The call is handed to SciPy: same result or same exception, and the route says so."""
    expected, error = scipy_outcome(function, *args, **kwargs)
    before = list(scimetal._dispatch.counters[f"ndimage.{function}"])
    if error is None:
        result = getattr(mn, function)(*args, **kwargs)
        assert type(result) is type(expected)
        np.testing.assert_array_equal(result, expected)
        assert np.asarray(result).dtype == np.asarray(expected).dtype
    else:
        with pytest.raises(type(error)) as raised:
            getattr(mn, function)(*args, **kwargs)
        assert str(raised.value) == str(error)
    assert scimetal.last_route().engine == "scipy"
    after = scimetal._dispatch.counters[f"ndimage.{function}"]
    assert after == [before[0], before[1] + 1]


def test_unsupported_inputs_fall_through_to_scipy():
    a = array((12, 15), np.float64, seed=14)
    for function, args in (("correlate1d", ([1.0, 2.0, 1.0],)), ("convolve1d", ([1.0, 2.0, 1.0],)),
                           ("gaussian_filter1d", (1.0,)), ("gaussian_filter", (1.0,)), ("uniform_filter1d", (3,)),
                           ("uniform_filter", (3,)), ("prewitt", ()), ("sobel", ())):
        for unusual in (a.astype(np.float16), a.astype(np.longdouble),
                        a.astype(np.complex128), a.astype(">f8"), a.tolist(), np.ma.masked_array(a),
                        np.empty((0, 4)), np.empty((3, 0), np.float32), np.float64(2.0), np.zeros((2,) * 9)):
            assert_declined_like_scipy(function, unusual, *args)
        unaligned = np.frombuffer(b"\0" + a.tobytes(), dtype=np.float64, offset=1).reshape(a.shape)
        assert not unaligned.flags.aligned
        assert_declined_like_scipy(function, unaligned, *args)


def test_invalid_arguments_raise_scipys_exceptions():
    a = array((12, 15), np.float64, seed=15)
    for function, args, kwargs in [
        ("correlate1d", ([1.0, 2.0],), dict(mode="bogus")),
        ("correlate1d", ([1.0, 2.0],), dict(mode=3)),
        ("correlate1d", ([],), {}),
        ("correlate1d", ([[1.0, 2.0]],), {}),
        ("correlate1d", (2.0,), {}),
        ("correlate1d", ([1.0, 2.0, 3.0],), dict(origin=2)),
        ("correlate1d", ([1.0, 2.0, 3.0],), dict(origin=-2)),
        ("correlate1d", ([1.0, 2.0],), dict(origin=1)),
        ("correlate1d", ([1.0, 2.0, 3.0],), dict(origin=0.0)),
        ("correlate1d", ([1.0, 2.0],), dict(axis=2)),
        ("correlate1d", ([1.0, 2.0],), dict(axis=-3)),
        ("correlate1d", ([1.0, 2.0],), dict(axis=1.0)),
        ("correlate1d", ([1.0, 2.0],), dict(cval="x")),
        ("correlate1d", ([1.0, 2.0],), dict(cval=1j)),
        ("correlate1d", ([1.0, 2j],), {}),               # complex weights: SciPy's complex path
        ("correlate1d", (["a", "b"],), {}),
        ("convolve1d", ([1.0, 2.0],), dict(origin=-2)),
        ("convolve1d", ([1.0, 2.0, 3.0],), dict(origin=2)),
        ("convolve1d", (3.0,), {}),
        ("gaussian_filter1d", (-1.0,), {}),
        ("gaussian_filter1d", (0.0,), {}),
        ("gaussian_filter1d", (float("nan"),), {}),
        ("gaussian_filter1d", ("wide",), {}),
        ("gaussian_filter1d", (1.0,), dict(order=-1)),
        ("gaussian_filter1d", (1.0,), dict(order=1.5)),
        ("gaussian_filter1d", (1.0,), dict(radius=-1)),
        ("gaussian_filter1d", (1.0,), dict(radius=1.5)),
        ("gaussian_filter1d", (1.0,), dict(truncate=-4.0)),
        ("gaussian_filter", ((1.0, 2.0, 3.0),), {}),
        ("gaussian_filter", (1.0,), dict(order=(0, 1, 2))),
        ("gaussian_filter", (1.0,), dict(mode=("reflect",))),
        ("gaussian_filter", (1.0,), dict(axes=(0, 0))),
        ("gaussian_filter", (1.0,), dict(axes=(0, 2))),
        ("gaussian_filter", (1.0,), dict(axes=5)),
        ("gaussian_filter", (1.0,), dict(axes="x")),
        ("gaussian_filter", (1.0,), dict(axes=1.5)),
        ("gaussian_filter", ((1.0, 2.0),), dict(axes=(0,))),
        ("gaussian_filter", (1.0,), dict(radius=(1, 2, 3))),
        ("gaussian_filter", (1.0,), dict(mode="grid-bogus")),
        ("uniform_filter1d", (0,), {}),
        ("uniform_filter1d", (-3,), {}),
        ("uniform_filter1d", (3.0,), {}),
        ("uniform_filter1d", (3,), dict(origin=2)),
        ("uniform_filter1d", (4,), dict(origin=2)),
        ("uniform_filter1d", (4,), dict(origin=-3)),
        ("uniform_filter1d", (3,), dict(mode="edge")),
        ("uniform_filter", ((3, 3, 3),), {}),
        ("uniform_filter", (3,), dict(origin=(1, 1, 1))),
        ("uniform_filter", (3,), dict(origin=5)),
        ("uniform_filter", (3,), dict(axes=(1, 1))),
        ("uniform_filter", ("big",), {}),
        ("prewitt", (2,), {}),
        ("prewitt", (), dict(mode=("reflect",) * 3)),
        ("sobel", (-3,), {}),
        ("sobel", (), dict(mode="spline")),
    ]:
        assert_declined_like_scipy(function, a, *args, **kwargs)
    # Arguments SciPy accepts although they look odd are accepted natively too.
    with scimetal.precision("strict"):
        for call in (Call("gaussian_filter", (1.0,), dict(axes=())), Call("gaussian_filter", ((),), dict(axes=[])),
                     Call("uniform_filter", (2.5,), {}), Call("uniform_filter", (0,), {}), Call("uniform_filter", (-4,), {}),
                     Call("gaussian_filter1d", (True,), {}), Call("gaussian_filter1d", (np.float32(1.3),), {}),
                     Call("gaussian_filter1d", (2,), dict(order=np.int64(2), radius=np.int8(3))),
                     Call("gaussian_filter1d", (-2.0,), dict(radius=3)),
                     Call("correlate1d", ([True, False, True],), dict(cval=True, mode=np.str_("constant"))),
                     Call("correlate1d", (np.array([1, 2, 3], dtype=np.int8),), dict(origin=np.int64(-1), axis=np.int32(0))),
                     Call("uniform_filter1d", (True,), {}), Call("sobel", (True,), {})):
            check(call, a, "cpu")


def test_wrappers_keep_scipys_signatures_and_are_registered():
    for name in FUNCTIONS:
        ours, theirs = getattr(mn, name), getattr(sn, name)
        assert ours is not theirs and callable(ours.__scimetal_native__)
        assert inspect.signature(ours) == inspect.signature(theirs)
        assert ours.__doc__ == theirs.__doc__ and ours.__name__ == name
        assert name in dir(mn)
    assert mn.median_filter is sn.median_filter  # everything else is SciPy's own object


# ---- properties -------------------------------------------------------------------------------

@st.composite
def random_call(draw):
    ndim = draw(st.integers(1, 4))
    shape = tuple(draw(st.lists(st.integers(1, 13), min_size=ndim, max_size=ndim)))
    dtype = draw(st.sampled_from([np.float32, np.float64]))
    function = draw(st.sampled_from(FUNCTIONS))
    mode = draw(st.sampled_from(MODES))
    axis = draw(st.integers(-ndim, ndim - 1))
    cval = draw(st.floats(-4, 4, width=32))
    common = dict(mode=mode, cval=cval)
    constant = mode in CONSTANT_MODES
    if function in ("correlate1d", "convolve1d"):
        weights = np.array(draw(st.lists(st.floats(-2, 2, width=32), min_size=1, max_size=30)))
        size = len(weights)
        origin = draw(st.integers(-(size // 2), (size - 1) // 2))
        return shape, dtype, Call(function, (weights, axis), dict(common, origin=origin), [weights], constant)
    if function == "gaussian_filter1d":
        sigma, order = draw(st.floats(0.3, 5.0)), draw(st.integers(0, 3))
        return shape, dtype, Call(function, (sigma, axis), dict(common, order=order), [gaussian_weights(sigma, order)], constant)
    if function == "gaussian_filter":
        sigmas = draw(st.lists(st.floats(0.3, 4.0), min_size=ndim, max_size=ndim))
        orders = draw(st.lists(st.integers(0, 3), min_size=ndim, max_size=ndim))
        windows = [gaussian_weights(s, o) for s, o in zip(sigmas, orders)]
        return shape, dtype, Call(function, (sigmas,), dict(common, order=orders), windows, constant)
    if function == "uniform_filter1d":
        size = draw(st.integers(1, 30))
        origin = draw(st.integers(-(size // 2), (size - 1) // 2))
        return shape, dtype, Call(function, (size, axis), dict(common, origin=origin), [uniform_weights(size)], constant)
    if function == "uniform_filter":
        sizes = draw(st.lists(st.integers(2, 20), min_size=ndim, max_size=ndim))
        return shape, dtype, Call(function, (sizes,), common, [uniform_weights(s) for s in sizes], constant)
    smoothing = [1, 1, 1] if function == "prewitt" else [1, 2, 1]
    return shape, dtype, Call(function, (axis,), common, [[-1, 0, 1]] + [smoothing] * (ndim - 1), constant)


@settings(max_examples=300, deadline=None, database=None)
@given(random_call(), st.integers(0, 2 ** 31))
def test_random_calls_cpu_engine_is_bit_identical(drawn, seed):
    shape, dtype, call = drawn
    with scimetal.precision("strict"), scimetal.device("cpu"):
        check(call, array(shape, dtype, seed), "cpu")


@settings(max_examples=300, deadline=None, database=None)
@given(random_call(), st.integers(0, 2 ** 31))
def test_random_calls_gpu_engine_within_bound(gpu, drawn, seed):
    shape, dtype, call = drawn
    with scimetal.precision("fast"), scimetal.device("gpu"):
        check(call, array(shape, dtype, seed), "gpu")


# ---- concurrency ------------------------------------------------------------------------------

def test_concurrent_calls_from_threads(gpu):
    a = array((300, 400), np.float64, seed=16)
    reference = sn.gaussian_filter(a, 2.0)
    bound = Call("gaussian_filter", (2.0,), {}, [gaussian_weights(2.0)] * 2).tolerance(a)
    failures = []

    def work(device):
        try:
            with scimetal.precision("fast"), scimetal.device(device):
                for _ in range(20):
                    result = mn.gaussian_filter(a, 2.0)
                    assert scimetal.last_route().engine == device
                    if device == "cpu":
                        assert result.tobytes() == reference.tobytes()
                    else:
                        assert np.abs(result - reference).max() <= bound
        except BaseException as error:  # noqa: BLE001 - reported in the main thread
            failures.append(error)

    threads = [threading.Thread(target=work, args=(device,)) for device in ("cpu", "gpu") * 4]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not failures, failures


def test_small_inputs_go_to_scipy_by_default():
    """Below the break-even size a call is SciPy's, unless small inputs are kept native (as conftest does)."""
    rng = np.random.default_rng(7)
    small, large = rng.standard_normal((8, 8)), rng.standard_normal((128, 128))
    scimetal.set_small_inputs("scipy")
    try:
        out = mn.gaussian_filter(small, 1.0)
        route = scimetal.last_route()
        assert route.engine == "scipy" and "below" in route.reason
        np.testing.assert_array_equal(out, sn.gaussian_filter(small, 1.0))
        with scimetal.precision("strict"):
            mn.gaussian_filter(large, 1.0)
        assert scimetal.last_route().engine == "cpu"
    finally:
        scimetal.set_small_inputs("native")
    mn.gaussian_filter(small, 1.0)
    assert scimetal.last_route().engine in ("cpu", "gpu")


# ---- A3: dense & composite filters ------------------------------------------------------------

A3_FUNCTIONS = ("correlate", "convolve", "laplace", "gaussian_laplace",
                "gaussian_gradient_magnitude")


def test_a3_wrappers_keep_scipys_signatures_and_are_registered():
    for name in A3_FUNCTIONS:
        ours, theirs = getattr(mn, name), getattr(sn, name)
        assert ours is not theirs and callable(ours.__scimetal_native__)
        assert inspect.signature(ours) == inspect.signature(theirs)
        assert ours.__doc__ == theirs.__doc__ and ours.__name__ == name
        assert name in dir(mn)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("policy, device", [("strict", "auto"), ("strict", "cpu"), ("fast", "cpu")])
def test_a3_cpu_engine_is_bit_identical_to_scipy(dtype, policy, device):
    rng = np.random.default_rng(100)
    with scimetal.precision(policy), scimetal.device(device):
        for ndim, shape in SHAPES.items():
            a = array(shape, dtype, seed=20 + ndim)
            w_shape = (3,) * ndim if ndim <= 3 else (3, 2, 3, 2)
            w = rng.standard_normal(w_shape)
            w.ravel()[0] = 0.0
            w.ravel()[-1] = 1e-17
            for fn in ("correlate", "convolve"):
                for mode in MODES:
                    for origin in (0, -1):
                        ours = getattr(mn, fn)(a, w, mode=mode, cval=CVAL, origin=origin)
                        ref = getattr(sn, fn)(a, w, mode=mode, cval=CVAL, origin=origin)
                        assert scimetal.last_route().engine == "cpu"
                        assert_identical(ours, ref, f"{fn} {shape} {mode} {origin}")
            if ndim >= 2:
                w1 = rng.standard_normal((2,))
                ours = mn.correlate(a, w1, mode="wrap", axes=0)
                ref = sn.correlate(a, w1, mode="wrap", axes=0)
                assert scimetal.last_route().engine == "cpu"
                assert_identical(ours, ref, f"correlate axes=0 {shape}")

            for mode in ("reflect", "constant", "wrap", "mirror", "nearest"):
                assert_identical(mn.laplace(a, mode=mode, cval=CVAL), sn.laplace(a, mode=mode, cval=CVAL))
                assert scimetal.last_route().engine == "cpu"
                assert_identical(mn.gaussian_laplace(a, 1.3, mode=mode, cval=CVAL),
                                 sn.gaussian_laplace(a, 1.3, mode=mode, cval=CVAL))
                assert scimetal.last_route().engine == "cpu"
                assert_identical(mn.gaussian_gradient_magnitude(a, 1.3, mode=mode, cval=CVAL),
                                 sn.gaussian_gradient_magnitude(a, 1.3, mode=mode, cval=CVAL))
                assert scimetal.last_route().engine == "cpu"

            # Zero-padded sigma in gaussian_laplace vs direct sigma in gaussian_gradient_magnitude.
            if ndim >= 2:
                assert_identical(mn.gaussian_laplace(a, 1.1, axes=(0, ndim - 1)),
                                 sn.gaussian_laplace(a, 1.1, axes=(0, ndim - 1)))
                assert_identical(mn.gaussian_gradient_magnitude(a, 1.1, axes=(0, ndim - 1)),
                                 sn.gaussian_gradient_magnitude(a, 1.1, axes=(0, ndim - 1)))
                assert_identical(mn.gaussian_laplace(a, 0.0, axes=(0,)),
                                 sn.gaussian_laplace(a, 0.0, axes=(0,)))
                assert_identical(mn.gaussian_laplace(a, 1.1, axes=()),
                                 sn.gaussian_laplace(a, 1.1, axes=()))


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_a3_gpu_engine_and_rank1_separable(gpu, dtype):
    rng = np.random.default_rng(200)
    with scimetal.precision("fast"), scimetal.device("gpu"):
        for ndim in (1, 2, 3):
            a = array(SHAPES[ndim], dtype, seed=30 + ndim)
            scale = max(float(np.abs(a).max()), 1.0)
            w = rng.standard_normal((3,) * ndim)
            for fn in ("correlate", "convolve"):
                for mode in ("reflect", "constant", "wrap", "mirror", "nearest"):
                    ours = getattr(mn, fn)(a, w, mode=mode, cval=CVAL)
                    ref = getattr(sn, fn)(a, w, mode=mode, cval=CVAL)
                    assert scimetal.last_route().engine == "gpu"
                    assert np.abs(ours - ref).max() <= 1e-4 * scale

            # Rank-1 separable kernel with non-unit sums and constant cval != 0.
            if ndim >= 2:
                vecs = [np.array([1.0, 2.0, 1.0]), np.array([-1.0, 3.0, 2.0]), np.array([0.5, 1.5])]
                w_sep = vecs[0]
                for ax in range(1, ndim):
                    w_sep = np.multiply.outer(w_sep, vecs[ax])
                for fn in ("correlate", "convolve"):
                    ours = getattr(mn, fn)(a, w_sep, mode="constant", cval=2.5)
                    ref = getattr(sn, fn)(a, w_sep, mode="constant", cval=2.5)
                    assert scimetal.last_route().engine == "gpu"
                    assert np.abs(ours - ref).max() <= 1e-3 * scale

            for fn, args in [("laplace", ()), ("gaussian_laplace", (1.2,)),
                             ("gaussian_gradient_magnitude", (1.2,))]:
                for mode in ("reflect", "constant", "wrap"):
                    ours = getattr(mn, fn)(a, *args, mode=mode, cval=CVAL)
                    ref = getattr(sn, fn)(a, *args, mode=mode, cval=CVAL)
                    assert scimetal.last_route().engine == "gpu"
                    assert np.abs(ours - ref).max() <= 5e-4 * scale




@pytest.mark.parametrize("name", ["generic_laplace", "generic_gradient_magnitude"])
@pytest.mark.parametrize("policy", ["strict", "fast"])
def test_generic_filters_run_stateful_callbacks_once_per_axis(name, policy):
    x = np.ones((128, 128))
    native = getattr(mn, name)
    reference = getattr(sn, name)

    def run(function):
        axes_called = []
        target = np.empty_like(x)

        def derivative(input, axis, output, mode, cval):
            axes_called.append(axis)
            scaled = input * len(axes_called)
            if isinstance(output, np.ndarray):
                output[...] = scaled
                return None
            return scaled.astype(output, copy=False)

        result = function(x, derivative, output=target, axes=(1, 0))
        assert result is target
        return result, axes_called

    expected, expected_axes = run(reference)
    with scimetal.precision(policy):
        actual, actual_axes = run(native)
    assert actual_axes == expected_axes == [1, 0]
    assert_identical(actual, expected)
    assert name not in scimetal.accelerated()["ndimage"]


@pytest.mark.parametrize("name", ["generic_laplace", "generic_gradient_magnitude"])
def test_generic_filters_propagate_callback_exception_once(name):
    x = np.ones((128, 128))
    calls = []
    error = RuntimeError("derivative failed")

    def derivative(input, axis, output, mode, cval):
        calls.append(axis)
        raise error

    with pytest.raises(RuntimeError) as caught:
        getattr(mn, name)(x, derivative)
    assert caught.value is error
    assert calls == [0]


def test_generic_callback_can_use_a_native_filter():
    x = array((128, 128), np.float64, seed=41)
    with scimetal.precision("strict"):
        actual = mn.generic_gradient_magnitude(x, mn.sobel)
    assert_identical(actual, sn.generic_gradient_magnitude(x, sn.sobel))
    route = scimetal.last_route()
    assert route.function == "ndimage.sobel" and route.engine == "cpu"


def test_a3_declines_and_errors():
    a = array((12, 15), np.float64, seed=42)
    for fn, args, kwargs in [
        ("correlate", (np.ones((3, 3)),), dict(mode=["reflect", "wrap"])),
        ("convolve", (np.ones((3, 3)),), dict(mode=("reflect", "wrap"))),
        ("correlate", (np.empty((0, 3)),), {}),
        ("convolve", (np.empty((3, 0)),), {}),
        ("correlate", (np.ones((3, 3)),), dict(origin=(2, 0))),
        ("convolve", (np.ones((2, 2)),), dict(origin=(0, -2))),
        ("correlate", (np.ones((3,)),), dict(axes=(0, 1))),
        ("correlate", (np.ones((3, 3)) + 1j,), {}),
        ("gaussian_laplace", (1.0,), dict(unknown_kw=1)),
        ("gaussian_gradient_magnitude", (1.0,), dict(unknown_kw=1)),
        ("laplace", (), dict(axes=(0, 0))),
    ]:
        assert_declined_like_scipy(fn, a, *args, **kwargs)
    assert_declined_like_scipy("correlate", a.astype(np.complex128), np.ones((3, 3)))
    assert_declined_like_scipy("laplace", a.astype(np.complex128))
    assert_declined_like_scipy("gaussian_laplace", a.astype(np.complex128), 1.0)
    assert_declined_like_scipy("gaussian_gradient_magnitude", a.astype(np.float16), 1.0)

    # In-place and output= target array work on A3 routes.
    with scimetal.precision("strict"):
        for fn, args in [("correlate", (np.ones((3, 3)),)), ("convolve", (np.ones((2, 2)),))]:
            work = a.copy()
            ref_work = a.copy()
            getattr(sn, fn)(ref_work, *args, output=ref_work)
            ret = getattr(mn, fn)(work, *args, output=work)
            assert ret is work and scimetal.last_route().engine == "cpu"
            assert_identical(work, ref_work)

        # Out-of-place output= target array runs natively on composite filters, while
        # multi-axis in-place output=input declines to match SciPy's sequential aliasing.
        for fn, args in [("laplace", ()), ("gaussian_laplace", (1.0,)), ("gaussian_gradient_magnitude", (1.0,))]:
            dest = np.empty_like(a)
            ref = getattr(sn, fn)(a, *args)
            ret = getattr(mn, fn)(a, *args, output=dest)
            assert ret is dest and scimetal.last_route().engine == "cpu"
            assert_identical(dest, ref)

            work = a.copy()
            ref_work = a.copy()
            getattr(sn, fn)(ref_work, *args, output=ref_work)
            ret = getattr(mn, fn)(work, *args, output=work)
            assert ret is work and scimetal.last_route().engine == "scipy"
            assert_identical(work, ref_work)


@st.composite
def random_a3_call(draw):
    ndim = draw(st.integers(1, 3))
    shape = tuple(draw(st.lists(st.integers(2, 12), min_size=ndim, max_size=ndim)))
    dtype = draw(st.sampled_from([np.float32, np.float64]))
    fn = draw(st.sampled_from(["correlate", "convolve", "laplace", "gaussian_laplace",
                               "gaussian_gradient_magnitude"]))
    mode = draw(st.sampled_from(MODES))
    cval = draw(st.floats(-3, 3, width=32))
    if fn in ("correlate", "convolve"):
        wshape = tuple(draw(st.lists(st.integers(1, 5), min_size=ndim, max_size=ndim)))
        w = np.array(draw(st.lists(st.floats(-2, 2, width=32),
                                   min_size=int(np.prod(wshape)), max_size=int(np.prod(wshape))))).reshape(wshape)
        origins = [draw(st.integers(-(s // 2), (s - 1) // 2)) for s in wshape]
        if fn == "convolve":
            # convolve reverses & adjusts origin; keep origins valid after adjustment.
            origins = [0] * ndim
        return shape, dtype, fn, (w,), dict(mode=mode, cval=cval, origin=origins)
    if fn == "laplace":
        return shape, dtype, fn, (), dict(mode=mode, cval=cval)
    sigma = draw(st.floats(0.4, 2.5))
    return shape, dtype, fn, (sigma,), dict(mode=mode, cval=cval)


@settings(max_examples=150, deadline=None, database=None)
@given(random_a3_call(), st.integers(0, 2 ** 31))
def test_random_a3_calls_cpu_and_gpu(gpu, drawn, seed):
    shape, dtype, fn, args, kwargs = drawn
    a = array(shape, dtype, seed)
    ref = getattr(sn, fn)(a, *args, **kwargs)
    with scimetal.precision("strict"), scimetal.device("cpu"):
        ours_cpu = getattr(mn, fn)(a, *args, **kwargs)
        assert scimetal.last_route().engine == "cpu"
        assert_identical(ours_cpu, ref, f"{fn} {shape}")
    with scimetal.precision("fast"), scimetal.device("gpu"):
        ours_gpu = getattr(mn, fn)(a, *args, **kwargs)
        assert scimetal.last_route().engine == "gpu"
        scale = max(float(np.abs(a).max()), abs(float(kwargs.get("cval", 0.0))), 1.0)
        assert np.abs(ours_gpu - ref).max() <= 5e-3 * scale


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_a3_gradient_magnitude_single_axis_multipass_gpu(gpu, dtype):
    """Regression: K=1 multi-pass SQRT_SUM_SQ must not alias ping/res0 on GPU."""
    rng = np.random.default_rng(777)
    a = rng.standard_normal((2048, 2048)).astype(dtype)
    ref = sn.gaussian_gradient_magnitude(a, 3.0, axes=(0,))
    with scimetal.precision("fast"), scimetal.device("auto"):
        got1 = mn.gaussian_gradient_magnitude(a, 3.0, axes=(0,))
        assert scimetal.last_route().engine == "gpu"
        got2 = mn.gaussian_gradient_magnitude(a, 3.0, axes=(0,))
    assert np.array_equal(got1, got2)
    assert np.abs(got1 - ref).max() <= 1e-4 * max(float(np.abs(ref).max()), 1.0)


def test_a3_rank1_dynamic_range_and_zero_tap_nan(gpu):
    """Regression: high-dynamic-range non-rank-1 weights and zero-tap NaN pruning on GPU."""
    w5 = np.zeros((5, 5), dtype=np.float64)
    w5[0, 0] = 1e16
    w5[1, 1:] = 1.0
    w5[2, 1:] = 1.0
    x = np.zeros((4096, 4096), dtype=np.float32)
    x[2000, 2000] = 42.0
    with scimetal.precision("fast"), scimetal.device("auto"):
        out = mn.correlate(x, w5, mode="constant", cval=0.0)
        assert scimetal.last_route().engine == "gpu"
    ref = sn.correlate(x, w5, mode="constant", cval=0.0)
    assert np.isclose(out[2000, 2000], ref[2000, 2000])

    # Rank-1 kernels with zero entries must prune zeros rather than leaking 0.0 * NaN.
    for dtype in (np.float32, np.float64):
        arr = np.ones((8, 8), dtype=dtype)
        arr[0, 0] = np.nan
        w_delta = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        w_sobel = np.outer([-1.0, 0.0, 1.0], [1.0, 2.0, 1.0])
        for w in (w_delta, w_sobel):
            with scimetal.precision("fast"), scimetal.device("gpu"), np.errstate(invalid="ignore"):
                ours = mn.correlate(arr, w, mode="constant", cval=0.0)
                assert scimetal.last_route().engine == "gpu"
                theirs = sn.correlate(arr, w, mode="constant", cval=0.0)
            np.testing.assert_array_equal(np.isnan(ours), np.isnan(theirs))
            finite = np.isfinite(theirs)
            assert np.abs(ours[finite] - theirs[finite]).max() <= 1e-5


# ---- A4: cross-float output=, NEON uniform_filter, GPU overhanging windows ------------------

def test_a4_cross_float_output_all_routes_and_guard_rails(gpu):
    import warnings

    for in_dtype, out_dtype in [(np.float64, np.float32), (np.float32, np.float64)]:
        a = array((24, 32), in_dtype, seed=401)
        scale = max(float(np.abs(a).max()), 1.0)
        calls = [
            ("gaussian_filter", (1.4,), {}),
            ("gaussian_filter", (0.0,), {}),   # empty passes -> astype
            ("uniform_filter", (5,), {}),
            ("uniform_filter", (1,), {}),      # empty passes -> astype
            ("correlate", (np.array([[0.1, -0.2, 0.3], [0.4, 0.5, -0.1], [0.2, 0.0, -0.3]]),), {}),
            ("convolve", (np.outer([1.0, 2.0, 1.0], [-1.0, 3.0, 2.0]),), dict(mode="constant", cval=1.5)),
            ("laplace", (), {}),
            ("laplace", (), dict(axes=())),
            ("gaussian_laplace", (1.2,), {}),
            ("gaussian_gradient_magnitude", (1.2,), {}),
        ]
        with scimetal.precision("strict"), scimetal.device("cpu"):
            for fn, args, kw in calls:
                ref = getattr(sn, fn)(a, *args, output=out_dtype, **kw)
                ours = getattr(mn, fn)(a, *args, output=out_dtype, **kw)
                assert scimetal.last_route().engine == "cpu"
                assert_identical(ours, ref, f"strict {fn} {in_dtype}->{out_dtype}")

                target = np.empty_like(a, dtype=out_dtype)
                ret = getattr(mn, fn)(a, *args, output=target, **kw)
                assert ret is target
                assert_identical(target, ref, f"strict array {fn} {in_dtype}->{out_dtype}")

        with scimetal.precision("fast"), scimetal.device("gpu"):
            for fn, args, kw in calls:
                ref = getattr(sn, fn)(a, *args, output=out_dtype, **kw)
                ours = getattr(mn, fn)(a, *args, output=out_dtype, **kw)
                assert ours.dtype == out_dtype and ours.shape == a.shape
                if fn in ("laplace", "gaussian_filter", "uniform_filter") and (args == (0.0,) or args == (1,) or kw.get("axes") == ()):
                    assert_identical(ours, ref)
                else:
                    assert scimetal.last_route().engine == "gpu"
                    assert scimetal.last_route().compute_dtype == "float32"
                    assert np.abs(ours.astype(np.float64) - ref.astype(np.float64)).max() <= 5e-3 * scale

    # A5 overflow guard rails + PrecisionWarning + explain() for float64->float32 and float32->float64.
    big = np.ones((1536, 2048), dtype=np.float64)
    big[0, 0] = 1e39
    with warnings.catch_warnings(record=True) as w_list:
        warnings.simplefilter("always", scimetal.PrecisionWarning)
        with scimetal.precision("fast"), scimetal.device("auto"):
            res = mn.gaussian_filter(big, 3.0, output=np.float32)
            route = scimetal.last_route()
            assert route.engine == "cpu" and route.compute_dtype == "float64"
            assert "overflow" in route.reason and "overflow" in scimetal.explain()
            assert np.all(np.isfinite(res)) and res.dtype == np.float32
        assert not [w for w in w_list if issubclass(w.category, scimetal.PrecisionWarning)]

    normal64 = array((1536, 2048), np.float64, seed=403)
    with warnings.catch_warnings(record=True) as w_list:
        warnings.simplefilter("always", scimetal.PrecisionWarning)
        with scimetal.precision("fast"), scimetal.device("auto"):
            res = mn.gaussian_filter(normal64, 3.0, output=np.float32)
            assert res.dtype == np.float32
            assert scimetal.last_route().engine == "gpu"
            assert scimetal.last_route().compute_dtype == "float32"
        pw = [w for w in w_list if issubclass(w.category, scimetal.PrecisionWarning)]
        assert len(pw) == 1 and "set_precision('strict')" in str(pw[0].message)

    normal32 = array((1536, 2048), np.float32, seed=404)
    with warnings.catch_warnings(record=True) as w_list:
        warnings.simplefilter("always", scimetal.PrecisionWarning)
        with scimetal.precision("fast"), scimetal.device("auto"):
            res = mn.gaussian_filter(normal32, 3.0, output=np.float64)
            assert res.dtype == np.float64
            assert scimetal.last_route().engine == "gpu"
            assert scimetal.last_route().compute_dtype == "float32"
        assert not [w for w in w_list if issubclass(w.category, scimetal.PrecisionWarning)]


def test_a4_uniform_neon_and_tiles_bit_identical():
    with scimetal.precision("strict"), scimetal.device("cpu"):
        for dtype in (np.float64, np.float32):
            for out_dtype in (np.float64, np.float32):
                # Stride == 1 across line counts 1..17 (4-line NEON bundles + 0..3 remainder lines).
                for lines in (1, 3, 4, 5, 7, 8, 9, 15, 17):
                    a = array((lines, 67), dtype, seed=500 + lines)
                    for sz in (3, 9, 75):
                        for mode in MODES[:5]:
                            ours = mn.uniform_filter1d(a, sz, axis=1, output=out_dtype, mode=mode, cval=CVAL)
                            ref = sn.uniform_filter1d(a, sz, axis=1, output=out_dtype, mode=mode, cval=CVAL)
                            assert scimetal.last_route().engine == "cpu"
                            assert_identical(ours, ref, f"uniform1d {lines}x67 sz={sz} {mode}")
                # Wide inner axis (uniform_tiles scaling across all lanes).
                wide = array((25, 4096), dtype, seed=555)
                for sz in (3, 9):
                    ours = mn.uniform_filter(wide, sz, output=out_dtype)
                    ref = sn.uniform_filter(wide, sz, output=out_dtype)
                    assert scimetal.last_route().engine == "cpu"
                    assert_identical(ours, ref, f"uniform wide {sz}")


def test_a4_overhanging_windows_cpu_and_gpu(gpu):
    rng = np.random.default_rng(600)
    for shape in [(1,), (1, 13), (13, 1), (5, 7), (12, 19)]:
        for dtype in (np.float32, np.float64):
            a = array(shape, dtype, seed=601)
            scale = max(float(np.abs(a).max()), abs(CVAL), 1.0)
            for axis in range(a.ndim):
                for taps in (5, 19, 65):
                    if taps <= shape[axis]:
                        continue
                    w_sym = gaussian_weights(3.0, order=0, radius=taps // 2)
                    w_odd = gaussian_weights(3.0, order=1, radius=taps // 2)
                    w_asym = rng.standard_normal(taps)
                    for w in (w_sym, w_odd, w_asym):
                        for mode in MODES:
                            for origin in sorted({-(taps // 2), 0, (taps - 1) // 2}):
                                ref = sn.correlate1d(a, w, axis=axis, mode=mode, cval=CVAL, origin=origin)
                                with scimetal.precision("strict"), scimetal.device("cpu"):
                                    ours_cpu = mn.correlate1d(a, w, axis=axis, mode=mode, cval=CVAL, origin=origin)
                                    assert scimetal.last_route().engine == "cpu"
                                    assert_identical(ours_cpu, ref)
                                with scimetal.precision("fast"), scimetal.device("gpu"):
                                    ours_gpu = mn.correlate1d(a, w, axis=axis, mode=mode, cval=CVAL, origin=origin)
                                    assert scimetal.last_route().engine == "gpu"
                                    bound = float(np.abs(w).sum()) * (taps + 6) * UNIT * scale * 2.0
                                    assert np.abs(ours_gpu - ref).max() <= bound


@st.composite
def random_a4_call(draw):
    ndim = draw(st.integers(1, 3))
    shape = tuple(draw(st.lists(st.integers(1, 10), min_size=ndim, max_size=ndim)))
    in_dtype = draw(st.sampled_from([np.float32, np.float64]))
    out_dtype = draw(st.sampled_from([np.float32, np.float64]))
    fn = draw(st.sampled_from(["correlate1d", "convolve1d", "uniform_filter1d", "gaussian_filter", "uniform_filter",
                               "correlate", "laplace", "gaussian_laplace", "gaussian_gradient_magnitude"]))
    mode = draw(st.sampled_from(MODES))
    cval = draw(st.floats(-3, 3, width=32))
    if fn in ("correlate1d", "convolve1d"):
        axis = draw(st.integers(-ndim, ndim - 1))
        taps = draw(st.integers(1, 25))
        w = np.array(draw(st.lists(st.floats(-2, 2, width=32), min_size=taps, max_size=taps)))
        origin = draw(st.integers(-(taps // 2), (taps - 1) // 2))
        return shape, in_dtype, out_dtype, fn, (w,), dict(axis=axis, mode=mode, cval=cval, origin=origin), [w]
    if fn == "uniform_filter1d":
        axis = draw(st.integers(-ndim, ndim - 1))
        sz = draw(st.integers(1, 25))
        origin = draw(st.integers(-(sz // 2), (sz - 1) // 2))
        return shape, in_dtype, out_dtype, fn, (sz,), dict(axis=axis, mode=mode, cval=cval, origin=origin), [uniform_weights(sz)]
    if fn == "uniform_filter":
        sz = draw(st.integers(1, 15))
        return shape, in_dtype, out_dtype, fn, (sz,), dict(mode=mode, cval=cval), (
            [uniform_weights(sz)] * ndim if sz > 1 else [])
    if fn == "gaussian_filter":
        sigma = draw(st.floats(0.0, 3.0))
        return shape, in_dtype, out_dtype, fn, (sigma,), dict(mode=mode, cval=cval), (
            [gaussian_weights(sigma)] * ndim if sigma > 1e-15 else [])
    if fn == "correlate":
        wshape = tuple(draw(st.lists(st.integers(1, 4), min_size=ndim, max_size=ndim)))
        w = np.array(draw(st.lists(st.floats(-2, 2, width=32),
                                   min_size=int(np.prod(wshape)), max_size=int(np.prod(wshape))))).reshape(wshape)
        return shape, in_dtype, out_dtype, fn, (w,), dict(mode=mode, cval=cval), [w.ravel()]
    if fn == "laplace":
        return shape, in_dtype, out_dtype, fn, (), dict(mode=mode, cval=cval), [np.array([1.0, -2.0, 1.0])]
    sigma = draw(st.floats(0.4, 2.0))
    return shape, in_dtype, out_dtype, fn, (sigma,), dict(mode=mode, cval=cval), [gaussian_weights(sigma, 2)] * ndim


@settings(max_examples=150, deadline=None, database=None)
@given(random_a4_call(), st.integers(0, 2 ** 31))
def test_a4_random_cross_float_output_and_overhanging(gpu, drawn, seed):
    shape, in_dtype, out_dtype, fn, args, kwargs, windows = drawn
    a = array(shape, in_dtype, seed)
    ref = getattr(sn, fn)(a, *args, output=out_dtype, **kwargs)
    with scimetal.precision("strict"), scimetal.device("cpu"):
        ours_cpu = getattr(mn, fn)(a, *args, output=out_dtype, **kwargs)
        assert scimetal.last_route().engine == "cpu"
        assert_identical(ours_cpu, ref, f"cpu {fn} {shape} {in_dtype}->{out_dtype}")
    with scimetal.precision("fast"), scimetal.device("gpu"):
        ours_gpu = getattr(mn, fn)(a, *args, output=out_dtype, **kwargs)
        assert ours_gpu.dtype == np.dtype(out_dtype) and ours_gpu.shape == shape
        if not windows:
            assert_identical(ours_gpu, ref)
        else:
            assert scimetal.last_route().engine == "gpu"
            scale = max(float(np.abs(a).max()), abs(float(kwargs.get("cval", 0.0))), 1.0)
            gain = 1.0
            for w in windows:
                gain *= max(float(np.abs(w).sum()), 1.0)
            assert np.abs(ours_gpu.astype(np.float64) - ref.astype(np.float64)).max() <= 5e-3 * scale * gain


def test_mixed_precision_float32_to_float64_overflow_guard(gpu):
    """float32 inputs with output=float64 whose gain/sum/cval overflows float32 must re-route under auto and decline under gpu."""
    with scimetal.precision("fast"):
        # 1. Rank-1 separable correlate with 1e38 float32 input and 5x5 * 10.0 float64 weights.
        a_1e38 = np.full((2500, 2500), 1e38, dtype=np.float32)
        w10 = np.full((5, 5), 10.0, dtype=np.float64)
        ref1 = sn.correlate(a_1e38, w10, mode="reflect", output=np.float64)
        with scimetal.device("auto"):
            got1 = mn.correlate(a_1e38, w10, mode="reflect", output=np.float64)
            r1 = scimetal.last_route()
            assert r1.engine == "cpu" and r1.compute_dtype == "float64" and r1.reason
            assert np.all(np.isfinite(got1))
            assert got1.tobytes() == ref1.tobytes()
        with scimetal.device("gpu"):
            got1_g = mn.correlate(a_1e38[:32, :32], w10, mode="reflect", output=np.float64)
            assert scimetal.last_route().engine == "scipy"
            assert got1_g.tobytes() == sn.correlate(a_1e38[:32, :32], w10, mode="reflect", output=np.float64).tobytes()

        # 2. Rank-1 separable correlate with constant cval=1e38 overflowing on pass 1 (cval_1 = 2.5e38 * 5).
        ones_f32 = np.ones((2500, 2500), dtype=np.float32)
        w05 = np.full((5, 5), 0.5, dtype=np.float64)
        ref2 = sn.correlate(ones_f32, w05, mode="constant", cval=1e38, output=np.float64)
        with scimetal.device("auto"):
            got2 = mn.correlate(ones_f32, w05, mode="constant", cval=1e38, output=np.float64)
            r2 = scimetal.last_route()
            assert r2.engine == "cpu" and r2.compute_dtype == "float64" and r2.reason
            assert np.all(np.isfinite(got2))
            assert got2.tobytes() == ref2.tobytes()
        with scimetal.device("gpu"):
            got2_g = mn.correlate(ones_f32[:32, :32], w05, mode="constant", cval=1e38, output=np.float64)
            assert scimetal.last_route().engine == "scipy"
            assert got2_g.tobytes() == sn.correlate(ones_f32[:32, :32], w05, mode="constant", cval=1e38, output=np.float64).tobytes()

        # 3. Composite filter (laplace) with alternating +-1e38 float32 input and output=float64.
        alt = np.full((2500, 2500), 1e38, dtype=np.float32)
        alt[::2, 1::2] = -1e38
        alt[1::2, ::2] = -1e38
        ref3 = sn.laplace(alt, mode="reflect", output=np.float64)
        with scimetal.device("auto"):
            got3 = mn.laplace(alt, mode="reflect", output=np.float64)
            r3 = scimetal.last_route()
            assert r3.engine == "cpu" and r3.compute_dtype == "float64" and r3.reason
            assert np.all(np.isfinite(got3))
            assert got3.tobytes() == ref3.tobytes()
        with scimetal.device("gpu"):
            got3_g = mn.laplace(alt[:32, :32], mode="reflect", output=np.float64)
            assert scimetal.last_route().engine == "scipy"
            assert got3_g.tobytes() == sn.laplace(alt[:32, :32], mode="reflect", output=np.float64).tobytes()

        # 4. Other separable & composite filters near float32 threshold with output=float64.
        a_2e38 = np.full((2500, 2500), 2e38, dtype=np.float32)
        a_1e20 = np.full((2500, 2500), 1e20, dtype=np.float32)
        alt_2e38 = alt * 2.0
        for fn, arr, args, kw in (
            ("correlate1d", a_1e38, (np.ones(29, dtype=np.float64),), dict(mode="reflect")),
            ("gaussian_filter", a_2e38, (2.0,), dict(mode="reflect")),
            ("uniform_filter", a_2e38, (5,), dict(mode="reflect")),
            ("gaussian_laplace", alt_2e38, (1.0,), dict(mode="reflect")),
            ("gaussian_gradient_magnitude", a_1e20, (1.0,), dict(mode="reflect")),
        ):
            ref = getattr(sn, fn)(arr, *args, output=np.float64, **kw)
            with scimetal.device("auto"):
                got = getattr(mn, fn)(arr, *args, output=np.float64, **kw)
                rt = scimetal.last_route()
                assert rt.engine == "cpu" and rt.compute_dtype == "float64" and rt.reason, f"{fn}: {rt}"
                assert np.all(np.isfinite(got))
                assert got.tobytes() == ref.tobytes()
            with scimetal.device("gpu"):
                sub = arr[:32, :32]
                got_g = getattr(mn, fn)(sub, *args, output=np.float64, **kw)
                assert scimetal.last_route().engine == "scipy", f"{fn} gpu did not decline"
                assert got_g.tobytes() == getattr(sn, fn)(sub, *args, output=np.float64, **kw).tobytes()

        # 5. Intentional NaN/Inf in float32 input with output=float64 stays on GPU and propagates.
        sp = np.ones((64, 64), dtype=np.float32)
        sp[5, 5] = np.inf
        sp[10, 10] = np.nan
        with scimetal.device("gpu"), np.errstate(invalid="ignore"):
            got_sp = mn.correlate(sp, np.full((3, 3), 0.25, np.float64), mode="reflect", output=np.float64)
            assert scimetal.last_route().engine == "gpu"
            ref_sp = sn.correlate(sp, np.full((3, 3), 0.25, np.float64), mode="reflect", output=np.float64)
            np.testing.assert_array_equal(np.isnan(got_sp), np.isnan(ref_sp))
            np.testing.assert_array_equal(np.isinf(got_sp), np.isinf(ref_sp))


# ---- A6: integer and boolean dtypes -----------------------------------------------------------

_INT_BOOL_DTYPES = (
    np.bool_, np.int8, np.int16, np.int32, np.int64,
    np.uint8, np.uint16, np.uint32, np.uint64,
)


def _int_bool_array(shape, dtype, seed=20261119):
    if dtype is np.bool_:
        rng = np.random.default_rng(seed)
        return (rng.integers(0, 4, size=shape, dtype=np.uint8) != 0)
    info = np.iinfo(dtype)
    rng = np.random.default_rng(seed)
    low = -40 if info.min < 0 else 0
    high = min(60, info.max)
    return rng.integers(low, high + 1, size=shape, dtype=dtype)


@pytest.mark.parametrize("dtype", _INT_BOOL_DTYPES)
def test_a6_integer_bool_strict_bit_identical(dtype):
    a = _int_bool_array((19, 23), dtype, seed=10 + int(np.dtype(dtype).num))
    w1d = np.array([-0.5, 1.25, -0.5], dtype=np.float64)
    w2d = np.array([[0.0, -0.5, 0.25], [0.5, 1.0, -0.25], [0.0, 0.5, -0.5]], dtype=np.float64)
    calls = (
        ("correlate1d", (w1d,), dict(axis=-1, mode="reflect")),
        ("convolve1d", (w1d,), dict(axis=0, mode="constant", cval=1.0)),
        ("correlate", (w2d,), dict(mode="reflect")),
        ("convolve", (w2d,), dict(mode="constant", cval=2.0)),
        ("gaussian_filter1d", (1.2,), dict(axis=-1)),
        ("gaussian_filter", (1.2,), dict()),
        ("uniform_filter1d", (5,), dict(axis=0)),
        ("uniform_filter", (4,), dict(mode="nearest")),
        ("prewitt", (), dict(mode="reflect")),
        ("sobel", (), dict(mode="constant", cval=1.0)),
        ("laplace", (), dict(mode="reflect")),
        ("gaussian_laplace", (1.2,), dict(mode="reflect")),
        ("gaussian_gradient_magnitude", (1.2,), dict(mode="reflect")),
    )
    with scimetal.precision("strict"), scimetal.device("auto"):
        for fn, args, kw in calls:
            ours = getattr(mn, fn)(a, *args, **kw)
            assert scimetal.last_route().engine == "cpu", f"{fn} {dtype}: {scimetal.last_route()}"
            ref = getattr(sn, fn)(a, *args, **kw)
            assert_identical(ours, ref, f"strict {fn} {dtype}")


def test_a6_bool_raw_bytes_and_cross_dtype_outputs():
    # Multi-byte raw bool view (values 0, 2, 5) matches SciPy's raw-byte NI_ArrayToLineBuffer semantics.
    raw = np.arange(15 * 19, dtype=np.uint8).reshape(15, 19) % 5
    b_raw = raw.view(np.bool_)
    with scimetal.precision("strict"), scimetal.device("cpu"):
        for fn, args in (("correlate1d", (np.array([0.25, 0.5, 0.25]),)),
                         ("correlate", (np.array([[1.0, -1.0], [0.5, 0.5]]),)),
                         ("gaussian_filter", (1.0,)),
                         ("laplace", ()),
                         ("uniform_filter", (3,))):
            ours = getattr(mn, fn)(b_raw, *args)
            ref = getattr(sn, fn)(b_raw, *args)
            assert scimetal.last_route().engine == "cpu"
            assert_identical(ours, ref, f"raw bool {fn}")

    # Cross-dtype output= combinations across separable, dense, and composite filters.
    combos = [
        (np.uint8, np.float64),
        (np.uint8, np.int16),
        (np.float64, np.int32),
        (np.bool_, np.uint8),
        (np.int32, np.float32),
    ]
    w2d = np.array([[0.25, -0.5], [1.0, 0.25]], dtype=np.float64)
    for in_dt, out_dt in combos:
        a = _int_bool_array((17, 21), in_dt, seed=55) if in_dt in _INT_BOOL_DTYPES else array((17, 21), in_dt, seed=55)
        with scimetal.precision("strict"), scimetal.device("cpu"):
            for fn, args, ref_fn in (
                ("correlate1d", (np.array([0.25, 0.5, 0.25]),), sn.correlate1d),
                ("gaussian_filter", (1.1,), sn.gaussian_filter),
                ("uniform_filter", (3,), sn.uniform_filter),
                ("sobel", (), sn.sobel),
                ("prewitt", (), sn.prewitt),
                ("correlate", (w2d,), sn.correlate),
                ("convolve", (w2d,), sn.convolve),
                ("laplace", (), sn.laplace),
                ("gaussian_laplace", (1.1,), sn.gaussian_laplace),
                ("gaussian_gradient_magnitude", (1.1,), sn.gaussian_gradient_magnitude),
            ):
                ours = getattr(mn, fn)(a, *args, output=out_dt)
                assert scimetal.last_route().engine == "cpu"
                ref = ref_fn(a, *args, output=out_dt)
                assert_identical(ours, ref, f"cross {fn} {in_dt}->{out_dt}")


def test_a6_gpu_fast_uint8_int16_and_large_int_guard(gpu):
    # GPU fast mode runs uint8 and int16 accurately.
    for dt in (np.uint8, np.int16):
        a = _int_bool_array((128, 128), dt, seed=77)
        with scimetal.precision("fast"), scimetal.device("gpu"):
            for fn, args in (("gaussian_filter", (1.5,)),
                             ("uniform_filter", (5,)),
                             ("correlate", (np.full((3, 3), 1.0 / 9.0),))):
                ours = getattr(mn, fn)(a, *args)
                rt = scimetal.last_route()
                assert rt.engine == "gpu" and rt.compute_dtype == "float32"
                ref = getattr(sn, fn)(a, *args)
                assert np.abs(ours.astype(np.float64) - ref.astype(np.float64)).max() <= 1.0
            if dt is np.int16:
                for fn, args in (("laplace", ()), ("gaussian_laplace", (1.2,)), ("gaussian_gradient_magnitude", (1.2,))):
                    ours = getattr(mn, fn)(a, *args)
                    rt = scimetal.last_route()
                    assert rt.engine == "gpu" and rt.compute_dtype == "float32"
                    ref = getattr(sn, fn)(a, *args)
                    ref_f64 = np.trunc(getattr(sn, fn)(a.astype(np.float64), *args)).astype(dt)
                    assert np.abs(ours.astype(np.float64) - ref_f64.astype(np.float64)).max() <= 1.0
                    assert np.abs(ours.astype(np.float64) - ref.astype(np.float64)).max() <= 3.0

    # |x| >= 2^24 guard rail re-routes AUTO -> CPU and declines under forced GPU.
    shape = (2048, 2048)
    for dt in (np.int32, np.uint32, np.int64, np.uint64):
        arr = np.full(shape, 7, dtype=dt)
        with scimetal.precision("fast"):
            if dt is np.int32:
                arr[0, 0] = (1 << 24) - 1
                with scimetal.device("auto"):
                    mn.gaussian_filter(arr, 1.0)
                    rt_ok = scimetal.last_route()
                    assert rt_ok.engine == "gpu" and rt_ok.compute_dtype == "float32"

            bad_values = (1 << 24, -(1 << 24)) if np.issubdtype(dt, np.signedinteger) else (1 << 24,)
            for bad in bad_values:
                arr[10, 20] = bad
                ref = sn.gaussian_filter(arr, 1.0)
                with scimetal.device("auto"):
                    got = mn.gaussian_filter(arr, 1.0)
                    rt = scimetal.last_route()
                    assert rt.engine == "cpu" and rt.compute_dtype == "float64" and rt.reason
                    assert "cpu" in scimetal.explain().lower() and rt.reason in scimetal.explain()
                    assert_identical(got, ref, f"auto guard {dt} {bad}")
                with scimetal.device("gpu"):
                    sub = arr[:64, :64]
                    got_g = mn.gaussian_filter(sub, 1.0)
                    assert scimetal.last_route().engine == "scipy"
                    assert_identical(got_g, sn.gaussian_filter(sub, 1.0), f"gpu guard {dt} {bad}")


def test_review_regressions():
    a = array((10, 12), np.float64, seed=123)

    # Negative and out-of-range scalar axes in gaussian_laplace / gaussian_gradient_magnitude.
    with scimetal.precision("strict"), scimetal.device("cpu"):
        for fn in ("gaussian_laplace", "gaussian_gradient_magnitude"):
            got = getattr(mn, fn)(a, 1.0, axes=-1)
            assert scimetal.last_route().engine == "cpu"
            assert_identical(got, getattr(sn, fn)(a, 1.0, axes=-1))
            assert_declined_like_scipy(fn, a, 1.0, axes=5)
            assert_declined_like_scipy(fn, a, 1.0, axes=-5)

    # String non-numeric output dtype "bool" raises RuntimeError like SciPy.
    assert_declined_like_scipy("gaussian_filter", a, 1.0, output="bool")

    # Self-overlapping strided output view declines cleanly to SciPy.
    base = np.zeros(20, dtype=np.float64)
    overlap = np.lib.stride_tricks.as_strided(base, shape=(10, 12), strides=(8, 0))
    base_ref = np.zeros(20, dtype=np.float64)
    overlap_ref = np.lib.stride_tricks.as_strided(base_ref, shape=(10, 12), strides=(8, 0))
    mn.gaussian_filter(a, 1.0, output=overlap)
    assert scimetal.last_route().engine == "scipy"
    sn.gaussian_filter(a, 1.0, output=overlap_ref)
    assert_identical(base, base_ref)

    # SciPy consumes generator extra_arguments on the first axis.
    def scale_deriv(inp, axis, output, mode, cval, factor):
        return mn.correlate1d(inp, [factor, -2.0 * factor, factor], axis=axis, output=output, mode=mode, cval=cval)

    with pytest.raises(TypeError, match="factor"):
        mn.generic_laplace(a, scale_deriv, extra_arguments=iter((2.5,)))

    # Unaligned float64 weights work cleanly in correlate1d and correlate.
    raw_w = bytearray(9 * 8 + 1)
    unaligned_w1 = np.frombuffer(raw_w, dtype=np.float64, count=3, offset=1)
    unaligned_w1[:] = [0.25, 0.5, 0.25]
    assert not unaligned_w1.flags.aligned
    unaligned_w2 = np.frombuffer(raw_w, dtype=np.float64, count=9, offset=1).reshape(3, 3)
    unaligned_w2[:] = 1.0 / 9.0
    assert not unaligned_w2.flags.aligned
    with scimetal.precision("strict"), scimetal.device("cpu"):
        assert_identical(mn.correlate1d(a, unaligned_w1), sn.correlate1d(a, unaligned_w1))
        assert scimetal.last_route().engine == "cpu"
        assert_identical(mn.correlate(a, unaligned_w2), sn.correlate(a, unaligned_w2))
        assert scimetal.last_route().engine == "cpu"



