"""Native routes for scipy.ndimage's separable, dense, and composite filters.

Every function here turns its arguments into native filter calls (``sm_ndimage_separable_filter``,
``sm_ndimage_dense_filter``, or ``sm_ndimage_composite_filter``) matching SciPy's exact
weights, axis order, origins, and boundary modes. A call these routes do not cover raises
``NotAccelerated`` before anything is computed, and the ``accelerated`` wrapper
hands it to SciPy with the caller's arguments untouched. That includes every
call SciPy would reject, so the exception a caller sees is SciPy's own.

Handled natively: real ``numpy.ndarray`` inputs (float32, float64, bool,
signed/unsigned 8..64-bit integers) of rank 1 to 8 with at least one element;
``output`` None, a supported real dtype (type, ``np.dtype``, or string), or a
writeable real array of the matching shape (the input itself for in-place use,
or memory that does not overlap it). Declined: any other input type or dtype
(complex, float16, longdouble, lists, array subclasses, other array libraries,
byte-swapped or unaligned data), empty and 0-d arrays, an ``output`` of a
non-real dtype (complex, float16, longdouble), wrong shape, or partial overlap,
complex weights, and invalid arguments of any kind.
"""
from __future__ import annotations

import numbers
import operator

import numpy as np

from .. import _native
from .._config import get_precision
from .._dispatch import NotAccelerated, Route, accelerated, record
from . import _native_ndimage as _nd

__all__ = ["convolve", "convolve1d", "correlate", "correlate1d", "gaussian_filter", "gaussian_filter1d",
           "gaussian_gradient_magnitude", "gaussian_laplace", "laplace", "prewitt", "sobel",
           "uniform_filter", "uniform_filter1d"]

# SciPy's mode names for these filters. Its line-extension code treats
# 'grid-wrap' as 'wrap' and 'grid-constant' as 'constant', and 'grid-mirror'
# is another name for 'reflect'.
_MODES = {"nearest": _nd.NEAREST, "wrap": _nd.WRAP, "grid-wrap": _nd.WRAP, "reflect": _nd.REFLECT,
          "grid-mirror": _nd.REFLECT, "mirror": _nd.MIRROR, "constant": _nd.CONSTANT,
          "grid-constant": _nd.CONSTANT}
# Type characters, not dtype equality: where long double is a 64-bit double
# (arm64 macOS) its dtype compares equal to float64, yet SciPy rejects it.
# Includes all 11 real dtypes (bool, signed/unsigned 8..64-bit int, float32/64)
# including platform aliases 'l'/'q' and 'L'/'Q'.
_REAL_CHARS = ("?", "b", "B", "h", "H", "i", "I", "l", "L", "q", "Q", "f", "d")
# Below this many elements SciPy is faster: a native call has a floor of about
# 16 microseconds (hand-off, buffer, thread wake-up) where SciPy's is 3. Measured
# on an M1 Ultra, the native CPU engine wins from 96 x 96 to 128 x 128 (2-D) and
# from 16384 elements (1-D) for every function here.
_BREAK_EVEN_ELEMENTS = 1 << 14


def _decline(reason: str):
    raise NotAccelerated(reason)


def _has_self_overlap(arr: np.ndarray) -> bool:
    if arr.size <= 1:
        return False
    if any(dim > 1 and stride == 0 for dim, stride in zip(arr.shape, arr.strides)):
        return True
    axes = sorted(((abs(st), dim) for dim, st in zip(arr.shape, arr.strides) if dim > 1), key=lambda t: t[0])
    span = arr.itemsize
    for st, dim in axes:
        if st < span:
            return True
        span += (dim - 1) * st
    return False


def _input(input):
    if type(input) is not np.ndarray:
        _decline("input is not a plain NumPy array")
    if input.dtype.char not in _REAL_CHARS or not input.dtype.isnative:
        _decline(f"no native filter for dtype {input.dtype}")
    if not 1 <= input.ndim <= _native.SM_MAX_NDIM or input.size == 0:
        _decline("0-d, empty and more than 8-dimensional arrays go to SciPy")
    if not input.flags.aligned:
        _decline("input data is not aligned")
    _native.decline_small(input.size, _BREAK_EVEN_ELEMENTS)
    return input


def _output(output, input):
    """Return ``(target, out_dtype)``: ``target`` is None for a new array or the caller's array to fill."""
    if output is None:
        return None, input.dtype
    if isinstance(output, np.ndarray):
        if (type(output) is not np.ndarray or output.dtype.char not in _REAL_CHARS
                or not output.dtype.isnative or output.shape != input.shape):
            _decline("output array of another type, dtype or shape")
        if not output.flags.writeable:
            _decline("output array is read-only")
        if _has_self_overlap(output):
            _decline("output array has self-overlapping strides")
        same_view = (output.dtype == input.dtype and
                     (output is input or
                      (output.ctypes.data == input.ctypes.data and output.strides == input.strides)))
        if not same_view and np.may_share_memory(output, input):
            _decline("output overlaps the input")
        return output, output.dtype
    if isinstance(output, (type, np.dtype, str)):
        try:
            dtype = np.dtype(output)
        except TypeError:
            _decline("output is not a dtype")
        if isinstance(output, str) and not issubclass(dtype.type, np.number):
            _decline("string non-numeric output dtype goes to SciPy")
        if dtype.char not in _REAL_CHARS or not dtype.isnative:
            _decline(f"no native filter output for dtype {dtype}")
        return None, dtype
    _decline("output is neither an array nor a dtype")


def _index(value, what: str) -> int:
    try:
        return operator.index(value)
    except TypeError:
        _decline(f"{what} is not an integer")


def _axis(axis, ndim: int) -> int:
    axis = _index(axis, "axis")
    if not -ndim <= axis < ndim:
        _decline("axis is out of range")
    return axis % ndim


def _mode(mode) -> int:
    if not isinstance(mode, str) or mode not in _MODES:
        _decline("boundary mode not supported")
    return _MODES[mode]


def _cval(cval) -> float:
    if not isinstance(cval, numbers.Real):
        _decline("cval is not a real number")
    return float(cval)


def _sequence(value, rank: int) -> list:
    """SciPy's rule for per-axis arguments: a scalar repeats, a sequence must have one entry per axis."""
    if not isinstance(value, str) and np.iterable(value):
        values = list(value)
        if len(values) != rank:
            _decline("sequence argument must have length equal to the number of axes")
        return values
    return [value] * rank


def _axes(axes, ndim: int, *, allow_negative_scalar: bool = True) -> tuple:
    if axes is None:
        return tuple(range(ndim))
    if np.isscalar(axes):
        ax = _index(axes, "axes")
        if not -ndim <= ax < ndim:
            _decline("an axis is out of range")
        if ax < 0 and not allow_negative_scalar:
            _decline("negative scalar axes goes to SciPy")
        axes = (ax % ndim,)
    elif isinstance(axes, (tuple, list, range, np.ndarray)):
        try:
            axes = tuple(_index(axis, "axes") for axis in axes)
        except TypeError:  # a 0-d array
            _decline("axes is not a sequence of integers")
        if any(not -ndim <= axis < ndim for axis in axes):
            _decline("an axis is out of range")
        axes = tuple(axis % ndim for axis in axes)
    else:
        _decline("axes is not an integer or a sequence of integers")
    if len(set(axes)) != len(axes):
        _decline("axes are not unique")
    return axes


def _weights(weights) -> np.ndarray:
    try:
        weights = np.asarray(weights)
        if weights.dtype.kind == "c":
            _decline("complex weights")
        weights = np.asarray(weights, dtype=np.float64)
    except NotAccelerated:
        raise
    except Exception:
        _decline("weights are not real numbers")
    if weights.ndim != 1 or weights.shape[0] < 1:
        _decline("no filter weights given")
    return weights


def _correlate_pass(ndim: int, weights, axis, mode, cval, origin) -> tuple:
    """One correlate1d call as a native pass: (kind, axis, mode, origin, size, weights, cval)."""
    weights = _weights(weights)
    axis = _axis(axis, ndim)
    origin = _index(origin, "origin")
    size = weights.shape[0]
    if origin < -(size // 2) or origin > (size - 1) // 2:
        _decline("invalid origin")
    if size > _nd.SM_NDIMAGE_MAX_WINDOW:
        _decline("more weights than the native filter takes")
    return (_nd.CORRELATE, axis, _mode(mode), origin, size, weights, _cval(cval))


def _uniform_pass(ndim: int, size, axis, mode, cval, origin) -> tuple:
    axis = _axis(axis, ndim)
    size = _index(size, "size")
    origin = _index(origin, "origin")
    if size < 1:
        _decline("incorrect filter size")
    if size // 2 + origin < 0 or size // 2 + origin >= size:
        _decline("invalid origin")
    if size > _nd.SM_NDIMAGE_MAX_WINDOW:
        _decline("a longer window than the native filter takes")
    return (_nd.UNIFORM, axis, _mode(mode), origin, size, None, _cval(cval))


def _gaussian_kernel(sigma, order, radius) -> np.ndarray:
    """The 1-D Gaussian (or its order-th derivative) sampled at -radius .. radius.

    The same NumPy operations in the same order as SciPy's kernel, so the
    weights, and with them the strict results, are identical to SciPy's.
    """
    if order < 0:
        raise ValueError("order must be non-negative")
    powers = np.arange(order + 1)
    variance = sigma * sigma
    x = np.arange(-radius, radius + 1)
    phi = np.exp(-0.5 / variance * x ** 2)
    phi = phi / phi.sum()
    if order == 0:
        return phi
    # The derivative is q(x) * phi(x) for a polynomial q: differentiating maps
    # q to q' - q * x / sigma^2, applied `order` times to q = 1 as a matrix on
    # the coefficients.
    q = np.zeros(order + 1)
    q[0] = 1
    step = np.diag(powers[1:], 1) + np.diag(np.ones(order) / -variance, -1)
    for _ in range(order):
        q = step.dot(q)
    return (x[:, None] ** powers).dot(q) * phi


def _gaussian_pass(ndim: int, sigma, axis, order, mode, cval, truncate, radius) -> tuple:
    try:
        deviation = float(sigma)
        reach = int(truncate * deviation + 0.5)
        if radius is not None:
            reach = radius
        if not isinstance(reach, numbers.Integral) or reach < 0:
            raise ValueError("Radius must be a nonnegative integer.")
        # SciPy correlates with the reversed kernel.
        weights = _gaussian_kernel(sigma, order, reach)[::-1]
    except Exception:
        _decline("sigma, order, truncate or radius is not usable")
    return _correlate_pass(ndim, weights, axis, mode, cval, 0)


def _run(function: str, array, passes: list, target, out_dtype=None):
    if out_dtype is None:
        out_dtype = array.dtype
    if passes:
        result = _nd.separable_filter(function, array, passes, out_dtype=out_dtype)
    else:
        # Nothing to filter (every sigma or size is trivial): SciPy copies/casts the input.
        try:
            result = _native.astype(array, out_dtype)
        except MemoryError as refusal:
            _decline(f"the native copy ran out of memory: {refusal}")
        record(Route(function, "cpu", np.dtype(out_dtype).name, get_precision(), "no axis to filter: copied"))
    if target is None:
        return result
    target[...] = result
    return target


def _run_composite(function: str, array: np.ndarray, chains: list, combine_op: int, target, out_dtype=None):
    if len(chains) > 1 and target is not None and np.may_share_memory(target, array):
        # SciPy's generic_laplace / generic_gradient_magnitude mutates output
        # in-place on axis 0 before reading input for subsequent axes.
        _decline("in-place multi-axis composite output goes to SciPy")
    result = _nd.composite_filter(function, array, chains, combine_op, out_dtype=out_dtype)
    if target is None:
        return result
    target[...] = result
    return target


@accelerated("ndimage.correlate1d")
def correlate1d(input, weights, axis=-1, output=None, mode="reflect", cval=0.0, origin=0):
    array = _input(input)
    target, out_dtype = _output(output, array)
    passes = [_correlate_pass(array.ndim, weights, axis, mode, cval, origin)]
    return _run("ndimage.correlate1d", array, passes, target, out_dtype)


@accelerated("ndimage.convolve1d")
def convolve1d(input, weights, axis=-1, output=None, mode="reflect", cval=0.0, origin=0):
    array = _input(input)
    target, out_dtype = _output(output, array)
    weights = _weights(weights)
    # Convolution is correlation with the reversed weights; an even window's
    # centre moves by one when it is reversed.
    origin = -_index(origin, "origin")
    if not weights.shape[0] & 1:
        origin -= 1
    passes = [_correlate_pass(array.ndim, weights[::-1], axis, mode, cval, origin)]
    return _run("ndimage.convolve1d", array, passes, target, out_dtype)


@accelerated("ndimage.gaussian_filter1d")
def gaussian_filter1d(input, sigma, axis=-1, order=0, output=None, mode="reflect", cval=0.0, truncate=4.0,
                      *, radius=None):
    array = _input(input)
    target, out_dtype = _output(output, array)
    passes = [_gaussian_pass(array.ndim, sigma, axis, order, mode, cval, truncate, radius)]
    return _run("ndimage.gaussian_filter1d", array, passes, target, out_dtype)


@accelerated("ndimage.gaussian_filter")
def gaussian_filter(input, sigma, order=0, output=None, mode="reflect", cval=0.0, truncate=4.0, *,
                    radius=None, axes=None):
    array = _input(input)
    target, out_dtype = _output(output, array)
    axes = _axes(axes, array.ndim)
    count = len(axes)
    orders, sigmas = _sequence(order, count), _sequence(sigma, count)
    modes, radiuses = _sequence(mode, count), _sequence(radius, count)
    try:
        active = [i for i in range(count) if sigmas[i] > 1e-15]
    except Exception:
        _decline("sigma is not a real number")
    passes = [_gaussian_pass(array.ndim, sigmas[i], axes[i], orders[i], modes[i], cval, truncate, radiuses[i])
              for i in active]
    return _run("ndimage.gaussian_filter", array, passes, target, out_dtype)


@accelerated("ndimage.uniform_filter1d")
def uniform_filter1d(input, size, axis=-1, output=None, mode="reflect", cval=0.0, origin=0):
    array = _input(input)
    target, out_dtype = _output(output, array)
    passes = [_uniform_pass(array.ndim, size, axis, mode, cval, origin)]
    return _run("ndimage.uniform_filter1d", array, passes, target, out_dtype)


@accelerated("ndimage.uniform_filter")
def uniform_filter(input, size=3, output=None, mode="reflect", cval=0.0, origin=0, *, axes=None):
    array = _input(input)
    target, out_dtype = _output(output, array)
    axes = _axes(axes, array.ndim)
    count = len(axes)
    sizes, origins, modes = _sequence(size, count), _sequence(origin, count), _sequence(mode, count)
    try:
        # SciPy skips sizes of at most 1 and truncates the others to integers.
        active = [(i, int(sizes[i])) for i in range(count) if sizes[i] > 1]
    except Exception:
        _decline("size is not a real number")
    passes = [_uniform_pass(array.ndim, size, axes[i], modes[i], cval, origins[i]) for i, size in active]
    return _run("ndimage.uniform_filter", array, passes, target, out_dtype)


def _derivative(function: str, smoothing, input, axis, output, mode, cval):
    """prewitt and sobel: a central difference along `axis`, then `smoothing` along every other axis."""
    array = _input(input)
    axis = _axis(axis, array.ndim)
    target, out_dtype = _output(output, array)
    modes = _sequence(mode, array.ndim)
    passes = [_correlate_pass(array.ndim, [-1, 0, 1], axis, modes[axis], cval, 0)]
    passes += [_correlate_pass(array.ndim, smoothing, other, modes[other], cval, 0)
               for other in range(array.ndim) if other != axis]
    return _run(function, array, passes, target, out_dtype)


@accelerated("ndimage.prewitt")
def prewitt(input, axis=-1, output=None, mode="reflect", cval=0.0):
    return _derivative("ndimage.prewitt", [1, 1, 1], input, axis, output, mode, cval)


@accelerated("ndimage.sobel")
def sobel(input, axis=-1, output=None, mode="reflect", cval=0.0):
    return _derivative("ndimage.sobel", [1, 2, 1], input, axis, output, mode, cval)


def _dense_weights(weights) -> np.ndarray:
    try:
        w = np.asarray(weights)
        if w.dtype.kind == "c":
            _decline("complex weights")
        w = np.asarray(w, dtype=np.float64)
    except NotAccelerated:
        raise
    except Exception:
        _decline("weights are not real numbers")
    return w


def _correlate_or_convolve(function: str, input, weights, output, mode, cval, origin, convolution: bool, axes):
    array = _input(input)
    target, out_dtype = _output(output, array)
    axes = _axes(axes, array.ndim, allow_negative_scalar=False)
    w = _dense_weights(weights)
    num_axes = len(axes)
    if num_axes < array.ndim:
        if w.ndim != num_axes:
            _decline("weights.ndim must match len(axes)")
        w = np.expand_dims(w, tuple(ax for ax in range(array.ndim) if ax not in axes))
    wshape = [s for s in w.shape if s > 0]
    if len(wshape) != array.ndim:
        _decline("weights.ndim must match input.ndim with positive extents")
    origins = [_index(o, "origin") for o in _sequence(origin, num_axes)]
    if num_axes < array.ndim:
        origins_temp = [0] * array.ndim
        for o, ax in zip(origins, axes):
            origins_temp[ax] = o
        origins = origins_temp
    if convolution:
        w = w[(slice(None, None, -1),) * w.ndim]
        for ii in range(len(origins)):
            origins[ii] = -origins[ii]
            if not w.shape[ii] & 1:
                origins[ii] -= 1
    for o, lenw in zip(origins, wshape):
        if o < -(lenw // 2) or o > (lenw - 1) // 2:
            _decline("invalid origin")
        if lenw > _nd.SM_NDIMAGE_MAX_WINDOW:
            _decline("more weights than the native filter takes")
    mode_code = _mode(mode)
    cval_f = _cval(cval)
    result = _nd.dense_filter(function, array, w, origins, mode_code, cval_f, out_dtype=out_dtype)
    if target is None:
        return result
    target[...] = result
    return target


@accelerated("ndimage.correlate")
def correlate(input, weights, output=None, mode='reflect', cval=0.0, origin=0, *, axes=None):
    return _correlate_or_convolve("ndimage.correlate", input, weights, output, mode, cval, origin, False, axes)


@accelerated("ndimage.convolve")
def convolve(input, weights, output=None, mode='reflect', cval=0.0, origin=0, *, axes=None):
    return _correlate_or_convolve("ndimage.convolve", input, weights, output, mode, cval, origin, True, axes)


@accelerated("ndimage.laplace")
def laplace(input, output=None, mode="reflect", cval=0.0, *, axes=None):
    array = _input(input)
    target, out_dtype = _output(output, array)
    axes = _axes(axes, array.ndim)
    if not axes:
        return _run("ndimage.laplace", array, [], target, out_dtype)
    modes = _sequence(mode, len(axes))
    chains = [[_correlate_pass(array.ndim, [1, -2, 1], ax, modes[i], cval, 0)] for i, ax in enumerate(axes)]
    return _run_composite("ndimage.laplace", array, chains, _nd.COMBINE_SUM, target, out_dtype)


def _gaussian_kwargs(kwargs: dict):
    kw = dict(kwargs)
    truncate = kw.pop("truncate", 4.0)
    radius = kw.pop("radius", None)
    if kw:
        _decline(f"unsupported keyword argument(s): {', '.join(kw)}")
    return truncate, radius


@accelerated("ndimage.gaussian_laplace")
def gaussian_laplace(input, sigma, output=None, mode="reflect", cval=0.0, *, axes=None, **kwargs):
    truncate, radius = _gaussian_kwargs(kwargs)
    array = _input(input)
    target, out_dtype = _output(output, array)
    axes = _axes(axes, array.ndim)
    num_axes = len(axes)
    sigmas = _sequence(sigma, num_axes)
    if not axes:
        _sequence(mode, 0)
        return _run("ndimage.gaussian_laplace", array, [], target, out_dtype)
    modes = _sequence(mode, num_axes)
    if num_axes < array.ndim:
        sigma_temp = [0] * array.ndim
        for s, ax in zip(sigmas, axes):
            sigma_temp[ax] = s
        sigmas_full = sigma_temp
    else:
        sigmas_full = sigmas
    radiuses = _sequence(radius, array.ndim)
    try:
        active_axes = [j for j in range(array.ndim) if sigmas_full[j] > 1e-15]
    except Exception:
        _decline("sigma is not a real number")
    chains = [
        [_gaussian_pass(array.ndim, sigmas_full[j], j, 2 if j == ax else 0, modes[i], cval, truncate, radiuses[j])
         for j in active_axes]
        for i, ax in enumerate(axes)
    ]
    return _run_composite("ndimage.gaussian_laplace", array, chains, _nd.COMBINE_SUM, target, out_dtype)


@accelerated("ndimage.gaussian_gradient_magnitude")
def gaussian_gradient_magnitude(input, sigma, output=None, mode="reflect", cval=0.0, *, axes=None, **kwargs):
    truncate, radius = _gaussian_kwargs(kwargs)
    array = _input(input)
    target, out_dtype = _output(output, array)
    axes = _axes(axes, array.ndim)
    if not axes:
        _sequence(mode, 0)
        return _run("ndimage.gaussian_gradient_magnitude", array, [], target, out_dtype)
    modes = _sequence(mode, len(axes))
    sigmas_full = _sequence(sigma, array.ndim)
    radiuses = _sequence(radius, array.ndim)
    try:
        active_axes = [j for j in range(array.ndim) if sigmas_full[j] > 1e-15]
    except Exception:
        _decline("sigma is not a real number")
    chains = [
        [_gaussian_pass(array.ndim, sigmas_full[j], j, 1 if j == ax else 0, modes[i], cval, truncate, radiuses[j])
         for j in active_axes]
        for i, ax in enumerate(axes)
    ]
    return _run_composite("ndimage.gaussian_gradient_magnitude", array, chains, _nd.COMBINE_SQRT_SUM_SQ, target,
                          out_dtype)
