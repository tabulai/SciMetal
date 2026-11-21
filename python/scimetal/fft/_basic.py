"""Native routes for scipy.fft's basic transforms.

Every function has SciPy's signature and SciPy's semantics for the calls it
accepts. A call it does not accept raises ``Declined`` (a ``NotAccelerated``),
and SciPy then produces the result or the error: anything SciPy would reject
is declined rather than diagnosed here, so the messages stay SciPy's.

What runs natively (GPU, single precision, see include/sm_fft.h):

- ``fft`` / ``ifft`` along any axis, ``rfft`` / ``irfft`` along the last axis,
  with ``n`` padding or truncation along the last axis;
- ``fftn`` / ``ifftn`` / ``fft2`` / ``ifft2`` over any axes, and
  ``rfftn`` / ``irfftn`` / ``rfft2`` / ``irfft2`` when the half-spectrum axis
  is the array's last, with ``s`` resizing along the last axis only;
- float32, float64, complex64 and complex128 arrays (float16 and integers
  with magnitude below 2^24 are promoted as SciPy promotes them).
  A double-precision input whose
  magnitudes single precision cannot carry (above 2^96 or all below 2^-100,
  infinities, NaNs) is declined.

Under the ``strict`` precision policy double-precision inputs stay in double
precision: they are declined and left with the caller's ``workers`` untouched
so the result is bit for bit SciPy's own (pocketfft takes rows through its
SIMD or its scalar path depending on how a batch is split between threads, and
the two round differently in the last place, about 1e-16 relative). Under the
``fast`` and ``refined`` policies, a declined call that SciPy can thread (a
batch of transforms, or one long 1-D transform of at least 2^17 points, e.g.
under ``device='cpu'`` or for lengths without a GPU plan) is handed to SciPy
with ``workers`` set to the performance-core count when the caller left it
unset. Set ``SCIMETAL_FFT_THREADED_FALLBACK=0`` to leave declined calls
exactly as the caller made them.
"""
from __future__ import annotations

import contextlib
import contextvars
import operator
import os

import numpy as np
import scipy.fft

from .. import _config, _native
from .._dispatch import NotAccelerated, accelerated
from . import _native_fft
from ._native_fft import C2R, FORWARD, INVERSE, R2C

__all__ = ["fft", "ifft", "rfft", "irfft", "fft2", "ifft2", "fftn", "ifftn",
           "rfft2", "irfft2", "rfftn", "irfftn"]

# SciPy's own threading starts paying at about this many points, over a batch
# or within one transform of a 1-D array.
_THREADED_MIN_POINTS = 1 << 17
_THREADED_FALLBACK = os.environ.get("SCIMETAL_FFT_THREADED_FALLBACK", "1").strip().lower() not in ("0", "false", "no", "off")
_CPU_COUNT = os.cpu_count() or 1


class Declined(NotAccelerated):
    """This call goes to SciPy; ``threaded`` says it is worth SciPy's worker threads."""

    def __init__(self, reason: str, threaded: bool = False):
        super().__init__(reason)
        self.threaded = threaded


def _workers(workers) -> int:
    """SciPy's effective worker count for this call (scipy.fft._duccfft.helper._workers)."""
    if workers is None:
        return scipy.fft.get_workers()
    try:
        workers = operator.index(workers)
    except TypeError:
        raise Declined("workers is not an integer") from None
    if workers < 0:
        if workers < -_CPU_COUNT:
            raise Declined("workers is out of range")
        workers += 1 + _CPU_COUNT
    elif workers == 0:
        raise Declined("workers is zero")
    return workers


def _input(x, norm, workers, plan, overwrite_x=False):
    """The array to transform, the normalisation code and SciPy's worker count."""
    if plan is not None:
        raise Declined("precomputed plans are SciPy's to reject")
    if not isinstance(x, np.ndarray):
        raise Declined("not a NumPy array")
    try:
        if overwrite_x and not x.flags.writeable:
            raise Declined("overwrite_x on a read-only array is SciPy's to handle")
    except (TypeError, ValueError):
        raise Declined("invalid overwrite_x") from None
    if x.ndim == 0 or x.ndim > _native.SM_MAX_NDIM:
        raise Declined(f"no native path for {x.ndim}-dimensional arrays")
    device = _config.get_device()
    if device == "cpu":
        raise Declined("device policy 'cpu': there is no native CPU FFT", threaded=True)
    if device == "auto":
        try:
            _native.decline_small(x.size, _native_fft.auto_min_points())
        except NotAccelerated as why:
            raise Declined(why.reason, threaded=True) from None
    try:
        norm = _native_fft.NORMS[norm]
    except (KeyError, TypeError):
        raise Declined("unknown norm") from None
    threads = _workers(workers)

    # Promotion as in scipy.fft._duccfft.helper._asfarray: float16 to float32,
    # integers and booleans to float64.
    kind = x.dtype.kind
    promoted = None
    if kind == "f" and x.dtype.itemsize == 2:
        promoted = np.float32
    elif kind in "iub":
        promoted = np.float64
    elif _native.native_dtype(x.dtype.newbyteorder("=")) is None:
        raise Declined(f"no native path for dtype {x.dtype}")
    double = np.dtype(promoted or x.dtype).itemsize // (2 if kind == "c" else 1) == 8
    if double and _config.get_precision() == "strict":
        raise Declined("strict precision keeps double-precision FFTs exact, on SciPy", threaded=True)
    if kind in "iu" and x.dtype.itemsize >= 4 and x.size:
        # Check before promotion while values retain their integer semantics.
        # Separate extrema avoid overflowing abs(INT_MIN) and allocating a mask.
        integers = np.asarray(x)
        limit = 1 << 24
        if integers.max() >= limit or (kind == "i" and integers.min() <= -limit):
            raise Declined("integer FFT input reaches the float32 guard (|x| >= 2^24)")
    if promoted is not None:
        x = np.asarray(x, promoted)
    elif not (x.dtype.isnative and x.flags.aligned):
        x = np.require(x, dtype=x.dtype.newbyteorder("="), requirements="A")
    return np.asarray(x), norm, threads


def _index(value, what):
    try:
        return operator.index(value)
    except TypeError:
        raise Declined(f"{what} is not an integer") from None


def _length(value):
    """A transform length the core can be asked for (it indexes with 32 bits)."""
    value = _index(value, "the transform length")
    if value < 1:
        raise Declined("invalid number of data points")
    if value >= 1 << 31:
        raise Declined("the transform is too long for the GPU kernels", threaded=True)
    return value


def _transform(x, kind, norm, axes, n, threads, function):
    try:
        return _native_fft.transform(x, _native_fft.options(kind, norm, axes, n, threads), function)
    except Declined:
        raise
    except NotAccelerated as why:
        # Whatever the core declines (no plan for the length, below the
        # crossover, too large to index) is still worth SciPy's threads.
        raise Declined(str(why), threaded=True) from None


def _transform_1d(function, kind, x, n, axis, norm, overwrite_x, workers, plan):
    x, norm, threads = _input(x, norm, workers, plan, overwrite_x=overwrite_x)
    axis = _index(axis, "axis")
    if not -x.ndim <= axis < x.ndim:
        raise Declined("axis is out of range")
    axis %= x.ndim
    if n is not None:
        n = _length(n)
    if kind == R2C and x.dtype.kind == "c":
        raise Declined("a real transform of complex input is SciPy's to reject")
    if axis != x.ndim - 1:
        if kind in (R2C, C2R):
            raise Declined("real transforms run natively along the last axis only", threaded=True)
        if n is not None and n != x.shape[axis]:
            raise Declined("padding or truncation runs natively along the last axis only", threaded=True)
        n = None
    return _transform(x, kind, norm, [axis], n or 0, threads, function)


def _shape_and_axes(x, s, axes):
    """SciPy's reading of ``s`` and ``axes`` (scipy.fft._duccfft.helper._init_nd_shape_and_axes)."""
    def integers(values, what):
        if isinstance(values, (int, np.integer)):
            values = (values,)
        try:
            return [operator.index(value) for value in values]
        except TypeError:
            raise Declined(f"{what} is not a sequence of integers") from None

    if axes is not None:
        axes = [axis + x.ndim if axis < 0 else axis for axis in integers(axes, "axes")]
        if any(axis >= x.ndim or axis < 0 for axis in axes) or len(set(axes)) != len(axes):
            raise Declined("axes are out of range or repeated")
    if s is not None:
        s = integers(s, "s")
        if axes and len(axes) != len(s):
            raise Declined("s and axes differ in length")
        if axes is None:
            if len(s) > x.ndim:
                raise Declined("s has more entries than the input has axes")
            axes = list(range(x.ndim - len(s), x.ndim))
        s = [x.shape[axis] if length == -1 else length for length, axis in zip(s, axes)]
    elif axes is None:
        axes, s = list(range(x.ndim)), list(x.shape)
    else:
        s = [x.shape[axis] for axis in axes]
    return [_length(length) for length in s], axes


def _transform_nd(function, kind, x, s, axes, norm, overwrite_x, workers, plan):
    x, norm, threads = _input(x, norm, workers, plan, overwrite_x=overwrite_x)
    natural = s is None
    s, axes = _shape_and_axes(x, s, axes)
    if not axes:
        raise Declined("no axes to transform")
    last = x.ndim - 1
    if kind == R2C and x.dtype.kind == "c":
        raise Declined("a real transform of complex input is SciPy's to reject")
    if kind in (R2C, C2R) and axes[-1] != last:
        raise Declined("real transforms run natively with the half-spectrum axis last", threaded=True)
    n = 0
    for length, axis in zip(s, axes):
        if axis == last:
            if kind == C2R:
                # s gives the output length; without it the core uses 2 (m - 1).
                n = 0 if natural else length
            elif length != x.shape[axis]:
                n = length
        elif length != x.shape[axis]:
            raise Declined("padding or truncation runs natively along the last axis only", threaded=True)
    return _transform(x, kind, norm, axes, n, threads, function)


# Set while a declined call is inside SciPy on our behalf: the uarray backend
# (see _backend.py) must then stand aside instead of taking the call again.
_delegating = contextvars.ContextVar("scimetal_fft_delegating", default=False)


@contextlib.contextmanager
def _delegating_to_scipy():
    token = _delegating.set(True)
    try:
        yield
    finally:
        _delegating.reset(token)


def run_threaded(reference, x, arguments, threads):
    """SciPy's own transform, threaded over the batch."""
    with _delegating_to_scipy():
        return reference(x, *arguments, threads)


def threaded_fallback(x, axes, why, workers, plan):
    """Worker count for a declined call, or None to leave the call as the caller made it.

    SciPy's threads split the transforms of a batch between them, and work
    inside one long transform of a 1-D array. They do not help (and cost time)
    when an axis of a multi-dimensional array holds a single transform.
    """
    if not (_THREADED_FALLBACK and getattr(why, "threaded", False)) or workers is not None or plan is not None \
            or scipy.fft.get_workers() != 1 or _config.get_precision() == "strict":
        return None
    if not isinstance(x, np.ndarray) or x.size < _THREADED_MIN_POINTS:
        return None
    if x.ndim > 1:
        try:
            axes = range(x.ndim) if axes is None else np.atleast_1d(axes).tolist()
            if min(x.size // x.shape[axis] for axis in axes) < 2:
                return None
        except (TypeError, ValueError, IndexError, ZeroDivisionError):
            return None  # arguments SciPy will reject
    threads = _native.performance_cores()
    return threads if threads > 1 else None


def _one_dimensional(name, kind):
    function = f"fft.{name}"
    reference = getattr(scipy.fft, name)

    def core(x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, *, plan=None):
        return _transform_1d(function, kind, x, n, axis, norm, overwrite_x, workers, plan)

    def implementation(x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, *, plan=None):
        try:
            return _transform_1d(function, kind, x, n, axis, norm, overwrite_x, workers, plan)
        except Declined as why:
            threads = threaded_fallback(x, axis, why, workers, plan)
            if threads is not None:
                raise NotAccelerated(
                    f"{why}; SciPy with workers={threads}",
                    context=_delegating_to_scipy,
                    workers=threads,
                ) from None
            raise NotAccelerated(str(why), context=_delegating_to_scipy) from None

    implementation.__name__ = implementation.__qualname__ = name
    # The native attempt alone, for the uarray backend: (result) or Declined.
    implementation.core = core
    implementation.reference = reference
    return accelerated(function)(implementation)


def _n_dimensional(name, kind, default_axes):
    function = f"fft.{name}"
    reference = getattr(scipy.fft, name)

    def core(x, s=None, axes=default_axes, norm=None, overwrite_x=False, workers=None, *, plan=None):
        return _transform_nd(function, kind, x, s, axes, norm, overwrite_x, workers, plan)

    def implementation(x, s=None, axes=default_axes, norm=None, overwrite_x=False, workers=None, *, plan=None):
        try:
            return _transform_nd(function, kind, x, s, axes, norm, overwrite_x, workers, plan)
        except Declined as why:
            threads = threaded_fallback(x, axes, why, workers, plan)
            if threads is not None:
                raise NotAccelerated(
                    f"{why}; SciPy with workers={threads}",
                    context=_delegating_to_scipy,
                    workers=threads,
                ) from None
            raise NotAccelerated(str(why), context=_delegating_to_scipy) from None

    implementation.__name__ = implementation.__qualname__ = name
    implementation.core = core
    implementation.reference = reference
    return accelerated(function)(implementation)


fft = _one_dimensional("fft", FORWARD)
ifft = _one_dimensional("ifft", INVERSE)
rfft = _one_dimensional("rfft", R2C)
irfft = _one_dimensional("irfft", C2R)
fftn = _n_dimensional("fftn", FORWARD, None)
ifftn = _n_dimensional("ifftn", INVERSE, None)
rfftn = _n_dimensional("rfftn", R2C, None)
irfftn = _n_dimensional("irfftn", C2R, None)
fft2 = _n_dimensional("fft2", FORWARD, (-2, -1))
ifft2 = _n_dimensional("ifft2", INVERSE, (-2, -1))
rfft2 = _n_dimensional("rfft2", R2C, (-2, -1))
irfft2 = _n_dimensional("irfft2", C2R, (-2, -1))

_PREPARE_KINDS = {
    "fft": (FORWARD, False, None),
    "ifft": (INVERSE, False, None),
    "rfft": (R2C, False, None),
    "irfft": (C2R, False, None),
    "fft2": (FORWARD, True, (-2, -1)),
    "ifft2": (INVERSE, True, (-2, -1)),
    "rfft2": (R2C, True, (-2, -1)),
    "irfft2": (C2R, True, (-2, -1)),
    "fftn": (FORWARD, True, None),
    "ifftn": (INVERSE, True, None),
    "rfftn": (R2C, True, None),
    "irfftn": (C2R, True, None),
}


def prepare(shape, dtype=np.complex128, *, kind="fft", axes=None, n=None, s=None, norm=None) -> bool:
    """Compile and cache GPU FFT pipelines ahead of time for the given request.

    Returns ``True`` when the current policies and device can run the transform
    natively and its pipelines are ready, or ``False`` when the request would
    fall back to SciPy. Raises ``ValueError`` on invalid arguments.
    """
    try:
        fft_kind, is_nd, default_axes = _PREPARE_KINDS[kind]
    except (KeyError, TypeError):
        raise ValueError(f"unknown FFT kind {kind!r}") from None

    if isinstance(shape, (int, np.integer)):
        shape_tuple = (operator.index(shape),)
    else:
        try:
            shape_tuple = tuple(operator.index(d) for d in shape)
        except TypeError:
            raise ValueError("shape must be an integer or sequence of integers") from None
    ndim = len(shape_tuple)
    if not 1 <= ndim <= _native.SM_MAX_NDIM:
        raise ValueError(f"shape rank must be between 1 and {_native.SM_MAX_NDIM}, got {ndim}")
    if any(d < 0 for d in shape_tuple):
        raise ValueError(f"shape dimensions must be non-negative, got {shape_tuple}")

    try:
        np_dtype = np.dtype(dtype)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid dtype {dtype!r}") from exc
    dtype_kind = np_dtype.kind
    if dtype_kind == "f" and np_dtype.itemsize == 2:
        np_dtype = np.dtype(np.float32)
    elif dtype_kind in "iub":
        np_dtype = np.dtype(np.float64)
    else:
        np_dtype = np_dtype.newbyteorder("=")
    if _native.native_dtype(np_dtype) is None:
        raise ValueError(f"unsupported dtype {dtype!r}")
    if fft_kind == R2C and np_dtype.kind == "c":
        raise ValueError("a real-to-complex FFT needs a real input")

    try:
        norm_code = _native_fft.NORMS[norm]
    except (KeyError, TypeError):
        raise ValueError(f"unknown norm {norm!r}") from None

    if n is not None and s is not None:
        raise ValueError("n and s cannot both be specified")

    def _int_seq(values, what):
        if isinstance(values, (int, np.integer)):
            return [operator.index(values)]
        try:
            return [operator.index(v) for v in values]
        except TypeError:
            raise ValueError(f"{what} must be an integer or sequence of integers") from None

    c_n = 0
    too_large = False
    off_last_declined = False

    if not is_nd:
        if s is not None:
            raise ValueError(f"s is not valid for 1-D FFT kind {kind!r}")
        if axes is None:
            axes_list = [ndim - 1]
        else:
            axes_list = _int_seq(axes, "axes")
            if len(axes_list) != 1:
                raise ValueError("1-D FFT expects a single axis")
            axis = axes_list[0]
            if not -ndim <= axis < ndim:
                raise ValueError(f"axis {axis} is out of bounds for array of dimension {ndim}")
            axes_list = [axis % ndim]
        if n is not None:
            try:
                n_val = operator.index(n)
            except TypeError:
                raise ValueError("n must be an integer") from None
            if n_val < 1:
                raise ValueError(f"invalid number of data points {n_val} specified")
            if n_val >= 1 << 31:
                too_large = True
            if axes_list[0] != ndim - 1:
                if fft_kind in (R2C, C2R) or n_val != shape_tuple[axes_list[0]]:
                    off_last_declined = True
            else:
                c_n = n_val
        elif axes_list[0] != ndim - 1 and fft_kind in (R2C, C2R):
            off_last_declined = True
    else:
        if n is not None:
            raise ValueError(f"n is not valid for multi-dimensional FFT kind {kind!r}")
        effective_axes = default_axes if axes is None else axes
        if effective_axes is not None:
            raw_axes = _int_seq(effective_axes, "axes")
            axes_list = [a + ndim if a < 0 else a for a in raw_axes]
            if any(a < 0 or a >= ndim for a in axes_list) or len(set(axes_list)) != len(axes_list):
                raise ValueError("axes are out of range or repeated")
        else:
            axes_list = None
        natural = s is None
        if s is not None:
            s_raw = _int_seq(s, "s")
            if axes_list is not None and len(s_raw) != len(axes_list):
                raise ValueError("when given, axes and s must have the same length")
            if axes_list is None:
                if len(s_raw) > ndim:
                    raise ValueError("s has more entries than the input has axes")
                axes_list = list(range(ndim - len(s_raw), ndim))
            if not axes_list:
                raise ValueError("at least one axis must be transformed")
            s_list = []
            for length, axis in zip(s_raw, axes_list):
                if length == -1:
                    s_list.append(shape_tuple[axis])
                elif length < 1:
                    raise ValueError(f"invalid number of data points {length} specified")
                else:
                    s_list.append(length)
        else:
            if axes_list is None:
                axes_list = list(range(ndim))
            if not axes_list:
                raise ValueError("at least one axis must be transformed")
            s_list = [shape_tuple[axis] for axis in axes_list]

        if any(length >= 1 << 31 for length in s_list):
            too_large = True
        last = ndim - 1
        if fft_kind in (R2C, C2R) and axes_list[-1] != last:
            off_last_declined = True
        for length, axis in zip(s_list, axes_list):
            if axis == last:
                if fft_kind == C2R:
                    c_n = 0 if natural else length
                elif length != shape_tuple[axis]:
                    c_n = length
            elif length != shape_tuple[axis]:
                off_last_declined = True

    if too_large or off_last_declined or any(d == 0 or d >= (1 << 64) for d in shape_tuple):
        return False
    device = _config.get_device()
    if device == "cpu":
        return False
    double = np_dtype.itemsize // (2 if np_dtype.kind == "c" else 1) == 8
    if double and _config.get_precision() == "strict":
        return False
    if device == "auto":
        total = 1
        for d in shape_tuple:
            total *= d
        try:
            _native.decline_small(total, _native_fft.auto_min_points())
        except NotAccelerated:
            return False

    opts = _native_fft.options(fft_kind, norm_code, axes_list, c_n, scipy.fft.get_workers())
    try:
        info = _native_fft.plan(np_dtype, shape_tuple, opts, prepare=True)
    except NotAccelerated:
        return False
    return info.status == _native.SM_OK
