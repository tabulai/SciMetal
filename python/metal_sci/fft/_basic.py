"""Native routes for scipy.fft's basic transforms.

Every function has SciPy's signature and SciPy's semantics for the calls it
accepts. A call it does not accept raises ``Declined`` (a ``NotAccelerated``),
and SciPy then produces the result or the error: anything SciPy would reject
is declined rather than diagnosed here, so the messages stay SciPy's.

What runs natively (GPU, single precision, see include/ms_fft.h):

- ``fft`` / ``ifft`` along any axis, ``rfft`` / ``irfft`` along the last axis,
  with ``n`` padding or truncation along the last axis;
- ``fftn`` / ``ifftn`` / ``fft2`` / ``ifft2`` over any axes, and
  ``rfftn`` / ``irfftn`` / ``rfft2`` / ``irfft2`` when the half-spectrum axis
  is the array's last, with ``s`` resizing along the last axis only;
- float32, float64, complex64 and complex128 arrays (integers and float16
  are promoted as SciPy promotes them). A double-precision input whose
  magnitudes single precision cannot carry (above 2^96 or all below 2^-100,
  infinities, NaNs) is declined.

Under the ``strict`` precision policy double-precision inputs stay in double
precision: they are declined. Since ``scipy.fft`` uses one worker by default,
a declined call that SciPy can thread (a batch of transforms, or one long
transform of a 1-D array) is handed to SciPy with ``workers`` set to the
performance-core count when the caller left it unset. That is SciPy's own
double-precision transform, but not always bit for bit the single-worker one:
pocketfft takes rows through its SIMD or its scalar path depending on how the
batch is split between threads, and the two round differently in the last
place (about 1e-16 relative). Set ``METAL_SCI_FFT_THREADED_FALLBACK=0`` to
leave declined calls exactly as the caller made them.
"""
from __future__ import annotations

import contextvars
import ctypes
import ctypes.util
import functools
import operator
import os

import numpy as np
import scipy.fft

from .. import _config, _native
from .._dispatch import NotAccelerated, Route, accelerated, counters, record
from . import _native_fft
from ._native_fft import C2R, FORWARD, INVERSE, R2C

__all__ = ["fft", "ifft", "rfft", "irfft", "fft2", "ifft2", "fftn", "ifftn",
           "rfft2", "irfft2", "rfftn", "irfftn"]

# SciPy's own threading starts paying at about this many points, over a batch
# or within one transform of a 1-D array.
_THREADED_MIN_POINTS = 1 << 17
_THREADED_FALLBACK = os.environ.get("METAL_SCI_FFT_THREADED_FALLBACK", "1").strip().lower() not in ("0", "false", "no", "off")
_CPU_COUNT = os.cpu_count() or 1


class Declined(NotAccelerated):
    """This call goes to SciPy; ``threaded`` says it is worth SciPy's worker threads."""

    def __init__(self, reason: str, threaded: bool = False):
        super().__init__(reason)
        self.threaded = threaded


@functools.cache
def performance_cores() -> int:
    """Physical performance cores: the worker count handed to SciPy for batches."""
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        value, size = ctypes.c_int(0), ctypes.c_size_t(ctypes.sizeof(ctypes.c_int))
        if libc.sysctlbyname(b"hw.perflevel0.physicalcpu", ctypes.byref(value), ctypes.byref(size), None, 0) == 0 \
                and value.value > 0:
            return value.value
    except (OSError, AttributeError):
        pass
    return _CPU_COUNT


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


def _input(x, norm, workers, plan):
    """The array to transform, the normalisation code and SciPy's worker count."""
    if plan is not None:
        raise Declined("precomputed plans are SciPy's to reject")
    if not isinstance(x, np.ndarray):
        raise Declined("not a NumPy array")
    if x.ndim == 0 or x.ndim > _native.MS_MAX_NDIM:
        raise Declined(f"no native path for {x.ndim}-dimensional arrays")
    device = _config.get_device()
    if device == "cpu":
        raise Declined("device policy 'cpu': there is no native CPU FFT", threaded=True)
    if device == "auto" and x.size < _native_fft.auto_min_points():
        raise Declined("below the GPU crossover", threaded=True)
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


def _transform_1d(function, kind, x, n, axis, norm, workers, plan):
    x, norm, threads = _input(x, norm, workers, plan)
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


def _transform_nd(function, kind, x, s, axes, norm, workers, plan):
    x, norm, threads = _input(x, norm, workers, plan)
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
_delegating = contextvars.ContextVar("metal_sci_fft_delegating", default=False)


def run_threaded(reference, x, arguments, threads):
    """SciPy's own transform, threaded over the batch."""
    token = _delegating.set(True)
    try:
        return reference(x, *arguments, threads)
    finally:
        _delegating.reset(token)


def _decline(name, reference, why, x, arguments, workers, plan):
    """What a declined call does next.

    Normally the exception goes on to the ``@accelerated`` wrapper, which calls
    SciPy with the caller's arguments. A call SciPy can thread is made here
    instead, with workers. ``arguments`` are the positional arguments between
    ``x`` and ``workers``: (n, axis, norm, overwrite_x) or (s, axes, norm,
    overwrite_x).
    """
    threads = threaded_fallback(x, arguments[1], why, workers, plan)
    if threads is None:
        raise why
    qualified = f"fft.{name}"
    result = run_threaded(reference, x, arguments, threads)
    record(Route(qualified, "scipy", "as given", _config.get_precision(), f"{why}; SciPy with workers={threads}"))
    # The @accelerated wrapper counts a returned result as a native call.
    tally = counters[qualified]
    tally[0] -= 1
    tally[1] += 1
    return result


def threaded_fallback(x, axes, why, workers, plan):
    """Worker count for a declined call, or None to leave the call as the caller made it.

    SciPy's threads split the transforms of a batch between them, and work
    inside one long transform of a 1-D array. They do not help (and cost time)
    when an axis of a multi-dimensional array holds a single transform.
    """
    if not (_THREADED_FALLBACK and why.threaded) or workers is not None or plan is not None \
            or scipy.fft.get_workers() != 1:
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
    threads = performance_cores()
    return threads if threads > 1 else None


def _public(wrapper, name):
    """The wrapper as ``metal_sci.fft.<name>``.

    It carries SciPy's name and documentation; left with SciPy's module too it
    would not pickle (pickle would find SciPy's own function under that name),
    and so could not be handed to ``multiprocessing``.
    """
    wrapper.__module__, wrapper.__qualname__ = "metal_sci.fft", name
    return wrapper


def _one_dimensional(name, kind):
    function = f"fft.{name}"
    reference = getattr(scipy.fft, name)

    def core(x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, *, plan=None):
        return _transform_1d(function, kind, x, n, axis, norm, workers, plan)

    def implementation(x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, *, plan=None):
        try:
            return _transform_1d(function, kind, x, n, axis, norm, workers, plan)
        except Declined as why:
            return _decline(name, reference, why, x, (n, axis, norm, overwrite_x), workers, plan)

    implementation.__name__ = implementation.__qualname__ = name
    # The native attempt alone, for the uarray backend: (result) or Declined.
    implementation.core = core
    implementation.reference = reference
    return _public(accelerated(function)(implementation), name)


def _n_dimensional(name, kind, default_axes):
    function = f"fft.{name}"
    reference = getattr(scipy.fft, name)

    def core(x, s=None, axes=default_axes, norm=None, overwrite_x=False, workers=None, *, plan=None):
        return _transform_nd(function, kind, x, s, axes, norm, workers, plan)

    def implementation(x, s=None, axes=default_axes, norm=None, overwrite_x=False, workers=None, *, plan=None):
        try:
            return _transform_nd(function, kind, x, s, axes, norm, workers, plan)
        except Declined as why:
            return _decline(name, reference, why, x, (s, axes, norm, overwrite_x), workers, plan)

    implementation.__name__ = implementation.__qualname__ = name
    implementation.core = core
    implementation.reference = reference
    return _public(accelerated(function)(implementation), name)


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
