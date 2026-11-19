"""Native routes for scipy.signal convolution and correlation.

Implements ``convolve``, ``correlate``, ``fftconvolve``, ``oaconvolve``,
``convolve2d``, ``correlate2d``, and ``choose_conv_method`` for plain native
``numpy.ndarray`` inputs of dtypes ``float32``, ``float64``, ``complex64``,
and ``complex128`` (ranks 1 to 8). Integer, boolean, ``float16``,
``longdouble``, empty, 0-D, non-NumPy, subclass, or invalid inputs decline to
SciPy.

Under ``strict`` precision, double-precision FFT convolution (and
``fftconvolve``/``oaconvolve``) declines to SciPy wrapped in
``scipy.fft.set_workers(performance_cores())`` for large calls (``>= 2^17``
elements when ``scipy.fft.get_workers() == 1`` and
``METAL_SCI_FFT_THREADED_FALLBACK != 0``). Note that multi-threaded pocketfft
can differ from single-worker SciPy in the last ulp. Under ``fast`` /
``refined`` precision, GPU FFT convolution runs in single precision (error
typically ``O(u * max|output|)`` with ``u = 2^-24``) and emits
``metal_sci.PrecisionWarning`` once by default on double-precision inputs.
"""
from __future__ import annotations

import operator
import os
import warnings

import numpy as np
import scipy.fft
import scipy.signal._signaltools as _sp_signaltools

from .. import _config, _native
from .._dispatch import NotAccelerated, Route, accelerated, record
from .._config import get_precision
from . import _native_signal as _sig

__all__ = [
    "choose_conv_method",
    "convolve",
    "convolve2d",
    "correlate",
    "correlate2d",
    "fftconvolve",
    "oaconvolve",
]

# Native float/complex type characters: excludes float16 ('e'), longdouble ('g'),
# clongdouble ('G'), integers, booleans, and objects.
_NATIVE_CHARS = ("f", "d", "F", "D")

_DIRECT_BREAK_EVEN_1D = 1 << 16
_DIRECT_BREAK_EVEN_ND = 1 << 12
_FFT_BREAK_EVEN_ELEMENTS = 1 << 15

_THREADED_MIN_POINTS = 1 << 17


def _decline(reason: str, *, context=None):
    raise NotAccelerated(reason, context=context)


def _threaded_fft_context(in1, in2):
    enabled = os.environ.get("METAL_SCI_FFT_THREADED_FALLBACK", "1").strip().lower() not in (
        "0", "false", "no", "off"
    )
    if not enabled or scipy.fft.get_workers() != 1:
        return None
    if isinstance(in1, np.ndarray) and isinstance(in2, np.ndarray):
        if max(in1.size, in2.size) >= _THREADED_MIN_POINTS:
            cores = _native.performance_cores()
            if cores > 1:
                return scipy.fft.set_workers(cores)
    return None


def _check_arrays(in1, in2):
    if type(in1) is not np.ndarray or type(in2) is not np.ndarray:
        _decline("inputs are not plain NumPy arrays")
    if in1.dtype.char not in _NATIVE_CHARS or not in1.dtype.isnative:
        _decline(f"no native signal path for dtype {in1.dtype}")
    if in2.dtype.char not in _NATIVE_CHARS or not in2.dtype.isnative:
        _decline(f"no native signal path for dtype {in2.dtype}")
    if not (in1.flags.aligned and in2.flags.aligned):
        _decline("input data is not aligned")
    if in1.ndim == 0 or in2.ndim == 0 or in1.ndim != in2.ndim:
        _decline("0-d or mismatched-rank inputs go to SciPy")
    if in1.ndim > _native.MS_MAX_NDIM:
        _decline(f"more than {_native.MS_MAX_NDIM} dimensions go to SciPy")
    if in1.size == 0 or in2.size == 0:
        _decline("empty inputs go to SciPy")
    return in1, in2


def _parse_mode(mode) -> int:
    if type(mode) is not str or mode not in _sig.MODES:
        _decline(f"unsupported mode {mode!r}")
    return _sig.MODES[mode]


def _parse_method(method) -> int:
    if type(method) is not str or method not in _sig.METHODS:
        _decline(f"unsupported method {method!r}")
    return _sig.METHODS[method]


def _valid_swap_needed(mode: str, s1: tuple[int, ...], s2: tuple[int, ...], axes) -> bool:
    if mode != "valid":
        return False
    ok1 = all(s1[a] >= s2[a] for a in axes)
    ok2 = all(s2[a] >= s1[a] for a in axes)
    if not (ok1 or ok2):
        _decline("invalid shapes for valid mode")
    return not ok1


def _parse_axes(axes, ndim: int) -> list[int] | None:
    if axes is None:
        return None
    if isinstance(axes, (int, np.integer)):
        vals = [operator.index(axes)]
    else:
        try:
            vals = [operator.index(a) for a in axes]
        except TypeError:
            _decline("axes is not an integer or sequence of integers")
    if not vals:
        _decline("axes cannot be empty")
    norm = [a + ndim if a < 0 else a for a in vals]
    if any(a < 0 or a >= ndim for a in norm) or len(set(norm)) != len(norm):
        _decline("axes out of range or repeated")
    return norm


def _direct_work_estimate(s1: tuple[int, ...], s2: tuple[int, ...], mode: str) -> int:
    if mode == "full":
        out_elems = 1
        for a, b in zip(s1, s2):
            out_elems *= (a + b - 1)
        return out_elems * min(int(np.prod(s1)), int(np.prod(s2)))
    if mode == "same":
        return int(np.prod(s1)) * int(np.prod(s2))
    out_elems = 1
    for a, b in zip(s1, s2):
        out_elems *= abs(a - b) + 1
    return out_elems * min(int(np.prod(s1)), int(np.prod(s2)))


def _warn_fft_nonfinite_flag(has_nonfinite: bool) -> None:
    if has_nonfinite:
        warnings.warn(
            "Use of fft convolution on input with NAN or inf results in NAN or inf output. "
            "Consider using method='direct' instead.",
            category=RuntimeWarning,
            stacklevel=3,
        )


def _warn_fft_nonfinite(out: np.ndarray) -> None:
    _warn_fft_nonfinite_flag(out.size > 0 and not np.isfinite(out).all())


def _convolve_or_correlate(function: str, in1, in2, mode: str, method: str, is_correlate: bool) -> np.ndarray:
    in1, in2 = _check_arrays(in1, in2)
    mode_code = _parse_mode(mode)
    method_code = _parse_method(method)
    _valid_swap_needed(mode, in1.shape, in2.shape, range(in1.ndim))

    threads = _native.performance_cores() if scipy.fft.get_workers() == 1 else scipy.fft.get_workers()
    if method_code == _sig.METHOD_FFT:
        if _config.get_device() == "auto":
            full_est = 1
            for a, b in zip(in1.shape, in2.shape):
                full_est *= a + b - 1
            try:
                _native.decline_small(full_est, _FFT_BREAK_EVEN_ELEMENTS)
            except NotAccelerated as why:
                _decline(why.reason, context=_threaded_fft_context(in1, in2))
        try:
            out, route_flags = _sig.convolve(
                function, in1, in2, mode_code, method_code, is_correlate,
                _sig.BOUNDARY_FILL, _sig.CENTER_ND, 0.0, 0.0, threads,
                return_route_flags=True,
            )
        except NotAccelerated as why:
            _decline(why.reason, context=_threaded_fft_context(in1, in2))
        if route_flags & _sig.SIGNAL_ROUTE_FFT:
            _warn_fft_nonfinite_flag(bool(route_flags & _sig.SIGNAL_ROUTE_NONFINITE))
        else:
            _warn_fft_nonfinite(out)
        return out

    work_est = _direct_work_estimate(in1.shape, in2.shape, mode)
    break_even = _DIRECT_BREAK_EVEN_1D if in1.ndim == 1 else _DIRECT_BREAK_EVEN_ND
    try:
        _native.decline_small(work_est, break_even)
    except NotAccelerated as why:
        ctx = _threaded_fft_context(in1, in2) if method_code == _sig.METHOD_AUTO else None
        _decline(why.reason, context=ctx)

    try:
        out, route_flags = _sig.convolve(
            function, in1, in2, mode_code, method_code, is_correlate,
            _sig.BOUNDARY_FILL, _sig.CENTER_ND, 0.0, 0.0, threads,
            return_route_flags=True,
        )
    except NotAccelerated as why:
        ctx = _threaded_fft_context(in1, in2) if method_code == _sig.METHOD_AUTO else None
        _decline(why.reason, context=ctx)
    if method_code == _sig.METHOD_AUTO and (route_flags & _sig.SIGNAL_ROUTE_FFT):
        _warn_fft_nonfinite_flag(bool(route_flags & _sig.SIGNAL_ROUTE_NONFINITE))
    return out


@accelerated("signal.convolve")
def convolve(in1, in2, mode="full", method="auto"):
    return _convolve_or_correlate("signal.convolve", in1, in2, mode, method, False)


@accelerated("signal.correlate")
def correlate(in1, in2, mode="full", method="auto"):
    return _convolve_or_correlate("signal.correlate", in1, in2, mode, method, True)


def _freq_convolve(function: str, in1, in2, mode, axes, overlap_add: bool) -> np.ndarray:
    in1, in2 = _check_arrays(in1, in2)
    mode_code = _parse_mode(mode)
    parsed_axes = _parse_axes(axes, in1.ndim)
    raw_axes = list(range(in1.ndim)) if parsed_axes is None else parsed_axes
    eff_axes = [a for a in raw_axes if in1.shape[a] != 1 and in2.shape[a] != 1]
    if not eff_axes:
        _decline("no axes with non-unit length in both inputs")
    for a in range(in1.ndim):
        if a not in eff_axes:
            s1a, s2a = in1.shape[a], in2.shape[a]
            if not (s1a == s2a or s1a == 1 or s2a == 1):
                _decline("incompatible non-transformed axis shapes")
    _valid_swap_needed(mode, in1.shape, in2.shape, eff_axes)

    promoted = np.result_type(in1.dtype, in2.dtype)
    double_out = promoted.itemsize // (2 if promoted.kind == "c" else 1) == 8
    device = _config.get_device()
    if device == "cpu":
        _decline("device policy 'cpu': there is no native CPU FFT convolution",
                 context=_threaded_fft_context(in1, in2))
    if double_out and _config.get_precision() == "strict":
        _decline("strict precision keeps double-precision FFT convolution on SciPy",
                 context=_threaded_fft_context(in1, in2))

    if device == "auto":
        full_est = 1
        for d in range(in1.ndim):
            if d in eff_axes:
                full_est *= in1.shape[d] + in2.shape[d] - 1
            else:
                full_est *= max(in1.shape[d], in2.shape[d])
        try:
            _native.decline_small(full_est, _FFT_BREAK_EVEN_ELEMENTS)
        except NotAccelerated as why:
            _decline(why.reason, context=_threaded_fft_context(in1, in2))

    threads = _native.performance_cores() if scipy.fft.get_workers() == 1 else scipy.fft.get_workers()
    try:
        return _sig.fftconvolve(
            function, in1, in2, mode_code, False, overlap_add, parsed_axes, threads
        )
    except NotAccelerated as why:
        _decline(why.reason, context=_threaded_fft_context(in1, in2))


@accelerated("signal.fftconvolve")
def fftconvolve(in1, in2, mode="full", axes=None):
    return _freq_convolve("signal.fftconvolve", in1, in2, mode, axes, False)


@accelerated("signal.oaconvolve")
def oaconvolve(in1, in2, mode="full", axes=None):
    return _freq_convolve("signal.oaconvolve", in1, in2, mode, axes, True)


def _convolve2d_or_correlate2d(function: str, in1, in2, mode, boundary, fillvalue, is_correlate: bool) -> np.ndarray:
    in1, in2 = _check_arrays(in1, in2)
    if in1.ndim != 2 or in2.ndim != 2:
        _decline("convolve2d/correlate2d inputs must be 2-D")
    mode_code = _parse_mode(mode)
    if type(boundary) is not str or boundary not in _sig.BOUNDARIES:
        _decline(f"unsupported boundary {boundary!r}")
    b_code = _sig.BOUNDARIES[boundary]
    swapped = _valid_swap_needed(mode, in1.shape, in2.shape, (0, 1))

    out_dtype = np.result_type(in1.dtype, in2.dtype)
    if is_correlate and mode == "valid" and swapped and out_dtype.kind == "c":
        # SciPy 1.18.1 correlate2d reverses without conjugating when swapped in 'valid' mode.
        _decline("complex swapped valid correlate2d uses SciPy parity")

    fill_real = 0.0
    fill_imag = 0.0
    if b_code == _sig.BOUNDARY_FILL:
        farr = np.asarray(fillvalue)
        if farr.ndim != 0 or farr.size != 1:
            _decline("non-scalar fillvalue goes to SciPy")
        if farr.dtype.kind not in ("b", "i", "u", "f", "c"):
            _decline("non-numeric fillvalue goes to SciPy")
        if out_dtype.kind == "f" and farr.dtype.kind == "c":
            _decline("complex fillvalue with real output goes to SciPy")
        c = complex(farr.item())
        fill_real, fill_imag = float(c.real), float(c.imag)

    work_est = _direct_work_estimate(in1.shape, in2.shape, mode)
    _native.decline_small(work_est, _DIRECT_BREAK_EVEN_ND)

    # convolve2d and correlate2d never swap in 'full' mode; convolve.cpp only
    # shifts 'same' centering to (k - 1) // 2 when options->correlate != 0.
    center = _sig.CENTER_2D_CORR
    return _sig.convolve(
        function, in1, in2, mode_code, _sig.METHOD_DIRECT, is_correlate,
        b_code, center, fill_real, fill_imag, 1,
    )


@accelerated("signal.convolve2d")
def convolve2d(in1, in2, mode="full", boundary="fill", fillvalue=0):
    return _convolve2d_or_correlate2d("signal.convolve2d", in1, in2, mode, boundary, fillvalue, False)


@accelerated("signal.correlate2d")
def correlate2d(in1, in2, mode="full", boundary="fill", fillvalue=0):
    return _convolve2d_or_correlate2d("signal.correlate2d", in1, in2, mode, boundary, fillvalue, True)


@accelerated("signal.choose_conv_method")
def choose_conv_method(in1, in2, mode="full", measure=False):
    in1, in2 = _check_arrays(in1, in2)
    mode_code = _parse_mode(mode)
    _valid_swap_needed(mode, in1.shape, in2.shape, range(in1.ndim))

    if measure:
        times = {}
        for m in ("fft", "direct"):
            times[m] = _sp_signaltools._timeit_fast(lambda m=m: convolve(in1, in2, mode=mode, method=m))
        chosen = "fft" if times["fft"] < times["direct"] else "direct"
        record(Route("signal.choose_conv_method", "cpu", "measured", get_precision(), f"measured: {chosen}"))
        return chosen, times

    threads = _native.performance_cores() if scipy.fft.get_workers() == 1 else scipy.fft.get_workers()
    method_code = _sig.choose_conv_method(in1.dtype, in1.shape, in2.dtype, in2.shape, mode_code, threads)
    chosen = "fft" if method_code == _sig.METHOD_FFT else "direct"
    promoted = np.result_type(in1.dtype, in2.dtype)
    record(Route("signal.choose_conv_method", "cpu", promoted.name, get_precision(), f"predicted: {chosen}"))
    return chosen
