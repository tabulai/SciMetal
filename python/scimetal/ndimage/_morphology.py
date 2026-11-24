"""Native routes for scipy.ndimage's min/max, rank/median, and grayscale morphology filters.

Covers 15 functions:
- ``minimum_filter1d``, ``maximum_filter1d``, ``minimum_filter``, ``maximum_filter``
- ``rank_filter``, ``median_filter``, ``percentile_filter``
- ``grey_erosion``, ``grey_dilation``, ``grey_opening``, ``grey_closing``,
  ``morphological_gradient``, ``morphological_laplace``, ``white_tophat``,
  ``black_tophat``

Separable rectangular operations (including multi-stage ``grey_opening``,
``grey_closing``, ``morphological_gradient``, ``morphological_laplace``,
``white_tophat``, and ``black_tophat`` when output dtype matches input dtype)
fuse into a single Metal command buffer on the GPU in ``fast`` / ``refined``
modes and run via threaded van Herk / Gil-Werman on the P-cores in ``strict``
mode (bit-identical to SciPy 1.18.1 arm64).

Dense non-separable footprints, non-flat structuring elements, and N-D
rank/median/percentile filters run via Metal compute kernels (with dedicated
tiled register-only sorting/Forgetful-Selection networks for 2-D 3x3, 5x5,
and 7x7 medians and general Forgetful Selection up to buffer size 64) in
``fast`` / ``refined`` modes and via threaded P-core CPU engines in ``strict``
mode.
"""
from __future__ import annotations

import numbers
import warnings

import numpy as np

from .._config import get_precision
from .._dispatch import NotAccelerated, Route, accelerated, last_route, record
from . import _native_ndimage as _nd
from ._filters import (
    _axis,
    _axes,
    _cval,
    _decline,
    _index,
    _input,
    _mode,
    _output,
    _run,
    _sequence,
)

__all__ = [
    "black_tophat",
    "grey_closing",
    "grey_dilation",
    "grey_erosion",
    "grey_opening",
    "maximum_filter",
    "maximum_filter1d",
    "median_filter",
    "minimum_filter",
    "minimum_filter1d",
    "morphological_gradient",
    "morphological_laplace",
    "percentile_filter",
    "rank_filter",
    "white_tophat",
]


def _minmax_pass(ndim: int, kind: int, size, axis, mode, cval, origin) -> tuple:
    axis = _axis(axis, ndim)
    size = _index(size, "size")
    origin = _index(origin, "origin")
    if size < 1:
        _decline("incorrect filter size")
    if size // 2 + origin < 0 or size // 2 + origin >= size:
        _decline("invalid origin")
    if size > _nd.SM_NDIMAGE_MAX_WINDOW:
        _decline("a longer window than the native filter takes")
    return (kind, axis, _mode(mode), origin, size, None, _cval(cval))


def _expand_footprint(ndim_image: int, axes: tuple, footprint: np.ndarray) -> np.ndarray:
    num_axes = len(axes)
    if num_axes < ndim_image:
        if footprint.ndim != num_axes:
            _decline("footprint.ndim must match len(axes)")
        missing = tuple(ax for ax in range(ndim_image) if ax not in axes)
        footprint = np.expand_dims(footprint, missing)
    return footprint


def _expand_origin(ndim_image: int, axes: tuple, origin) -> list[int]:
    num_axes = len(axes)
    origins = [_index(o, "origin") for o in _sequence(origin, num_axes)]
    if num_axes < ndim_image:
        out = [0] * ndim_image
        for o, ax in zip(origins, axes):
            out[ax] = o
        return out
    return origins


def _prepare_minmax_spec(ndim: int, size, footprint, structure, mode, cval, origin, axes, *,
                         is_max: bool, is_dilation: bool = False):
    """Validate and classify a min/max or erosion/dilation call.

    Returns ``(separable, spec, warn_size_ignored)`` where:
    - if ``separable`` is True: ``spec`` is a list of 1-D passes;
    - if ``separable`` is False: ``spec`` is ``(fshape, origins, footprint_u8, ss_f64_or_None, mode_code, cval_f)``.
    """
    warn_ignored = (size is not None) and (footprint is not None)
    axes_t = _axes(axes, ndim, allow_negative_scalar=False)
    num_axes = len(axes_t)

    fp_arr = None
    st_arr = None
    if structure is not None:
        try:
            st_raw = np.asarray(structure)
            if st_raw.dtype.kind == "c":
                _decline("complex structure")
            st_arr = np.asarray(st_raw, dtype=np.float64)
        except NotAccelerated:
            raise
        except Exception:
            _decline("structure is not convertible to float64")
        if is_dilation:
            st_arr = st_arr[(slice(None, None, -1),) * st_arr.ndim]
    if footprint is not None:
        try:
            fp_arr = np.asarray(footprint, dtype=bool)
        except Exception:
            _decline("footprint is not convertible to bool")
        if is_dilation:
            fp_arr = fp_arr[(slice(None, None, -1),) * fp_arr.ndim]

    if is_dilation:
        orig_seq = [_index(o, "origin") for o in _sequence(origin, num_axes)]
        flipped = []
        for ii in range(num_axes):
            if fp_arr is not None:
                if ii >= fp_arr.ndim:
                    _decline("footprint rank mismatch")
                sz = fp_arr.shape[ii]
            elif st_arr is not None:
                if ii >= st_arr.ndim:
                    _decline("structure rank mismatch")
                sz = st_arr.shape[ii]
            else:
                sizes_tmp = _sequence(size, num_axes)
                sz = _index(sizes_tmp[ii], "size")
            o = -orig_seq[ii]
            if not (sz & 1):
                o -= 1
            flipped.append(o)
        origin = flipped

    if st_arr is None:
        if fp_arr is None:
            if size is None:
                _decline("no footprint or size provided")
            separable = True
        else:
            if fp_arr.size == 0 or not fp_arr.any():
                _decline("all-zero or empty footprint")
            if fp_arr.all():
                size = fp_arr.shape
                fp_arr = None
                separable = True
            else:
                separable = False
    else:
        separable = False
        if fp_arr is None:
            fp_arr = np.ones(st_arr.shape, dtype=bool)
        if fp_arr.size == 0 or not fp_arr.any():
            _decline("all-zero or empty footprint")

    cval_f = _cval(cval)

    if separable:
        sizes = _sequence(size, num_axes)
        origins = _sequence(origin, num_axes)
        modes = _sequence(mode, num_axes)
        kind = _nd.MAXIMUM if is_max else _nd.MINIMUM
        active = []
        for i in range(num_axes):
            try:
                sz = int(sizes[i])
            except Exception:
                _decline("size is not a real number")
            if sizes[i] != sz and not isinstance(sizes[i], (int, np.integer, float, np.floating)):
                _decline("size is not a real number")
            if sizes[i] < 1:
                _decline("non-positive filter size goes to SciPy")
            if sizes[i] > 1:
                active.append((i, sz))
        passes = [_minmax_pass(ndim, kind, sz, axes_t[i], modes[i], cval_f, origins[i]) for i, sz in active]
        return True, passes, warn_ignored

    # Non-separable dense footprint or non-flat structure.
    mode_code = _mode(mode)
    orig_fp_shape = fp_arr.shape
    fp_exp = _expand_footprint(ndim, axes_t, fp_arr)
    fshape = [s for s in fp_exp.shape if s > 0]
    if len(fshape) != ndim:
        _decline("footprint.ndim must match len(axes)")
    origins_exp = _expand_origin(ndim, axes_t, origin)
    for o, lenf in zip(origins_exp, fshape):
        if lenf // 2 + o < 0 or lenf // 2 + o >= lenf:
            _decline("invalid origin")
        if lenf > _nd.SM_NDIMAGE_MAX_WINDOW:
            _decline("footprint exceeds maximum window")
    ss = None
    if st_arr is not None:
        if st_arr.ndim != num_axes or st_arr.shape != orig_fp_shape:
            _decline("structure shape does not match footprint")
        ss = st_arr if is_max else -st_arr
    return False, (tuple(fshape), origins_exp, fp_exp, ss, mode_code, cval_f), warn_ignored


def _run_minmax_spec(function: str, array: np.ndarray, separable: bool, spec, is_max: bool,
                     target, out_dtype, warn_ignored: bool) -> np.ndarray:
    if warn_ignored:
        warnings.warn("ignoring size because footprint is set", UserWarning, stacklevel=3)
    if separable:
        return _run(function, array, spec, target, out_dtype)
    fshape, origins, fp_exp, ss, mode_code, cval_f = spec
    result = _nd.minmax_filter(function, array, is_max, fshape, origins, fp_exp, ss,
                               mode_code, cval_f, out_dtype=out_dtype)
    if target is None:
        return result
    target[...] = result
    return target


@accelerated("ndimage.minimum_filter1d")
def minimum_filter1d(input, size, axis=-1, output=None, mode="reflect", cval=0.0, origin=0):
    array = _input(input)
    target, out_dtype = _output(output, array)
    passes = [_minmax_pass(array.ndim, _nd.MINIMUM, size, axis, mode, cval, origin)]
    return _run("ndimage.minimum_filter1d", array, passes, target, out_dtype)


@accelerated("ndimage.maximum_filter1d")
def maximum_filter1d(input, size, axis=-1, output=None, mode="reflect", cval=0.0, origin=0):
    array = _input(input)
    target, out_dtype = _output(output, array)
    passes = [_minmax_pass(array.ndim, _nd.MAXIMUM, size, axis, mode, cval, origin)]
    return _run("ndimage.maximum_filter1d", array, passes, target, out_dtype)


@accelerated("ndimage.minimum_filter")
def minimum_filter(input, size=None, footprint=None, output=None, mode="reflect", cval=0.0, origin=0, *, axes=None):
    array = _input(input)
    target, out_dtype = _output(output, array)
    separable, spec, warn_ignored = _prepare_minmax_spec(
        array.ndim, size, footprint, None, mode, cval, origin, axes, is_max=False)
    return _run_minmax_spec("ndimage.minimum_filter", array, separable, spec, False, target, out_dtype, warn_ignored)


@accelerated("ndimage.maximum_filter")
def maximum_filter(input, size=None, footprint=None, output=None, mode="reflect", cval=0.0, origin=0, *, axes=None):
    array = _input(input)
    target, out_dtype = _output(output, array)
    separable, spec, warn_ignored = _prepare_minmax_spec(
        array.ndim, size, footprint, None, mode, cval, origin, axes, is_max=True)
    return _run_minmax_spec("ndimage.maximum_filter", array, separable, spec, True, target, out_dtype, warn_ignored)


def _rank_common(function: str, input, rank_arg, size, footprint, output, mode, cval, origin, operation: str, axes):
    warn_ignored = (size is not None) and (footprint is not None)
    array = _input(input)
    target, out_dtype = _output(output, array)
    axes_t = _axes(axes, array.ndim, allow_negative_scalar=False)
    num_axes = len(axes_t)

    if footprint is None:
        if size is None:
            _decline("no footprint or filter size provided")
        sizes = [_index(s, "size") for s in _sequence(size, num_axes)]
        if any(s < 1 for s in sizes):
            _decline("non-positive filter size")
        all_true = True
        fp_arr = None
        fp_shape_axes = tuple(sizes)
        # Expand rectangular shape across axes.
        fshape_list = [1] * array.ndim
        for s, ax in zip(sizes, axes_t):
            fshape_list[ax] = s
        fshape = tuple(fshape_list)
        filter_size = 1
        for s in fshape:
            filter_size *= s
    else:
        try:
            fp_raw = np.asarray(footprint, dtype=bool)
        except Exception:
            _decline("footprint is not convertible to bool")
        fp_exp = _expand_footprint(array.ndim, axes_t, fp_raw)
        fshape = tuple(s for s in fp_exp.shape if s > 0)
        if len(fshape) != array.ndim:
            _decline("footprint.ndim must match len(axes)")
        filter_size = int(np.count_nonzero(fp_exp))
        if filter_size == 0:
            _decline("all-zero footprint")
        all_true = bool(fp_exp.all())
        fp_arr = None if all_true else fp_exp
        fp_shape_axes = fp_raw.shape

    origins = _expand_origin(array.ndim, axes_t, origin)
    for o, lenf in zip(origins, fshape):
        if lenf // 2 + o < 0 or lenf // 2 + o >= lenf:
            _decline("invalid origin")
        if lenf > _nd.SM_NDIMAGE_MAX_WINDOW:
            _decline("footprint exceeds maximum window")

    if operation == "median":
        rank = filter_size // 2
    elif operation == "percentile":
        if not isinstance(rank_arg, numbers.Real):
            _decline("percentile is not a real number")
        percentile = float(rank_arg)
        if percentile < 0.0:
            percentile += 100.0
        if percentile < 0.0 or percentile > 100.0 or np.isnan(percentile):
            _decline("invalid percentile")
        if percentile == 100.0:
            rank = filter_size - 1
        else:
            rank = int(float(filter_size) * percentile / 100.0)
    else:
        rank = _index(rank_arg, "rank")

    if rank < 0:
        rank += filter_size
    if rank < 0 or rank >= filter_size:
        _decline("rank not within filter footprint size")

    # Extreme ranks reduce directly to minimum_filter / maximum_filter (which
    # also allows a sequence of modes when the footprint is separable).
    if rank == 0 or rank == filter_size - 1:
        is_max = (rank == filter_size - 1)
        fp_for_mm = np.ones(fp_shape_axes, dtype=bool) if all_true else np.asarray(footprint, dtype=bool)
        separable, spec, _ = _prepare_minmax_spec(
            array.ndim, None, fp_for_mm, None, mode, cval, origin, axes, is_max=is_max)
        return _run_minmax_spec(function, array, separable, spec, is_max, target, out_dtype, warn_ignored)

    # Non-extreme rank: SciPy rejects sequences of modes.
    mode_code = _mode(mode)
    cval_f = _cval(cval)

    # Match SciPy 1.18.1 1-D _rank_filter_1d corner cases cleanly by declining
    # where SciPy's C++ 1-D path diverges from NI_RankFilter (non-all-true
    # 1-D footprints and uint64 cast to int64).
    if array.ndim == 1:
        f_total = fshape[0]
        lim2 = array.size - ((f_total - 1) // 2 - origins[0])
        if lim2 >= 0 or array.size == 1:
            if not all_true:
                _decline("1-D non-rectangular rank footprint goes to SciPy")
            if array.dtype == np.uint64:
                _decline("1-D uint64 rank filter goes to SciPy")
            if mode_code == _nd.CONSTANT and array.dtype.kind in "iub":
                if not (-9223372036854775808.0 <= cval_f < 9223372036854775808.0):
                    _decline("1-D integer rank cval out of int64 range")

    if warn_ignored:
        warnings.warn("ignoring size because footprint is set", UserWarning, stacklevel=3)
    result = _nd.rank_filter(function, array, rank, fshape, origins, fp_arr, mode_code, cval_f, out_dtype=out_dtype)
    if target is None:
        return result
    target[...] = result
    return target


@accelerated("ndimage.rank_filter")
def rank_filter(input, rank, size=None, footprint=None, output=None, mode="reflect", cval=0.0, origin=0, *, axes=None):
    return _rank_common("ndimage.rank_filter", input, rank, size, footprint, output, mode, cval, origin, "rank", axes)


@accelerated("ndimage.median_filter")
def median_filter(input, size=None, footprint=None, output=None, mode="reflect", cval=0.0, origin=0, *, axes=None):
    return _rank_common("ndimage.median_filter", input, 0, size, footprint, output, mode, cval, origin, "median", axes)


@accelerated("ndimage.percentile_filter")
def percentile_filter(input, percentile, size=None, footprint=None, output=None, mode="reflect", cval=0.0, origin=0,
                      *, axes=None):
    return _rank_common("ndimage.percentile_filter", input, percentile, size, footprint, output, mode, cval, origin,
                        "percentile", axes)


def _check_grey_args(size, footprint, structure):
    if size is None and footprint is None and structure is None:
        _decline("size, footprint, or structure must be specified")


def _retag_route(function: str):
    r = last_route()
    if r is not None:
        record(Route(function, r.engine, r.compute_dtype, r.policy, r.reason))


@accelerated("ndimage.grey_erosion")
def grey_erosion(input, size=None, footprint=None, structure=None, output=None, mode="reflect", cval=0.0, origin=0,
                 *, axes=None):
    _check_grey_args(size, footprint, structure)
    array = _input(input)
    target, out_dtype = _output(output, array)
    separable, spec, warn_ignored = _prepare_minmax_spec(
        array.ndim, size, footprint, structure, mode, cval, origin, axes, is_max=False, is_dilation=False)
    return _run_minmax_spec("ndimage.grey_erosion", array, separable, spec, False, target, out_dtype, warn_ignored)


@accelerated("ndimage.grey_dilation")
def grey_dilation(input, size=None, footprint=None, structure=None, output=None, mode="reflect", cval=0.0, origin=0,
                  *, axes=None):
    _check_grey_args(size, footprint, structure)
    array = _input(input)
    target, out_dtype = _output(output, array)
    separable, spec, warn_ignored = _prepare_minmax_spec(
        array.ndim, size, footprint, structure, mode, cval, origin, axes, is_max=True, is_dilation=True)
    return _run_minmax_spec("ndimage.grey_dilation", array, separable, spec, True, target, out_dtype, warn_ignored)


def _grey_open_or_close(function: str, input, size, footprint, structure, output, mode, cval, origin, axes,
                        *, opening: bool):
    _check_grey_args(size, footprint, structure)
    array = _input(input)
    target, out_dtype = _output(output, array)
    sep_er, spec_er, warn_er = _prepare_minmax_spec(
        array.ndim, size, footprint, structure, mode, cval, origin, axes, is_max=False, is_dilation=False)
    sep_di, spec_di, _ = _prepare_minmax_spec(
        array.ndim, size, footprint, structure, mode, cval, origin, axes, is_max=True, is_dilation=True)

    # Fuse separable opening/closing into a single command buffer when out_dtype == array.dtype.
    if sep_er and sep_di and out_dtype == array.dtype:
        fused = (spec_er + spec_di) if opening else (spec_di + spec_er)
        if len(fused) <= _nd.SM_NDIMAGE_MAX_PASSES:
            if warn_er:
                warnings.warn("ignoring size because footprint is set", UserWarning, stacklevel=3)
            return _run(function, array, fused, target, out_dtype)

    if warn_er:
        warnings.warn("ignoring size because footprint is set", UserWarning, stacklevel=3)
    if opening:
        tmp = _run_minmax_spec(function, array, sep_er, spec_er, False, None, array.dtype, False)
        return _run_minmax_spec(function, tmp, sep_di, spec_di, True, target, out_dtype, False)
    tmp = _run_minmax_spec(function, array, sep_di, spec_di, True, None, array.dtype, False)
    return _run_minmax_spec(function, tmp, sep_er, spec_er, False, target, out_dtype, False)


@accelerated("ndimage.grey_opening")
def grey_opening(input, size=None, footprint=None, structure=None, output=None, mode="reflect", cval=0.0, origin=0,
                 *, axes=None):
    return _grey_open_or_close("ndimage.grey_opening", input, size, footprint, structure, output, mode, cval, origin,
                               axes, opening=True)


@accelerated("ndimage.grey_closing")
def grey_closing(input, size=None, footprint=None, structure=None, output=None, mode="reflect", cval=0.0, origin=0,
                 *, axes=None):
    return _grey_open_or_close("ndimage.grey_closing", input, size, footprint, structure, output, mode, cval, origin,
                               axes, opening=False)


@accelerated("ndimage.morphological_gradient")
def morphological_gradient(input, size=None, footprint=None, structure=None, output=None, mode="reflect", cval=0.0,
                           origin=0, *, axes=None):
    _check_grey_args(size, footprint, structure)
    if output is not None and not isinstance(output, np.ndarray):
        _decline("non-array output goes to SciPy")
    array = _input(input)
    target, out_dtype = _output(output, array)
    if array.dtype == np.bool_ or out_dtype == np.bool_:
        _decline("boolean morphological_gradient goes to SciPy")
    if target is not None and np.may_share_memory(target, array):
        _decline("in-place morphological_gradient goes to SciPy")
    sep_di, spec_di, warn_di = _prepare_minmax_spec(
        array.ndim, size, footprint, structure, mode, cval, origin, axes, is_max=True, is_dilation=True)
    sep_er, spec_er, _ = _prepare_minmax_spec(
        array.ndim, size, footprint, structure, mode, cval, origin, axes, is_max=False, is_dilation=False)

    if sep_di and sep_er and out_dtype == array.dtype:
        if warn_di:
            warnings.warn("ignoring size because footprint is set", UserWarning, stacklevel=2)
        result = _nd.composite_filter(
            "ndimage.morphological_gradient", array, [spec_di, spec_er], _nd.COMBINE_SUB, out_dtype=out_dtype)
        if target is None:
            return result
        target[...] = result
        return target

    if warn_di:
        warnings.warn("ignoring size because footprint is set", UserWarning, stacklevel=2)
    dil = _run_minmax_spec("ndimage.morphological_gradient", array, sep_di, spec_di, True, None, array.dtype, False)
    ero = _run_minmax_spec("ndimage.morphological_gradient", array, sep_er, spec_er, False, None, out_dtype, False)
    if target is not None:
        np.subtract(dil, ero, out=target)
        return target
    return dil - ero


@accelerated("ndimage.morphological_laplace")
def morphological_laplace(input, size=None, footprint=None, structure=None, output=None, mode="reflect", cval=0.0,
                          origin=0, *, axes=None):
    _check_grey_args(size, footprint, structure)
    if output is not None and not isinstance(output, np.ndarray):
        _decline("non-array output goes to SciPy")
    array = _input(input)
    target, out_dtype = _output(output, array)
    if array.dtype == np.bool_ or out_dtype == np.bool_:
        _decline("boolean morphological_laplace goes to SciPy")
    if target is not None and np.may_share_memory(target, array):
        _decline("in-place morphological_laplace goes to SciPy")
    sep_di, spec_di, warn_di = _prepare_minmax_spec(
        array.ndim, size, footprint, structure, mode, cval, origin, axes, is_max=True, is_dilation=True)
    sep_er, spec_er, _ = _prepare_minmax_spec(
        array.ndim, size, footprint, structure, mode, cval, origin, axes, is_max=False, is_dilation=False)

    if sep_di and sep_er and out_dtype == array.dtype:
        if warn_di:
            warnings.warn("ignoring size because footprint is set", UserWarning, stacklevel=2)
        result = _nd.composite_filter(
            "ndimage.morphological_laplace", array, [spec_di, spec_er], _nd.COMBINE_MORPH_LAPLACE, out_dtype=out_dtype)
        if target is None:
            return result
        target[...] = result
        return target

    if warn_di:
        warnings.warn("ignoring size because footprint is set", UserWarning, stacklevel=2)
    dil = _run_minmax_spec("ndimage.morphological_laplace", array, sep_di, spec_di, True, None, array.dtype, False)
    ero = _run_minmax_spec("ndimage.morphological_laplace", array, sep_er, spec_er, False, None, out_dtype, False)
    out_arr = target if target is not None else ero
    np.add(dil, ero, out=out_arr)
    np.subtract(out_arr, array, out=out_arr)
    np.subtract(out_arr, array, out=out_arr)
    return out_arr


def _tophat(function: str, input, size, footprint, structure, output, mode, cval, origin, axes, *, white: bool):
    _check_grey_args(size, footprint, structure)
    array = _input(input)
    target, out_dtype = _output(output, array)
    if target is not None and np.may_share_memory(target, array):
        _decline(f"in-place {function} goes to SciPy")
    sep_er, spec_er, warn_er = _prepare_minmax_spec(
        array.ndim, size, footprint, structure, mode, cval, origin, axes, is_max=False, is_dilation=False)
    sep_di, spec_di, _ = _prepare_minmax_spec(
        array.ndim, size, footprint, structure, mode, cval, origin, axes, is_max=True, is_dilation=True)

    if sep_er and sep_di and out_dtype == array.dtype:
        chain = (spec_er + spec_di) if white else (spec_di + spec_er)
        if len(chain) <= _nd.SM_NDIMAGE_MAX_PASSES:
            if warn_er:
                warnings.warn("ignoring size because footprint is set", UserWarning, stacklevel=3)
            chains = [[], chain] if white else [chain, []]
            combine_op = _nd.COMBINE_XOR if array.dtype == np.bool_ else _nd.COMBINE_SUB
            result = _nd.composite_filter(function, array, chains, combine_op, out_dtype=out_dtype)
            if target is None:
                return result
            target[...] = result
            return target

    if warn_er:
        warnings.warn("ignoring size because footprint is set", UserWarning, stacklevel=3)
    if white:
        tmp = _run_minmax_spec(function, array, sep_er, spec_er, False, None, array.dtype, False)
        opened = _run_minmax_spec(function, tmp, sep_di, spec_di, True, target, out_dtype, False)
        if array.dtype == np.bool_ and opened.dtype == np.bool_:
            np.bitwise_xor(array, opened, out=opened)
        else:
            np.subtract(array, opened, out=opened)
        return opened
    tmp = _run_minmax_spec(function, array, sep_di, spec_di, True, None, array.dtype, False)
    closed = _run_minmax_spec(function, tmp, sep_er, spec_er, False, target, out_dtype, False)
    if array.dtype == np.bool_ and closed.dtype == np.bool_:
        np.bitwise_xor(closed, array, out=closed)
    else:
        np.subtract(closed, array, out=closed)
    return closed


@accelerated("ndimage.white_tophat")
def white_tophat(input, size=None, footprint=None, structure=None, output=None, mode="reflect", cval=0.0, origin=0,
                 *, axes=None):
    return _tophat("ndimage.white_tophat", input, size, footprint, structure, output, mode, cval, origin, axes,
                   white=True)


@accelerated("ndimage.black_tophat")
def black_tophat(input, size=None, footprint=None, structure=None, output=None, mode="reflect", cval=0.0, origin=0,
                 *, axes=None):
    return _tophat("ndimage.black_tophat", input, size, footprint, structure, output, mode, cval, origin, axes,
                   white=False)
