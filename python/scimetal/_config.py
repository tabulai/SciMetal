"""Precision and device policy.

``set_precision`` / ``set_device`` change the process-wide default;
``precision(...)`` / ``device(...)`` override it for the current context
(thread or task) only, so nested and concurrent uses do not leak.
"""
from __future__ import annotations

import contextlib
import contextvars
import os

PRECISIONS = ("fast", "refined", "strict")
DEVICES = ("auto", "cpu", "gpu")


def _check(value, allowed, what):
    if value not in allowed:
        raise ValueError(f"{what} must be one of {', '.join(map(repr, allowed))}; got {value!r}")
    return value


def _from_env(name, allowed, default):
    value = os.environ.get(name)
    return default if value is None else _check(value.strip().lower(), allowed, name)


# fast:    float64/complex128 inputs are computed in float32/complex64 (GPU, or
#          Accelerate single precision below the crossover) and returned as
#          float64 with roughly 6-7 significant digits.
# refined: float32 work plus a float64 correction where one exists (solves).
# strict:  exact float64 on the CPU engine.
_default_precision = _from_env("SCIMETAL_PRECISION", PRECISIONS, "fast")
# auto: the router picks; cpu / gpu force an engine (gpu never falls back).
_default_device = _from_env("SCIMETAL_DEVICE", DEVICES, "auto")
# Whether inputs too small to win natively are handed to SciPy (the default).
# A native call costs some 10-20 microseconds of hand-off whatever its size, so
# below a per-function size SciPy is faster; tests of the native code turn
# this off to reach it with small arrays.
_small_to_scipy = _from_env("SCIMETAL_SMALL_INPUTS", ("scipy", "native"), "scipy") == "scipy"
_precision_override = contextvars.ContextVar("scimetal_precision", default=None)
_device_override = contextvars.ContextVar("scimetal_device", default=None)


def get_precision() -> str:
    """The precision policy in effect: 'fast', 'refined' or 'strict'."""
    return _precision_override.get() or _default_precision


def set_precision(value: str) -> None:
    """Set the process-wide default precision policy."""
    global _default_precision
    _default_precision = _check(value, PRECISIONS, "precision")


@contextlib.contextmanager
def precision(value: str):
    """Use a precision policy inside a ``with`` block."""
    token = _precision_override.set(_check(value, PRECISIONS, "precision"))
    try:
        yield
    finally:
        _precision_override.reset(token)


def get_device() -> str:
    """The device policy in effect: 'auto', 'cpu' or 'gpu'."""
    return _device_override.get() or _default_device


def set_device(value: str) -> None:
    """Set the process-wide default device policy."""
    global _default_device
    _default_device = _check(value, DEVICES, "device")


@contextlib.contextmanager
def device(value: str):
    """Use a device policy inside a ``with`` block."""
    token = _device_override.set(_check(value, DEVICES, "device"))
    try:
        yield
    finally:
        _device_override.reset(token)


def set_small_inputs(route: str) -> None:
    """Where inputs too small to gain from the native code go: 'scipy' (default) or 'native'."""
    global _small_to_scipy
    _small_to_scipy = _check(route, ("scipy", "native"), "small-input route") == "scipy"


def small_to_scipy() -> bool:
    """True when native routes should decline inputs below their break-even size."""
    return _small_to_scipy
