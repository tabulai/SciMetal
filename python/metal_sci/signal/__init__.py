"""Drop-in for ``scipy.signal``: native routes where they exist, SciPy otherwise.

Native: ``convolve``, ``correlate``, ``fftconvolve``, ``oaconvolve``,
``convolve2d``, ``correlate2d``, and ``choose_conv_method`` (see
``_signaltools`` for supported dtypes, policies, and declines).
"""
from .._fallback import install as _install
from ._signaltools import (
    choose_conv_method,
    convolve,
    convolve2d,
    correlate,
    correlate2d,
    fftconvolve,
    oaconvolve,
)

_install(globals(), "scipy.signal")
