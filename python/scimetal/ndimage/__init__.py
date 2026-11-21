"""Drop-in for ``scipy.ndimage``: native routes where they exist, SciPy otherwise.

Native: ``correlate1d``, ``convolve1d``, ``gaussian_filter1d``, ``gaussian_filter``,
``uniform_filter1d``, ``uniform_filter``, ``prewitt``, ``sobel``, ``correlate``,
``convolve``, ``laplace``, ``gaussian_laplace``, ``gaussian_gradient_magnitude``,
``generic_laplace``, and ``generic_gradient_magnitude`` (see ``_filters`` for what
each handles and what it leaves to SciPy).
"""
from .._fallback import install as _install
from ._filters import (convolve, convolve1d, correlate, correlate1d, gaussian_filter, gaussian_filter1d,
                       gaussian_gradient_magnitude, gaussian_laplace, generic_gradient_magnitude,
                       generic_laplace, laplace, prewitt, sobel, uniform_filter, uniform_filter1d)

_install(globals(), "scipy.ndimage")
