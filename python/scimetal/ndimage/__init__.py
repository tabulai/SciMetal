"""Drop-in for ``scipy.ndimage``: native routes where they exist, SciPy otherwise.

Native: ``correlate1d``, ``convolve1d``, ``gaussian_filter1d``, ``gaussian_filter``,
``uniform_filter1d``, ``uniform_filter``, ``prewitt``, ``sobel``, ``correlate``,
``convolve``, ``laplace``, ``gaussian_laplace``, and ``gaussian_gradient_magnitude``.
SciPy handles callback-based filters such as ``generic_laplace`` and
``generic_gradient_magnitude`` directly.
"""
from .._fallback import install as _install
from ._filters import (convolve, convolve1d, correlate, correlate1d, gaussian_filter, gaussian_filter1d,
                       gaussian_gradient_magnitude, gaussian_laplace, laplace, prewitt, sobel,
                       uniform_filter, uniform_filter1d)

_install(globals(), "scipy.ndimage")
