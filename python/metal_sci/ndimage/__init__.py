"""Drop-in for ``scipy.ndimage``: native routes where they exist, SciPy otherwise.

Native: ``correlate1d``, ``convolve1d``, ``gaussian_filter1d``, ``gaussian_filter``,
``uniform_filter1d``, ``uniform_filter``, ``prewitt`` and ``sobel`` (see ``_filters``
for what each handles and what it leaves to SciPy).
"""
from .._fallback import install as _install
from ._filters import (convolve1d, correlate1d, gaussian_filter, gaussian_filter1d, prewitt, sobel,
                       uniform_filter, uniform_filter1d)

_install(globals(), "scipy.ndimage")
