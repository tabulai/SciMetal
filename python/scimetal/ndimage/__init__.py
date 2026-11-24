"""Drop-in for ``scipy.ndimage``: native routes where they exist, SciPy otherwise.

Native: ``correlate1d``, ``convolve1d``, ``gaussian_filter1d``, ``gaussian_filter``,
``uniform_filter1d``, ``uniform_filter``, ``prewitt``, ``sobel``, ``correlate``,
``convolve``, ``laplace``, ``gaussian_laplace``, ``gaussian_gradient_magnitude``,
``minimum_filter1d``, ``maximum_filter1d``, ``minimum_filter``, ``maximum_filter``,
``rank_filter``, ``median_filter``, ``percentile_filter``, ``grey_erosion``,
``grey_dilation``, ``grey_opening``, ``grey_closing``, ``morphological_gradient``,
``morphological_laplace``, ``white_tophat``, and ``black_tophat``.
SciPy handles callback-based filters such as ``generic_laplace`` and
``generic_gradient_magnitude`` directly.
"""
from .._fallback import install as _install
from ._filters import (convolve, convolve1d, correlate, correlate1d, gaussian_filter, gaussian_filter1d,
                       gaussian_gradient_magnitude, gaussian_laplace, laplace, prewitt, sobel,
                       uniform_filter, uniform_filter1d)
from ._morphology import (black_tophat, grey_closing, grey_dilation, grey_erosion, grey_opening,
                          maximum_filter, maximum_filter1d, median_filter, minimum_filter,
                          minimum_filter1d, morphological_gradient, morphological_laplace,
                          percentile_filter, rank_filter, white_tophat)

_install(globals(), "scipy.ndimage")
