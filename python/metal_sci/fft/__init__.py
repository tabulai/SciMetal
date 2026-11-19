"""Drop-in for ``scipy.fft``: native routes where they exist, SciPy otherwise.

``fft``, ``ifft``, ``rfft``, ``irfft`` and their 2-D and N-D forms run on the
Apple GPU in single precision when the input is large enough to win there
(see ``_basic``); every other name is SciPy's own object. ``prepare()``
compiles and caches GPU pipelines ahead of time so the first transform does
not pay pipeline compilation cost.

``register()`` installs the same routes as a ``scipy.fft`` backend, so code
that calls ``scipy.fft`` directly is accelerated without being changed;
``unregister()`` hands the calls back to SciPy.
"""
from .._fallback import install as _install
from . import _backend, _basic
from ._backend import backend, register, registered, unregister
from ._basic import (fft, fft2, fftn, ifft, ifft2, ifftn, irfft, irfft2, irfftn, prepare, rfft, rfft2,
                     rfftn)

__all__ = sorted({"prepare", *_backend.__all__, *_basic.__all__})

_install(globals(), "scipy.fft")
