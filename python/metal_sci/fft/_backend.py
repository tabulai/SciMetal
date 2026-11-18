"""uarray backend: ``scipy.fft`` itself uses the native routes.

    import metal_sci.fft
    metal_sci.fft.register()
    scipy.fft.fft(x)            # native where metal-sci accepts the call, SciPy otherwise

SciPy is not modified: the backend is installed through its public
``scipy.fft.register_backend``. Registered backends are tried before SciPy's
own, and a call this backend answers with ``NotImplemented`` continues to it.

For a limited scope, ``metal_sci.fft.backend`` works with SciPy's context
manager: ``with scipy.fft.set_backend(metal_sci.fft.backend): ...``.
"""
from __future__ import annotations

import inspect

import scipy.fft

from .. import _config
from .._dispatch import Route, counters, record
from . import _basic

__all__ = ["backend", "register", "registered", "unregister"]


class _Backend:
    """The ``numpy.scipy.fft`` backend object (also usable with ``scipy.fft.set_backend``)."""

    __ua_domain__ = "numpy.scipy.fft"

    def __init__(self):
        self.enabled = True
        self._routes = {name: getattr(_basic, name).__metal_sci_native__ for name in _basic.__all__}
        self._signatures = {name: inspect.signature(route.core) for name, route in self._routes.items()}

    def __ua_function__(self, method, args, kwargs):
        route = self._routes.get(method.__name__)
        if route is None or not self.enabled or _basic._delegating.get():
            return NotImplemented
        qualified = f"fft.{method.__name__}"
        tally = counters[qualified]
        try:
            result = route.core(*args, **kwargs)
        except _basic.Declined as why:
            tally[1] += 1
            threads = self._threads(method.__name__, why, args, kwargs)
            if threads is None:
                record(Route(qualified, "scipy", "as given", _config.get_precision(), str(why)))
                return NotImplemented
            arguments, x = threads[1], threads[2]
            result = _basic.run_threaded(route.reference, x, arguments, threads[0])
            record(Route(qualified, "scipy", "as given", _config.get_precision(),
                         f"{why}; SciPy with workers={threads[0]}"))
            return result
        except TypeError:
            # Arguments that do not fit the signature are SciPy's to reject.
            tally[1] += 1
            return NotImplemented
        tally[0] += 1
        return result

    def _threads(self, name, why, args, kwargs):
        """(workers, positional arguments after x, x) for a threaded SciPy call, or None."""
        try:
            bound = self._signatures[name].bind(*args, **kwargs)
        except TypeError:
            return None
        bound.apply_defaults()
        values = list(bound.arguments.values())  # x, n|s, axis|axes, norm, overwrite_x, workers, plan
        x, arguments, workers, plan = values[0], tuple(values[1:5]), values[5], values[6]
        threads = _basic.threaded_fallback(x, arguments[1], why, workers, plan)
        return None if threads is None else (threads, arguments, x)


# For scipy.fft.set_backend(...): always answers.
backend = _Backend()
# The instance register() installs. uarray keeps it for the life of the
# process, so unregister() switches it off instead.
_installed = _Backend()
_installed.enabled = False
_registered = False


def register() -> None:
    """Make ``scipy.fft``'s own functions use the native routes.

    Calls metal-sci does not accept (sizes without a plan, inputs below the
    GPU crossover, double precision under the ``strict`` policy, ...) run on
    SciPy's backend exactly as before.
    """
    global _registered
    if not _registered:
        scipy.fft.register_backend(_installed)
        _registered = True
    _installed.enabled = True


def unregister() -> None:
    """Hand ``scipy.fft`` back to SciPy's backend.

    uarray has no public call that removes one registered backend (clearing
    the domain would also drop other libraries' backends), so the backend
    stays in SciPy's list and answers ``NotImplemented`` to every call, which
    SciPy's own backend then handles.
    """
    _installed.enabled = False


def registered() -> bool:
    """Whether ``scipy.fft`` currently routes through metal-sci."""
    return _installed.enabled
