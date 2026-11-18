"""Routing of accelerated functions, with fall-through to stock SciPy.

An accelerated function is written as an implementation that either returns a
result (after calling ``record``) or raises ``NotAccelerated`` to say this call
is one it does not handle (unsupported arguments, a dtype with no native path,
an input too small to win). The wrapper then calls SciPy's own function with
the caller's arguments untouched, so behaviour never regresses.
"""
from __future__ import annotations

import contextvars
import dataclasses
import functools
import importlib

__all__ = ["NotAccelerated", "Route", "accelerated", "counters", "explain", "last_route", "record",
           "registry"]


class NotAccelerated(Exception):
    """Raised by a native route to hand the call to stock SciPy."""


@dataclasses.dataclass(frozen=True)
class Route:
    """How the most recent accelerated call ran."""

    function: str        # e.g. 'ndimage.gaussian_filter'
    engine: str          # 'gpu', 'cpu' or 'scipy' (fell through)
    compute_dtype: str   # dtype the arithmetic ran in, e.g. 'float32'
    policy: str          # precision policy in effect
    reason: str = ""     # why this route, or why it fell through

    def __str__(self) -> str:
        text = f"{self.function}: engine={self.engine}, computed in {self.compute_dtype}, policy={self.policy}"
        return f"{text} ({self.reason})" if self.reason else text


_last = contextvars.ContextVar("metal_sci_last_route", default=None)
# namespace -> names with a native route, e.g. {'ndimage': {'gaussian_filter'}}
registry: dict[str, set[str]] = {}
# 'ndimage.gaussian_filter' -> [calls that ran natively, calls handed to SciPy]
counters: dict[str, list[int]] = {}


def record(route: Route) -> None:
    _last.set(route)


def last_route() -> Route | None:
    """The Route of the most recent accelerated call in this context, if any."""
    return _last.get()


def explain() -> str:
    """One line describing how the most recent accelerated call ran."""
    route = _last.get()
    return str(route) if route else "no accelerated call has run in this context"


def accelerated(qualified_name: str):
    """Register ``impl`` as the native route for ``scipy.<qualified_name>``."""
    namespace, _, name = qualified_name.rpartition(".")
    reference = getattr(importlib.import_module(f"scipy.{namespace}"), name)

    def decorate(impl):
        from ._config import get_precision

        tally = counters.setdefault(qualified_name, [0, 0])

        @functools.wraps(reference)
        def wrapper(*args, **kwargs):
            try:
                result = impl(*args, **kwargs)
            except NotAccelerated as why:
                tally[1] += 1
                record(Route(qualified_name, "scipy", "as given", get_precision(), str(why)))
                return reference(*args, **kwargs)
            tally[0] += 1
            return result

        # functools.wraps copied SciPy's module and qualified name; under those
        # pickle would find SciPy's own function, not this wrapper, and refuse it
        # (multiprocessing pickles the functions it is handed).
        wrapper.__module__, wrapper.__qualname__ = f"metal_sci.{namespace}", name
        wrapper.__metal_sci_native__ = impl
        registry.setdefault(namespace, set()).add(name)
        return wrapper

    return decorate
