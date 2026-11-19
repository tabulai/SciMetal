"""metal-sci: SciPy's API, accelerated on Apple Silicon.

Every namespace mirrors the SciPy one of the same name::

    import metal_sci.ndimage as ndimage      # instead of scipy.ndimage
    from metal_sci import signal, linalg

Functions with a native route run on the Apple GPU or on a multithreaded
Accelerate/NEON CPU engine; everything else is SciPy's own function.

Precision policy (``set_precision`` / ``with precision(...)``):

- ``'fast'`` (default): float64 inputs are computed in float32 and returned as
  float64 with roughly 6-7 significant digits.
- ``'refined'``: float32 work plus a float64 correction where one exists.
- ``'strict'``: exact float64 on the CPU engine.

``explain()`` reports how the most recent accelerated call ran.
"""
from __future__ import annotations

import importlib as _importlib

from ._config import (device, get_device, get_precision, precision, set_device,
                      set_precision, set_small_inputs)
from ._dispatch import NotAccelerated, PrecisionWarning, Route, explain, last_route
from ._fallback import install_finder as _install_finder

__version__ = "0.1.0.dev0"

_install_finder()

__all__ = ["NotAccelerated", "PrecisionWarning", "Route", "accelerated", "device", "explain", "get_device",
           "get_precision", "last_route", "precision", "set_device", "set_precision", "set_small_inputs"]


def accelerated() -> dict[str, list[str]]:
    """The functions that have a native route, by namespace."""
    from ._dispatch import registry

    for name in ("ndimage", "signal", "fft", "linalg", "sparse"):
        _importlib.import_module(f"{__name__}.{name}")
    return {namespace: sorted(names) for namespace, names in sorted(registry.items())}


def __getattr__(name):
    # Namespaces: a real metal_sci module when there is one, else SciPy's.
    if not name.startswith("_"):
        try:
            return _importlib.import_module(f"{__name__}.{name}")
        except ImportError:
            pass
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
