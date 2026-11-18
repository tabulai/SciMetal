"""Makes every ``metal_sci`` namespace a drop-in for the SciPy one it mirrors.

- ``install(globals(), 'scipy.ndimage')`` in a namespace module gives it a
  module ``__getattr__`` / ``__dir__`` that fall through to SciPy for every
  name the module does not define itself.
- A meta-path finder resolves imports such as ``import metal_sci.signal.windows``
  or ``import metal_sci.stats`` that have no file here to the SciPy module of the
  same relative name.
"""
from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import pathlib
import sys

_PREFIX = "metal_sci."
_PACKAGE_DIR = pathlib.Path(__file__).resolve().parent


def install(module_globals: dict, scipy_name: str) -> None:
    scipy_module = importlib.import_module(scipy_name)
    own_name = module_globals["__name__"]

    def __getattr__(name):
        # Never forward dunders: handing out SciPy's __path__ would make the
        # import system load SciPy submodules a second time under our name.
        if not name.startswith("__"):
            try:
                return getattr(scipy_module, name)
            except AttributeError:
                pass
        raise AttributeError(f"module {own_name!r} has no attribute {name!r}")

    def __dir__():
        own = {key for key in module_globals if not key.startswith("_")}
        return sorted(own | set(dir(scipy_module)))

    module_globals["__getattr__"] = __getattr__
    module_globals["__dir__"] = __dir__
    module_globals.setdefault("__doc__", scipy_module.__doc__)


class _AliasLoader(importlib.abc.Loader):
    def __init__(self, scipy_name):
        self._scipy_name = scipy_name

    def create_module(self, spec):
        return importlib.import_module(self._scipy_name)

    def exec_module(self, module):
        pass  # already executed as the SciPy module


def _has_own_module(relative: str) -> bool:
    base = _PACKAGE_DIR.joinpath(*relative.split("."))
    if (base / "__init__.py").exists():
        return True
    return any(base.parent.glob(base.name + ".py")) or any(base.parent.glob(base.name + ".*.so"))


class _ScipyAliasFinder(importlib.abc.MetaPathFinder):
    """First on sys.meta_path, but only claims names metal_sci has no file for.

    It has to run before the path finder: an aliased parent such as
    ``metal_sci.stats`` *is* ``scipy.stats``, so the path finder would otherwise
    find ``distributions`` on SciPy's path and execute a second copy of it.
    """

    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(_PREFIX):
            return None
        relative = fullname[len(_PREFIX):]
        if _has_own_module(relative):
            return None
        scipy_name = "scipy." + relative
        if any(part.startswith("_") for part in scipy_name.split(".")):
            return None
        try:
            if importlib.util.find_spec(scipy_name) is None:
                return None
        except (ImportError, ValueError):
            return None
        return importlib.util.spec_from_loader(fullname, _AliasLoader(scipy_name))


def install_finder() -> None:
    if not any(isinstance(finder, _ScipyAliasFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _ScipyAliasFinder())
