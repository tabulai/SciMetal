"""Makes every ``scimetal`` namespace a drop-in for the SciPy one it mirrors.

- ``install(globals(), 'scipy.ndimage')`` in a namespace module gives it a
  module ``__getattr__`` / ``__dir__`` that fall through to SciPy for every
  name the module does not define itself.
- A meta-path finder resolves imports such as ``import scimetal.signal.windows``
  or ``import scimetal.stats`` that have no file here to the SciPy module of the
  same relative name.
"""
from __future__ import annotations

import glob
import importlib
import importlib.abc
import importlib.util
import pathlib
import sys

_PREFIX = "scimetal."
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
    own_all = set(module_globals.get("__all__", ()))
    scipy_all = set(getattr(scipy_module, "__all__", ()))
    if own_all or scipy_all:
        module_globals["__all__"] = sorted(own_all | scipy_all)
    module_globals.setdefault("__doc__", scipy_module.__doc__)


class _AliasLoader(importlib.abc.Loader):
    def __init__(self, scipy_name):
        self._scipy_name = scipy_name
        self._saved_attrs = None

    def create_module(self, spec):
        module = importlib.import_module(self._scipy_name)
        self._saved_attrs = (
            getattr(module, "__spec__", None),
            getattr(module, "__loader__", None),
            getattr(module, "__package__", None),
        )
        return module

    def exec_module(self, module):
        # importlib._bootstrap._init_module_attrs overwrites __spec__/__loader__/__package__
        # on the returned SciPy module object; restore SciPy's own metadata immediately so
        # importlib.resources, importlib.reload, and introspection on scipy.<pkg> stay intact.
        if self._saved_attrs is not None:
            orig_spec, orig_loader, orig_package = self._saved_attrs
            if orig_spec is not None:
                module.__spec__ = orig_spec
            if orig_loader is not None:
                module.__loader__ = orig_loader
            if orig_package is not None:
                module.__package__ = orig_package
            self._saved_attrs = None


def _has_own_module(relative: str) -> bool:
    parts = relative.split(".")
    if any(not part or part in (".", "..") or "/" in part or "\\" in part for part in parts):
        return False
    base = _PACKAGE_DIR.joinpath(*parts)
    if (base / "__init__.py").exists() or base.with_suffix(".py").exists():
        return True
    escaped = glob.escape(base.name)
    return any(base.parent.glob(escaped + ".*.so"))


class _ScipyAliasFinder(importlib.abc.MetaPathFinder):
    """First on sys.meta_path, but only claims names scimetal has no file for.

    It has to run before the path finder: an aliased parent such as
    ``scimetal.stats`` *is* ``scipy.stats``, so the path finder would otherwise
    find submodules (including private underscore ones) on SciPy's path and
    execute a second copy of them.
    """

    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(_PREFIX):
            return None
        relative = fullname[len(_PREFIX):]
        if not relative or relative.startswith("_"):
            return None
        if _has_own_module(relative):
            return None
        scipy_name = "scipy." + relative
        try:
            scipy_spec = importlib.util.find_spec(scipy_name)
        except (ImportError, ValueError):
            return None
        if scipy_spec is None:
            return None
        is_package = scipy_spec.submodule_search_locations is not None
        return importlib.util.spec_from_loader(fullname, _AliasLoader(scipy_name), is_package=is_package)


def install_finder() -> None:
    if not any(isinstance(finder, _ScipyAliasFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _ScipyAliasFinder())
