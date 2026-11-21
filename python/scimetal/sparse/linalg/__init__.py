"""Drop-in for ``scipy.sparse.linalg``: native routes where they exist, SciPy otherwise."""
from ..._fallback import install as _install

_install(globals(), "scipy.sparse.linalg")
