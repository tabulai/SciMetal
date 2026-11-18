import pathlib
import sys

# Run against the source tree: python/ on the path, native library from .build/.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))

import metal_sci  # noqa: E402

# These tests reach the native code with small arrays; by default metal-sci
# would hand those to SciPy.
metal_sci.set_small_inputs("native")
