import os
import pathlib
import sys

# Run against the source tree unless explicitly testing an installed package/wheel.
if not os.environ.get("METAL_SCI_TEST_INSTALLED"):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))

import metal_sci  # noqa: E402

# These tests reach the native code with small arrays; by default metal-sci
# would hand those to SciPy.
metal_sci.set_small_inputs("native")
