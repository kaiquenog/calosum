from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Suppress matplotlib MPLCONFIGDIR warning by pointing to a writable tmp dir
# before any test module has a chance to import matplotlib.
if not os.environ.get("MPLCONFIGDIR"):
    _mpl_dir = os.path.join(tempfile.gettempdir(), "calosum_mplcfg")
    os.makedirs(_mpl_dir, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = _mpl_dir
