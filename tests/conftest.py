"""Test suite bootstrap: sets MPLCONFIGDIR to suppress matplotlib temp-dir warnings."""
import os
import tempfile

# Provide a writable matplotlib config dir before any test imports matplotlib.
# Without this, matplotlib creates a tmpdir and emits a UserWarning on every run.
_mpl_dir = os.environ.get("MPLCONFIGDIR")
if not _mpl_dir:
    _mpl_dir = os.path.join(tempfile.gettempdir(), "calosum_mplcfg")
    os.makedirs(_mpl_dir, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = _mpl_dir
