"""Guard the zero-dependency-core contract.

`import openworld` and everything the core pulls in must use only the stdlib.
numpy-backed extras (wavelets/sheaf/infogeom/transport) are loaded lazily and
must not be dragged in by a bare import. This test fails loudly if any core
module starts importing a third-party package at module load time.
"""

import subprocess
import sys


def test_core_import_is_stdlib_only():
    # Run in a fresh interpreter so nothing this test session imported leaks in.
    code = (
        "import sys; import openworld; "
        "third_party = [m for m in ('numpy', 'fastapi', 'uvicorn', 'click', 'rich') "
        "if m in sys.modules]; "
        "print(','.join(third_party))"
    )
    out = subprocess.check_output([sys.executable, "-c", code], text=True).strip()
    assert out == "", f"core import pulled third-party modules: {out}"


def test_lazy_numpy_extras_still_accessible():
    import openworld
    # Accessing a numpy-backed name works (and pulls numpy on demand).
    for name in ("wasserstein1", "wavelet_denoise", "glue", "bayes_update", "dwt"):
        assert callable(getattr(openworld, name)), name
    assert "wasserstein1" in dir(openworld)
