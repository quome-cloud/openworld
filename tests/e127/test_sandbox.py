# tests/e127/test_sandbox.py
import importlib.util, os
import pytest
from experiments.e127 import sandbox

def _arc_available():
    return importlib.util.find_spec("arc_agi") is not None or os.path.exists(sandbox.ARC_VENV)

def test_module_surface():
    assert hasattr(sandbox, "SandboxGame") and hasattr(sandbox, "ARC_VENV")
    # GameLike methods exist
    for m in ("reset", "step", "close"):
        assert callable(getattr(sandbox.SandboxGame, m))

@pytest.mark.skipif(not _arc_available(), reason="arc venv / arc_agi not available")
def test_smoke_real_game_steps():
    # Integration smoke: a real game resets and steps, exposing only the sandbox surface.
    g = sandbox.SandboxGame("ar25")
    f = g.reset()
    assert f.shape == (64, 64)
    assert isinstance(g.levels, int) and isinstance(g.avail, list)
    a = g.avail[0] if g.avail else 7
    g.step(a if a != 6 else 6, 0, 0)
    g.close()
