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

def test_init_failure_reaps_worker(monkeypatch):
    # A worker that dies before "ready" (readline -> "") must be terminated,
    # not left orphaned. No arc_agi / real env needed.
    class _FakeStdout:
        def readline(self):
            return ""  # EOF: worker died before sending {"ready": True}
    class _FakeProc:
        def __init__(self, *a, **k):
            self.stdout = _FakeStdout()
            self.stdin = None
            self.terminated = False
        def terminate(self):
            self.terminated = True
        def wait(self, timeout=None):
            return 0
    procs = []
    def _fake_popen(*a, **k):
        p = _FakeProc()
        procs.append(p)
        return p
    monkeypatch.setattr(sandbox.subprocess, "Popen", _fake_popen)
    with pytest.raises(RuntimeError, match="worker died before ready"):
        sandbox.SandboxGame("fake")
    assert procs and procs[0].terminated, "worker subprocess was not reaped on init failure"

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
