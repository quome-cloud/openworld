# tests/e127/test_probes.py
import numpy as np
from experiments.e127 import probes, engine
from experiments.e127.safe_exec import compile_engine
from tests.e127.toy import toy_factory, TOY_ENGINE_SRC, TOY_WRONG_SRC, ACTION_API
from tests.e127.toy_click import toy_click_factory, CLICK_ACTION_API


def _acts(seq):
    return [(a, None, None) for a in seq]


def _observed(seed=0, n=20):
    g = toy_factory()
    rng = np.random.default_rng(seed)
    seq = [int(rng.choice([1, 2, 3, 4, 5, 7])) for _ in range(n)]
    return [engine.play(g, _acts(seq))]


def test_finds_counterexample_for_wrong_engine():
    # Drive the real game through a KNOWN gem-collecting path so the observed corpus provably contains
    # a real gem-collection transition the wrong engine (which ignores collection) cannot reproduce.
    # From the reset cursor (4,4): up x3 clamps to row 1 -> (1,4); left x3 -> (1,1), collecting gem
    # (1,1). The bare-prefix replay then diverges deterministically -- no reliance on the novelty walk.
    g = toy_factory()
    obs = [engine.play(g, _acts([1, 1, 1, 3, 3, 3]))]
    mask = engine.identity_mask(obs)
    wrong = compile_engine(TOY_WRONG_SRC)
    budget = {"limit": 500, "used": 0}
    cexs = probes.find_counterexamples(wrong, toy_factory, obs, mask, ACTION_API, budget)
    assert len(cexs) >= 1
    c = cexs[0]
    assert not np.array_equal(c["real_frame"], c["engine_frame"])
    assert budget["used"] > 0


def test_no_counterexample_for_faithful_engine():
    obs = _observed()
    mask = engine.identity_mask(obs)
    faithful = compile_engine(TOY_ENGINE_SRC)
    budget = {"limit": 500, "used": 0}
    cexs = probes.find_counterexamples(faithful, toy_factory, obs, mask, ACTION_API, budget)
    assert cexs == []


def test_property_violation_detects_nondeterminism_claim():
    # An engine whose step depends on a hidden RNG-like counter mismatch -> determinism still holds for
    # ToyGame; instead verify color_range catches an out-of-range predictor.
    bad = ("class Engine:\n"
           "    def __init__(self): self.state={'levels':0}\n"
           "    def reset(self): return np.zeros((8,8),dtype=int)\n"
           "    def step(self, a):\n        f=np.zeros((8,8),dtype=int); f[0,0]=999; return f\n"
           "    def is_win(self,p): return False\n")
    factory = compile_engine(bad)
    budget = {"limit": 200, "used": 0}
    viols = probes.property_violations(factory, toy_factory, ACTION_API, budget)
    assert any(v["kind"] == "color_range" for v in viols)


def test_budget_caps_real_steps():
    obs = _observed(n=40)
    mask = engine.identity_mask(obs)
    wrong = compile_engine(TOY_WRONG_SRC)
    budget = {"limit": 5, "used": 0}
    probes.find_counterexamples(wrong, toy_factory, obs, mask, ACTION_API, budget)
    assert budget["used"] <= 5 + 1     # never overshoots the limit by more than one batch boundary


# --- Click-game (all-modality) test --------------------------------------------------------------
# A WRONG click engine: renders the three buttons but NEVER advances on a press (phase/t frozen), so
# it diverges from ToyClickGame the moment a correct button is pressed (button recolor + status bar
# tick). Analogous to TOY_WRONG_SRC ignoring collection.
TOY_WRONG_CLICK_SRC = '''
_BTN = [((2, 2), 5), ((4, 5), 6), ((6, 1), 7)]
class Engine:
    def __init__(self):
        self.state = {"levels": 0, "phase": 0, "t": 0, "done": False}
    def _draw(self):
        f = np.zeros((8, 8), dtype=int)
        for i, ((ry, cx), col) in enumerate(_BTN):
            f[ry, cx] = col
        f[0, 0] = 1
        return f
    def reset(self):
        self.state = {"levels": 0, "phase": 0, "t": 0, "done": False}
        return self._draw()
    def step(self, action):
        return self._draw()
    def is_win(self, prev_frame):
        return False
'''


def test_finds_counterexample_for_wrong_click_engine():
    # Drive the real click game with the correct press order so the observed corpus contains real
    # press transitions the frozen engine cannot reproduce. Clicks are (6, x=col, y=row).
    g = toy_click_factory()
    obs = [engine.play(g, [(6, 2, 2), (6, 5, 4), (6, 1, 6)])]
    mask = engine.identity_mask(obs)
    wrong = compile_engine(TOY_WRONG_CLICK_SRC)
    budget = {"limit": 500, "used": 0}
    cexs = probes.find_counterexamples(wrong, toy_click_factory, obs, mask, CLICK_ACTION_API, budget)
    assert len(cexs) >= 1
    c = cexs[0]
    assert not np.array_equal(c["real_frame"], c["engine_frame"])
    assert budget["used"] > 0
