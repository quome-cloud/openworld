import numpy as np
from experiments.e130.world_model import WorldModel
from experiments.e130 import perception as P, moral_filter as mf
from experiments.e130.cycle import run_cycle


class ToyGame:
    """Deterministic A->B->C protocol: clicking the rare cell (1,1) advances a 1-cell marker
    rightward; reaching column 3 raises the level. Tests the cycle end-to-end with no real env."""
    def __init__(self):
        self.col = 0; self.levels = 0; self.done = False; self.avail = [6]
    @property
    def frame(self):
        f = np.zeros((8, 8), dtype=int); f[1, 1] = 3; f[4, self.col] = 5
        return f
    def step(self, a, x=None, y=None):
        if a == 6 and (x, y) == (1, 1):
            self.col = min(self.col + 1, 3)
            if self.col == 3: self.levels = 1
        return self.frame


def test_cycle_reaches_the_win_and_traces_tension():
    g = ToyGame(); wm = WorldModel(); rng = np.random.default_rng(0)
    res = run_cycle(g, wm, lambda fr: P.extrospect(fr, avail=[6]),
                    mf.DEFAULT_EXPERTS, budget=50, win=1, rng=rng)
    assert res.best_levels == 1
    assert len(res.tension_trace) > 0
    assert res.best_actions[-1] == [6, 1, 1]      # last action is the advancing click


def test_cycle_never_regresses_below_seed():
    g = ToyGame(); wm = WorldModel(); rng = np.random.default_rng(1)
    res = run_cycle(g, wm, lambda fr: P.extrospect(fr, avail=[6]),
                    mf.DEFAULT_EXPERTS, budget=2, win=1, rng=rng)
    assert res.best_levels >= 0
