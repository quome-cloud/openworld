# tests/e130/test_cycle_v2.py
import numpy as np
from experiments.e130.world_model import WorldModel
from experiments.e130 import perception as P, moral_filter as mf
from experiments.e130.cycle import run_cycle_v2


class WalkGame:
    """Directional toy: avatar (color 5) must reach column 6 to win. dir_map must be used to navigate."""
    def __init__(self): self.x = 0; self.levels = 0; self.done = False; self.avail = [1, 2, 3, 4]
    @property
    def frame(self):
        f = np.zeros((8, 8), dtype=int); f[3, self.x] = 5; f[3, 6] = 9   # 9 = the rare goal target
        return f
    def step(self, a, x=None, y=None):
        if a == 3: self.x = min(self.x + 1, 6)
        elif a == 4: self.x = max(self.x - 1, 0)
        if self.x == 6: self.levels = 1
        return self.frame


def test_v2_cycle_navigates_to_win():
    g = WalkGame(); wm = WorldModel(); rng = np.random.default_rng(0)
    dir_map = {3: (0, 1), 4: (0, -1), 1: (-1, 0), 2: (1, 0)}
    res = run_cycle_v2(g, wm, lambda fr: P.extrospect(fr, avail=[1, 2, 3, 4]),
                       mf.DEFAULT_EXPERTS, budget=50, win=1, rng=rng, avatar=5, dir_map=dir_map)
    assert res.best_levels == 1                      # reached the goal by NAVIGATING (reach via dir_map)
    assert wm.predict_rel(5, 3) == (0, 1)            # learned the lifted rule by acting
