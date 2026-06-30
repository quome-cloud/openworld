import numpy as np
from experiments.e130 import perception as P
from experiments.e130.world_model import WorldModel
from experiments.e132.plan import solve_hybrid


class ChainGame:
    """Win requires the length-4 directional chain (action 1 four times). A depth>=4 plan-in-model
    after exploring the chain must find it; verify on the real env must confirm levels rise."""
    def __init__(self): self.s = 0; self.levels = 0; self.done = False; self.avail = [1, 2]
    @property
    def frame(self):
        f = np.zeros((8, 8), dtype=int); f[0, self.s] = 5; return f
    def reset(self): self.s = 0; self.levels = 0; self.done = False; return self.frame
    def step(self, a, x=None, y=None):
        if a == 1: self.s = min(self.s + 1, 4)
        if self.s == 4: self.levels = 1
        return self.frame


def test_hybrid_explores_then_plans_then_verifies_the_win():
    g = ChainGame(); wm = WorldModel()
    res = solve_hybrid(g, lambda fr: P.extrospect(fr, avail=[1, 2]), wm, frontier_path=[],
                       seed_levels=0, win=1, depth=8, beam=8, rounds=6, explore_budget=60)
    assert res.best_levels == 1                       # found + VERIFIED the deep win on the real env
    assert res.best_actions[-4:] == [[1], [1], [1], [1]]
