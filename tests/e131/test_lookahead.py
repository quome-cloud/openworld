# tests/e131/test_lookahead.py
import numpy as np
from experiments.e131.lookahead import best_sequence, FrontierCache
from experiments.e130 import perception as P


class TwoStepGame:
    """Level rises ONLY after the ordered pair (action 1, then action 2). Neither alone helps, so a
    1-step greedy sees no signal but a depth-2 lookahead finds it via level-delta."""
    def __init__(self): self.s = 0; self.levels = 0; self.done = False; self.avail = [1, 2, 3]
    @property
    def frame(self):
        f = np.zeros((8, 8), dtype=int); f[0, self.s] = 5; return f
    def reset(self): self.s = 0; self.levels = 0; self.done = False; return self.frame
    def step(self, a, x=None, y=None):
        self.s = 1 if (a == 1 and self.s == 0) else (2 if (a == 2 and self.s == 1) else 0)
        if self.s == 2: self.levels = 1
        return self.frame


def test_depth2_finds_the_ordered_pair():
    g = TwoStepGame(); perceive = lambda fr: P.extrospect(fr, avail=[1, 2, 3])
    cache = FrontierCache()
    s = perceive(g.frame); cache.seen.add(s.key)
    first, val = best_sequence(g, perceive, [], s.key, 0, [1, 2, 3], cache, depth=2, beam=4)
    assert first == [1]            # the only first move that (via action 2) raises the level
    assert val[0] == 1             # a +1 level-delta leaf was found within the horizon


def test_returns_novelty_move_when_no_level_in_horizon():
    # depth 1: no pair reachable, so level-delta is 0 everywhere; it should still return a real action
    g = TwoStepGame(); perceive = lambda fr: P.extrospect(fr, avail=[1, 2, 3])
    cache = FrontierCache(); s = perceive(g.frame); cache.seen.add(s.key)
    first, val = best_sequence(g, perceive, [], s.key, 0, [1, 2, 3], cache, depth=1, beam=4)
    assert first is not None and val[0] == 0
