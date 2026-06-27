import numpy as np
from e119 import planner, perceive


class FakeGame:
    """Deterministic toy: a token on a 1-D track; action 7 moves right; reaching x==3 is a level."""
    def __init__(self): self.win = 1; self.reset()
    def reset(self):
        self.pos = 0; self.levels = 0; self.done = False; self._render(); return self.frame
    def _render(self):
        f = np.zeros((64, 64), int); f[0, self.pos] = 4; self.frame = f
    def step(self, a, x=None, y=None):
        if a == 7 and self.pos < 63: self.pos += 1
        if self.pos == 3 and self.levels == 0: self.levels = 1; self.done = True
        self._render(); return self.frame


def test_replay_levels_counts_max_and_done():
    g = FakeGame()
    mx, done = planner.replay_levels(g, [(7,), (7,), (7,)])
    assert mx == 1 and done is True


def test_search_level_finds_action_sequence_that_levels_up():
    g = FakeGame()
    cands = lambda frame: [(7,), (1,)]           # 7 helps, 1 is a no-op
    key = lambda frame: frame.astype(np.int16).tobytes()
    seq = planner.search_level(g, cands, key, {"max_nodes": 200, "max_depth": 6})
    assert seq is not None
    mx, _ = planner.replay_levels(g, seq)
    assert mx == 1


def test_search_level_respects_node_budget_and_returns_none():
    g = FakeGame()
    cands = lambda frame: [(1,)]                  # never progresses
    key = lambda frame: frame.astype(np.int16).tobytes()
    seq = planner.search_level(g, cands, key, {"max_nodes": 10, "max_depth": 3})
    assert seq is None


def test_probe_collects_one_transition_per_directional_action():
    g = FakeGame(); g.avail = [7, 1]
    trans = perceive.probe(g)
    actions = {t["action"] for t in trans}
    assert (7,) in actions and (1,) in actions
    moved = [t for t in trans if not np.array_equal(t["before"], t["after"])]
    assert any(t["action"] == (7,) for t in moved)   # action 7 changed the board
