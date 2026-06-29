import numpy as np
from e119 import proxy_probe


def _frame(cells):
    """64x64 grid; cells = {(r,c): color}. Background 0."""
    g = np.zeros((64, 64), int)
    for (r, c), v in cells.items():
        g[r, c] = v
    return g


def test_enumerate_covers_present_colors_and_kinds():
    frames = [_frame({(0, 0): 4, (1, 1): 4, (2, 2): 7})]
    preds = proxy_probe.enumerate_predicates(frames)
    kinds = {p["type"] for p in preds}
    assert kinds == {"reach", "count", "align"}
    reach_colors = {p["color"] for p in preds if p["type"] == "reach"}
    assert reach_colors == {0, 4, 7}                      # every observed color
    assert any(p["type"] == "align" and p["a"] == 4 and p["b"] == 7 for p in preds)


def test_scan_satisfiable_filters_to_true_on_some_frame():
    frames = [_frame({(0, 0): 4}), _frame({(0, 0): 4, (0, 1): 4})]  # color 4 count is 1 then 2
    preds = proxy_probe.enumerate_predicates(frames)
    sat = proxy_probe.scan_satisfiable(preds, frames)
    assert {"type": "reach", "color": 4} in sat
    assert {"type": "count", "color": 4, "op": "==", "k": 2} in sat   # true on frame 2
    # a count that is never observed is not satisfiable
    assert {"type": "count", "color": 4, "op": "==", "k": 5} not in sat


class CorridorGame:
    """1-D corridor length L; pos starts 0. action 7=right, 1=left. No reward (levels stay 0).
    Mirrors the Game/_PrefixGame surface search_stats needs."""
    def __init__(self, L=12): self.L = L; self.win = 1; self.gid = "corridor"; self.reset()
    def reset(self):
        self.pos = 0; self.levels = 0; self.done = False; self.avail = [7, 1]; self._r(); return self.frame
    def _r(self):
        g = np.zeros((64, 64), int); g[0, self.pos] = 4; self.frame = g
    def step(self, a, x=None, y=None):
        if a == 7 and self.pos < self.L - 1: self.pos += 1
        if a == 1 and self.pos > 0: self.pos -= 1
        self._r(); return self.frame


class BinaryPathGame:
    """Binary tree of L/R moves; every distinct path is a distinct state (no dedup), so the
    search BRANCHES. action 1 = append-L, 7 = append-R. No reward (levels stay 0). Used to
    show best-first dives one branch deep while BFS spreads breadth-first within a node budget."""
    def __init__(self, maxlen=30): self.maxlen = maxlen; self.win = 1; self.gid = "tree"; self.reset()
    def reset(self):
        self.path = []; self.levels = 0; self.done = False; self.avail = [1, 7]; self._r(); return self.frame
    def _r(self):
        g = np.zeros((64, 64), int)
        for i, m in enumerate(self.path): g[0, i] = m
        self.frame = g
    def step(self, a, x=None, y=None):
        if a in (1, 7) and len(self.path) < self.maxlen: self.path = self.path + [a]
        self._r(); return self.frame


def test_search_stats_blind_exhausts_small_corridor():
    cands = lambda f: [(7,), (1,)]
    key = lambda f: int(np.asarray(f).reshape(64, 64)[0].argmax())  # pos is the only state
    s = proxy_probe.search_stats(CorridorGame(L=6), cands, key, {"max_nodes": 500, "max_depth": 20})
    assert s["states"] == 6 and s["frontier_exhausted"] is True and s["solved"] is False


def test_search_stats_guided_reaches_depth_faster_than_blind():
    # Branching tree: BFS spreads breadth-first; best-first scored by depth dives one branch.
    cands = lambda f: [(1,), (7,)]
    key = lambda f: np.asarray(f).reshape(64, 64)[0].tobytes()           # distinct path -> distinct state
    depth = lambda f: float((np.asarray(f).reshape(64, 64)[0] != 0).sum())
    budget = {"max_nodes": 12, "max_depth": 30}         # tight: cuts off before full exploration
    blind = proxy_probe.search_stats(BinaryPathGame(), cands, key, budget, None)
    guided = proxy_probe.search_stats(BinaryPathGame(), cands, key, budget, depth)
    assert guided["max_depth"] > blind["max_depth"]      # best-first dives; BFS spreads


def test_probe_game_reports_signals_on_corridor():
    # Corridor: blind explores all positions (frontier exhausts -> no novelty headroom),
    # and a gradient predicate ("reach color 4 far right") is FALSE at start, TRUE later.
    g = CorridorGame(L=8)
    # monkeypatch perception/candidates to the corridor's 1-D state via proxy_probe seams:
    import numpy as np
    from e119 import proxy_probe as pp
    row = pp.probe_game(g, {"max_nodes": 500, "max_depth": 20}, max_preds=10)
    assert row["game"] == "corridor"
    assert row["blind"]["frontier_exhausted"] is True
    assert row["novelty_headroom"] is False
    assert "best_depth_gain" in row and "best_novel_gain" in row
