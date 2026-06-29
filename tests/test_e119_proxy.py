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
    # Corridor: blind explores all positions (frontier exhausts -> no novelty headroom).
    # All reach(4) predicates are TRUE at start (color 4 is always present), so n_gradient==0;
    # this exercises the no-gradient / novelty-headroom path, not the guided-search loop.
    g = CorridorGame(L=8)
    # monkeypatch perception/candidates to the corridor's 1-D state via proxy_probe seams:
    import numpy as np
    from e119 import proxy_probe as pp
    row = pp.probe_game(g, {"max_nodes": 500, "max_depth": 20}, max_preds=10)
    assert row["game"] == "corridor"
    assert row["blind"]["frontier_exhausted"] is True
    assert row["novelty_headroom"] is False
    assert "best_depth_gain" in row and "best_novel_gain" in row


def test_decide_go_subgoal_signal():
    rows = [{"game": "g50t", "n_satisfiable": 3, "best_depth_gain": 4,
             "best_novel_gain": 0.0, "novelty_headroom": False}]
    d = proxy_probe.decide_go(rows)
    assert d["go"] is True and d["signal"] == "subgoal"


def test_decide_go_novelty_default_when_both():
    rows = [{"game": "g50t", "n_satisfiable": 3, "best_depth_gain": 4,
             "best_novel_gain": 0.0, "novelty_headroom": True}]
    d = proxy_probe.decide_go(rows)
    assert d["go"] is True and d["signal"] == "novelty"   # novelty wins ties (brainstorm default)


def test_decide_go_no_go_when_flat_and_exhausted():
    rows = [{"game": "g50t", "n_satisfiable": 0, "best_depth_gain": 0,
             "best_novel_gain": 0.0, "novelty_headroom": False}]
    d = proxy_probe.decide_go(rows)
    assert d["go"] is False and d["signal"] == "none"


def test_run_probe_aggregates_and_decides(tmp_path):
    import e119_proxy_probe as drv
    def fake_make(gid):
        g = CorridorGame(L=6); g.gid = gid; return g
    payload = drv.run_probe(["g50t", "tr87"], make=fake_make,
                            budget={"max_nodes": 500, "max_depth": 20})
    assert payload["n_games"] == 2
    assert payload["decision"]["go"] in (True, False)
    assert any(r["game"] == "g50t" for r in payload["rows"])


# ── Finding 1: decide_go handles error rows / missing keys for primary ──────────

def test_decide_go_primary_error_row_returns_no_go():
    """Error row for the primary must not raise; returns go=False, signal='none'."""
    rows = [{"game": "g50t", "error": "timeout after 30s"}]
    d = proxy_probe.decide_go(rows)
    assert d["go"] is False
    assert d["signal"] == "none"
    assert "error" in d["reason"].lower() or "errored" in d["reason"].lower()


def test_decide_go_primary_missing_keys_returns_no_go():
    """Row with missing signal keys for the primary must not raise KeyError."""
    rows = [{"game": "g50t"}]          # no n_satisfiable, best_depth_gain, etc.
    d = proxy_probe.decide_go(rows)
    assert d["go"] is False
    assert d["signal"] == "none"
    assert "missing" in d["reason"].lower()


# ── Finding 2: probe_game gradient loop is exercised ────────────────────────────

class OneStepColorGame:
    """Minimal game: a single right-move (action 7) changes cell (0,0) from color 3 to color 5.
    Color 5 is absent at the start frame, so reach(5) is FALSE at start but satisfiable after
    one step — ensuring n_gradient >= 1 and exercising the guided-search loop in probe_game."""
    gid = "one_step_color"
    win = 1
    avail = [7]

    def reset(self):
        self.stepped = False; self.levels = 0; self.done = False; self._r(); return self.frame
    def _r(self):
        g = np.zeros((64, 64), int)
        g[0, 0] = 5 if self.stepped else 3
        self.frame = g
    def step(self, a, x=None, y=None):
        if a == 7: self.stepped = True
        self._r(); return self.frame


def test_probe_game_gradient_loop_exercised():
    """At least one predicate is false-at-start but satisfiable (n_gradient >= 1),
    confirming the guided-search loop inside probe_game runs."""
    g = OneStepColorGame()
    g.reset()
    row = proxy_probe.probe_game(g, {"max_nodes": 200, "max_depth": 10}, max_preds=20)
    assert row["game"] == "one_step_color"
    assert row["n_gradient"] >= 1, "expected at least one gradient predicate (reach 5 is false at start)"
    assert isinstance(row["best_depth_gain"], int) and row["best_depth_gain"] >= 0
    assert isinstance(row["best_novel_gain"], float) and row["best_novel_gain"] >= 0.0


# ── Finding 4: probe_game handles empty perceive.probe result ────────────────────

class NoActionGame:
    """Game that perceive.probe will return [] for (no avail actions yields no transitions)."""
    gid = "no_action"
    win = 1
    avail = []      # no available actions -> perceive.probe returns []

    def reset(self):
        self.levels = 0; self.done = False
        self.frame = np.zeros((64, 64), int); return self.frame
    def step(self, a, x=None, y=None):
        return self.frame


def test_probe_game_empty_probe_returns_error_row():
    """When perceive.probe returns [], probe_game returns an error row instead of IndexError."""
    from unittest.mock import patch
    g = NoActionGame()
    with patch("e119.perceive.probe", return_value=[]):
        row = proxy_probe.probe_game(g, {"max_nodes": 200, "max_depth": 10})
    assert row["game"] == "no_action"
    assert "error" in row
    assert "empty probe" in row["error"]
