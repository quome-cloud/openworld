import json, numpy as np
from e119 import solve


class TrackGame:
    """2 levels: walk right to x==3 (level 1), then to x==6 (level 2). Action 7 = right."""
    def __init__(self): self.win = 2; self.reset()
    def reset(self):
        self.pos = 0; self.levels = 0; self.done = False; self.avail = [7, 1]; self._r(); return self.frame
    def _r(self):
        f = np.zeros((64, 64), int); f[0, self.pos] = 4; self.frame = f
    def step(self, a, x=None, y=None):
        if a == 7 and self.pos < 63: self.pos += 1
        if self.pos == 3 and self.levels == 0: self.levels = 1
        if self.pos == 6 and self.levels == 1: self.levels = 2; self.done = True
        self._r(); return self.frame


def test_solve_game_search_only_chains_all_levels(tmp_path):
    res = solve.solve_game(TrackGame(), mode="search",
                           budget={"max_nodes": 500, "max_depth": 8}, logdir=tmp_path)
    assert res["levels"] == 2 and res["win"] == 2
    assert res["verified"] is True
    # banked solved.json round-trips
    saved = json.loads((tmp_path / "TrackGame_solved.json").read_text())
    assert saved["levels"] == 2


def test_solve_game_search_mode_never_calls_llm(tmp_path):
    class Boom:
        def ask(self, *a, **k): raise AssertionError("llm must not be called in search mode")
    res = solve.solve_game(TrackGame(), llm=Boom(), mode="search",
                           budget={"max_nodes": 500, "max_depth": 8}, logdir=tmp_path)
    assert res["levels"] == 2


def test_entry_run_pilot_aggregates(monkeypatch, tmp_path):
    import e119_slm_solver as entry

    def fake_make(gid):
        return TrackGame()
    payload = entry.run_pilot(["g1", "g2"], mode="search", make=fake_make,
                              budget={"max_nodes": 500, "max_depth": 8}, logdir=tmp_path)
    assert payload["n_games"] == 2
    assert payload["levels_solved"] == 4          # 2 levels each
    assert all(r["verified"] for r in payload["results"])
