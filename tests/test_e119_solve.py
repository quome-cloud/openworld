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


class DeadGame:
    """No action ever raises levels; search must return 0 honestly (the control-rung baseline case)."""
    def __init__(self): self.win = 3; self.reset()
    def reset(self):
        self.levels = 0; self.done = False; self.avail = [7, 1]
        self.frame = np.zeros((64, 64), int); return self.frame
    def step(self, a, x=None, y=None): return self.frame   # frame never changes, levels never rise


def test_zero_solve_is_honest_not_a_failed_assert(tmp_path):
    """A game the control rung cannot solve yields levels=0 with no error; the driver's
    honesty invariant must ACCEPT that, while still rejecting a fabricated unverified solve."""
    import e119_slm_solver as entry
    payload = entry.run_pilot(["d1"], mode="search", make=lambda gid: DeadGame(),
                              budget={"max_nodes": 200, "max_depth": 6}, logdir=tmp_path)
    assert payload["levels_solved"] == 0
    r = payload["results"][0]
    assert r["levels"] == 0 and r["verified"] is False and "error" not in r
    assert entry._is_honest(r)                                   # honest zero passes
    assert not entry._is_honest({"levels": 2, "verified": False})  # unverified non-zero solve fails


class CheckpointGame:
    """Mimics the real arc env: reset() restores the board to the start of the CURRENT
    level but RETAINS completed-level progress (it does not zero levels_completed). Only a
    fresh make() is a true game start. 1 level: walk right to x==3. Action 7 = right."""
    def __init__(self): self.win = 2; self.gid = "cp"; self.levels = 0; self._init()
    def _init(self):
        self.pos = 0; self.done = False; self.avail = [7, 1]; self._r()
    def reset(self):
        self._init()                # board resets; self.levels deliberately NOT zeroed
        return self.frame
    def _r(self):
        f = np.zeros((64, 64), int); f[0, self.pos] = 4; self.frame = f
    def step(self, a, x=None, y=None):
        if a == 7 and self.pos < 63: self.pos += 1
        if self.pos == 3 and self.levels == 0: self.levels = 1
        self._r(); return self.frame


def test_verify_uses_fresh_env_when_reset_retains_progress(tmp_path):
    """Root cause of the 0-vs-0 mis-measure: the arc env's reset() keeps completed-level
    progress, so verifying on the reused env makes replay_levels' delta collapse to 0.
    solve_game must verify on a FRESH env (via make) so a real solve isn't reported as 0."""
    res = solve.solve_game(CheckpointGame(), mode="search", make=lambda gid: CheckpointGame(),
                           budget={"max_nodes": 200, "max_depth": 8}, logdir=tmp_path)
    assert res["levels"] == 1, f"expected 1 replay-verified level, got {res['levels']}"
    assert res["verified"] is True


def test_entry_run_pilot_aggregates(monkeypatch, tmp_path):
    import e119_slm_solver as entry

    def fake_make(gid):
        return TrackGame()
    payload = entry.run_pilot(["g1", "g2"], mode="search", make=fake_make,
                              budget={"max_nodes": 500, "max_depth": 8}, logdir=tmp_path)
    assert payload["n_games"] == 2
    assert payload["levels_solved"] == 4          # 2 levels each
    assert all(r["verified"] for r in payload["results"])
