import json, numpy as np, hashlib
from e119 import solve, trace


class MacroGame:
    """Level 1 needs walking to pos 6 via action 7. Tight budget => blind BFS can't assemble it."""
    def __init__(self): self.win = 1; self.gid = "mg"; self.reset()
    def reset(self): self.pos = 0; self.levels = 0; self.done = False; self.avail = [7, 1]; self._r(); return self.frame
    def _r(self): g = np.zeros((64, 64), int); g[0, self.pos] = 4; self.frame = g
    def step(self, a, x=None, y=None):
        if a == 7 and self.pos < 63: self.pos += 1
        if a == 1 and self.pos > 0: self.pos -= 1
        if self.pos == 6 and self.levels == 0: self.levels = 1; self.done = True
        self._r(); return self.frame


def test_random_macro_mode_is_seed_deterministic():
    # Same seed -> identical banked result; no LLM is consulted in random-macro mode.
    class Boom:
        def ask(self, *a, **k): raise AssertionError("random-macro mode must not call the LLM")
    r1 = solve.solve_game(MacroGame(), llm=Boom(), mode="random-macro", seed=7,
                          budget={"max_nodes": 3, "max_depth": 10}, make=lambda gid: MacroGame())
    r2 = solve.solve_game(MacroGame(), llm=Boom(), mode="random-macro", seed=7,
                          budget={"max_nodes": 3, "max_depth": 10}, make=lambda gid: MacroGame())
    assert r1["actions"] == r2["actions"] and r1["levels"] == r2["levels"]


def test_prompt_digest_is_stable():
    d = trace.prompt_digest("hello world")
    assert d["chars"] == 11 and d["lines"] == 1
    assert d["sha256"] == hashlib.sha256(b"hello world").hexdigest()
    assert d["approx_tokens"] >= 1


def test_provenance_captures_config():
    p = trace.provenance("qwen2.5-coder:7b", {"num_ctx": 8192, "temperature": 0.7}, [0, 1, 2], {"max_nodes": 6000})
    assert p["model"] == "qwen2.5-coder:7b" and p["seeds"] == [0, 1, 2]
    assert p["options"]["num_ctx"] == 8192 and p["budget"]["max_nodes"] == 6000
    assert "python" in p["versions"]              # version block present
    assert "digest" in p                          # best-effort key always present (may be None)


def test_log_run_appends_one_json_line(tmp_path):
    f = tmp_path / "runs.jsonl"
    trace.log_run(f, {"run_id": "tr87__macro__t", "game": "tr87", "verified": True})
    trace.log_run(f, {"run_id": "tr87__search__t", "game": "tr87", "verified": False})
    lines = f.read_text().strip().splitlines()
    assert len(lines) == 2 and json.loads(lines[0])["run_id"] == "tr87__macro__t"


def test_run_sweep_aggregates_arms_and_seeds():
    import e119_macro_sweep as sweep
    from openworld import MockLLM
    # SLM arm: 12 replies of the winning 6-step macro -> solves; search arm: tight budget -> 0.
    def llm_factory(seed):
        return MockLLM([json.dumps(["a7", "a7", "a7", "a7", "a7", "a7"])] * 12)
    payload = sweep.run_sweep(["mg"], seeds=[0, 1, 2], make=lambda gid: MacroGame(),
                              llm_factory=llm_factory, budget={"max_nodes": 3, "max_depth": 10})
    mg = payload["by_game_arm"]["mg"]
    assert mg["search"]["k_solved"] == 0                      # blind cannot, deterministic
    assert mg["macro"]["k_solved"] == 3 and mg["macro"]["m"] == 3   # SLM solves every seed
    assert mg["macro"]["levels_mean"] == 1.0
    assert "provenance" in payload and payload["provenance"]["seeds"] == [0, 1, 2]


def test_reverify_replays_banked_solutions(tmp_path):
    from e119 import reverify
    # a banked solve: the 6-step macro that wins MacroGame
    (tmp_path / "mg_solved.json").write_text(json.dumps(
        {"game": "mg", "levels": 1, "actions": [[7], [7], [7], [7], [7], [7]]}))
    res = reverify.reverify_solves(tmp_path, make=lambda gid: MacroGame())
    assert res["ok"] == 1 and res["n"] == 1 and res["fail"] == []
    # a bogus banked solve fails re-verification
    (tmp_path / "bad_solved.json").write_text(json.dumps(
        {"game": "bad", "levels": 1, "actions": [[1], [1]]}))
    res2 = reverify.reverify_solves(tmp_path, make=lambda gid: MacroGame())
    assert res2["ok"] == 1 and res2["n"] == 2 and "bad" in res2["fail"]
