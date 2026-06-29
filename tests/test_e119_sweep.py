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
