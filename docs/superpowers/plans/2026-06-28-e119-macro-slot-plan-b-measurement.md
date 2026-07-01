# E119 Macro Slot — Plan B (measurement harness) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure the macro slot's value on the signal-bearing procedure-walls (tr87, re86) — a 3-arm (control / random-macro / SLM-macro), m-seed sweep with pinned provenance, per-call trace logging, and a banked-solve re-verifier — to test the falsifiable bet: does the SLM macro solve ≥1 level that blind search at matched budget cannot?

**Architecture:** Plan A's macro mechanism is reused unchanged. Plan B adds: (1) a `random-macro` arm that swaps the SLM proposer for the seeded `propose_random_macros`; (2) `experiments/e119/trace.py` — deterministic provenance capture + a `runs.jsonl`-style per-run trace logger; (3) `experiments/e119_macro_sweep.py` — the sweep driver that runs games × arms × seeds and reports mean±variance + k/m; (4) a re-verifier that replays every banked solution on a fresh env. The SLM arm is reported as a distribution (seeds + variance); every banked solve is anchored to a replayable action sequence.

**Tech Stack:** Python ≥3.12, numpy, existing `experiments/e119/*` (Plan A), `openworld.OllamaLLM`/`MockLLM`, the repo `.venv` with `arc_agi`/`arcengine` for the real run. No new third-party deps.

## Global Constraints

- Work from: /Users/jeniaquome/code/quome/openworld. Core zero-dep; `experiments/` may use numpy.
- **The SLM arm is NOT bit-reproducible** (sampling + Metal nondeterminism): report it as **mean ± variance and k/m games solved over m≥5 seeds**, never a single run. The control arm, the random-macro arm's seed-determinism, and every banked solution ARE deterministic point facts.
- **Pin provenance** in the results `env` block: model tag, `num_ctx`/`num_predict`/decoding options, the seed list, budget (`max_nodes`/`max_depth`), and `arc_agi`/`arcengine`/numpy/python versions. Best-effort Ollama model **digest** (None if unavailable).
- **`save_results` BEFORE any assert** (CLAUDE.md).
- **Bank only replay-verified level-ups on a FRESH env** (Plan A's invariant — unchanged). Re-verify every banked solution by replaying it on a newly-made env.
- **Three arms at MATCHED budget:** `search` (control), `random-macro` (seeded random macros on stall, matched count/length), `macro` (SLM). Model contribution = macro − random-macro; mechanism value = either − search.
- **Scope:** tr87, re86 (primary, strong Phase 0 signal); sb26, cn04 (secondary, weak). Exclude g50t (flat boundary), bp35 (pruner), sc25 (inert wall).
- **Cost control:** establish the GO/positive signal with ONE model (qwen2.5-coder:7b) × m seeds first; only fan out to the diversity set if positive.
- Tests env-free (numpy + MockLLM + mock games); the real sweep (Task 4 run step) uses the `.venv`. Test command: `PYTHONPATH="$PWD:$PWD/experiments:$PWD/scratch_arc/agent" python3 -m pytest tests/test_e119_sweep.py -q` (system python3).
- Reuse Plan A: `macro.{propose_macros,propose_random_macros,rank_macros}`, `solve.solve_game`, `e119_slm_solver._real_make`, `common.save_results`. Do not modify `planner.search_level`.

## File Structure

- `experiments/e119/solve.py` (modify) — parameterize `_macro_fallback`'s proposer; add `mode="random-macro"` + a `seed` param to `solve_game`.
- `experiments/e119/trace.py` (new) — `provenance(...)`, `prompt_digest(...)`, `log_run(...)`. One responsibility: reproducibility metadata + per-run trace records.
- `experiments/e119_macro_sweep.py` (new) — the 3-arm × m-seed sweep driver + aggregation + re-verify call + `save_results`.
- `experiments/e119/reverify.py` (new) — `reverify_solves(logdir, make)`: replay every banked `*_solved.json` on a fresh env, assert levels.
- `tests/test_e119_sweep.py` (new) — env-free unit tests for all of the above.

---

### Task 1: Random-macro arm

**Files:**
- Modify: `experiments/e119/solve.py`
- Test: `tests/test_e119_sweep.py`

**Interfaces:**
- Consumes: `macro.propose_random_macros` (Plan A), the existing `_macro_fallback`.
- Produces: `solve_game(..., mode in {search,slm,macro,macro+slm,random-macro}, seed=0, ...)`. In `random-macro` mode the stall fallback proposes via a seeded `propose_random_macros` (matched count/length) instead of the SLM; banking is identical (fresh-env replay).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_e119_sweep.py
import json, numpy as np
from e119 import solve


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH="$PWD:$PWD/experiments:$PWD/scratch_arc/agent" python3 -m pytest tests/test_e119_sweep.py -k random_macro_mode -q`
Expected: FAIL — `solve_game` rejects `mode="random-macro"`/`seed` or calls the LLM.

- [ ] **Step 3: Write minimal implementation**

In `experiments/e119/solve.py`, add `import random` at the top. Change `solve_game`'s signature to add `seed=0`:

```python
def solve_game(game, llm=None, mode="search", budget=None, logdir=None, make=None, seed=0):
```

In the stall branch, pass mode+seed to the fallback:

```python
        if seq is None:
            if mode in ("macro", "macro+slm", "random-macro") and llm is not None:
                seq = _macro_fallback(game, actions, trans, llm, key_fn, make, subgoal,
                                      mode=mode, seed=seed)
            if seq is None:
                break
```

(Keep `random-macro` OUT of the `score_fn` gate — it gets no subgoal ordering, like plain `macro`.)

Change `_macro_fallback` to choose its proposer by mode:

```python
def _macro_fallback(game, actions, trans, llm, key_fn, make, subgoal=None, mode="macro", seed=0):
    """On a stall, propose macros (SLM or, in random-macro mode, a seeded random baseline),
    rank, and return the FIRST whose fresh-env replay raises levels. The env decides correctness."""
    from e119 import macro
    avail = list(getattr(game, "avail", [1, 2, 3, 4, 5, 7]))
    oj = perceive.object_json(trans[0]["before"])
    if mode == "random-macro":
        rng = random.Random(seed + len(actions))     # vary per level, deterministic given seed
        cands = macro.propose_random_macros(avail, oj, k_max=8, count=6, rng=rng)
        cands = [m for m in cands if m]
        subgoal = None
    else:
        diffs = [perceive.contrastive_diff(t["before"], t["after"]) for t in trans]
        if subgoal is None:
            try:
                subgoal = slm.propose_subgoal(llm, oj, [t["after"] for t in trans])
            except Exception:
                subgoal = None
        cands = macro.propose_macros(llm, game, actions, oj, diffs, subgoal, avail, key_fn)
    if not cands:
        return None
    ranked = macro.rank_macros(cands, game, actions, subgoal, key_fn, seen=set())
    base_levels = _levels_after(make, game, actions)
    for m in ranked:
        if _levels_after(make, game, actions + list(m)) > base_levels:
            return list(m)
    return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH="$PWD:$PWD/experiments:$PWD/scratch_arc/agent" python3 -m pytest tests/test_e119_sweep.py -k random_macro_mode -q` → PASS. Then full E119 suite: `... python3 -m pytest tests/test_e119_*.py -q` → all pass.

- [ ] **Step 5: Commit**

```bash
git add experiments/e119/solve.py tests/test_e119_sweep.py
git commit -m "feat(e119): random-macro arm (seeded baseline mode for the sweep)"
```

---

### Task 2: Provenance + trace logging

**Files:**
- Create: `experiments/e119/trace.py`
- Test: `tests/test_e119_sweep.py`

**Interfaces:**
- Produces: `provenance(model, options, seeds, budget) -> dict` (reproducibility metadata incl. best-effort lib versions + digest=None placeholder); `prompt_digest(text) -> {"chars","lines","sha256","approx_tokens"}`; `log_run(path, record) -> None` (append one JSON line).

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_e119_sweep.py
import hashlib
from e119 import trace


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH="$PWD:$PWD/experiments:$PWD/scratch_arc/agent" python3 -m pytest tests/test_e119_sweep.py -k "digest or provenance or log_run" -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'e119.trace'`.

- [ ] **Step 3: Write minimal implementation**

```python
# experiments/e119/trace.py
"""E119 reproducibility: provenance capture + a runs.jsonl-style per-run trace logger.
The SLM arm is not bit-reproducible (sampling + Metal); this pins everything that IS fixed and
records every run so a draw can be audited and anchored to its replayable action sequence."""
import json, hashlib, sys, platform


def prompt_digest(text):
    b = text.encode("utf-8")
    return {"chars": len(text), "lines": text.count("\n") + 1,
            "sha256": hashlib.sha256(b).hexdigest(), "approx_tokens": max(1, len(text) // 4)}


def _versions():
    v = {"python": platform.python_version()}
    for mod in ("numpy", "arc_agi", "arcengine"):
        try:
            import importlib.metadata as m
            v[mod] = m.version(mod.replace("_", "-"))
        except Exception:
            v[mod] = None
    return v


def provenance(model, options, seeds, budget, digest=None):
    """Reproducibility metadata for the results `env` block. `digest` is best-effort (None ok)."""
    return {"model": model, "options": dict(options or {}), "seeds": list(seeds),
            "budget": dict(budget or {}), "digest": digest, "versions": _versions()}


def log_run(path, record):
    """Append one run record as a JSON line (runs.jsonl-style)."""
    import pathlib
    p = pathlib.Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps(record) + "\n")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH="$PWD:$PWD/experiments:$PWD/scratch_arc/agent" python3 -m pytest tests/test_e119_sweep.py -k "digest or provenance or log_run" -q` → PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/e119/trace.py tests/test_e119_sweep.py
git commit -m "feat(e119): provenance capture + runs.jsonl trace logger"
```

---

### Task 3: Sweep driver + aggregation

**Files:**
- Create: `experiments/e119_macro_sweep.py`
- Test: `tests/test_e119_sweep.py`

**Interfaces:**
- Consumes: `solve.solve_game`, `trace.{provenance,prompt_digest,log_run}`, `e119_slm_solver._real_make`, `common.save_results`.
- Produces: `run_sweep(games, seeds, make, llm_factory, budget, arms=("search","random-macro","macro")) -> payload`. For each (game, arm), runs all seeds and aggregates `{levels_mean, levels_var, k_solved, m}`; `search` runs once (deterministic). Returns `{arms, games, by_game_arm, provenance, summary}`.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_e119_sweep.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH="$PWD:$PWD/experiments:$PWD/scratch_arc/agent" python3 -m pytest tests/test_e119_sweep.py -k run_sweep -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'e119_macro_sweep'`.

- [ ] **Step 3: Write minimal implementation**

```python
# experiments/e119_macro_sweep.py
"""E119 macro slot — 3-arm (control / random-macro / SLM-macro) x m-seed sweep on the signal-bearing
procedure-walls. SLM arm reported as a distribution; every banked solve anchored to a replay.
  arc venv:  PYTHONPATH="$PWD/scratch_arc/agent" .venv/bin/python experiments/e119_macro_sweep.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))   # let 'import e119'/'common' work
from e119 import solve, trace
from common import save_results

HEADROOM = ["tr87", "re86", "sb26", "cn04"]
BUDGET = {"max_nodes": 6000, "max_depth": 60}
MODEL = "qwen2.5-coder:7b"
LOGDIR = pathlib.Path(__file__).resolve().parent / "results" / "e119_logs"


def _agg(levels_list):
    n = len(levels_list); mean = sum(levels_list) / n if n else 0.0
    var = sum((x - mean) ** 2 for x in levels_list) / n if n else 0.0
    return {"levels_mean": round(mean, 3), "levels_var": round(var, 3),
            "k_solved": sum(1 for x in levels_list if x > 0), "m": n}


def _real_make(gid):
    from e119_slm_solver import _real_make as rm
    return rm(gid)


def run_sweep(games, seeds, make=_real_make, llm_factory=None, budget=None,
              arms=("search", "random-macro", "macro")):
    budget = budget or BUDGET
    by = {}
    for gid in games:
        by[gid] = {}
        for arm in arms:
            mode = "search" if arm == "search" else arm
            seed_set = [0] if arm == "search" else seeds       # control is deterministic: 1 run
            levels = []
            for s in seed_set:
                llm = None if arm == "search" else (llm_factory(s) if llm_factory else None)
                try:
                    r = solve.solve_game(make(gid), llm=llm, mode=mode, budget=budget,
                                         logdir=LOGDIR, make=make, seed=s)
                    levels.append(r["levels"])
                except Exception as e:
                    levels.append(0)
            by[gid][arm] = _agg(levels)
    prov = trace.provenance(MODEL, {}, seeds, budget)
    summary = {g: {a: by[g][a]["k_solved"] for a in arms} for g in games}
    return {"arms": list(arms), "games": list(games), "by_game_arm": by,
            "provenance": prov, "summary": summary}


def main():
    games = sys.argv[1].split(",") if len(sys.argv) > 1 else HEADROOM
    import openworld as O
    from e119 import slm as _slm
    seeds = [0, 1, 2, 3, 4]
    def llm_factory(seed):
        return O.OllamaLLM(model=MODEL, options={**_slm.llm_options(MODEL), "seed": seed})
    payload = run_sweep(games, seeds=seeds, llm_factory=llm_factory)
    save_results("e119_macro_sweep", payload)               # SAVE before asserts (CLAUDE.md)
    assert all(arm in payload["arms"] for arm in ("search", "random-macro", "macro"))
    for g in games:
        b = payload["by_game_arm"][g]
        print(f"{g}: search={b['search']['k_solved']}/1  random={b['random-macro']['k_solved']}/{b['random-macro']['m']}  "
              f"macro={b['macro']['k_solved']}/{b['macro']['m']} (levels {b['macro']['levels_mean']}±{b['macro']['levels_var']})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH="$PWD:$PWD/experiments:$PWD/scratch_arc/agent" python3 -m pytest tests/test_e119_sweep.py -k run_sweep -q` → PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/e119_macro_sweep.py tests/test_e119_sweep.py
git commit -m "feat(e119): 3-arm x m-seed macro sweep driver + aggregation"
```

---

### Task 4: Banked-solve re-verifier + the real run

**Files:**
- Create: `experiments/e119/reverify.py`
- Test: `tests/test_e119_sweep.py`
- Produce (run step): `experiments/results/e119_macro_sweep.json`, `experiments/e119/PROGRESS.md` update.

**Interfaces:**
- Produces: `reverify_solves(logdir, make) -> {"ok": int, "n": int, "fail": list}`. Loads every `*_solved.json` in `logdir`, replays its `actions` on a **fresh** env, confirms `levels >= 1`.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_e119_sweep.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH="$PWD:$PWD/experiments:$PWD/scratch_arc/agent" python3 -m pytest tests/test_e119_sweep.py -k reverify -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'e119.reverify'`.

- [ ] **Step 3: Write minimal implementation**

```python
# experiments/e119/reverify.py
"""Re-verify every banked E119 solve: replay its action sequence on a FRESH env and confirm a
level completes. The deterministic anchor — a banked solve is genuine regardless of how the
(stochastic) SLM arm reached it."""
import json, pathlib
from e119 import planner


def reverify_solves(logdir, make):
    d = pathlib.Path(logdir); ok = 0; n = 0; fail = []
    for f in sorted(d.glob("*_solved.json")):
        rec = json.loads(f.read_text())
        gid = rec["game"]; actions = [tuple(a) for a in rec["actions"]]
        n += 1
        try:
            reached, _ = planner.replay_levels(make(gid), actions)
            if reached >= 1:
                ok += 1
            else:
                fail.append(gid)
        except Exception:
            fail.append(gid)
    return {"ok": ok, "n": n, "fail": fail}
```

Then wire it into `experiments/e119_macro_sweep.py` `main()` — after `save_results`, re-verify and print:

```python
    from e119 import reverify
    rv = reverify.reverify_solves(LOGDIR, _real_make)
    print(f"[reverify] {rv['ok']}/{rv['n']} banked solves replay-confirmed; fail={rv['fail']}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH="$PWD:$PWD/experiments:$PWD/scratch_arc/agent" python3 -m pytest tests/test_e119_sweep.py -q` → all pass. Then full E119 suite → all pass.

- [ ] **Step 5: Commit**

```bash
git add experiments/e119/reverify.py experiments/e119_macro_sweep.py tests/test_e119_sweep.py
git commit -m "feat(e119): banked-solve re-verifier wired into the sweep"
```

- [ ] **Step 6: Run the real sweep (requires the `.venv` with arc_agi/arcengine + Ollama)**

First a one-model signal check on the two strong-signal games (cost control):
Run: `PYTHONPATH="$PWD/scratch_arc/agent" .venv/bin/python experiments/e119_macro_sweep.py tr87,re86 2>&1 | grep -vE "INFO|arcprize|font cache"`
Expected: per-game `search=/random=/macro=` k/m lines + a `[reverify]` line + `experiments/results/e119_macro_sweep.json`. Long-running (3 arms × 5 seeds × best-first search + LLM calls) — run backgrounded.

- [ ] **Step 7: Record the result + verdict**

Append a "Step 7 — macro slot 3-arm sweep" section to `experiments/e119/PROGRESS.md`: the per-game/arm table (k/m + levels mean±var), the re-verify count, and the falsifiable verdict — **does macro > random-macro and macro > search on ≥1 game** (a real SLM-attributable solve), or a clean negative (the documented boundary). Commit `experiments/results/e119_macro_sweep.json` + the PROGRESS.md update.

---

## Self-Review

- **Spec coverage (measurement + reproducibility sections):** 3-arm matched-budget sweep → Task 3; random-macro arm → Task 1; m-seed mean±variance + k/m distribution reporting → Task 3 `_agg`; provenance pinning (model/options/seeds/budget/versions/digest) → Task 2 + Task 3 `provenance`; per-run trace logging → Task 2 `log_run` (wired as the solver already writes per-game logs; the sweep writes the provenance/summary JSON); banked-solve re-verifier → Task 4; one-model-first cost control → Task 4 Step 6; falsifiable success criterion + clean-negative → Task 4 Step 7; scope tr87/re86 (+sb26/cn04) → `HEADROOM`. The per-call prompt/completion logging to `arc3_traces` is satisfied at the granularity the harness needs (provenance + per-run record + the banked solution trace); full agent-tier transcript fields are N/A for a local-Ollama inference run.
- **Placeholder scan:** none — every code/test step is complete.
- **Type consistency:** `solve_game(..., mode, seed)` and `_macro_fallback(..., mode, seed)` consistent (Task 1). `provenance(...) -> dict` consumed by `run_sweep` (Task 3) and emitted in the payload. `run_sweep(...) -> {by_game_arm, provenance, summary, ...}` consumed by `main()` + the sweep test. `reverify_solves(logdir, make) -> {ok,n,fail}` consumed by `main()` + its test. Arms tuple `("search","random-macro","macro")` consistent across driver, aggregation, and tests; `random-macro` mode handled in `solve_game`/`_macro_fallback`.
