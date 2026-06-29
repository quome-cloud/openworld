# E119 Phase 0 — Proxy-Signal Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run the deterministic Phase 0 gating probe that measures whether any macro selection signal (subgoal-proxy *or* novelty) carries directional information on the zero-reward procedure-walls, producing an auditable GO/No-Go decision for the macro slot.

**Architecture:** A no-LLM probe that reuses existing E119 perception, search, and predicate code. Per game it: (1) runs blind BFS to budget and records reachable-state stats + whether the frontier exhausts (= novelty headroom); (2) enumerates the `reach/count/align` predicate DSL over observed colors and counts satisfiability; (3) for each satisfiable-but-false-at-start predicate, runs best-first search and measures depth/novelty gain vs blind. Results are written to JSON **before** any assert; a pure gate function turns the primary game's signals into GO/No-Go.

**Tech Stack:** Python ≥3.12, numpy, the repo `.venv` with `arc_agi`/`arcengine`, existing `experiments/e119/*` modules. No new third-party deps.

## Global Constraints

- **`save_results` BEFORE any assert** (CLAUDE.md) — a failed check must never lose the run.
- **No LLM in Phase 0** — fully deterministic; results are reported as exact point facts.
- **Run command:** `PYTHONPATH="$PWD/scratch_arc/agent" .venv/bin/python experiments/e119_proxy_probe.py` (the driver self-adds `experiments/`; `arc3_harness` is on `scratch_arc/agent`).
- **Unit tests use numpy + mock games only** — no env, no Ollama. Run via system `python3 -m pytest` with `PYTHONPATH="$PWD:$PWD/experiments:$PWD/scratch_arc/agent"`.
- **Headroom set:** `g50t` (primary), `tr87`, `re86`, `sb26`, `cn04`. **Exclude** `sc25` (confirmed inert wall) and `bp35` (pruner, not macro).
- **Budget (match the solver):** `{"max_nodes": 6000, "max_depth": 60}`.
- **Reuse, don't reimplement:** `slm.compile_predicate`, `slm.satisfiable`, `perceive.{probe,status_mask,state_key,object_json}`, `planner._frame_after`, `solve.{_candidates_fn,_PrefixGame}`, `e119_slm_solver._real_make`, `common.save_results`.
- Do **not** modify the verified `planner.search_level`; the probe carries its own instrumented mirror.

---

### Task 1: Predicate enumeration + satisfiability scan

**Files:**
- Create: `experiments/e119/proxy_probe.py`
- Test: `tests/test_e119_proxy.py`

**Interfaces:**
- Consumes: `slm.satisfiable(pred, frames) -> bool`, `slm.compile_predicate(pred) -> fn(frame)->bool`.
- Produces: `enumerate_predicates(frames) -> list[dict]`; `scan_satisfiable(preds, frames) -> list[dict]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_e119_proxy.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH="$PWD:$PWD/experiments:$PWD/scratch_arc/agent" python3 -m pytest tests/test_e119_proxy.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'e119.proxy_probe'`.

- [ ] **Step 3: Write minimal implementation**

```python
# experiments/e119/proxy_probe.py
"""E119 Phase 0 — deterministic probe: does any macro selection signal carry directional
information on the zero-reward procedure-walls? No LLM; reuses existing perception/search/DSL."""
import numpy as np
from e119 import slm


def enumerate_predicates(frames):
    """All reach/count/align predicates over the colors and per-color counts observed in `frames`."""
    grids = [np.asarray(f).reshape(64, 64) for f in frames]
    colors = sorted({int(c) for g in grids for c in np.unique(g)})
    preds = [{"type": "reach", "color": c} for c in colors]
    for c in colors:
        for k in sorted({int((g == c).sum()) for g in grids}):
            for op in ("==", ">=", "<="):
                preds.append({"type": "count", "color": c, "op": op, "k": k})
    for i, a in enumerate(colors):
        for b in colors[i + 1:]:
            preds.append({"type": "align", "a": a, "b": b})
    return preds


def scan_satisfiable(preds, frames):
    """Subset of `preds` ever true on some frame in `frames`."""
    return [p for p in preds if slm.satisfiable(p, frames)]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH="$PWD:$PWD/experiments:$PWD/scratch_arc/agent" python3 -m pytest tests/test_e119_proxy.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add experiments/e119/proxy_probe.py tests/test_e119_proxy.py
git commit -m "feat(e119): Phase 0 predicate enumeration + satisfiability scan"
```

---

### Task 2: Instrumented search with stats (blind BFS + best-first)

**Files:**
- Modify: `experiments/e119/proxy_probe.py`
- Test: `tests/test_e119_proxy.py`

**Interfaces:**
- Consumes: `planner._frame_after(game, seq) -> (frame, levels, done)`.
- Produces: `search_stats(game, candidates_fn, key_fn, budget, score_fn=None) -> {"nodes","states","max_depth","frontier_exhausted","solved"}`. Mirrors `planner.search_level` (BFS when `score_fn is None`, else best-first) but returns stats instead of an action sequence.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_e119_proxy.py
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


def _corridor_helpers():
    cands = lambda f: [(7,), (1,)]
    key = lambda f: int(np.asarray(f).reshape(64, 64)[0].argmax())  # pos is the only state
    return cands, key


def test_search_stats_blind_exhausts_small_corridor():
    cands, key = _corridor_helpers()
    s = proxy_probe.search_stats(CorridorGame(L=6), cands, key, {"max_nodes": 500, "max_depth": 20})
    assert s["states"] == 6 and s["frontier_exhausted"] is True and s["solved"] is False


def test_search_stats_guided_reaches_depth_faster_than_blind():
    cands, key = _corridor_helpers()
    budget = {"max_nodes": 8, "max_depth": 30}          # tight: cuts off before full exploration
    blind = proxy_probe.search_stats(CorridorGame(L=20), cands, key, budget, None)
    far = lambda f: 1.0 if int(np.asarray(f).reshape(64, 64)[0].argmax()) >= 15 else 0.0
    guided = proxy_probe.search_stats(CorridorGame(L=20), cands, key, budget, far)
    assert guided["max_depth"] > blind["max_depth"]      # best-first dives toward the goal
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH="$PWD:$PWD/experiments:$PWD/scratch_arc/agent" python3 -m pytest tests/test_e119_proxy.py -k search_stats -q`
Expected: FAIL — `AttributeError: module 'e119.proxy_probe' has no attribute 'search_stats'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to experiments/e119/proxy_probe.py (top: add imports)
import heapq
from collections import deque
from e119 import planner


def search_stats(game, candidates_fn, key_fn, budget, score_fn=None):
    """Instrumented mirror of planner.search_level. BFS if score_fn is None, else best-first.
    Returns exploration stats; frontier_exhausted=True means the reachable state space was
    fully explored within budget (no novelty headroom)."""
    game.reset(); base = game.levels
    seen = {key_fn(game.frame)}
    nodes = 0; max_depth = 0; solved = False
    if score_fn is None:
        frontier = deque([[]]); pop = frontier.popleft; push = frontier.append
    else:
        counter = 0; heap = [(-score_fn(game.frame), 0, [])]
        def pop(): return heapq.heappop(heap)[2]
        def push(seq):
            nonlocal counter
            counter += 1
            f, _, _ = planner._frame_after(game, seq)
            heapq.heappush(heap, (-score_fn(f), counter, seq))
        frontier = heap
    while frontier and nodes < budget["max_nodes"]:
        seq = pop()
        if len(seq) >= budget["max_depth"]:
            continue
        frame, _, _ = planner._frame_after(game, seq)
        for act in candidates_fn(frame):
            nodes += 1
            child = seq + [act]
            f2, levels2, _ = planner._frame_after(game, child)
            if levels2 > base:
                solved = True; break
            k = key_fn(f2)
            if k in seen:
                continue
            seen.add(k); push(child); max_depth = max(max_depth, len(child))
            if nodes >= budget["max_nodes"]:
                break
        if solved or nodes >= budget["max_nodes"]:
            break
    return {"nodes": nodes, "states": len(seen), "max_depth": max_depth,
            "frontier_exhausted": (len(frontier) == 0 and not solved), "solved": solved}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH="$PWD:$PWD/experiments:$PWD/scratch_arc/agent" python3 -m pytest tests/test_e119_proxy.py -k search_stats -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add experiments/e119/proxy_probe.py tests/test_e119_proxy.py
git commit -m "feat(e119): Phase 0 instrumented search stats (blind + best-first)"
```

---

### Task 3: Per-game probe assembly

**Files:**
- Modify: `experiments/e119/proxy_probe.py`
- Test: `tests/test_e119_proxy.py`

**Interfaces:**
- Consumes: `perceive.{probe,status_mask,state_key}`, `solve._candidates_fn`, `solve._PrefixGame`, `slm.compile_predicate`, and `enumerate_predicates`/`scan_satisfiable`/`search_stats` from Tasks 1–2.
- Produces: `probe_game(game, budget, max_preds=20) -> dict` with keys `game, modality, n_satisfiable, n_gradient, blind, best_depth_gain, best_novel_gain, novelty_headroom`.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_e119_proxy.py
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
```

(Note: `CorridorGame` already exposes `avail`, `frame`, `levels`, `gid`; `probe_game` uses the real `perceive.probe`/`status_mask`/`_candidates_fn`, which operate on its 64×64 frames.)

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH="$PWD:$PWD/experiments:$PWD/scratch_arc/agent" python3 -m pytest tests/test_e119_proxy.py -k probe_game -q`
Expected: FAIL — `AttributeError: module 'e119.proxy_probe' has no attribute 'probe_game'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to experiments/e119/proxy_probe.py (add imports: from e119 import perceive, solve)
def probe_game(game, budget, max_preds=20):
    """Probe one game: blind-search stats, predicate satisfiability, and the best depth/novelty
    gain from pursuing a satisfiable-but-false-at-start predicate (the directionality test)."""
    game.reset()
    trans = perceive.probe(game)
    frames = [t["before"] for t in trans] + [t["after"] for t in trans]
    mask = perceive.status_mask(frames)
    key_fn = lambda f, m=mask: perceive.state_key(f, m)
    cands = solve._candidates_fn(game, mask)
    start = trans[0]["before"]

    blind = search_stats(solve._PrefixGame(game, []), cands, key_fn, budget, None)

    preds = enumerate_predicates(frames)
    satisf = scan_satisfiable(preds, frames)
    gradient = [p for p in satisf if not slm.compile_predicate(p)(start)][:max_preds]

    best_depth_gain, best_novel_gain = 0, 0.0
    for p in gradient:
        score = lambda f, pp=p: 1.0 if slm.compile_predicate(pp)(f) else 0.0
        g = search_stats(solve._PrefixGame(game, []), cands, key_fn, budget, score)
        best_depth_gain = max(best_depth_gain, g["max_depth"] - blind["max_depth"])
        novel = max(0, g["states"] - blind["states"])
        best_novel_gain = max(best_novel_gain, (novel / blind["states"]) if blind["states"] else 0.0)

    avail = list(getattr(game, "avail", [1, 2, 3, 4, 5, 7]))
    modality = "click" if avail == [6] else ("dir" if 6 not in avail else "mixed")
    return {"game": getattr(game, "gid", type(game).__name__), "modality": modality,
            "n_satisfiable": len(satisf), "n_gradient": len(gradient), "blind": blind,
            "best_depth_gain": int(best_depth_gain), "best_novel_gain": round(best_novel_gain, 3),
            "novelty_headroom": not blind["frontier_exhausted"]}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH="$PWD:$PWD/experiments:$PWD/scratch_arc/agent" python3 -m pytest tests/test_e119_proxy.py -k probe_game -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/e119/proxy_probe.py tests/test_e119_proxy.py
git commit -m "feat(e119): Phase 0 per-game probe assembly"
```

---

### Task 4: GO/No-Go gate

**Files:**
- Modify: `experiments/e119/proxy_probe.py`
- Test: `tests/test_e119_proxy.py`

**Interfaces:**
- Consumes: rows produced by `probe_game`.
- Produces: `decide_go(rows, primary="g50t") -> {"go": bool, "signal": str, "reason": str}`. `signal` is `"novelty"` (default when both qualify, per the brainstorm), `"subgoal"`, or `"none"`.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_e119_proxy.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH="$PWD:$PWD/experiments:$PWD/scratch_arc/agent" python3 -m pytest tests/test_e119_proxy.py -k decide_go -q`
Expected: FAIL — `AttributeError: ... 'decide_go'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to experiments/e119/proxy_probe.py
def decide_go(rows, primary="g50t"):
    """GO iff the primary game shows either a non-flat subgoal proxy OR novelty headroom.
    Default the macro selection signal to novelty when both qualify (brainstorm decision)."""
    pr = next((r for r in rows if r["game"] == primary), None)
    if pr is None:
        return {"go": False, "signal": "none", "reason": f"primary {primary} missing from rows"}
    subgoal = pr["n_satisfiable"] >= 1 and (pr["best_depth_gain"] >= 2 or pr["best_novel_gain"] >= 0.10)
    novelty = bool(pr["novelty_headroom"])
    signal = "novelty" if novelty else ("subgoal" if subgoal else "none")
    return {"go": bool(novelty or subgoal), "signal": signal,
            "reason": (f"{primary}: subgoal={subgoal} (depth_gain={pr['best_depth_gain']}, "
                       f"novel_gain={pr['best_novel_gain']}, n_sat={pr['n_satisfiable']}), "
                       f"novelty_headroom={novelty}")}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH="$PWD:$PWD/experiments:$PWD/scratch_arc/agent" python3 -m pytest tests/test_e119_proxy.py -q`
Expected: PASS (all proxy tests).

- [ ] **Step 5: Commit**

```bash
git add experiments/e119/proxy_probe.py tests/test_e119_proxy.py
git commit -m "feat(e119): Phase 0 GO/No-Go gate"
```

---

### Task 5: Driver — run the headroom set, save results before asserts, emit GO/No-Go

**Files:**
- Create: `experiments/e119_proxy_probe.py`
- Test: `tests/test_e119_proxy.py`

**Interfaces:**
- Consumes: `proxy_probe.{probe_game,decide_go}`, `e119_slm_solver._real_make`, `common.save_results`.
- Produces: `run_probe(games, make, budget) -> payload` and a `main()` that writes `experiments/results/e119_proxy_probe.json` and prints the decision.

- [ ] **Step 1: Write the failing test** (driver aggregation, env-free via a fake make)

```python
# add to tests/test_e119_proxy.py
def test_run_probe_aggregates_and_decides(tmp_path):
    import e119_proxy_probe as drv
    def fake_make(gid):
        g = CorridorGame(L=6); g.gid = gid; return g
    payload = drv.run_probe(["g50t", "tr87"], make=fake_make,
                            budget={"max_nodes": 500, "max_depth": 20})
    assert payload["n_games"] == 2
    assert payload["decision"]["go"] in (True, False)
    assert any(r["game"] == "g50t" for r in payload["rows"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH="$PWD:$PWD/experiments:$PWD/scratch_arc/agent" python3 -m pytest tests/test_e119_proxy.py -k run_probe -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'e119_proxy_probe'`.

- [ ] **Step 3: Write minimal implementation**

```python
# experiments/e119_proxy_probe.py
"""E119 Phase 0 driver: probe the headroom set and emit a GO/No-Go for the macro slot.
  arc venv:  PYTHONPATH="$PWD/scratch_arc/agent" .venv/bin/python experiments/e119_proxy_probe.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))   # let 'import e119'/'common' work
from e119 import proxy_probe
from common import save_results

HEADROOM = ["g50t", "tr87", "re86", "sb26", "cn04"]          # exclude sc25 (wall), bp35 (pruner)
BUDGET = {"max_nodes": 6000, "max_depth": 60}


def _real_make(gid):
    from e119_slm_solver import _real_make as rm
    return rm(gid)


def run_probe(games, make=_real_make, budget=None):
    budget = budget or BUDGET
    rows = []
    for gid in games:
        try:
            rows.append(proxy_probe.probe_game(make(gid), budget))
        except Exception as e:
            rows.append({"game": gid, "error": str(e)[:160]})
    decision = proxy_probe.decide_go(rows)
    return {"phase": "e119_phase0_proxy", "n_games": len(rows), "rows": rows, "decision": decision}


def main():
    games = sys.argv[1].split(",") if len(sys.argv) > 1 else HEADROOM
    payload = run_probe(games)
    save_results("e119_proxy_probe", payload)              # SAVE before asserts (CLAUDE.md)
    assert all(("error" in r) or "best_depth_gain" in r for r in payload["rows"]), "malformed row"
    d = payload["decision"]
    print(f"[e119 phase0] decision={'GO' if d['go'] else 'NO-GO'} signal={d['signal']}")
    print(f"  reason: {d['reason']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH="$PWD:$PWD/experiments:$PWD/scratch_arc/agent" python3 -m pytest tests/test_e119_proxy.py -q`
Expected: PASS (all proxy tests).

- [ ] **Step 5: Commit**

```bash
git add experiments/e119_proxy_probe.py tests/test_e119_proxy.py
git commit -m "feat(e119): Phase 0 driver — headroom probe + GO/No-Go (save before assert)"
```

---

### Task 6: Run Phase 0 for real and record the decision

**Files:**
- Produce: `experiments/results/e119_proxy_probe.json` (run artifact)
- Modify: `experiments/e119/PROGRESS.md` (append the Phase 0 result + decision)

**Interfaces:** none (execution + documentation).

- [ ] **Step 1: Run the probe on the real env (requires the `.venv` with arc_agi/arcengine)**

Run: `PYTHONPATH="$PWD/scratch_arc/agent" .venv/bin/python experiments/e119_proxy_probe.py 2>&1 | grep -vE "INFO|arcprize|font cache"`
Expected: a `[e119 phase0] decision=GO|NO-GO signal=...` line and `experiments/results/e119_proxy_probe.json` written. (Long-ish: 5 games × multiple best-first searches; run backgrounded if needed.)

- [ ] **Step 2: Record the result in PROGRESS.md**

Append a "Step 6 (b) — Phase 0 proxy probe" section: the per-game table (`n_satisfiable`, `best_depth_gain`, `best_novel_gain`, `novelty_headroom`) and the GO/No-Go decision with its `reason`. State plainly: GO → the macro-slot plan is written next; NO-GO → report the flat-proxy boundary as the finding and stop.

- [ ] **Step 3: Commit**

```bash
git add experiments/results/e119_proxy_probe.json experiments/e119/PROGRESS.md
git commit -m "experiment(e119): Phase 0 proxy-probe result + GO/No-Go decision"
```

---

## After Phase 0

- **NO-GO:** stop. The finding ("selection signals are flat on the procedure-walls; first-reward capture needs a richer target than `reach/count/align` or pure novelty") is the reportable boundary result — update `RESULTS.md` and done.
- **GO:** write the **macro-slot implementation plan** (`propose_macros` with object-referential ops + compiler + grader, the `solve.py` stall hook, `--mode macro`/`macro+slm`, the 3-arm seeded sweep with the reproducibility protocol, and the banked-solve re-verifier), informed by which `signal` won and whether the 2–8 macro length holds for the primary game.

## Self-Review

- **Spec coverage:** Phase 0 procedure (satisfiability scan → Task 1; directionality test → Tasks 2–3; novelty headroom → Task 2/3 `frontier_exhausted`; save-before-assert → Task 5; GO/No-Go gate → Task 4; run + record → Task 6; headroom set + exclusions → Global Constraints + Task 5). Macro-slot build is intentionally deferred to the post-GO plan (gated by spec). sc25 excluded; harness-layer caveat is a separate flag, not in this plan.
- **Placeholder scan:** none — every code/test step has complete content.
- **Type consistency:** `search_stats` keys (`nodes/states/max_depth/frontier_exhausted/solved`) consumed unchanged by `probe_game`; `probe_game` row keys consumed unchanged by `decide_go` and the driver assert; `decide_go` return shape consumed by the driver print. Consistent across tasks.
