# E119 Macro Slot — Plan A (mechanism) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the SLM macro/procedure mechanism for E119 — when blind search stalls, the SLM proposes a short action *procedure* (graded behaviorally, ranked by a subgoal proxy, banked only on replay-verified level-up) — wired into `solve.py` behind a `--mode macro` / `--mode macro+slm` flag.

**Architecture:** A new `experiments/e119/macro.py` holds an action/object-referential op grammar + compiler, an SLM proposer (best-of-N + behavioral clustering + τ-abstention, mirroring `slm.propose_subgoal`), a subgoal-proxy ranker, and a seeded random-macro baseline. `solve.py` gains a post-stall hook: when `planner.search_level` returns `None`, it synthesizes a subgoal, proposes+ranks macros, replay-verifies each from the banked prefix on a fresh env, and banks the first that raises `levels`. The env is ground truth throughout; macros only add reachable states.

**Tech Stack:** Python ≥3.12, numpy, existing `experiments/e119/*` modules, `openworld.OllamaLLM`/`MockLLM`. No new third-party deps. This plan (Plan A) is env-free-testable with `MockLLM` + mock games; the real-model 3-arm sweep + reproducibility is Plan B.

## Global Constraints

- Work from: /Users/jeniaquome/code/quome/openworld. Core stays zero-dep; `experiments/` may use numpy.
- **Macro form:** action/object-referential ops compiled to primitives. Op vocabulary: `"aN"` (directional ACTION N, N in the game's `avail`, N≠6), `"aN xK"` (repeat K, K capped at 4), `"click #I"` (click object I's centroid; only if 6 in `avail`), `"click #I xK"`. Directional games (the tr87/re86 targets) use `aN` ops; the SLM is grounded by the probe's per-action contrastive diffs in the prompt, NOT by semantic directions. **Never put the raw grid in the prompt** — relational only (object_json + diffs + subgoal + stall context).
- **Grading:** strict level-up is the ONLY thing banked (a verified solve). The **subgoal proxy ranks** which macros get tried (Phase 0 chose subgoal-proxy over novelty). Macros only ADD reachable states — `macro-search ⊒ control` always.
- **Verify on a FRESH env** for banking (per the Bug #2 fix): replay the banked action list on a newly-made game, never the polluted reused one.
- **Macro length 2–8 ops.** Abstain (propose nothing) when best-of-N doesn't agree — fall back to plain search.
- **Reuse, don't reimplement:** `slm.propose_subgoal`/`compile_predicate`, `perceive.{probe,object_json,contrastive_diff,status_mask,state_key}`, `planner.{search_level,_frame_after,replay_levels}`, `solve.{_candidates_fn,_PrefixGame}`, `e119_slm_solver._real_make`. Do NOT modify `planner.search_level`.
- **Tests env-free:** `MockLLM` + numpy mock games only (no Ollama/arc_agi). Run: `PYTHONPATH="$PWD:$PWD/experiments:$PWD/scratch_arc/agent" python3 -m pytest tests/test_e119_macro.py -q` (system python3; `.venv` lacks pytest).
- **Build scope (target games):** tr87, re86 (Plan B runs them). Plan A is game-agnostic mechanism + mocks.

## File Structure

- `experiments/e119/macro.py` (new) — `compile_macro`, `propose_macros`, `rank_macros`, `propose_random_macros`, the prompt + op grammar. One responsibility: turning a stalled state into ranked, compiled candidate macros.
- `experiments/e119/solve.py` (modify) — `solve_game` gains modes `"macro"`/`"macro+slm"` and the post-stall hook. No other file owns the search loop.
- `experiments/e119_slm_solver.py` (modify) — argparse `--mode` choices + construct `llm` for any non-`search` mode.
- `tests/test_e119_macro.py` (new) — all Plan A unit tests.

---

### Task 1: Op grammar + compiler (`compile_macro`)

**Files:**
- Create: `experiments/e119/macro.py`
- Test: `tests/test_e119_macro.py`

**Interfaces:**
- Produces: `compile_macro(ops, obj_json, avail) -> list[tuple]`. `ops` is a list of op strings; returns a flat list of action tuples (`(N,)` or `(6,x,y)`). Unresolvable ops are dropped; an all-dropped macro returns `[]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_e119_macro.py
from e119 import macro


OJ = {"bg": 0, "objects": [{"id": 0, "color": 5, "centroid": (10, 20)},
                           {"id": 1, "color": 3, "centroid": (40, 8)}], "relations": []}


def test_compile_directional_and_repeat():
    assert macro.compile_macro(["a1", "a3 x2"], OJ, [1, 2, 3, 4]) == [(1,), (3,), (3,)]


def test_compile_click_resolves_object_centroid():
    # centroid is (row, col) = (y, x); click tuple is (6, x=col, y=row)
    assert macro.compile_macro(["click #0"], OJ, [6]) == [(6, 20, 10)]


def test_compile_drops_unresolvable_ops():
    # a5 not in avail -> dropped; click when 6 not in avail -> dropped; missing obj -> dropped
    assert macro.compile_macro(["a5", "a2", "click #9"], OJ, [1, 2, 3, 4]) == [(2,)]
    assert macro.compile_macro(["click #0"], OJ, [1, 2]) == []


def test_compile_caps_repeat_at_four():
    assert macro.compile_macro(["a1 x99"], OJ, [1]) == [(1,), (1,), (1,), (1,)]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH="$PWD:$PWD/experiments:$PWD/scratch_arc/agent" python3 -m pytest tests/test_e119_macro.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'e119.macro'`.

- [ ] **Step 3: Write minimal implementation**

```python
# experiments/e119/macro.py
"""E119 macro slot: object/action-referential op grammar + compiler, SLM proposer (best-of-N +
behavioral clustering + abstention), subgoal-proxy ranker, and a seeded random-macro baseline.
The env decides correctness; macros only ORDER/extend search and are replay-verified before banking."""
import json, re
import numpy as np

_MAX_REPEAT = 4


def _parse_op(op):
    """'a3 x2' -> ('a', 3, 2); 'click #1 x2' -> ('click', 1, 2); times defaults to 1."""
    m = re.match(r"\s*a(\d+)(?:\s*x\s*(\d+))?\s*$", op)
    if m:
        return ("a", int(m.group(1)), min(int(m.group(2) or 1), _MAX_REPEAT))
    m = re.match(r"\s*click\s*#(\d+)(?:\s*x\s*(\d+))?\s*$", op)
    if m:
        return ("click", int(m.group(1)), min(int(m.group(2) or 1), _MAX_REPEAT))
    return None


def compile_macro(ops, obj_json, avail):
    """Compile object/action-referential ops to primitive action tuples. Unresolvable ops dropped."""
    objs = {o["id"]: o for o in obj_json.get("objects", [])}
    out = []
    for op in ops:
        parsed = _parse_op(op) if isinstance(op, str) else None
        if parsed is None:
            continue
        kind, idx, times = parsed
        if kind == "a":
            if idx in avail and idx != 6:
                out += [(idx,)] * times
        else:  # click
            if 6 in avail and idx in objs:
                cy, cx = objs[idx]["centroid"]
                out += [(6, int(round(cx)), int(round(cy)))] * times
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH="$PWD:$PWD/experiments:$PWD/scratch_arc/agent" python3 -m pytest tests/test_e119_macro.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add experiments/e119/macro.py tests/test_e119_macro.py
git commit -m "feat(e119): macro op grammar + compiler"
```

---

### Task 2: SLM macro proposer (`propose_macros`)

**Files:**
- Modify: `experiments/e119/macro.py`
- Test: `tests/test_e119_macro.py`

**Interfaces:**
- Consumes: `compile_macro` (Task 1); `planner._frame_after`, `solve._PrefixGame`.
- Produces: `propose_macros(llm, game, prefix, obj_json, diffs, subgoal, avail, key_fn, k_max=8, n=6, tau=0.5) -> list[list[tuple]]`. Samples `n` op-lists from the LLM, compiles each (truncated to `k_max` actions), clusters by BEHAVIORAL effect (replayed endpoint masked-state + level delta) from the banked `prefix`, returns the cluster representatives whose top cluster clears τ — sorted by cluster mass — or `[]` (abstain) if none agree.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_e119_macro.py
import json, numpy as np
from openworld import MockLLM


class StepGame:
    """Each action 7 advances pos by 1 (deterministic). frame[0,pos]=4. No reward path here.
    Mirrors the Game/_PrefixGame surface propose_macros replays against."""
    def __init__(self): self.win = 1; self.gid = "step"; self.reset()
    def reset(self): self.pos = 0; self.levels = 0; self.done = False; self.avail = [7, 1]; self._r(); return self.frame
    def _r(self): g = np.zeros((64, 64), int); g[0, self.pos] = 4; self.frame = g
    def step(self, a, x=None, y=None):
        if a == 7 and self.pos < 63: self.pos += 1
        if a == 1 and self.pos > 0: self.pos -= 1
        self._r(); return self.frame


def _key(f): return int(np.asarray(f).reshape(64, 64)[0].argmax())


def test_propose_macros_returns_consensus_macro():
    # 4 of 6 samples agree on ["a7","a7"] (-> pos 2); they cluster, clear tau=0.5, and survive.
    replies = [json.dumps(["a7", "a7"])] * 4 + [json.dumps(["a1"]), json.dumps(["a7", "a7", "a7"])]
    llm = MockLLM(replies)
    macros = macro.propose_macros(llm, StepGame(), [], {"objects": []}, [], None,
                                  avail=[7, 1], key_fn=_key, k_max=8, n=6, tau=0.5)
    assert [(7,), (7,)] in macros            # the consensus macro survived
    assert macros[0] == [(7,), (7,)]         # ranked first by cluster mass


def test_propose_macros_abstains_on_disagreement():
    replies = [json.dumps(["a7"]), json.dumps(["a7", "a7"]), json.dumps(["a1"]),
               json.dumps(["a7", "a7", "a7"]), json.dumps(["a1", "a1"]), json.dumps(["a7", "a1"])]
    llm = MockLLM(replies)
    macros = macro.propose_macros(llm, StepGame(), [], {"objects": []}, [], None,
                                  avail=[7, 1], key_fn=_key, k_max=8, n=6, tau=0.6)
    assert macros == []                       # no cluster clears tau -> abstain
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH="$PWD:$PWD/experiments:$PWD/scratch_arc/agent" python3 -m pytest tests/test_e119_macro.py -k propose_macros -q`
Expected: FAIL — `AttributeError: module 'e119.macro' has no attribute 'propose_macros'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to experiments/e119/macro.py (add imports: from collections import defaultdict; from e119 import planner, solve)
from collections import defaultdict
from e119 import planner, solve

_PROMPT = (
    "Blind search STALLED on an interactive puzzle. Propose ONE short action PROCEDURE (a macro) "
    "to make progress. Relational scene:\n{oj}\nWhat each action did from the current state:\n{diffs}\n"
    "Goal to pursue: {subgoal}\n"
    'Reply ONLY a JSON list of {k} ops max. Ops: "aN" (do action N), "aN xK" (repeat K), '
    '"click #I" (click object I). Example: ["a7","a7","a1"].'
)


def _endpoint(game, prefix, macro_actions, key_fn):
    """Replay prefix+macro from reset on a FRESH _PrefixGame view; return (masked key, level delta)."""
    pg = solve._PrefixGame(game, prefix)
    base = pg.levels
    frame, levels, _ = planner._frame_after(pg, list(macro_actions))
    return key_fn(frame), levels - base


def propose_macros(llm, game, prefix, obj_json, diffs, subgoal, avail, key_fn,
                   k_max=8, n=6, tau=0.5):
    prompt = _PROMPT.format(oj=json.dumps(obj_json)[:1200], diffs=json.dumps(diffs)[:800],
                            subgoal=json.dumps(subgoal), k=k_max)
    clusters = defaultdict(list)      # behavioral signature -> [compiled macro, ...]
    drawn = 0
    for _ in range(n):
        try:
            ops = json.loads(re.search(r"\[.*\]", llm.ask(prompt), re.S).group(0))
            m = compile_macro(ops, obj_json, avail)[:k_max]
        except Exception:
            continue
        drawn += 1
        if not m:                     # empty/ungradeable macro discarded, not fatal
            continue
        try:
            sig = _endpoint(game, prefix, m, key_fn)
        except Exception:
            continue
        clusters[sig].append(m)
    if not clusters:
        return []
    ranked = sorted(clusters.values(), key=len, reverse=True)
    top = len(ranked[0])
    if drawn == 0 or top / drawn < tau:    # no consensus -> abstain
        return []
    return [reps[0] for reps in ranked if len(reps) >= 1]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH="$PWD:$PWD/experiments:$PWD/scratch_arc/agent" python3 -m pytest tests/test_e119_macro.py -k propose_macros -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add experiments/e119/macro.py tests/test_e119_macro.py
git commit -m "feat(e119): SLM macro proposer with behavioral clustering + abstention"
```

---

### Task 3: Subgoal-proxy ranker (`rank_macros`)

**Files:**
- Modify: `experiments/e119/macro.py`
- Test: `tests/test_e119_macro.py`

**Interfaces:**
- Consumes: `_endpoint` (Task 2), `slm.compile_predicate`.
- Produces: `rank_macros(macros, game, prefix, subgoal, key_fn, seen) -> list[list[tuple]]`. Orders macros: those whose replayed endpoint satisfies the `subgoal` predicate first, then those reaching a masked state not in `seen` (novelty tiebreak), preserving input order within a tier. `subgoal=None` ranks purely by novelty.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_e119_macro.py
from e119 import slm


def test_rank_macros_subgoal_satisfier_first():
    g = StepGame()
    # subgoal: reach color 5. StepGame never produces color 5, so make a game variant:
    class ColorAtThree(StepGame):
        def _r(self):
            v = 5 if self.pos == 3 else 4
            x = np.zeros((64, 64), int); x[0, self.pos] = v; self.frame = x
    sub = {"type": "reach", "color": 5}
    m_far = [(7,), (7,), (7,)]    # reaches pos 3 -> color 5 -> satisfies subgoal
    m_near = [(7,)]               # reaches pos 1 -> color 4 -> does not
    ranked = macro.rank_macros([m_near, m_far], ColorAtThree(), [], sub, _key, seen=set())
    assert ranked[0] == m_far     # subgoal-satisfier ranked first
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH="$PWD:$PWD/experiments:$PWD/scratch_arc/agent" python3 -m pytest tests/test_e119_macro.py -k rank_macros -q`
Expected: FAIL — `AttributeError: ... 'rank_macros'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to experiments/e119/macro.py (add import: from e119 import slm)
from e119 import slm


def rank_macros(macros, game, prefix, subgoal, key_fn, seen):
    """Order macros: subgoal-satisfying endpoints first, then novel (unseen) endpoints."""
    pred = slm.compile_predicate(subgoal) if subgoal else (lambda f: False)
    scored = []
    for i, m in enumerate(macros):
        pg = solve._PrefixGame(game, prefix)
        frame, _, _ = planner._frame_after(pg, list(m))
        sat = 1 if pred(frame) else 0
        novel = 1 if key_fn(frame) not in seen else 0
        scored.append((-sat, -novel, i, m))         # stable within tier via original index i
    scored.sort(key=lambda t: t[:3])
    return [t[3] for t in scored]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH="$PWD:$PWD/experiments:$PWD/scratch_arc/agent" python3 -m pytest tests/test_e119_macro.py -k rank_macros -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/e119/macro.py tests/test_e119_macro.py
git commit -m "feat(e119): subgoal-proxy macro ranker"
```

---

### Task 4: Random-macro baseline (`propose_random_macros`)

**Files:**
- Modify: `experiments/e119/macro.py`
- Test: `tests/test_e119_macro.py`

**Interfaces:**
- Produces: `propose_random_macros(avail, obj_json, k_max, count, rng) -> list[list[tuple]]`. `count` macros of random length in `[2, k_max]`, each action a random choice among directional `avail` actions (and `click #I` for present objects if 6 in `avail`), compiled. Deterministic given `rng` (a `random.Random`).

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_e119_macro.py
import random


def test_random_macros_seeded_and_bounded():
    rng1 = random.Random(0); rng2 = random.Random(0)
    a = macro.propose_random_macros([1, 2, 3, 4], {"objects": []}, k_max=8, count=5, rng=rng1)
    b = macro.propose_random_macros([1, 2, 3, 4], {"objects": []}, k_max=8, count=5, rng=rng2)
    assert a == b                                 # same seed -> identical
    assert len(a) == 5
    assert all(2 <= len(m) <= 8 for m in a)       # length bounds
    assert all(act[0] in (1, 2, 3, 4) for m in a for act in m)   # only avail directional actions
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH="$PWD:$PWD/experiments:$PWD/scratch_arc/agent" python3 -m pytest tests/test_e119_macro.py -k random_macros -q`
Expected: FAIL — `AttributeError: ... 'propose_random_macros'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to experiments/e119/macro.py
def propose_random_macros(avail, obj_json, k_max, count, rng):
    """Matched-budget baseline: `count` random op-lists (len 2..k_max), compiled. Seeded by `rng`."""
    dirs = [f"a{a}" for a in avail if a != 6]
    clicks = [f"click #{o['id']}" for o in obj_json.get("objects", [])] if 6 in avail else []
    vocab = dirs + clicks
    out = []
    for _ in range(count):
        length = rng.randint(2, k_max)
        ops = [rng.choice(vocab) for _ in range(length)] if vocab else []
        out.append(compile_macro(ops, obj_json, avail))
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH="$PWD:$PWD/experiments:$PWD/scratch_arc/agent" python3 -m pytest tests/test_e119_macro.py -k random_macros -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/e119/macro.py tests/test_e119_macro.py
git commit -m "feat(e119): seeded random-macro baseline"
```

---

### Task 5: Stall hook in `solve_game` + driver modes

**Files:**
- Modify: `experiments/e119/solve.py` (the `solve_game` while-loop and signature)
- Modify: `experiments/e119_slm_solver.py` (argparse `--mode` choices; `llm` for non-search modes)
- Test: `tests/test_e119_macro.py`

**Interfaces:**
- Consumes: `macro.{propose_macros,rank_macros,propose_random_macros}`, `slm.propose_subgoal`, `perceive.contrastive_diff`, `make` (fresh-env factory, already a `solve_game` param).
- Produces: `solve_game(..., mode in {"search","slm","macro","macro+slm"}, ...)` — on `search_level` returning `None` in a macro mode, runs the macro fallback; banks a macro only if a fresh-env replay of `actions + macro` raises levels. `"macro+slm"` also passes the subgoal `score_fn` to `search_level` (existing path); `"macro"` does not.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_e119_macro.py
class MacroGame:
    """Level 1 needs the exact 6-action sequence (7,7,7,7,7,7) (walk to pos 6). With a tight
    node budget blind BFS cannot assemble it, but a single banked macro of that sequence does."""
    def __init__(self): self.win = 1; self.gid = "mg"; self.reset()
    def reset(self): self.pos = 0; self.levels = 0; self.done = False; self.avail = [7, 1]; self._r(); return self.frame
    def _r(self): g = np.zeros((64, 64), int); g[0, self.pos] = 4; self.frame = g
    def step(self, a, x=None, y=None):
        if a == 7 and self.pos < 63: self.pos += 1
        if a == 1 and self.pos > 0: self.pos -= 1
        if self.pos == 6 and self.levels == 0: self.levels = 1; self.done = True
        self._r(); return self.frame


def test_macro_mode_banks_a_verified_macro_solve(tmp_path):
    from e119 import solve
    replies = [json.dumps(["a7", "a7", "a7", "a7", "a7", "a7"])] * 6   # consensus 6-step macro
    llm = MockLLM(replies)
    res = solve.solve_game(MacroGame(), llm=llm, mode="macro",
                           budget={"max_nodes": 3, "max_depth": 8},   # tight: blind cannot reach pos 6
                           make=lambda gid: MacroGame())
    assert res["levels"] == 1 and res["verified"] is True


def test_macro_mode_never_banks_unverified(tmp_path):
    from e119 import solve
    replies = [json.dumps(["a1", "a1"])] * 6      # macro that never raises levels
    llm = MockLLM(replies)
    res = solve.solve_game(MacroGame(), llm=llm, mode="macro",
                           budget={"max_nodes": 3, "max_depth": 8}, make=lambda gid: MacroGame())
    assert res["levels"] == 0 and res["verified"] is False     # honest stop, nothing banked
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH="$PWD:$PWD/experiments:$PWD/scratch_arc/agent" python3 -m pytest tests/test_e119_macro.py -k macro_mode -q`
Expected: FAIL — `solve_game` does not run a macro fallback, so `levels == 0` on the first test.

- [ ] **Step 3: Write minimal implementation**

In `experiments/e119/solve.py`, add `from e119 import macro` at the top with the other `e119` imports. Replace the body of the `while not game.done and game.levels < win:` loop's tail so that, after `seq = planner.search_level(...)`, a stall triggers the macro fallback. The current tail is:

```python
        if seq is None:
            break
        actions += seq
        # re-apply to advance the real game state for the next iteration
        game.reset()
        for a in actions: game.step(*a)
```

Replace with:

```python
        if seq is None:
            if mode in ("macro", "macro+slm") and llm is not None:
                seq = _macro_fallback(game, actions, trans, llm, key_fn, make)
            if seq is None:
                break
        actions += seq
        # re-apply to advance the real game state for the next iteration
        game.reset()
        for a in actions: game.step(*a)
```

Also extend the `score_fn` gate so the subgoal ordering runs for `slm` AND `macro+slm` (not `macro`):

```python
        if mode in ("slm", "macro+slm") and llm is not None:
```

(the rest of that block — building `frames`, `oj`, `subgoal`, `score_fn` — is unchanged; for `macro` mode `score_fn` stays `None`.)

Add the helper near `_PrefixGame`:

```python
def _macro_fallback(game, actions, trans, llm, key_fn, make):
    """On a stall, synthesize a subgoal, propose+rank macros, and return the FIRST macro whose
    fresh-env replay of (actions+macro) raises levels. Returns the macro (list of action tuples)
    or None (honest stop). The env decides correctness."""
    from e119 import macro
    avail = list(getattr(game, "avail", [1, 2, 3, 4, 5, 7]))
    oj = perceive.object_json(trans[0]["before"])
    diffs = [perceive.contrastive_diff(t["before"], t["after"]) for t in trans]
    subgoal = slm.propose_subgoal(llm, oj, [t["after"] for t in trans])
    cands = macro.propose_macros(llm, game, actions, oj, diffs, subgoal, avail, key_fn)
    if not cands:
        return None
    ranked = macro.rank_macros(cands, game, actions, subgoal, key_fn, seen=set())
    base_levels = _levels_after(make, game, actions)
    for m in ranked:
        reached = _levels_after(make, game, actions + list(m))
        if reached > base_levels:
            return list(m)
    return None


def _levels_after(make, game, action_list):
    """Fresh-env replay (Bug #2): make a new game from gid when possible, else reuse `game`."""
    gid = getattr(game, "gid", None)
    g = make(gid) if (make is not None and isinstance(gid, str)) else game
    reached, _ = planner.replay_levels(g, action_list)
    return reached
```

In `experiments/e119_slm_solver.py`, update argparse and llm construction:

```python
    ap.add_argument("--mode", choices=["search", "slm", "macro", "macro+slm"], default="search")
    ...
    llm = None
    if a.mode != "search":
        import openworld as O
        from e119 import slm as _slm
        llm = O.OllamaLLM(model=a.model, options=_slm.llm_options(a.model))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH="$PWD:$PWD/experiments:$PWD/scratch_arc/agent" python3 -m pytest tests/test_e119_macro.py -q`
Expected: PASS (all macro tests). Then run the full E119 suite to confirm no regression:
`PYTHONPATH="$PWD:$PWD/experiments:$PWD/scratch_arc/agent" python3 -m pytest tests/test_e119_*.py -q` → all pass.

- [ ] **Step 5: Commit**

```bash
git add experiments/e119/solve.py experiments/e119_slm_solver.py tests/test_e119_macro.py
git commit -m "feat(e119): post-stall macro fallback + --mode macro/macro+slm"
```

---

## Self-Review

- **Spec coverage:** macro form/grammar + directional grounding via diffs (Tasks 1, 5 prompt); proposer with best-of-N + behavioral clustering + abstention (Task 2); subgoal-proxy ranker (Task 3); random-macro baseline (Task 4); post-stall hook + `--mode macro`/`macro+slm` + fresh-env banking (Task 5). Reproducibility protocol, the 3-arm seeded sweep, `arc3_traces` logging, and the banked-solve re-verifier are **Plan B** (the spec's measurement section) — out of scope here by the agreed split. g50t excluded; scope tr87/re86 is a Plan B run-target, not mechanism.
- **Placeholder scan:** none — every code/test step is complete.
- **Type consistency:** `compile_macro(ops, obj_json, avail) -> list[tuple]` consumed by `propose_macros`, `rank_macros`, `propose_random_macros`, and `_macro_fallback`. `propose_macros(...) -> list[list[tuple]]` consumed by `rank_macros`. `rank_macros(...) -> list[list[tuple]]` consumed by the fallback loop. `_endpoint`/`_levels_after` use `planner._frame_after`/`replay_levels` and `solve._PrefixGame` with their real signatures. `key_fn` threaded from `solve_game` (`perceive.state_key`-based) into the proposer/ranker. Modes `search|slm|macro|macro+slm` consistent across `solve_game` and the driver.
