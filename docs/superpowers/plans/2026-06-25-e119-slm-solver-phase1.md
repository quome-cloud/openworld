# E119 Neuro-symbolic SLM Solver — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the deterministic search core + an abstaining SLM subgoal layer that solves ARC-AGI-3 levels on a 5-game pilot set, with a search-only control rung and replay-verified banking.

**Architecture:** A deterministic harness (`perceive` → `planner`) owns search and verification against the replay-only env; the SLM is confined to one execution-graded, abstaining slot (`subgoal`) that only *orders* the search. All correctness comes from env replay; the SLM can only make search faster, never wrong.

**Tech Stack:** Python stdlib + numpy; `arc_agi` (in the arc venv) via `experiments/arc3_harness.py`; `openworld.OllamaLLM` for local models; pytest for tests.

## Global Constraints

- **Pixel-honest:** infer click targets/objects from frames only — never call `env._game._get_valid_clickable_actions` or read internal level index. (spec §9)
- **No self-repair loops, no model-judge, no few-shot-by-default.** Every "is this better?" decision is execution-graded. (spec §2)
- **Abstention is behavioral:** cluster best-of-N candidates by observed effect, vote by cluster mass, abstain below τ. (spec §4.2)
- **Frame shape:** `(1,64,64)` → use `np.asarray(o.frame)[-1].reshape(64,64)`; colors 0–15. Reward signal = `levels_completed` rising. (spec §3)
- **Env is replay-only & slow to make:** `arc.make` once per game, then `reset()` + replay; never `arc.make` in a loop. (spec §7)
- **`save_results` BEFORE asserts** so a failed check never loses a run. (CLAUDE.md)
- **Core stays zero-dep:** this code lives under `experiments/` (may use numpy); do not touch `openworld/` core imports.
- **Arc venv python (runs `arc_agi`):** `/private/tmp/claude-501/-Users-jim-Desktop-openworld/71e8c8de-fcca-4c0d-b13e-d3aae6071546/scratchpad/arcv/bin/python`. Plain `python` cannot import `arc_agi`. Pure-logic tests (Tasks 1–5) run under any python with numpy; env tests (Task 6–7 smoke) need the arc venv.
- **Pilot game set (Phase 1):** `tn36`, `ar25` (directional), `vc33`, `lp85` (click), `sk48` (deeper, partial). (spec §8)

---

## File Structure

- `experiments/e119/__init__.py` — marks the subpackage.
- `experiments/e119/perceive.py` — masking, object-JSON, contrastive diff, click candidates, probe. (deterministic)
- `experiments/e119/planner.py` — `GameLike` protocol, `replay_levels`, `search_level` (env-ground-truth BFS/best-first). (deterministic)
- `experiments/e119/abstain.py` — `best_of_n` behavioral voting + τ-gate. (model-agnostic)
- `experiments/e119/slm.py` — per-family decoding config, predicate schema + grader, `propose_subgoal`. (uses abstain)
- `experiments/e119/solve.py` — per-game orchestration, banking, JSONL logging.
- `experiments/e119/world.py` — emit a solved game as an `openworld.World` (+ `to_spec`).
- `experiments/e119_slm_solver.py` — experiment entry (rungs, `save_results`, asserts).
- `tests/conftest.py` — put `experiments/` on `sys.path` so tests can `import e119.*`.
- `tests/test_e119_perceive.py`, `tests/test_e119_planner.py`, `tests/test_e119_abstain.py`, `tests/test_e119_slm.py`, `tests/test_e119_solve.py`.

**Interfaces locked across tasks (copy names verbatim):**
- `perceive.status_mask(frames: list[np.ndarray], thresh=0.95) -> np.ndarray`  # bool (64,64), True = zero it
- `perceive.state_key(frame: np.ndarray, mask: np.ndarray) -> bytes`
- `perceive.object_json(frame, bg=None) -> dict`  # {bg, objects:[{id,color,size,centroid,bbox}], relations:[...]}
- `perceive.contrastive_diff(before, after, bg=None) -> dict`  # {moved, appeared, vanished, recolored}
- `perceive.click_candidates(frame, bg=None, max_size=40) -> list[tuple[int,int]]`  # (x=col, y=row)
- `perceive.probe(game) -> list[dict]`  # each {action, before, after, dlevels}
- `planner.replay_levels(game, actions) -> tuple[int,bool]`  # (max_levels_reached, done)
- `planner.search_level(game, candidates_fn, key_fn, budget, score_fn=None) -> list | None`  # action list raising levels
- `abstain.best_of_n(sample_fn, behavior_fn, n, tau) -> tuple[object|None, dict]`
- `slm.llm_options(model, thinking=False) -> dict`
- `slm.compile_predicate(pred: dict) -> "Callable[[np.ndarray], bool]"`
- `slm.satisfiable(pred, frames) -> bool`
- `slm.propose_subgoal(llm, obj_json, frames, n=6, tau=0.5) -> dict | None`  # `llm` is an `openworld.BaseLLM`
- `solve.solve_game(game, llm=None, mode="search", budget=None, logdir=None) -> dict`
- `world.solver_world(result, transitions) -> "openworld.World"`  # emit the solved game as a World

Action encoding everywhere: a directional action is `(a,)` with `a in {1,2,3,4,5,7}`; a click is `(6, x, y)`.

**OpenWorld binding (SLM-only — Global):** the proposer takes an `openworld.BaseLLM`,
so `OllamaLLM` (SLMs) and `MockLLM` (tests) are interchangeable; **no Anthropic/Claude
backbone is added or used**. The solved solver is emitted as an `openworld.World`
(Task 8) per CLAUDE.md "build solvers as OpenWorld".

---

### Task 1: Scaffolding + perceive masking & state key

**Files:**
- Create: `experiments/e119/__init__.py`
- Create: `tests/conftest.py`
- Create: `experiments/e119/perceive.py`
- Test: `tests/test_e119_perceive.py`

**Interfaces:**
- Produces: `perceive.status_mask`, `perceive.state_key` (signatures above).

- [ ] **Step 1: Create the package marker and test path shim**

`experiments/e119/__init__.py`:
```python
"""E119 neuro-symbolic SLM solver (Phase 1)."""
```

`tests/conftest.py`:
```python
import sys, pathlib
# Let tests import the e119 subpackage: put experiments/ on sys.path.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "experiments"))
```

- [ ] **Step 2: Write the failing test**

`tests/test_e119_perceive.py`:
```python
import numpy as np
from e119 import perceive


def test_status_mask_flags_only_always_changing_cells():
    # cell (0,0) flips every step; everything else constant.
    frames = []
    for t in range(10):
        f = np.zeros((64, 64), int)
        f[0, 0] = t % 2          # changes every step
        f[5, 5] = 7              # constant
        frames.append(f)
    mask = perceive.status_mask(frames, thresh=0.95)
    assert mask.shape == (64, 64)
    assert mask[0, 0] == True
    assert mask[5, 5] == False


def test_state_key_ignores_masked_cells():
    mask = np.zeros((64, 64), bool)
    mask[0, 0] = True
    a = np.zeros((64, 64), int); a[0, 0] = 1
    b = np.zeros((64, 64), int); b[0, 0] = 9   # differs only in masked cell
    assert perceive.state_key(a, mask) == perceive.state_key(b, mask)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_e119_perceive.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'e119'` then (after Step 4) collection passes.

- [ ] **Step 4: Write minimal implementation**

`experiments/e119/perceive.py`:
```python
"""Deterministic perception: masking, object-JSON, diffs, click candidates, probe."""
import numpy as np
import arc3_graph  # sibling module on sys.path when run from experiments/


def status_mask(frames, thresh=0.95):
    """Cells that change on >thresh of step-to-step transitions -> mask (zero before hashing)."""
    arr = np.stack([np.asarray(f).reshape(64, 64) for f in frames])
    if len(arr) < 2:
        return np.zeros((64, 64), bool)
    changes = (arr[1:] != arr[:-1]).mean(axis=0)   # fraction of steps each cell changed
    return changes > thresh


def state_key(frame, mask):
    g = np.asarray(frame).reshape(64, 64).copy()
    g[mask] = 0
    return g.astype(np.int16).tobytes()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_e119_perceive.py -q`
Expected: PASS (2 passed)

- [ ] **Step 6: Commit**

```bash
git add experiments/e119/__init__.py tests/conftest.py experiments/e119/perceive.py tests/test_e119_perceive.py
git commit -m "feat(e119): perceive masking + state key, package scaffold"
```

---

### Task 2: perceive object-JSON, contrastive diff, click candidates

**Files:**
- Modify: `experiments/e119/perceive.py`
- Test: `tests/test_e119_perceive.py`

**Interfaces:**
- Consumes: `arc3_graph.objects(frame, bg=None) -> (objs, bg)` where each obj is `{color,size,centroid,bbox,shape}`.
- Produces: `perceive.object_json`, `perceive.contrastive_diff`, `perceive.click_candidates`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_e119_perceive.py`:
```python
def test_object_json_is_relational_to_largest_object():
    f = np.zeros((64, 64), int)
    f[10:14, 10:14] = 3        # big object (agent proxy) size 16
    f[2, 40] = 5               # tiny object size 1
    oj = perceive.object_json(f)
    assert oj["bg"] == 0
    ids = {o["color"]: o for o in oj["objects"]}
    assert 3 in ids and 5 in ids
    # relations are expressed relative to the largest object (id 0 by sort order)
    assert any("of #0" in r for r in oj["relations"])


def test_contrastive_diff_detects_a_move():
    a = np.zeros((64, 64), int); a[10, 10] = 4
    b = np.zeros((64, 64), int); b[10, 11] = 4    # same color moved +1 col
    d = perceive.contrastive_diff(a, b)
    assert d["moved"], "expected one moved object"
    assert d["moved"][0]["color"] == 4


def test_click_candidates_are_small_components_as_xy():
    f = np.zeros((64, 64), int)
    f[0:30, 0:30] = 2          # big region -> NOT a candidate
    f[2, 40] = 5               # tiny sprite -> candidate at (x=40, y=2)
    cands = perceive.click_candidates(f, max_size=40)
    assert (40, 2) in cands
    assert (0, 0) not in cands
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_e119_perceive.py -q`
Expected: FAIL with `AttributeError: module 'e119.perceive' has no attribute 'object_json'`

- [ ] **Step 3: Implement**

Append to `experiments/e119/perceive.py`:
```python
def object_json(frame, bg=None):
    g = np.asarray(frame).reshape(64, 64)
    objs, bg = arc3_graph.objects(g, bg=bg)
    out = []
    for i, o in enumerate(objs):
        out.append({"id": i, "color": o["color"], "size": o["size"],
                    "centroid": o["centroid"], "bbox": o["bbox"]})
    relations = []
    if out:
        ref = out[0]["centroid"]                       # largest object is the anchor
        for o in out[1:]:
            dy = round(o["centroid"][0] - ref[0], 1)
            dx = round(o["centroid"][1] - ref[1], 1)
            relations.append(f"#{o['id']}(c{o['color']}) at dy={dy},dx={dx} of #0")
    return {"bg": bg, "objects": out, "relations": relations}


def contrastive_diff(before, after, bg=None):
    ba, aa = object_json(before, bg)["objects"], object_json(after, bg)["objects"]
    by_color_b, by_color_a = {}, {}
    for o in ba: by_color_b.setdefault(o["color"], []).append(o)
    for o in aa: by_color_a.setdefault(o["color"], []).append(o)
    moved, appeared, vanished = [], [], []
    for color, alist in by_color_a.items():
        blist = by_color_b.get(color, [])
        if blist and alist:
            b0, a0 = blist[0], alist[0]
            if b0["centroid"] != a0["centroid"]:
                moved.append({"color": color, "from": b0["centroid"], "to": a0["centroid"]})
        elif alist and not blist:
            appeared.append({"color": color, "at": alist[0]["centroid"]})
    for color in by_color_b:
        if color not in by_color_a:
            vanished.append({"color": color})
    return {"moved": moved, "appeared": appeared, "vanished": vanished, "recolored": []}


def click_candidates(frame, bg=None, max_size=40):
    g = np.asarray(frame).reshape(64, 64)
    objs, bg = arc3_graph.objects(g, bg=bg)
    color_counts = {}
    for o in objs:
        color_counts[o["color"]] = color_counts.get(o["color"], 0) + o["size"]
    cands = []
    for o in objs:
        small = o["size"] <= max_size
        rare = color_counts[o["color"]] <= max_size
        if small or rare:
            cy, cx = o["centroid"]
            cands.append((int(round(cx)), int(round(cy))))   # (x=col, y=row)
    # dedup, stable order
    seen, out = set(), []
    for c in cands:
        if c not in seen:
            seen.add(c); out.append(c)
    return out
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_e119_perceive.py -q`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add experiments/e119/perceive.py tests/test_e119_perceive.py
git commit -m "feat(e119): object-json, contrastive diff, click candidates"
```

---

### Task 3: perceive.probe + planner (env-ground-truth search)

**Files:**
- Modify: `experiments/e119/perceive.py`
- Create: `experiments/e119/planner.py`
- Test: `tests/test_e119_planner.py`

**Interfaces:**
- Produces: `perceive.probe(game)`, `planner.replay_levels`, `planner.search_level`.
- `GameLike`: attributes `levels:int`, `win:int`, `done:bool`, `frame:np.ndarray`; methods `reset()`, `step(a, x=None, y=None)`. (`arc3_harness.Game` satisfies this.)
- `budget` is `{"max_nodes": int, "max_depth": int}`.
- `candidates_fn(frame) -> list[action]`; `key_fn(frame) -> bytes`; `score_fn(frame) -> float | None` (higher = closer to goal; optional).

- [ ] **Step 1: Write the failing test (FakeGame replays deterministically)**

`tests/test_e119_planner.py`:
```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_e119_planner.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'e119.planner'`

- [ ] **Step 3: Implement planner**

`experiments/e119/planner.py`:
```python
"""Env-ground-truth search over the replay-only ARC env. Correctness via replay, never the model."""
import heapq
from collections import deque


def replay_levels(game, actions):
    """Replay an action list from reset(); return (max levels reached, done)."""
    game.reset(); base = game.levels; mx = base
    for act in actions:
        game.step(*act)
        if game.levels > mx: mx = game.levels
        if game.done: break
    return mx - base, game.done


def _frame_after(game, actions):
    game.reset()
    for act in actions:
        game.step(*act)
        if game.done: break
    return game.frame, game.levels, game.done


def search_level(game, candidates_fn, key_fn, budget, score_fn=None):
    """Find an action sequence that raises levels by >=1. BFS, or best-first when score_fn given.
    Each node is an action prefix; we replay it from reset() to expand (env is replay-only)."""
    game.reset(); base = game.levels
    start_frame = game.frame
    seen = {key_fn(start_frame)}
    nodes = 0
    if score_fn is None:
        frontier = deque([[]])
        pop = frontier.popleft
        push = frontier.append
    else:
        counter = 0
        heap = [(-score_fn(start_frame), 0, [])]
        def pop():
            return heapq.heappop(heap)[2]
        def push(seq):
            nonlocal counter
            counter += 1
            f, _, _ = _frame_after(game, seq)
            heapq.heappush(heap, (-score_fn(f), counter, seq))
        frontier = heap
    while frontier and nodes < budget["max_nodes"]:
        seq = pop()
        if len(seq) >= budget["max_depth"]:
            continue
        frame, _, _ = _frame_after(game, seq)
        for act in candidates_fn(frame):
            nodes += 1
            child = seq + [act]
            f2, levels2, done2 = _frame_after(game, child)
            if levels2 > base:
                return child
            k = key_fn(f2)
            if k in seen:
                continue
            seen.add(k)
            push(child)
            if nodes >= budget["max_nodes"]:
                break
    return None
```

- [ ] **Step 4: Implement perceive.probe**

Append to `experiments/e119/perceive.py`:
```python
def probe(game):
    """Single-step transitions from reset(): each directional avail action + each click candidate.
    Returns list of {action, before, after, dlevels}. Replays from reset() per probe (env is replay-only)."""
    game.reset()
    base_frame = np.asarray(game.frame).reshape(64, 64).copy()
    base_levels = game.levels
    avail = [a for a in getattr(game, "avail", [1, 2, 3, 4, 5, 7]) if a != 6]
    transitions = []
    for a in avail:
        game.reset()
        game.step(a)
        transitions.append({"action": (a,), "before": base_frame,
                            "after": np.asarray(game.frame).reshape(64, 64).copy(),
                            "dlevels": game.levels - base_levels})
    if 6 in getattr(game, "avail", []):
        for (x, y) in click_candidates(base_frame):
            game.reset()
            game.step(6, x, y)
            transitions.append({"action": (6, x, y), "before": base_frame,
                                "after": np.asarray(game.frame).reshape(64, 64).copy(),
                                "dlevels": game.levels - base_levels})
    return transitions
```

- [ ] **Step 5: Add a probe test**

Append to `tests/test_e119_planner.py`:
```python
def test_probe_collects_one_transition_per_directional_action():
    g = FakeGame(); g.avail = [7, 1]
    trans = perceive.probe(g)
    actions = {t["action"] for t in trans}
    assert (7,) in actions and (1,) in actions
    moved = [t for t in trans if not np.array_equal(t["before"], t["after"])]
    assert any(t["action"] == (7,) for t in moved)   # action 7 changed the board
```

- [ ] **Step 6: Run to verify pass**

Run: `python -m pytest tests/test_e119_planner.py -q`
Expected: PASS (4 passed)

- [ ] **Step 7: Commit**

```bash
git add experiments/e119/planner.py experiments/e119/perceive.py tests/test_e119_planner.py
git commit -m "feat(e119): env-ground-truth search_level + probe"
```

---

### Task 4: abstain (behavioral best-of-N + τ-gate)

**Files:**
- Create: `experiments/e119/abstain.py`
- Test: `tests/test_e119_abstain.py`

**Interfaces:**
- Produces: `abstain.best_of_n(sample_fn, behavior_fn, n, tau) -> (winner|None, meta)`.
- `sample_fn() -> candidate` (called up to `n` times). `behavior_fn(candidate) -> hashable` (its observed effect). Winner = a candidate from the largest behavior cluster iff `cluster_size/n >= tau`, else `None` (ABSTAIN). `meta = {"agreement": float, "clusters": int, "samples": int}`. Adaptive stop once a cluster is mathematically guaranteed to clear τ.

- [ ] **Step 1: Write the failing tests**

`tests/test_e119_abstain.py`:
```python
from e119 import abstain


def test_best_of_n_returns_majority_behavior_when_agree():
    seq = iter(["a", "a", "a", "b"])
    winner, meta = abstain.best_of_n(lambda: next(seq), behavior_fn=lambda c: c, n=4, tau=0.5)
    assert winner == "a"
    assert meta["agreement"] >= 0.5


def test_best_of_n_abstains_when_no_cluster_clears_tau():
    seq = iter(["a", "b", "c", "d"])
    winner, meta = abstain.best_of_n(lambda: next(seq), behavior_fn=lambda c: c, n=4, tau=0.5)
    assert winner is None
    assert meta["agreement"] < 0.5


def test_best_of_n_clusters_by_behavior_not_text():
    # different text, same behavior signature -> they agree
    cand = iter(["x=1", "x = 1", "x= 1", "y=9"])
    winner, meta = abstain.best_of_n(
        lambda: next(cand),
        behavior_fn=lambda c: c.replace(" ", "").split("=")[0],  # variable name = behavior
        n=4, tau=0.5)
    assert winner is not None
    assert winner.replace(" ", "").startswith("x")
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_e119_abstain.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'e119.abstain'`

- [ ] **Step 3: Implement**

`experiments/e119/abstain.py`:
```python
"""Best-of-N with BEHAVIORAL clustering and a tau abstention gate (spec law: route to the executor)."""
from collections import defaultdict


def best_of_n(sample_fn, behavior_fn, n, tau):
    clusters = defaultdict(list)   # behavior signature -> [candidates]
    drawn = 0
    need = -(-int(tau * n) // 1)   # ceil(tau*n): min cluster size to clear tau
    need = int(need) if need >= 1 else 1
    for _ in range(n):
        try:
            c = sample_fn()
        except StopIteration:
            break
        drawn += 1
        try:
            sig = behavior_fn(c)
        except Exception:
            continue               # ungradeable candidate is discarded, not fatal
        clusters[sig].append(c)
        if len(clusters[sig]) >= need:    # adaptive stop: this cluster already clears tau
            break
    if not clusters:
        return None, {"agreement": 0.0, "clusters": 0, "samples": drawn}
    best_sig = max(clusters, key=lambda s: len(clusters[s]))
    top = len(clusters[best_sig])
    agreement = top / drawn if drawn else 0.0
    winner = clusters[best_sig][0] if agreement >= tau else None
    return winner, {"agreement": agreement, "clusters": len(clusters), "samples": drawn}
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_e119_abstain.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add experiments/e119/abstain.py tests/test_e119_abstain.py
git commit -m "feat(e119): behavioral best-of-N abstention gate"
```

---

### Task 5: slm (decoding config, predicate schema + grader, propose_subgoal)

**Files:**
- Create: `experiments/e119/slm.py`
- Test: `tests/test_e119_slm.py`

**Interfaces:**
- Produces: `slm.llm_options`, `slm.compile_predicate`, `slm.satisfiable`, `slm.propose_subgoal`.
- Predicate JSON: `{"type": "reach", "color": int}` | `{"type": "count", "color": int, "op": "==", "k": int}` | `{"type": "align", "a": int, "b": int}`.
- `compile_predicate(pred) -> fn(frame)->bool`. `satisfiable(pred, frames)` = True if some frame satisfies it.
- `propose_subgoal(llm, obj_json, frames, n=6, tau=0.5)`: `llm` is an `openworld.BaseLLM` (uses `.ask(prompt)->str`); samples N predicates, behavior = `(pred satisfiable on frames, canonical pred tuple)`, abstains via `best_of_n`. Returns a predicate dict or `None`.
- Tests use `openworld.MockLLM(responses)` (the framework's scripted `BaseLLM`), not a bespoke stub — this is the OpenWorld binding.

- [ ] **Step 1: Write the failing tests**

`tests/test_e119_slm.py`:
```python
import json, numpy as np
from e119 import slm
from openworld import MockLLM   # framework's scripted BaseLLM — bind to OpenWorld, no bespoke stub


def test_llm_options_pins_gemma_differently_from_qwen():
    q = slm.llm_options("qwen2.5-coder:7b")
    gm = slm.llm_options("gemma2:9b")
    assert q["temperature"] == 0.7 and q["top_k"] == 20
    assert gm["temperature"] == 1.0 and gm["top_k"] == 64   # Gemma defaults differ


def test_compile_and_satisfiable_reach_color():
    pred = {"type": "reach", "color": 5}
    f_no = np.zeros((64, 64), int)
    f_yes = np.zeros((64, 64), int); f_yes[10, 10] = 5
    assert slm.satisfiable(pred, [f_no, f_yes]) is True
    assert slm.satisfiable(pred, [f_no]) is False


def test_propose_subgoal_votes_and_returns_predicate():
    frames = [np.zeros((64, 64), int)]
    frames[0][2, 2] = 5
    oj = {"objects": [{"id": 0, "color": 5}], "relations": []}
    replies = [json.dumps({"type": "reach", "color": 5})] * 4
    llm = MockLLM(replies)
    pred = slm.propose_subgoal(llm, oj, frames, n=4, tau=0.5)
    assert pred == {"type": "reach", "color": 5}


def test_propose_subgoal_abstains_on_disagreement():
    frames = [np.zeros((64, 64), int)]
    oj = {"objects": [], "relations": []}
    replies = [json.dumps({"type": "reach", "color": c}) for c in (1, 2, 3, 4)]
    llm = MockLLM(replies)
    pred = slm.propose_subgoal(llm, oj, frames, n=4, tau=0.6)
    assert pred is None
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_e119_slm.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'e119.slm'`

- [ ] **Step 3: Implement**

`experiments/e119/slm.py`:
```python
"""SLM proposer: per-family decoding, a tiny predicate DSL with an executable grader, abstaining subgoal."""
import json, re
import numpy as np
from e119 import abstain

# Per-family decoding (spec §5): pin every family; Gemma differs sharply from Qwen.
_FAMILY = [
    ("qwen3", {"temperature": 0.6, "top_p": 0.95, "top_k": 20}),   # thinking default; see thinking flag
    ("qwen",  {"temperature": 0.7, "top_p": 0.8,  "top_k": 20, "repeat_penalty": 1.05}),
    ("gemma", {"temperature": 1.0, "top_p": 0.95, "top_k": 64}),
    ("llama", {"temperature": 0.6, "top_p": 0.9,  "top_k": 40}),
    ("phi",   {"temperature": 0.7, "top_p": 0.9,  "top_k": 40}),
]


def llm_options(model, thinking=False):
    name = model.lower()
    for key, opts in _FAMILY:
        if key in name:
            o = dict(opts)
            if key == "qwen3" and not thinking:
                o.update({"temperature": 0.7, "top_p": 0.8})       # non-thinking Qwen3
            o["num_predict"] = 4096 if thinking else 1024          # generous; thinking needs room
            return o
    return {"temperature": 0.7, "top_p": 0.9, "top_k": 40, "num_predict": 1024}


def compile_predicate(pred):
    t = pred.get("type")
    if t == "reach":
        c = pred["color"]
        return lambda f: bool((np.asarray(f).reshape(64, 64) == c).any())
    if t == "count":
        c, k, op = pred["color"], pred["k"], pred.get("op", "==")
        def fn(f):
            n = int((np.asarray(f).reshape(64, 64) == c).sum())
            return {"==": n == k, ">=": n >= k, "<=": n <= k}.get(op, False)
        return fn
    if t == "align":
        import arc3_graph
        a, b = pred["a"], pred["b"]
        def fn(f):
            objs, _ = arc3_graph.objects(np.asarray(f).reshape(64, 64))
            ca = [o for o in objs if o["color"] == a]
            cb = [o for o in objs if o["color"] == b]
            return bool(ca and cb and round(ca[0]["centroid"][1]) == round(cb[0]["centroid"][1]))
        return fn
    return lambda f: False


def satisfiable(pred, frames):
    fn = compile_predicate(pred)
    return any(fn(f) for f in frames)


def _canon(pred):
    return tuple(sorted((k, str(v)) for k, v in pred.items()))


def _parse(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no json")
    return json.loads(m.group(0))


_PROMPT = (
    "You pick the GOAL of one ARC level as a JSON predicate. Objects (relational):\n{oj}\n"
    'Allowed: {{"type":"reach","color":N}} | {{"type":"count","color":N,"op":"==|>=|<=","k":N}} '
    '| {{"type":"align","a":N,"b":N}}. Output ONLY the JSON.'
)


def propose_subgoal(llm, obj_json, frames, n=6, tau=0.5):
    prompt = _PROMPT.format(oj=json.dumps(obj_json)[:1500])

    def sample():
        return _parse(llm.ask(prompt))

    def behavior(pred):
        # behavior signature = (is it satisfiable on observed frames, canonical predicate)
        return (satisfiable(pred, frames), _canon(pred))

    winner, _meta = abstain.best_of_n(sample, behavior, n=n, tau=tau)
    return winner
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_e119_slm.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add experiments/e119/slm.py tests/test_e119_slm.py
git commit -m "feat(e119): slm decoding config, predicate grader, abstaining subgoal"
```

---

### Task 6: solve (orchestration, banking, logging)

**Files:**
- Create: `experiments/e119/solve.py`
- Test: `tests/test_e119_solve.py`

**Interfaces:**
- Produces: `solve.solve_game(game, llm=None, mode="search", budget=None, logdir=None) -> dict`.
- Returns `{"game", "mode", "levels", "win", "actions", "verified"}`. When `mode="search"` the SLM is never called (rung 1 control). When `mode="slm"` and `llm` given, a subgoal orders the search via a `score_fn`. Writes `solved.json` only after `replay_levels` reconfirms the banked `actions` reach `levels`. Appends one JSONL record per level to `logdir/<game>.jsonl` when `logdir` set.
- `candidates_fn(frame)` = directional `avail` ∪ `perceive.click_candidates(frame)`; `key_fn` = `perceive.state_key(frame, mask)`.

- [ ] **Step 1: Write the failing tests (multi-level FakeGame)**

`tests/test_e119_solve.py`:
```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_e119_solve.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'e119.solve'`

- [ ] **Step 3: Implement**

`experiments/e119/solve.py`:
```python
"""Per-game orchestration: probe -> (optional subgoal) -> search each level -> bank replay-verified."""
import json, time
import numpy as np
from e119 import perceive, planner, slm


def _candidates_fn(game, mask):
    avail_dir = [a for a in getattr(game, "avail", [1, 2, 3, 4, 5, 7]) if a != 6]
    has_click = 6 in getattr(game, "avail", [])

    def fn(frame):
        acts = [(a,) for a in avail_dir]
        if has_click:
            acts += [(6, x, y) for (x, y) in perceive.click_candidates(frame)]
        return acts
    return fn


def solve_game(game, llm=None, mode="search", budget=None, logdir=None):
    budget = budget or {"max_nodes": 4000, "max_depth": 40}
    game.reset()
    win = game.win
    name = type(game).__name__ if not isinstance(getattr(game, "gid", None), str) else game.gid
    actions = []
    log = []
    while not game.done and game.levels < win:
        trans = perceive.probe(game)
        mask = perceive.status_mask([t["before"] for t in trans] + [t["after"] for t in trans])
        key_fn = lambda f, m=mask: perceive.state_key(f, m)
        cands = _candidates_fn(game, mask)
        score_fn = None
        subgoal = None
        if mode == "slm" and llm is not None:
            frames = [t["after"] for t in trans]
            oj = perceive.object_json(trans[0]["before"])
            subgoal = slm.propose_subgoal(llm, oj, frames)
            if subgoal is not None:
                pred = slm.compile_predicate(subgoal)
                score_fn = lambda f, p=pred: 1.0 if p(f) else 0.0   # frontier prefers goal-satisfying frames
        # search from the CURRENT progress: replay known actions, then search the next level
        def fresh():
            g2 = game; g2.reset()
            for a in actions: g2.step(*a)
            return g2
        seq = planner.search_level(_PrefixGame(game, actions), cands, key_fn, budget, score_fn)
        rec = {"level_index": game.levels, "subgoal": subgoal, "found": seq is not None,
               "ts": None}
        log.append(rec)
        if seq is None:
            break
        actions += seq
        # re-apply to advance the real game state for the next iteration
        game.reset()
        for a in actions: game.step(*a)
    # verify before banking
    reached, _ = planner.replay_levels(game, actions)
    verified = reached >= game.levels and reached > 0
    result = {"game": name, "mode": mode, "levels": reached, "win": win,
              "actions": actions, "verified": bool(verified)}
    if logdir is not None and verified:
        import pathlib
        d = pathlib.Path(logdir); d.mkdir(parents=True, exist_ok=True)
        (d / f"{name}_solved.json").write_text(json.dumps(result))
        (d / f"{name}.jsonl").write_text("\n".join(json.dumps(r) for r in log))
    return result


class _PrefixGame:
    """Wraps a GameLike so search starts AFTER a fixed action prefix (the levels already solved)."""
    def __init__(self, game, prefix):
        self._g = game; self._prefix = list(prefix)
        self.win = game.win; self.reset()
    def reset(self):
        self._g.reset()
        for a in self._prefix: self._g.step(*a)
        self.levels = self._g.levels; self.done = self._g.done; self.frame = self._g.frame
        self.avail = getattr(self._g, "avail", [1, 2, 3, 4, 5, 7])
        return self.frame
    def step(self, a, x=None, y=None):
        self._g.step(a, x, y)
        self.levels = self._g.levels; self.done = self._g.done; self.frame = self._g.frame
        return self.frame
```

> Note: remove the unused `fresh()` helper if your linter flags it; it documents intent but `_PrefixGame` is what the search consumes.

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_e119_solve.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add experiments/e119/solve.py tests/test_e119_solve.py
git commit -m "feat(e119): orchestration, level chaining, replay-verified banking"
```

---

### Task 7: experiment entry (rungs, save_results, asserts) + real-env smoke

**Files:**
- Create: `experiments/e119_slm_solver.py`
- Test: `tests/test_e119_solve.py` (add an entry-level unit test)

**Interfaces:**
- Consumes: `solve.solve_game`, `experiments/common.py:save_results`.
- Produces: a `main()` that runs the pilot set in `--mode {search,slm}` and writes `experiments/results/e119_slm_solver.json` via `save_results("e119_slm_solver", payload)`; asserts come AFTER the save.

- [ ] **Step 1: Write the failing test (entry runs on the FakeGame registry)**

Append to `tests/test_e119_solve.py`:
```python
def test_entry_run_pilot_aggregates(monkeypatch, tmp_path):
    import e119_slm_solver as entry

    def fake_make(gid):
        return TrackGame()
    payload = entry.run_pilot(["g1", "g2"], mode="search", make=fake_make,
                              budget={"max_nodes": 500, "max_depth": 8}, logdir=tmp_path)
    assert payload["n_games"] == 2
    assert payload["levels_solved"] == 4          # 2 levels each
    assert all(r["verified"] for r in payload["results"])
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest "tests/test_e119_solve.py::test_entry_run_pilot_aggregates" -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'e119_slm_solver'`

- [ ] **Step 3: Implement the entry**

`experiments/e119_slm_solver.py`:
```python
"""E119 entry: run the pilot ARC-AGI-3 games under the search-only control and the SLM-in-loop rung.

  arc venv python -- needs arc_agi:
  .../arcv/bin/python experiments/e119_slm_solver.py --mode search
  .../arcv/bin/python experiments/e119_slm_solver.py --mode slm --model qwen2.5-coder:7b
"""
import argparse, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))   # let 'import e119' work
from e119 import solve
from common import save_results

PILOT = ["tn36", "ar25", "vc33", "lp85", "sk48"]


def _real_make(gid):
    from arc3_harness import Game
    g = Game(gid); g.reset(); g.gid = gid
    return g


def run_pilot(games, mode="search", make=_real_make, llm=None, budget=None, logdir=None):
    results = []
    for gid in games:
        try:
            g = make(gid)
            r = solve.solve_game(g, llm=llm, mode=mode, budget=budget, logdir=logdir)
        except Exception as e:
            r = {"game": gid, "mode": mode, "levels": 0, "win": 0,
                 "actions": [], "verified": False, "error": str(e)[:160]}
        results.append(r)
    return {"mode": mode, "n_games": len(results),
            "levels_solved": sum(r["levels"] for r in results),
            "full_games": sum(1 for r in results if r["win"] and r["levels"] >= r["win"]),
            "results": results}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["search", "slm"], default="search")
    ap.add_argument("--model", default="qwen2.5-coder:7b")
    ap.add_argument("--games", default=",".join(PILOT))
    a = ap.parse_args()
    llm = None
    if a.mode == "slm":
        import openworld as O
        from e119 import slm as _slm
        llm = O.OllamaLLM(model=a.model, options=_slm.llm_options(a.model))
    logdir = pathlib.Path(__file__).resolve().parent / "results" / "e119_logs"
    payload = run_pilot(a.games.split(","), mode=a.mode, llm=llm,
                        budget={"max_nodes": 6000, "max_depth": 60}, logdir=logdir)
    save_results("e119_slm_solver", payload)          # SAVE before asserts (CLAUDE.md)
    assert payload["levels_solved"] >= 0
    assert all(("error" in r) or r["verified"] for r in payload["results"]), "unverified non-error solve"
    print(f"[e119] mode={a.mode} levels={payload['levels_solved']} full={payload['full_games']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest "tests/test_e119_solve.py::test_entry_run_pilot_aggregates" -q`
Expected: PASS (1 passed)

- [ ] **Step 5: Run the full hermetic suite**

Run: `python -m pytest tests/test_e119_perceive.py tests/test_e119_planner.py tests/test_e119_abstain.py tests/test_e119_slm.py tests/test_e119_solve.py -q`
Expected: PASS (all green, no network/env needed)

- [ ] **Step 6: Real-env smoke (search-only control on one game)**

Run:
```bash
/private/tmp/claude-501/-Users-jim-Desktop-openworld/71e8c8de-fcca-4c0d-b13e-d3aae6071546/scratchpad/arcv/bin/python \
  experiments/e119_slm_solver.py --mode search --games tn36
```
Expected: prints `[e119] mode=search levels=N full=M` with `N>=1` on at least one directional game; writes `experiments/results/e119_slm_solver.json`. If `levels=0` everywhere, the candidate set or budget needs tuning (not a code bug) — record it, don't force a number.

- [ ] **Step 7: Commit**

```bash
git add experiments/e119_slm_solver.py tests/test_e119_solve.py
git commit -m "feat(e119): experiment entry with search/slm rungs + save_results"
```

---

### Task 8: emit the solved solver as an `openworld.World` (the framework binding)

**Files:**
- Create: `experiments/e119/world.py`
- Test: `tests/test_e119_world.py`

**Interfaces:**
- Consumes: `openworld.World`, `openworld.FunctionTransition`, `openworld.CodeObjective`, `openworld.to_spec`.
- Produces: `world.action_name(act) -> str`, `world.solver_world(game_name, chain) -> openworld.World`.
- `chain` is the solved path as a list of `{"key": str, "action": tuple, "next_key": str, "levels": int}` (hex state keys from `perceive.state_key`, in order). The World's `FunctionTransition` looks up the learned table; its `CodeObjective` rewards a level-up. This satisfies CLAUDE.md "build solvers as OpenWorld"; the closure over the table is flagged `lossy` by `to_spec` in Phase 1 (lossless `CodeTransition` is Phase 2).

- [ ] **Step 1: Write the failing test**

`tests/test_e119_world.py`:
```python
from e119 import world
import openworld as O


def _chain():
    # 2-step solved path; the second step raises levels 0 -> 1.
    return [
        {"key": "aa", "action": (7,), "next_key": "bb", "levels": 0},
        {"key": "bb", "action": (6, 60, 32), "next_key": "cc", "levels": 1},
    ]


def test_action_name_encodes_directional_and_click():
    assert world.action_name((7,)) == "a7"
    assert world.action_name((6, 60, 32)) == "click_60_32"


def test_solver_world_rollout_follows_the_learned_table():
    w = world.solver_world("tn36", _chain())
    assert isinstance(w, O.World)
    s0 = w.initial_state
    s1 = w.transition.step(s0, O.Action("a7", agent="solver"))
    assert s1["key"] == "bb"
    s2 = w.transition.step(s1, O.Action("click_60_32", agent="solver"))
    assert s2["key"] == "cc" and s2["levels"] == 1


def test_solver_world_serializes_to_spec():
    w = world.solver_world("tn36", _chain())
    spec = O.to_spec(w)
    assert spec["name"] == "arc_tn36"
    assert "a7" in spec["actions"] and "click_60_32" in spec["actions"]
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_e119_world.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'e119.world'`

- [ ] **Step 3: Implement**

`experiments/e119/world.py` (uses only the verified `World` signature
`World(name, description, initial_state, actions, rules=None, transition=None, llm=None)` —
the `CodeObjective` is attached by the scorer, not the constructor):
```python
"""Emit a solved ARC game as an openworld.World (CLAUDE.md: build solvers as OpenWorld)."""
import openworld as O


def action_name(act):
    if act[0] == 6:
        return f"click_{act[1]}_{act[2]}"
    return f"a{act[0]}"


def solver_world(game_name, chain):
    """Materialize the solved path as a World: masked-frame key = state; the learned
    (key, action_name) -> (next_key, levels) table = FunctionTransition dynamics."""
    table = {(t["key"], action_name(t["action"])): (t["next_key"], t["levels"]) for t in chain}
    actions = sorted({action_name(t["action"]) for t in chain})
    start_key = chain[0]["key"] if chain else ""

    def fn(state, action):
        nxt = table.get((state.get("key"), action.get("name")))
        return dict(state) if nxt is None else {"key": nxt[0], "levels": nxt[1]}

    return O.World(
        name=f"arc_{game_name}",
        description=f"Learned state-graph solver for ARC-AGI-3 game {game_name}.",
        initial_state={"key": start_key, "levels": 0},
        actions=actions,
        rules=[f"Masked-frame key = state; raising 'levels' wins. {len(chain)} learned transitions."],
        transition=O.FunctionTransition(fn),
    )
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_e119_world.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add experiments/e119/world.py tests/test_e119_world.py
git commit -m "feat(e119): emit solved solver as an openworld.World (framework binding)"
```

---

## Phase 1 Done-When

- All six hermetic test files pass under plain `python -m pytest` (no env, no network):
  `test_e119_perceive`, `_planner`, `_abstain`, `_slm`, `_solve`, `_world`.
- The proposer runs through `openworld.BaseLLM` — tests use `openworld.MockLLM`, the entry uses `openworld.OllamaLLM`. **No Claude/Anthropic backbone exists or is used.**
- `--mode search` runs the pilot under the arc venv and banks ≥1 replay-verified level on the directional games (the search-only control rung).
- `--mode slm --model qwen2.5-coder:7b` runs end-to-end (subgoal proposed *or* cleanly abstained; never crashes a game), writing per-level JSONL.
- `experiments/results/e119_slm_solver.json` exists via `save_results`.
- `world.solver_world(...)` produces an `openworld.World` whose rollout follows the learned table and serializes via `to_spec` (the framework binding).

## Out of scope (later plans)
- **Phase 2:** full 25-game sweep, the capability-substitution curve, the diverse-model voting pool, per-family decoding for all families wired into the entry, Pareto-vs-compute + Wilson CIs reporting, the hidden held-out game, lossless `CodeTransition` (Phase 1 emits a closure flagged `lossy` by `to_spec`), and serving the solver Worlds in `serve /view`.
- **Phase 3:** LoRA distillation of search-verified subgoal/macro traces into a small Gemma; the vision-representation ablation.
- **`macro` slot** (when search stalls) and the **in-model `rule` prediction** acceleration — added once the subgoal rung is shown to help.

## Self-Review notes
- Spec coverage: §3 masking/object-JSON/diff → Tasks 1–2; §3.5 OpenWorld binding (BaseLLM proposer + World emission) → Tasks 5 & 8; §4.1 probe → Task 3; §4.3 search → Task 3; §4.2 abstention behavioral + subgoal → Tasks 4–5; §5 per-family decoding (Qwen+Gemma pinned) → Task 5; §4.4 banking+JSONL + build-solver-as-World → Tasks 6 & 8; §6 rungs (search-only control + SLM) + save_results → Task 7. Pareto/CIs/held-out/distillation/vision/serve are explicitly Phase 2–3.
- Type consistency: action tuples `(a,)`/`(6,x,y)`, `action_name((a,))="a{a}"`/`action_name((6,x,y))="click_{x}_{y}"`, `search_level(game,candidates_fn,key_fn,budget,score_fn)`, `best_of_n(sample_fn,behavior_fn,n,tau)`, `propose_subgoal(llm:BaseLLM,obj_json,frames,n,tau)`, `solve_game(game,llm,mode,budget,logdir)`, `solver_world(game_name,chain)` used identically in every consumer.
- OpenWorld binding: proposer takes `openworld.BaseLLM` (tests `MockLLM`, entry `OllamaLLM`); solver emitted via `openworld.World`/`FunctionTransition`/`to_spec`. SLM-only, no Claude.
- Pixel-honest: no privileged engine calls anywhere; candidates come from `click_candidates` only.
