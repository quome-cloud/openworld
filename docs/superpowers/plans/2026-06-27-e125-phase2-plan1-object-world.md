# E125 Phase-2 — Plan 1: Object-centric verified OpenWorld world (foundation)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace E125's brittle pixel-exact world model with an **object-centric** state, a **decision-equivalent** verifier gate, and an `openworld.World` wrapper that round-trips (to_spec/from_spec) and yields the map (`preview.graph`).

**Architecture:** A frame is abstracted to an **object state** (entities = connected color components with position/size). A synthesized `predict(state, action) -> (next_state, level_up)` is verified by matching the **decision-relevant** projection of the object state + the win predicate (not pixels). The verified program + goal energy + object perceptor are assembled into an `openworld.World` (FunctionTransition + CodeObjective + CodePerceptor) that serializes losslessly.

**Tech Stack:** Python 3.14 (`~/.arcv/bin/python`, `DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib`), `openworld` (World/FunctionTransition/CodeObjective/CodePerceptor/to_spec/from_spec), pytest. Reuses `experiments/e125/{verify,synth}`, `experiments/arc3_graph`.

## Global Constraints

- **Source-free + solution-free:** no game source, no banked solutions; only frames the agent observed.
- **`predict` contract (NEW, object-state):** `predict(state: dict, action: list[int]) -> (next_state: dict, level_up: bool)`, where `state = {"bg": int, "objects": [{"color","size","y","x"}, ...], "level_up": bool}`.
- **Decision-equivalent gate:** a program is accepted iff it reproduces the **`state_key`** projection (default fields `("color","y","x")`) of every held-out transition's next_state AND the `level_up` flag. Pixel-exact match is NOT required (kept only as a future diagnostic).
- **Zero new core deps;** `experiments/` may use arc-venv packages. The object perceptor source must be **self-contained stdlib** (it runs in the OpenWorld sandbox; no imports).
- **Run with** `~/.arcv/bin/python`. **Commit only when asked** (CLAUDE.md) — the "Commit" steps below are gated on the human's go-ahead; do the add/commit only after approval.

## File Structure

- Create `experiments/e125/objstate.py` — `object_state(frame)`/`state_key(...)` (pure-stdlib) + `PERCEIVE_SRC` (the same logic as a self-contained `perceive(data)` for `CodePerceptor`).
- Modify `experiments/e125/verify.py` — add `score_obj` + `check_obj` (decision-equivalent scoring over object states).
- Create `experiments/e125/world.py` — `build_world(...)` → `openworld.World` (+ attach perceptor/objective) and `round_trip_ok(world)`.
- Tests: `tests/test_e125_objstate.py`, `tests/test_e125_verify_obj.py`, `tests/test_e125_world.py`.

A **transition (object form)** is `{"state": dict, "action": list[int], "next_state": dict, "level_up": bool}`.

---

## Task 1: Object-state perceptor

**Files:**
- Create: `experiments/e125/objstate.py`
- Test: `tests/test_e125_objstate.py`

**Interfaces:**
- Produces: `objstate.object_state(frame, ignore_colors=()) -> dict` — `{"bg": int, "objects": [{"color","size","y","x"}, ...]}`, objects sorted canonically by `(color,y,x,size)`. `objstate.state_key(s, fields=("color","y","x")) -> tuple` — a hashable decision-relevant projection. `objstate.PERCEIVE_SRC: str` — source defining `def perceive(data) -> dict` equivalent to `object_state`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_e125_objstate.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
from e125 import objstate

def _grid():
    g = [[0]*8 for _ in range(8)]      # bg=0
    g[1][1] = 3                         # a size-1 object color 3 at (1,1)
    g[5][5] = 3; g[5][6] = 3            # a size-2 object color 3 at (5, 5.5->6)
    g[2][6] = 7                         # a size-1 object color 7 at (2,6)
    return g

def test_object_state_extracts_entities_and_bg():
    s = objstate.object_state(_grid())
    assert s["bg"] == 0
    cols = sorted((o["color"], o["size"]) for o in s["objects"])
    assert cols == [(3, 1), (3, 2), (7, 1)]

def test_object_state_is_canonically_sorted():
    s = objstate.object_state(_grid())
    keys = [(o["color"], o["y"], o["x"]) for o in s["objects"]]
    assert keys == sorted(keys)

def test_state_key_projects_decision_relevant_fields():
    s = objstate.object_state(_grid())
    k = objstate.state_key(s)
    assert k == (0, ((3, 1, 1), (3, 5, 6), (7, 2, 6)))   # (bg, ((color,y,x)...))

def test_ignore_colors_drops_entities():
    s = objstate.object_state(_grid(), ignore_colors=(7,))
    assert all(o["color"] != 7 for o in s["objects"])

def test_perceive_src_matches_object_state():
    ns = {}
    exec(objstate.PERCEIVE_SRC, ns)
    assert ns["perceive"](_grid()) == objstate.object_state(_grid())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.arcv/bin/python -m pytest tests/test_e125_objstate.py -q`
Expected: FAIL (`e125.objstate` missing).

- [ ] **Step 3: Write minimal implementation**

```python
# experiments/e125/objstate.py
"""Object-centric perception: a 64x64 (or NxN) frame -> {"bg", "objects":[{color,size,y,x}]} via 4-connectivity
connected components, canonically sorted. state_key projects the DECISION-RELEVANT fields (positions) the
verifier gate compares -- abstracting away pixels/animation. PERCEIVE_SRC is the SAME logic as a self-contained
`perceive(data)` for an OpenWorld CodePerceptor (stdlib only, no imports -- runs in the sandbox)."""

_BODY = '''
    g = [list(map(int, row)) for row in (data[0] if (len(data) == 1 and hasattr(data[0], "__len__")
                                                     and hasattr(data[0][0], "__len__")) else data)]
    h = len(g); w = len(g[0])
    cnt = {}
    for row in g:
        for c in row:
            cnt[c] = cnt.get(c, 0) + 1
    bg = max(cnt, key=cnt.get)
    seen = [[False] * w for _ in range(h)]
    ents = []
    for i in range(h):
        for j in range(w):
            if seen[i][j] or g[i][j] == bg:
                continue
            color = g[i][j]; stack = [(i, j)]; seen[i][j] = True; cells = []
            while stack:
                y, x = stack.pop(); cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny][nx] and g[ny][nx] == color:
                        seen[ny][nx] = True; stack.append((ny, nx))
            if color in ignore_colors:
                continue
            ys = [c[0] for c in cells]; xs = [c[1] for c in cells]
            ents.append({"color": int(color), "size": len(cells),
                         "y": int(round(sum(ys) / len(ys))), "x": int(round(sum(xs) / len(xs)))})
    ents.sort(key=lambda e: (e["color"], e["y"], e["x"], e["size"]))
    return {"bg": int(bg), "objects": ents}
'''


exec("def _extract(data, ignore_colors):" + _BODY, globals())   # compiled ONCE at import (not per call)


def object_state(frame, ignore_colors=()):
    return _extract(frame, set(ignore_colors))


def state_key(s, fields=("color", "y", "x")):
    return (int(s.get("bg", -1)),
            tuple(tuple(o[f] for f in fields) for o in s.get("objects", [])))


# Self-contained perceive(data) for a CodePerceptor (ignore_colors fixed to none at the boundary).
PERCEIVE_SRC = "def perceive(data):\n    ignore_colors = set()" + _BODY
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.arcv/bin/python -m pytest tests/test_e125_objstate.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit (after human approval)**

```bash
git add experiments/e125/objstate.py tests/test_e125_objstate.py
git commit -m "E125 P2.1: object-centric perceptor (object_state + state_key + PERCEIVE_SRC)"
```

---

## Task 2: Decision-equivalent verifier gate

**Files:**
- Modify: `experiments/e125/verify.py` (add functions; leave existing pixel functions intact)
- Test: `tests/test_e125_verify_obj.py`

**Interfaces:**
- Consumes: `verify.compile_predict` (existing — execs source, returns `predict`); `objstate.state_key`.
- Produces: `verify.score_obj(predict_fn, transitions, fields=("color","y","x")) -> (n_matched: int, fails: list[(transition, predicted_next_state|None)])`; `verify.check_obj(predict_fn, transitions, fields=("color","y","x")) -> (ok: bool, counterexample|None)`. A transition is `{"state","action","next_state","level_up"}`. Match iff `state_key(pred_next)==state_key(real_next)` AND `bool(pred_lu)==bool(real_lu)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_e125_verify_obj.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
from e125 import verify

def _t(s, a, ns, lu): return {"state": s, "action": a, "next_state": ns, "level_up": lu}

S0 = {"bg": 0, "objects": [{"color": 3, "size": 1, "y": 1, "x": 1}]}
S1 = {"bg": 0, "objects": [{"color": 3, "size": 1, "y": 1, "x": 2}]}   # moved x:1->2
S2 = {"bg": 0, "objects": [{"color": 3, "size": 1, "y": 1, "x": 3}]}
TR = [_t(S0, [4], S1, False), _t(S1, [4], S2, False)]

# correct: action [4] increments object x by 1 (decision-relevant move); win never
GOODP = ("def predict(state, action):\n"
         "    ns = {'bg': state['bg'], 'objects': [dict(o) for o in state['objects']]}\n"
         "    if action == [4]:\n"
         "        for o in ns['objects']: o['x'] += 1\n"
         "    return ns, False")
# cosmetically different but decision-equivalent: also bumps 'size' (NOT a decision field) -> still passes
COSMETIC = GOODP.replace("o['x'] += 1", "o['x'] += 1; o['size'] += 9")
# wrong on a decision field (x): mispredicts -> fails
BADP = ("def predict(state, action):\n    return {'bg': state['bg'], 'objects': [dict(o) for o in state['objects']]}, False")

def test_check_obj_accepts_decision_correct():
    ok, ce = verify.check_obj(verify.compile_predict(GOODP), TR)
    assert ok is True and ce is None

def test_check_obj_ignores_non_decision_fields():
    ok, ce = verify.check_obj(verify.compile_predict(COSMETIC), TR)
    assert ok is True                      # size differs but is not a decision field

def test_check_obj_rejects_decision_wrong():
    ok, ce = verify.check_obj(verify.compile_predict(BADP), TR)
    assert ok is False and ce is not None and ce["action"] == [4]

def test_score_obj_counts_matches():
    n, fails = verify.score_obj(verify.compile_predict(GOODP), TR)
    assert n == 2 and fails == []
    n, fails = verify.score_obj(verify.compile_predict(BADP), TR)
    assert n == 0 and len(fails) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.arcv/bin/python -m pytest tests/test_e125_verify_obj.py -q`
Expected: FAIL (`verify.check_obj` missing).

- [ ] **Step 3: Write minimal implementation** (append to `experiments/e125/verify.py`)

```python
# --- decision-equivalent gate over OBJECT states (value-equivalent, not pixel reconstruction) ---
from e125 import objstate as _objstate


def score_obj(predict_fn, transitions, fields=("color", "y", "x")):
    """(n_matched, fails). Match iff the DECISION-RELEVANT state_key of the predicted next_state equals the
    real one AND level_up matches. fails = [(transition, predicted_next_state|None)]."""
    if predict_fn is None:
        return 0, [(t, None) for t in transitions]
    n, fails = 0, []
    for t in transitions:
        try:
            ns, lu = predict_fn(dict(t["state"]), list(t["action"]))
        except Exception:
            fails.append((t, None)); continue
        if (_objstate.state_key(ns, fields) == _objstate.state_key(t["next_state"], fields)
                and bool(lu) == bool(t["level_up"])):
            n += 1
        else:
            fails.append((t, ns))
    return n, fails


def check_obj(predict_fn, transitions, fields=("color", "y", "x")):
    """(ok, counterexample). ok iff every transition matches on the decision-relevant key + level_up."""
    n, fails = score_obj(predict_fn, transitions, fields)
    return (len(fails) == 0 and predict_fn is not None), (fails[0][0] if fails else None)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.arcv/bin/python -m pytest tests/test_e125_verify_obj.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit (after human approval)**

```bash
git add experiments/e125/verify.py tests/test_e125_verify_obj.py
git commit -m "E125 P2.2: decision-equivalent verifier gate over object states"
```

---

## Task 3: `build_world` → `openworld.World` (+ lossless round-trip)

**Files:**
- Create: `experiments/e125/world.py`
- Test: `tests/test_e125_world.py`

**Interfaces:**
- Consumes: `openworld.World`, `openworld.FunctionTransition`, `openworld.CodeObjective`, `openworld.CodePerceptor`, `openworld.to_spec`, `openworld.from_spec`; `objstate.PERCEIVE_SRC`; `verify.compile_predict`.
- Produces: `world.build_world(predict_src, goal_src, initial_state, actions, game) -> openworld.World` — a World whose `transition` is a `FunctionTransition` over object state (carrying `level_up` in state), with `world.perceptors=[CodePerceptor(PERCEIVE_SRC)]` and `world.objectives=[CodeObjective(goal_src)]` attached for `to_spec`. `world.round_trip_ok(w, steps=8) -> bool` — `from_spec(to_spec(w), allow_code=True)` reproduces `w`'s rollout under a fixed action cycle.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_e125_world.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
import openworld as O
from e125 import world

PRED = ("def predict(state, action):\n"
        "    ns = {'bg': state['bg'], 'objects': [dict(o) for o in state['objects']]}\n"
        "    if action == [4]:\n"
        "        for o in ns['objects']: o['x'] = min(9, o['x'] + 1)\n"
        "    return ns, bool(ns['objects'] and ns['objects'][0]['x'] == 5)")
GOAL = ("def reward(state, action, next_state):\n"
        "    o = next_state['objects'][0]\n    return float(-(5 - o['x']))")
S0 = {"bg": 0, "objects": [{"color": 3, "size": 1, "y": 1, "x": 1}]}

def test_build_world_returns_openworld_world():
    w = world.build_world(PRED, GOAL, S0, [[1],[2],[3],[4]], "synthA")
    assert isinstance(w, O.World)
    assert w.perceptors and w.objectives

def test_world_transition_steps_object_state():
    w = world.build_world(PRED, GOAL, S0, [[4]], "synthA")
    ns = w.transition.step(dict(S0, level_up=False), {"name": "[4]"})
    assert ns["objects"][0]["x"] == 2

def test_to_spec_round_trips_and_has_map():
    w = world.build_world(PRED, GOAL, S0, [[1],[2],[3],[4]], "synthA")
    spec = O.to_spec(w)
    assert O.from_spec(spec, allow_code=True) is not None
    assert spec.get("preview", {}).get("graph") is not None   # the MAP exists
    assert world.round_trip_ok(w)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib ~/.arcv/bin/python -m pytest tests/test_e125_world.py -q`
Expected: FAIL (`e125.world` missing).

- [ ] **Step 3: Write minimal implementation**

```python
# experiments/e125/world.py
"""Assemble a synthesized object-state predict() + goal energy into an openworld.World: FunctionTransition over
object state (carrying level_up), a CodePerceptor (frame->object state) and a CodeObjective (goal energy) for
to_spec. to_spec(world).preview.graph is the MAP; render_card the atlas; serve /view the UI. round_trip_ok
checks lossless serialization (from_spec(to_spec(w)) reproduces the rollout) -- structural integrity,
independent of real-data fidelity."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import openworld as O
from openworld import FunctionTransition, CodeObjective, CodePerceptor
from e125 import verify, objstate


def _act_name(action):
    return action.get("name") if isinstance(action, dict) else getattr(action, "name", action)


def build_world(predict_src, goal_src, initial_state, actions, game):
    predict_fn = verify.compile_predict(predict_src)

    def trans(state, action):
        nm = _act_name(action)
        act = nm if isinstance(nm, list) else list(eval(nm)) if isinstance(nm, str) and nm.startswith("[") else [nm]
        ns, lu = predict_fn({k: state[k] for k in ("bg", "objects")}, list(act))
        out = {"bg": ns["bg"], "objects": [dict(o) for o in ns["objects"]], "level_up": bool(lu)}
        return out

    w = O.World(name=f"e125-{game}", description=f"E125 object-state world model for {game}",
                initial_state={**initial_state, "level_up": False},
                actions=[str(a) for a in actions], transition=FunctionTransition(trans))
    w.perceptors = [CodePerceptor(objstate.PERCEIVE_SRC, modality="grid", produces={"objects": list})]
    w.objectives = [CodeObjective(goal_src, name="goal_energy")]
    return w


def round_trip_ok(w, steps=8):
    """from_spec(to_spec(w)) reproduces w's rollout under a fixed action cycle."""
    spec = O.to_spec(w)
    w2 = O.from_spec(spec, allow_code=True)
    if w2 is None:
        return False
    acts = w.actions
    s1 = dict(w.initial_state); s2 = dict(w2.initial_state)
    for i in range(steps):
        a = {"name": acts[i % len(acts)]}
        s1 = w.transition.step(s1, a); s2 = w2.transition.step(s2, a)
        if objstate.state_key(s1) != objstate.state_key(s2) or bool(s1.get("level_up")) != bool(s2.get("level_up")):
            return False
    return True
```

- [ ] **Step 4: Run test to verify it passes**

Run: `DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib ~/.arcv/bin/python -m pytest tests/test_e125_world.py -q`
Expected: PASS (3 passed). If `CodePerceptor`/`CodeObjective`/`to_spec` reject a kwarg or the perceptor must be sandbox-clean, adjust the constructor call to match the actual signature (inspect `openworld/perceive.py`, `openworld/reward.py`, `openworld/spec.py`) — keep the object-state contract identical.

- [ ] **Step 5: Run the full E125 suite (no regressions)**

Run: `DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib ~/.arcv/bin/python -m pytest tests/test_e125_*.py -q`
Expected: all pass (existing 48 + new objstate/verify_obj/world tests).

- [ ] **Step 6: Commit (after human approval)**

```bash
git add experiments/e125/world.py tests/test_e125_world.py
git commit -m "E125 P2.3: build_world -> openworld.World (object-state, round-trip verified)"
```

---

## Self-Review

**Spec coverage (Plan 1's slice):** object-centric state (decision 5) → Task 1; decision-equivalent gate (decision 6) → Task 2; OpenWorld-native World + map + lossless round-trip (decision 4) → Task 3. Deferred to Plan 2 (synth-over-objects + top-k ensemble + `goal_score` shaping) and Plan 3 (`plan_in_world` imagination-primary + `traverse.py` codex fallback + `solve_game` + rule-library transfer + live head-to-head). Each later plan produces working, testable software on its own.

**Placeholder scan:** none — every step has runnable test + implementation code. Task 3 Step 4 names a concrete fallback (inspect the real openworld signatures) rather than a vague "handle errors", because the exact `CodePerceptor`/`CodeObjective` kwargs must be confirmed against the installed `openworld` at implementation time.

**Type consistency:** `predict(state, action)->(next_state, level_up)` and the `{state,action,next_state,level_up}` transition shape are identical across Tasks 2–3; `state_key(s, fields)` signature matches between `objstate` (Task 1) and `verify.score_obj/check_obj` (Task 2); `build_world(predict_src, goal_src, initial_state, actions, game)` (Task 3) matches the interfaces block. `object_state` returns `{"bg","objects"}`; the World carries the extra `"level_up"` key in state (set by the transition, ignored by `state_key`).

## Follow-on plans (to be written after Plan 1 lands)

- **Plan 2 — object-state synthesis + ensemble:** adapt `synth` to render object-state transitions and synthesize `predict(state,action)` under the decision-equivalent gate (reusing the FunSearch DB / failure-memory / seeding); keep the **top-k** verified programs as an ensemble + a `disagreement(state, action)` signal; shape `goal_score` from grounded outcomes. Live: synthesize a verified object-state World for dc22.
- **Plan 3 — traversal + game loop:** `plan_in_world` (imagination-primary, energy heuristic, ensemble-gated) + `traverse.py` (codex-as-tool macro fallback, source-free, M0-audited) + `agent.solve_game` (commit+reset per level, rule-library transfer) + the head-to-head harness (levels solved · RHAE · fidelity · disagreement · model-based-plan success) vs the loose sweep agent.
