# E125 Phase-2 — Plan 3: Traversal engine (object planner + executor + traverse loop)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Phase-2 traversal engine: plan in the verified object-world by energy-descent (imagination), execute plans/macros against the real env halting on level-up or surprise, and a `traverse_level` loop that is imagination-primary with an LLM **macro fallback** gated by ensemble disagreement.

**Architecture:** `simworld.plan_obj` does best-first energy descent over OBJECT states. `execute.execute_obj` runs an action list against the real env in object-state, halting on a real level-up (solved) or a decision-key surprise (records an object transition). `traverse.traverse_level` plans in imagination first; if a plan exists and the ensemble agrees along it, it executes the verified plan; otherwise it asks an LLM (codex, with the Claude fallback already wired in synth) for a short macro and executes that — the env's `g.levels` decides the win.

**Tech Stack:** Python 3.14 (`~/.arcv/bin/python`, `DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib`), pytest. Reuses `experiments/e125/{objstate,verify,simworld,synth,execute,e124/codex_iso}`.

## Global Constraints

- **Object-state everywhere:** state = `{"bg":int,"objects":[{color,size,y,x},...]}` (+ `level_up` carried in World state). `predict(state,action)->(next_state,level_up)`; decision key via `objstate.state_key`.
- **Source-free + solution-free:** the macro LLM sees only object-states/the world derived from the agent's own exploration; never game source or banked answers; the env (`g.levels`) decides the win.
- **Mutation-safe:** always pass `copy.deepcopy(state)` into a synthesized predict (it may mutate in place).
- **Hermetic tests:** synthetic games + mock LLM runners; never call real codex/claude or the arc env in tests.
- **Don't break** the ~100 existing e125 tests. Add new modules/functions; reuse, don't rewrite.
- **Run with** `~/.arcv/bin/python`. **Commit only when asked** — per-task commits authorized; append `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

## File Structure

- Modify `experiments/e125/simworld.py` — add `plan_obj` (object-state best-first energy descent).
- Modify `experiments/e125/execute.py` — add `execute_obj` (object-state real-env executor + halt).
- Create `experiments/e125/traverse.py` — `MACRO_SCHEMA`, `_macro_prompt`, `traverse_level`.
- Tests: `tests/test_e125_plan_obj.py`, `tests/test_e125_execute_obj.py`, `tests/test_e125_traverse.py`.

Shapes: object transition `{"state","action","next_state","level_up"}`; a **world-model** dict `wm = {"predict_src","predict_fn","goal_src","goal_fn","ensemble"}`.

---

## Task 1: `simworld.plan_obj` — object-state energy-descent planner

**Files:**
- Modify: `experiments/e125/simworld.py`
- Test: `tests/test_e125_plan_obj.py`

**Interfaces:**
- Consumes: `objstate.state_key`.
- Produces: `simworld.plan_obj(predict_fn, initial_state, candidates_fn, budget, max_depth=40, goal_fn=None) -> list[action]|None` — best-first (heapq by `goal_fn(state)` energy, then depth) over object states; dedup by `state_key`; each node stores its own state (one `predict` call per expansion); returns the action list whose predicted `level_up` fires, or `None`. `candidates_fn(state) -> list[action]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_e125_plan_obj.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
from e125 import simworld, verify

# object world: action [4] moves the single object +1 in x (clamped 10); win when x==10.
PRED = ("def predict(state, action):\n"
        "    ns={'bg':state['bg'],'objects':[dict(o) for o in state['objects']]}\n"
        "    if action==[4]:\n        o=ns['objects'][0]; o['x']=min(10,o['x']+1)\n"
        "    return ns, bool(ns['objects'][0]['x']==10)")
GOAL = "def goal_score(state):\n    return float(10 - state['objects'][0]['x'])"
S0 = {"bg":0, "objects":[{"color":3,"size":1,"y":1,"x":1}]}
fn = verify.compile_predict(PRED); goal = verify.compile_goal(GOAL)

def test_plan_obj_finds_win_via_energy_descent():
    plan = simworld.plan_obj(fn, S0, lambda s:[[4],[2],[3]], budget=200, goal_fn=goal)
    assert plan == [[4]]*9

def test_plan_obj_dedups_noops_and_returns_none_when_unreachable():
    stay = verify.compile_predict("def predict(state, action):\n    return {'bg':state['bg'],'objects':[dict(o) for o in state['objects']]}, False")
    assert simworld.plan_obj(stay, S0, lambda s:[[2],[3]], budget=50, goal_fn=goal) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.arcv/bin/python -m pytest tests/test_e125_plan_obj.py -q` → FAIL (`plan_obj` missing).

- [ ] **Step 3: Write minimal implementation** (append to `experiments/e125/simworld.py`)

```python
import copy as _copy_sw
import math as _math_sw


def plan_obj(predict_fn, initial_state, candidates_fn, budget, max_depth=40, goal_fn=None):
    """Best-first energy descent over OBJECT states. Dedup by objstate.state_key; each frontier node carries its
    own state (one predict() per expansion). Returns the winning action list (predicted level_up) or None."""
    from e125 import objstate
    import heapq

    def energy(s):
        if goal_fn is None:
            return 0.0
        try:
            v = float(goal_fn(s))
        except Exception:
            return 1e18
        return v if _math_sw.isfinite(v) else 1e18

    seen = {objstate.state_key(initial_state)}
    counter = 0
    heap = [(energy(initial_state), 0, counter, initial_state, [])]
    n = 0
    while heap and n < budget:
        _, depth, _, state, actions = heapq.heappop(heap)
        if depth >= max_depth:
            continue
        for st in (s if isinstance(s, list) else [s] for s in candidates_fn(state)):
            try:
                ns, lu = predict_fn(_copy_sw.deepcopy(state), list(st))
            except Exception:
                continue
            n += 1
            if lu:
                return actions + [st]
            key = objstate.state_key(ns)
            if key not in seen:
                seen.add(key); counter += 1
                heapq.heappush(heap, (energy(ns), depth + 1, counter, ns, actions + [st]))
            if n >= budget:
                break
    return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.arcv/bin/python -m pytest tests/test_e125_plan_obj.py -q` → PASS (2).

- [ ] **Step 5: Commit (authorized)**

```bash
git add experiments/e125/simworld.py tests/test_e125_plan_obj.py
git commit -m "E125 P3.1: plan_obj (object-state energy-descent planner)"
```

---

## Task 2: `execute.execute_obj` — object-state real-env executor + halt

**Files:**
- Modify: `experiments/e125/execute.py`
- Test: `tests/test_e125_execute_obj.py`

**Interfaces:**
- Consumes: `objstate.state_key`.
- Produces: `execute.execute_obj(real_game, actions, predict_fn, perceive, do_reset=True) -> {"solved":bool, "verified_prefix":list, "new_transitions":list[objtrans], "halt_step":int|None}`. `perceive(frame)->state` (e.g. `objstate.object_state`). Steps the real game per action; **solved** on a real `levels` bump; **halt** (record an object transition `{state,action,next_state,level_up:False}`) on a decision-key mismatch (`state_key(pred)!=state_key(real)`) OR a refuted win (predict said `level_up` but env didn't bump). `do_reset=False` continues from the game's current state.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_e125_execute_obj.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
import numpy as np
from e125 import execute, verify, objstate

# real game: action [4] moves a color-3 size-1 object right along row 1; win at x==4 (col 4).
class RealObjGame:
    def __init__(self): self.reset()
    def reset(self): self.x=1; self.levels=0; self.done=False; self._draw()
    def _draw(self):
        self.frame=np.zeros((8,8),dtype=int); self.frame[1,self.x]=3
    def step(self,a,x=None,y=None):
        if a==4: self.x=min(7,self.x+1)
        self._draw()
        if self.x==4: self.levels=1; self.done=True

PRED = ("def predict(state, action):\n"
        "    ns={'bg':state['bg'],'objects':[dict(o) for o in state['objects']]}\n"
        "    if action==[4]:\n        o=ns['objects'][0]; o['x']=o['x']+1\n"
        "    return ns, bool(ns['objects'][0]['x']==4)")
fn = verify.compile_predict(PRED)
perc = lambda f: objstate.object_state(f)

def test_execute_obj_solves_on_real_levelup():
    r = execute.execute_obj(RealObjGame(), [[4],[4],[4]], fn, perc)
    assert r["solved"] is True and r["halt_step"] is None and r["verified_prefix"] == [[4],[4],[4]]

def test_execute_obj_halts_on_surprise():
    # model says [4] moves +1 but real game also has a wall: redefine a game where [4] does NOTHING -> surprise
    class Stuck(RealObjGame):
        def step(self,a,x=None,y=None): self._draw()   # never moves
    r = execute.execute_obj(Stuck(), [[4]], fn, perc)
    assert r["solved"] is False and r["halt_step"] == 1 and len(r["new_transitions"]) == 1
    assert r["new_transitions"][0]["level_up"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.arcv/bin/python -m pytest tests/test_e125_execute_obj.py -q` → FAIL (`execute_obj` missing).

- [ ] **Step 3: Write minimal implementation** (append to `experiments/e125/execute.py`)

```python
import copy as _copy_ex


def execute_obj(real_game, actions, predict_fn, perceive, do_reset=True):
    """Run an action list vs the REAL env in OBJECT state. Solved on a real levels bump; halt+record an object
    transition on a decision-key surprise or a refuted win hypothesis. do_reset=False continues from current."""
    from e125 import objstate
    if do_reset:
        real_game.reset()
    base = real_game.levels
    cur = perceive(real_game.frame)
    verified, new_trans = [], []
    for i, a in enumerate(actions):
        try:
            pred_ns, pred_lu = predict_fn(_copy_ex.deepcopy(cur), list(a))
        except Exception:
            pred_ns, pred_lu = cur, False
        real_game.step(*a)
        real_ns = perceive(real_game.frame)
        if real_game.levels > base:
            verified.append(a)
            return {"solved": True, "verified_prefix": verified, "new_transitions": new_trans, "halt_step": None}
        if objstate.state_key(pred_ns) != objstate.state_key(real_ns) or pred_lu:
            new_trans.append({"state": cur, "action": list(a), "next_state": real_ns, "level_up": False})
            return {"solved": False, "verified_prefix": verified, "new_transitions": new_trans, "halt_step": i + 1}
        verified.append(a); cur = real_ns
    return {"solved": False, "verified_prefix": verified, "new_transitions": new_trans, "halt_step": None}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.arcv/bin/python -m pytest tests/test_e125_execute_obj.py -q` → PASS (2).

- [ ] **Step 5: Commit (authorized)**

```bash
git add experiments/e125/execute.py tests/test_e125_execute_obj.py
git commit -m "E125 P3.2: execute_obj (object-state real-env executor + halt)"
```

---

## Task 3: `traverse.traverse_level` — imagination-primary loop + macro fallback

**Files:**
- Create: `experiments/e125/traverse.py`
- Test: `tests/test_e125_traverse.py`

**Interfaces:**
- Consumes: `simworld.plan_obj`, `execute.execute_obj`, `synth.ensemble_disagreement`, `synth._objs`, `objstate.object_state`, `codex_iso.run`.
- Produces: `traverse.MACRO_SCHEMA`; `traverse._macro_prompt(state, action_api, predict_src, goal_src, history) -> str`; `traverse.traverse_level(game_factory, candidates_fn, wm, action_api, game, macro_runner=None, perceive=None, committed=None, budget_plan=20000, max_macros=8, stall_macros=3, disagreement_thresh=0.0, traces_dir=None) -> {"solved":bool,"actions":list,"new_transitions":list,"reason":str,"macros_used":int}`. `wm = {"predict_src","predict_fn","goal_src","goal_fn","ensemble"}`. Strategy each round: (1) `plan_obj` in imagination; if a plan exists AND `max ensemble_disagreement` along it `<= disagreement_thresh`, execute it (`execute_obj`); (2) else ask `macro_runner` for a 3–5 action macro and execute that. Solved on a real level-up → return solved+actions. A halt records `new_transitions` and returns (so the caller re-synthesizes). Stall (no progress for `stall_macros` rounds) or `max_macros` exhausted → return unsolved with a reason. **No banked answers.**

- [ ] **Step 1: Write the failing test**

```python
# tests/test_e125_traverse.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
import numpy as np
from e125 import traverse, verify, objstate

class RealObjGame:
    def __init__(self): self.reset()
    def reset(self): self.x=1; self.levels=0; self.done=False; self._draw()
    def _draw(self):
        self.frame=np.zeros((8,8),dtype=int); self.frame[1,self.x]=3
    def step(self,a,x=None,y=None):
        if a==4: self.x=min(7,self.x+1)
        self._draw()
        if self.x==4: self.levels=1; self.done=True

PRED=("def predict(state, action):\n    ns={'bg':state['bg'],'objects':[dict(o) for o in state['objects']]}\n"
      "    if action==[4]:\n        o=ns['objects'][0]; o['x']=o['x']+1\n    return ns, bool(ns['objects'][0]['x']==4)")
GOAL="def goal_score(state):\n    return float(4 - state['objects'][0]['x'])"
fn=verify.compile_predict(PRED); goal=verify.compile_goal(GOAL)
WM={"predict_src":PRED,"predict_fn":fn,"goal_src":GOAL,"goal_fn":goal,"ensemble":[fn]}
perc=lambda f: objstate.object_state(f)

def test_traverse_solves_via_imagination_plan():
    # ensemble agrees (single fn), plan_obj finds [4]*3 to x==4 -> executes verified plan -> real level-up
    r = traverse.traverse_level(RealObjGame, lambda s:[[4],[2]], WM, "actions=[2,4]", "g",
                                perceive=perc, budget_plan=200)
    assert r["solved"] is True and r["actions"] == [[4],[4],[4]] and r["macros_used"] == 0

def test_traverse_uses_macro_fallback_when_no_plan():
    # goal_fn=None + a predict whose level_up NEVER fires -> no imagination plan -> macro fallback solves
    nowin=verify.compile_predict("def predict(state, action):\n    ns={'bg':state['bg'],'objects':[dict(o) for o in state['objects']]}\n    if action==[4]:\n        ns['objects'][0]['x']=ns['objects'][0]['x']+1\n    return ns, False")
    wm2={"predict_src":"src","predict_fn":nowin,"goal_src":None,"goal_fn":None,"ensemble":[nowin]}
    def macro_runner(prompt, schema, model, game, **kw):
        return {"final":{"macro":[[4],[4],[4]],"rationale":"go right","goal_note":"x->4"},
                "events":[],"tainted":False,"raw":"","model_version":""}
    r = traverse.traverse_level(RealObjGame, lambda s:[[4],[2]], wm2, "actions=[2,4]", "g",
                                macro_runner=macro_runner, perceive=perc, budget_plan=50)
    assert r["solved"] is True and r["macros_used"] >= 1

def test_traverse_abandons_without_banked_answers():
    nowin=verify.compile_predict("def predict(state, action):\n    return {'bg':state['bg'],'objects':[dict(o) for o in state['objects']]}, False")
    wm2={"predict_src":"s","predict_fn":nowin,"goal_src":None,"goal_fn":None,"ensemble":[nowin]}
    def macro_runner(prompt, schema, model, game, **kw):
        return {"final":{"macro":[[2]],"rationale":"x","goal_note":"x"},"events":[],"tainted":False,"raw":"","model_version":""}
    r = traverse.traverse_level(RealObjGame, lambda s:[[2]], wm2, "actions=[2]", "g",
                                macro_runner=macro_runner, perceive=perc, budget_plan=20, max_macros=3)
    assert r["solved"] is False and "macros_used" in r
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.arcv/bin/python -m pytest tests/test_e125_traverse.py -q` → FAIL (`traverse` missing).

- [ ] **Step 3: Write minimal implementation**

```python
# experiments/e125/traverse.py
"""Phase-2 traversal: drive a level by PLANNING in the verified object-world (imagination-primary), executing
verified plans/macros against the REAL env. Imagination plan first; if the ensemble agrees along it, execute it;
else ask an LLM (codex; the Claude fallback is already wired in synth) for a short MACRO and execute that. The
env's g.levels decides the win. Source-free + solution-free -- the LLM never sees game source or banked answers."""
import os, sys, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from e125 import simworld, execute, synth, objstate
from e124 import codex_iso

MACRO_SCHEMA = {"type": "object", "additionalProperties": False,
                "required": ["macro", "rationale", "goal_note"],
                "properties": {"macro": {"type": "array", "items": {"type": "array", "items": {"type": "integer"}}},
                               "rationale": {"type": "string"}, "goal_note": {"type": "string"}}}


def _macro_prompt(state, action_api, predict_src, goal_src, history):
    hist = "\n".join(f"- tried {h['macro']} -> {h['outcome']}" for h in history[-6:]) or "(none yet)"
    return ("You are solving an unknown grid game by acting. You have a VERIFIED world model (predict) and a goal "
            f"energy (goal_score) of the current OBJECT state. Propose a SHORT macro (3-5 actions) toward the win.\n\n"
            f"predict():\n```python\n{predict_src}\n```\ngoal_score():\n```python\n{goal_src}\n```\n"
            f"Current state: {synth._objs(state)}\nActions: {action_api}\n"
            f"Macros already tried (do not repeat fruitless ones):\n{hist}\n\n"
            "Return JSON {macro: [[a],...], rationale, goal_note}. macro is a list of actions like [[4],[4],[6,3,5]].")


def _max_disagreement(plan, predict_fn, ensemble, initial_state):
    """Replay the plan through predict_fn; at each state measure ensemble disagreement; return the max."""
    if not ensemble or len(ensemble) <= 1:
        return 0.0
    s = initial_state; worst = 0.0
    for a in plan:
        worst = max(worst, synth.ensemble_disagreement(ensemble, s, a))
        try:
            s, _ = predict_fn(copy.deepcopy(s), list(a))
        except Exception:
            break
    return worst


def traverse_level(game_factory, candidates_fn, wm, action_api, game, macro_runner=None, perceive=None,
                   committed=None, budget_plan=20000, max_macros=8, stall_macros=3, disagreement_thresh=0.0,
                   traces_dir=None):
    perceive = perceive or objstate.object_state
    run = macro_runner or codex_iso.run
    committed = list(committed or [])
    predict_fn = wm["predict_fn"]; goal_fn = wm.get("goal_fn"); ensemble = wm.get("ensemble") or [predict_fn]
    history = []; new_trans = []; macros_used = 0; stall = 0

    def _state_after(prefix):
        g = game_factory(); g.reset()
        for a in prefix:
            g.step(*a)
        return g, perceive(g.frame)

    for _ in range(max_macros):
        _, init_state = _state_after(committed)
        plan = simworld.plan_obj(predict_fn, init_state, candidates_fn, budget_plan, goal_fn=goal_fn)
        use_plan = plan is not None and _max_disagreement(plan, predict_fn, ensemble, init_state) <= disagreement_thresh
        if use_plan:
            actions = plan
        else:
            res = run(_macro_prompt(init_state, action_api, wm.get("predict_src"), wm.get("goal_src"), history),
                      MACRO_SCHEMA, "gpt-5.5", game)
            macro = (res.get("final") or {}).get("macro") or []
            actions = [list(a) for a in macro if a]
            macros_used += 1
            if not actions:
                stall += 1
                if stall >= stall_macros:
                    return {"solved": False, "actions": committed, "new_transitions": new_trans,
                            "reason": "no macro", "macros_used": macros_used}
                continue
        rg = game_factory(); rg.reset()
        for a in committed:
            rg.step(*a)
        r = execute.execute_obj(rg, actions, predict_fn, perceive, do_reset=False)
        committed += r["verified_prefix"]
        if r["solved"]:
            return {"solved": True, "actions": committed, "new_transitions": new_trans, "reason": "solved",
                    "macros_used": macros_used}
        if r["new_transitions"]:
            new_trans += r["new_transitions"]
            return {"solved": False, "actions": committed, "new_transitions": new_trans, "reason": "surprise",
                    "macros_used": macros_used}
        history.append({"macro": actions, "outcome": "no progress"})
        stall += 1
        if stall >= stall_macros:
            return {"solved": False, "actions": committed, "new_transitions": new_trans, "reason": "stall",
                    "macros_used": macros_used}
    return {"solved": False, "actions": committed, "new_transitions": new_trans, "reason": "max_macros",
            "macros_used": macros_used}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.arcv/bin/python -m pytest tests/test_e125_traverse.py -q` → PASS (3).

- [ ] **Step 5: Run full e125 suite (no regressions)**

Run: `DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib ~/.arcv/bin/python -m pytest tests/test_e125_*.py -q` → all pass.

- [ ] **Step 6: Commit (authorized)**

```bash
git add experiments/e125/traverse.py tests/test_e125_traverse.py
git commit -m "E125 P3.3: traverse_level (imagination-primary + macro fallback)"
```

---

## Self-Review

**Spec coverage:** plan-in-imagination primary (decision 2) → Task 1 + Task 3 (plan_obj used first); ensemble-gated fallback (decision 7) → Task 3 (`_max_disagreement` vs `disagreement_thresh`); real-env execution + halt-on-surprise/refuted-win + online oracle (decisions 2,9) → Task 2 + Task 3; per-macro cadence (decision 3) → Task 3 macro path. Deferred to Plan 3.5: `solve_game` (commit+reset per level, rule-library transfer), entry `--mode traverse`, head-to-head metrics, live dc22 run.

**Placeholder scan:** none — every step has runnable test + implementation code; live runs are explicitly controller-run in Plan 3.5, not subagent steps.

**Type consistency:** object transition `{state,action,next_state,level_up}` and `state_key` usage match across plan_obj/execute_obj/traverse and Plan 1-2 (`verify.score_obj`, `objstate`); `predict(state,action)->(next_state,level_up)` consistent; `wm` dict keys (`predict_src/predict_fn/goal_src/goal_fn/ensemble`) are produced by Plan 2's `synthesize_obj` 4-tuple (the caller in Plan 3.5 assembles the dict); `ensemble_disagreement(fns,state,action)` and `_objs(state)` reused from synth; `execute_obj(real_game,actions,predict_fn,perceive,do_reset)` matches its test and the traverse call.
