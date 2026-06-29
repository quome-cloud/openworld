# E125 Phase-2 — Plan 3.5: Game loop + entry + live run

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the Phase-1 (object-state FunSearch world) + Phase-2 (traversal) pieces into a per-level `solve_game` loop with object exploration, commit+reset per level, and rule-library transfer; expose it via the entry point; then run it live on dc22.

**Architecture:** `explorer.collect_obj` gathers object-state transitions from the real env. `agent.solve_game` loops per level: explore → `synth_obj_fn` (object FunSearch, seeded from the prior level's verified program) → assemble a `wm` → `traverse_level` → on a real level-up, extend the solution and carry the verified program forward; on surprise, re-synthesize; on stall, stop. The entry wires `SandboxGame` + an object candidates_fn + a metrics dict.

**Tech Stack:** Python 3.14 (`~/.arcv/bin/python`, `DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib`), pytest. Reuses `experiments/e125/{objstate,synth,traverse,explorer}`, `experiments/arc3_sandbox`, `experiments/e119/perceive`.

## Global Constraints

- **Object-state everywhere:** `candidates_fn(state)` takes an OBJECT state; transitions are `{state,action,next_state,level_up}`; `perceive(frame)->state` = `objstate.object_state`.
- **Source-free + solution-free:** explore by acting; the LLM (codex/Claude) sees only object-states + the agent's own programs; the env (`g.levels`) decides the win; never banked answers.
- **`synth_obj_fn` contract:** `synth_obj_fn(transitions, action_api, game, seed_src=None) -> (src, predict_fn, goal_fn, ensemble)` (Plan 2's `synthesize_obj` shape; the entry wraps it).
- **Hermetic tests:** synthetic multi-level games + mock `synth_obj_fn`/`macro_runner`; never the real arc env or LLMs.
- **Don't break** the ~110 existing e125 tests. **Run with** `~/.arcv/bin/python`. **Commit only when asked** — per-task commits authorized; append `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

## File Structure

- Modify `experiments/e125/explorer.py` — add `collect_obj`.
- Modify `experiments/e125/agent.py` — add `solve_game`.
- Modify `experiments/e125_executable_world.py` — add `--mode traverse`, `_obj_candidates_fn`, object `solve_game` wiring + metrics.
- Tests: `tests/test_e125_collect_obj.py`, `tests/test_e125_solve_game.py`, extend `tests/test_e125_entry.py`.

---

## Task 1: `explorer.collect_obj` — object-state exploration

**Files:**
- Modify: `experiments/e125/explorer.py`
- Test: `tests/test_e125_collect_obj.py`

**Interfaces:**
- Consumes: `objstate.state_key`.
- Produces: `explorer.collect_obj(game_factory, candidates_fn, budget, perceive, prefix=None) -> list[objtrans]` — replays `prefix` then round-robins candidates, perceiving each frame to an object state; dedups by `(state_key(state), tuple(action))`; records `{state,action,next_state,level_up}`. `candidates_fn(state)->list[action]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_e125_collect_obj.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
import numpy as np
from e125 import explorer, objstate

class G:
    def __init__(self): self.reset()
    def reset(self): self.x=1; self.levels=0; self.done=False; self._d()
    def _d(self): self.frame=np.zeros((8,8),dtype=int); self.frame[1,self.x]=3
    def step(self,a,x=None,y=None):
        if a==4: self.x=min(7,self.x+1)
        self._d()
perc = lambda f: objstate.object_state(f)

def test_collect_obj_records_object_transitions():
    tr = explorer.collect_obj(G, lambda s:[[4]], budget=3, perceive=perc)
    assert len(tr) == 3
    assert tr[0]["state"]["objects"][0]["x"] == 1 and tr[0]["next_state"]["objects"][0]["x"] == 2
    assert all(set(t) == {"state","action","next_state","level_up"} for t in tr)

def test_collect_obj_replays_prefix():
    tr = explorer.collect_obj(G, lambda s:[[4]], budget=1, perceive=perc, prefix=[[4],[4]])
    assert tr[0]["state"]["objects"][0]["x"] == 3   # started after 2x [4]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.arcv/bin/python -m pytest tests/test_e125_collect_obj.py -q` → FAIL.

- [ ] **Step 3: Write minimal implementation** (append to `experiments/e125/explorer.py`)

```python
def collect_obj(game_factory, candidates_fn, budget, perceive, prefix=None):
    """Object-state exploration: replay prefix, then round-robin candidates, perceiving each frame to an object
    state; dedup by (state_key, action). Returns object transitions {state,action,next_state,level_up}."""
    from e125 import objstate
    prefix = list(prefix or [])

    def _fresh():
        g = game_factory(); g.reset()
        for a in prefix:
            g.step(*a)
        return g

    g = _fresh()
    trans, seen = [], set()
    for _ in range(budget):
        state = perceive(g.frame)
        cands = [s if isinstance(s, list) else [s] for s in candidates_fn(state)]
        if not cands:
            break
        a = cands[len(trans) % len(cands)]
        lv = g.levels
        g.step(*a)
        nstate = perceive(g.frame)
        key = (objstate.state_key(state), tuple(a))
        if key not in seen:
            seen.add(key)
            trans.append({"state": state, "action": list(a), "next_state": nstate, "level_up": g.levels > lv})
        if g.done:
            g = _fresh()
    return trans
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.arcv/bin/python -m pytest tests/test_e125_collect_obj.py -q` → PASS (2).

- [ ] **Step 5: Commit (authorized)**

```bash
git add experiments/e125/explorer.py tests/test_e125_collect_obj.py
git commit -m "E125 P3.5.1: collect_obj (object-state exploration)"
```

---

## Task 2: `agent.solve_game` — per-level loop + transfer

**Files:**
- Modify: `experiments/e125/agent.py`
- Test: `tests/test_e125_solve_game.py`

**Interfaces:**
- Consumes: `explorer.collect_obj`, `traverse.traverse_level`, `objstate`.
- Produces: `agent.solve_game(game_factory, candidates_fn, action_api, game, synth_obj_fn, perceive=None, macro_runner=None, budget_explore=60, budget_plan=20000, rounds_per_level=4, max_levels=9, max_macros=8, traces_dir=None) -> {"levels_solved":int, "solution":list, "levels":list[dict], "real_actions":int}`. Per level: explore (with the running solution as prefix) → up to `rounds_per_level` of `synth_obj_fn` (seeded from the prior level's verified `src` = rule-library transfer) → build `wm` → `traverse_level(committed=solution)`. On a real level-up: `solution = result["actions"]`, carry `src` forward, advance level. On surprise: append `new_transitions`, re-synth. On stall/no-model: stop. Drops trailing no-op actions is NOT required (traverse handles win detection); just record what solved.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_e125_solve_game.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
import numpy as np
from e125 import agent, verify, objstate

# 2-level game: level 0 win at x==3 (move [4]); level 1 win at x==6 (keep moving [4]). object color 3 row 1.
class TwoLevel:
    def __init__(self): self.reset()
    def reset(self): self.x=1; self.levels=0; self.done=False; self._d()
    def _d(self): self.frame=np.zeros((8,8),dtype=int); self.frame[1,self.x]=3
    def step(self,a,x=None,y=None):
        if a==4: self.x=min(7,self.x+1)
        self._d()
        if self.levels==0 and self.x==3: self.levels=1
        elif self.levels==1 and self.x==6: self.levels=2; self.done=True

PRED=("def predict(state, action):\n    ns={'bg':state['bg'],'objects':[dict(o) for o in state['objects']]}\n"
      "    if action==[4]:\n        ns['objects'][0]['x']=ns['objects'][0]['x']+1\n"
      "    return ns, False")   # dynamics correct; win discovered by the env oracle, not the model
GOAL="def goal_score(state):\n    return float(9 - state['objects'][0]['x'])"
perc=lambda f: objstate.object_state(f)

def _synth(transitions, action_api, game, seed_src=None, **kw):
    fn=verify.compile_obj_predict(PRED); goal=verify.compile_goal(GOAL)
    return PRED, fn, goal, [fn]

def _macro(prompt, schema, model, game, **kw):
    return {"final":{"macro":[[4],[4],[4]],"rationale":"right","goal_note":"x up"},
            "events":[],"tainted":False,"raw":"","model_version":""}

def test_solve_game_solves_two_levels_with_transfer():
    r = agent.solve_game(TwoLevel, lambda s:[[4],[2]], "actions=[2,4]", "g", _synth, perceive=perc,
                         macro_runner=_macro, budget_explore=6, budget_plan=50, rounds_per_level=3, max_levels=2)
    assert r["levels_solved"] == 2
    assert r["real_actions"] > 0 and len(r["levels"]) >= 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.arcv/bin/python -m pytest tests/test_e125_solve_game.py -q` → FAIL.

- [ ] **Step 3: Write minimal implementation** (append to `experiments/e125/agent.py`)

```python
def solve_game(game_factory, candidates_fn, action_api, game, synth_obj_fn, perceive=None, macro_runner=None,
               budget_explore=60, budget_plan=20000, rounds_per_level=4, max_levels=9, max_macros=8,
               traces_dir=None):
    """Per-level loop over the verified object-world. Explore -> synth_obj (seeded from the prior level's
    verified program = rule-library transfer) -> traverse_level (committed=solution so far). On a real level-up
    extend the solution and carry the program forward; on surprise re-synth; on stall stop. The env decides wins."""
    from e125 import explorer, traverse, objstate
    perceive = perceive or objstate.object_state
    solution = []                 # actions through solved levels
    rule_src = None               # last verified predict src (transfer seed for the next level)
    levels = []; real_actions = 0
    for level in range(max_levels):
        trans = explorer.collect_obj(game_factory, candidates_fn, budget_explore, perceive, prefix=solution)
        real_actions += budget_explore
        last_src = rule_src; solved = False; reason = "no model"
        for _ in range(rounds_per_level):
            src, fn, goal_fn, ensemble = synth_obj_fn(trans, action_api, game, seed_src=last_src,
                                                      traces_dir=traces_dir)
            if fn is None:
                reason = "no verified predict()"; break
            last_src = src
            wm = {"predict_src": src, "predict_fn": fn, "goal_src": None, "goal_fn": goal_fn, "ensemble": ensemble}
            res = traverse.traverse_level(game_factory, candidates_fn, wm, action_api, game,
                                          macro_runner=macro_runner, perceive=perceive, committed=list(solution),
                                          budget_plan=budget_plan, max_macros=max_macros, traces_dir=traces_dir)
            real_actions += max(0, len(res["actions"]) - len(solution)) + res["macros_used"]
            reason = res["reason"]
            if res["solved"]:
                solution = res["actions"]; rule_src = src; solved = True; break
            if res["new_transitions"]:
                trans = trans + res["new_transitions"]      # surprise -> re-synthesize (seeded)
            else:
                break                                        # stall/no progress
        levels.append({"level": level, "solved": solved, "reason": reason})
        if not solved:
            break
    return {"levels_solved": sum(1 for l in levels if l["solved"]), "solution": solution,
            "levels": levels, "real_actions": real_actions}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.arcv/bin/python -m pytest tests/test_e125_solve_game.py -q` → PASS.

- [ ] **Step 5: Run full e125 suite (no regressions)**

Run: `DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib ~/.arcv/bin/python -m pytest tests/test_e125_*.py -q` → all pass.

- [ ] **Step 6: Commit (authorized)**

```bash
git add experiments/e125/agent.py tests/test_e125_solve_game.py
git commit -m "E125 P3.5.2: solve_game (per-level loop + rule-library transfer)"
```

---

## Task 3: Entry `--mode traverse` + object candidates + metrics

**Files:**
- Modify: `experiments/e125_executable_world.py`
- Test: `tests/test_e125_entry.py` (extend)

**Interfaces:**
- Produces: `_obj_candidates_fn(avail) -> (state)->list[action]` (directional simple actions; plus click `[6,x,y]` targets from small objects in the state when 6 is available); a `--mode {structured,traverse}` arg whose `traverse` branch runs `agent.solve_game` over `SandboxGame` with `synth.synthesize_obj` and `claude_iso.run` as the fallback runner, and writes a metrics dict via `save_results`.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_e125_entry.py
import numpy as np
def test_obj_candidates_directional():
    cands = entry._obj_candidates_fn([1,2,3,4])
    st = {"bg":0,"objects":[{"color":3,"size":1,"y":2,"x":5}]}
    assert cands(st) == [[1],[2],[3],[4]]

def test_obj_candidates_includes_clicks_from_objects():
    cands = entry._obj_candidates_fn([1,2,3,4,6])
    st = {"bg":0,"objects":[{"color":3,"size":1,"y":2,"x":5}]}
    out = cands(st)
    assert [1] in out and [6,5,2] in out          # click target at (x=5,y=2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib ~/.arcv/bin/python -m pytest tests/test_e125_entry.py -q` → FAIL (`_obj_candidates_fn` missing).

- [ ] **Step 3: Write minimal implementation** (add to `experiments/e125_executable_world.py`)

```python
def _obj_candidates_fn(avail):
    """Action candidates from an OBJECT state: directional/simple actions always; plus click [6,x,y] targets at
    each small object's position when action 6 is available (x=col, y=row)."""
    simple = [x for x in avail if x in (1, 2, 3, 4, 5, 7)]
    if 6 in avail:
        return lambda s: ([[x] for x in simple]
                          + [[6, int(o["x"]), int(o["y"])] for o in s.get("objects", []) if o.get("size", 99) <= 40])
    return lambda s: [[x] for x in simple]
```

Then add a `--mode` arg in `main()` and a `traverse` branch (place after the existing structured path):

```python
    ap.add_argument("--mode", default="structured", choices=["structured", "traverse"])
    # ... inside the per-game loop, when a.mode == "traverse":
    #   from e125 import agent, synth, claude_iso, objstate
    #   avail = list(g.avail); g.close()
    #   cands = _obj_candidates_fn(avail)
    #   sfn = lambda tr, api, gm, **kw: synth.synthesize_obj(tr, api, gm, model=a.model,
    #             fallback_runner=lambda p,s,m,gg,**k: claude_iso.run(p, s, model="claude-opus-4-8", game=gg),
    #             **kw)
    #   res = agent.solve_game(lambda: SandboxGame(gid), cands, f"actions={avail}", gid, sfn,
    #             perceive=objstate.object_state, macro_runner=None,
    #             budget_explore=a.budget_explore, budget_plan=a.budget_plan, rounds_per_level=a.rounds,
    #             max_levels=a.max_levels, traces_dir=a.traces)
    #   results[gid] = {k: res[k] for k in ("levels_solved","real_actions","levels")}
```

Add `--max-levels` (default 9). Keep the existing `structured` path unchanged. `save_results("e125_executable_world", {...})` before any assert.

- [ ] **Step 4: Run test to verify it passes**

Run: `DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib ~/.arcv/bin/python -m pytest tests/test_e125_entry.py -q` → PASS.

- [ ] **Step 5: Run full e125 suite + import-smoke the entry**

Run: `DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib ~/.arcv/bin/python -m pytest tests/test_e125_*.py -q` → all pass.
Run: `~/.arcv/bin/python -c "import sys; sys.path.insert(0,'experiments'); import e125_executable_world"` → no error.

- [ ] **Step 6: Commit (authorized)**

```bash
git add experiments/e125_executable_world.py tests/test_e125_entry.py
git commit -m "E125 P3.5.3: entry --mode traverse + object candidates + metrics"
```

---

## Self-Review

**Spec coverage:** object exploration → Task 1; per-level loop + commit/reset/transfer (decisions 1,8) → Task 2; entry + object candidates + metrics (decision: head-to-head metrics) → Task 3. Live dc22 run is controller-run after the plan lands (below).

**Placeholder scan:** Task 3's `--mode traverse` branch is given as a precise comment-block edit against the existing `main()`; the implementer wires it verbatim. No vague directives.

**Type consistency:** `candidates_fn(state)` is object-state throughout (collect_obj, solve_game, traverse, plan_obj); `synth_obj_fn(...)->(src,fn,goal_fn,ensemble)` matches Plan 2 `synthesize_obj`; `wm` keys match Plan 3 `traverse_level`; `perceive=objstate.object_state`; `claude_iso.run` as `fallback_runner` matches Plan 2.5's runner contract; metrics dict keys (`levels_solved/real_actions/levels`) match `solve_game`'s return.

## Live validation (controller-run, after the plan lands — the "try it")

On the cleanest pilot dc22 (object-clean): run
`DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib ~/.arcv/bin/python experiments/e125_executable_world.py --games dc22 --mode traverse --max-levels 1 --rounds 3 --budget-explore 40 --budget-plan 20000`.
Report **honestly**: did Phase-1 synthesize a verified object-world? did Phase-2 reach a real level-up (g.levels bump)? levels solved · real-env actions · codex/Claude calls. Watch the arc3 reset-pollution caveat — for L0 a shared `SandboxGame` reset is fine; deeper levels need a fresh-process replayer (note if it bites). No banked answers.
