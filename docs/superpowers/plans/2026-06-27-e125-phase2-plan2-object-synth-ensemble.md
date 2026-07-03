# E125 Phase-2 — Plan 2: Object-state FunSearch synthesis + ensemble

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Synthesize a verified **object-state** `predict(state, action) -> (next_state, level_up)` via FunSearch under the decision-equivalent gate, and keep the **top-k** verified programs as an ensemble with a `disagreement` signal (epistemic uncertainty for Plan 3's confidence gate).

**Architecture:** Reuse Plan 1's object representation (`objstate`), decision-equivalent gate + sandbox-env compile (`verify.score_obj`/`check_obj`/`compile_obj_predict`), and `build_world`. Add an OBJECT-state synthesis path to `synth.py` that renders object-state transitions for codex, runs the existing FunSearch `_Database` loop (clusters + Boltzmann + length pref + k-shot ascending-versioned prompt + failure memory + seed-within-level), scores via the object gate, and returns the verified program plus a top-k ensemble.

**Tech Stack:** Python 3.14 (`~/.arcv/bin/python`, `DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib`), pytest. Reuses `experiments/e125/{synth,verify,objstate,world}`, `experiments/e124/codex_iso`, `scripts/capture_lib`.

## Global Constraints

- **Source-free + solution-free:** codex sees only object-states derived from observed frames; M0 audit via `codex_iso`; tainted call discarded.
- **`predict` contract (object-state):** `predict(state, action) -> (next_state, level_up)`, `state = {"bg": int, "objects": [{"color","size","y","x"}, ...]}` (the World adds `level_up`); compiled with `verify.compile_obj_predict` (OpenWorld SAFE_BUILTINS — **no numpy/import**; gate env ⊆ World sandbox).
- **Decision-equivalent gate:** accept iff `verify.check_obj` passes (object `state_key` + `level_up`), default `fields=("color","y","x")`.
- **Object transition shape:** `{"state": dict, "action": list[int], "next_state": dict, "level_up": bool}`.
- **Do not break the existing FRAME synth path** (`synthesize`, `_prompt`, `_funsearch_prompt`, `score_program`, the 70 passing tests). Add the object path alongside; share `_Database`/`_softmax`/`_rename_fn`/`failed_summaries`.
- **Run with** `~/.arcv/bin/python`. **Commit only when asked** — per-task commits are authorized for this plan; append `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

## File Structure

- Modify `experiments/e125/synth.py` — add object rendering/prompts (Task 1), ensemble primitives (Task 2), `synthesize_obj` (Task 3). Leave the frame path intact.
- Tests: `tests/test_e125_synth_obj.py` (Tasks 1 & 3), `tests/test_e125_ensemble.py` (Task 2).

---

## Task 1: Object-state rendering + prompts

**Files:**
- Modify: `experiments/e125/synth.py` (add functions; do not touch existing ones)
- Test: `tests/test_e125_synth_obj.py`

**Interfaces:**
- Consumes: `objstate.state_key`.
- Produces: `synth.render_obj_transitions(transitions, k=12) -> str`; `synth._obj_prompt(transitions, action_api, counterexample=None) -> str`; `synth._obj_diff(fails, fields=("color","y","x"), k=3) -> str`; `synth._obj_funsearch_prompt(samples, action_api, failed=None) -> str`. `samples` are program dicts `{"src","score","fails"}` (as produced by `_Database`). All prompts instruct the `predict(state, action)` object-state contract and ask for `{predict_src, goal_score_src, rationale}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_e125_synth_obj.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
from e125 import synth

S0 = {"bg": 0, "objects": [{"color": 3, "size": 1, "y": 1, "x": 1}]}
S1 = {"bg": 0, "objects": [{"color": 3, "size": 1, "y": 1, "x": 2}]}
def _t(s, a, ns, lu): return {"state": s, "action": a, "next_state": ns, "level_up": lu}
TR = [_t(S0, [4], S1, False)]

def test_render_obj_transitions_shows_objects_and_action():
    out = synth.render_obj_transitions(TR)
    assert "action=[4]" in out and "c3" in out and "x1" in out and "x2" in out

def test_obj_prompt_states_object_contract_and_goal():
    p = synth._obj_prompt(TR, "actions=[1,2,3,4]")
    assert "predict(state, action)" in p and "next_state" in p
    assert "goal_score" in p and "{predict_src, goal_score_src, rationale}" in p

def test_obj_diff_lists_mispredicted_objects():
    bad_next = {"bg": 0, "objects": [{"color": 3, "size": 1, "y": 1, "x": 9}]}   # wrong x
    d = synth._obj_diff([(TR[0], bad_next)])
    assert "action=[4]" in d and ("9" in d or "real" in d)

def test_obj_funsearch_prompt_kshot_and_failed_block():
    samples = [{"src": "def predict(state, action):\n    return state, False", "score": 0, "fails": []},
               {"src": "def predict(state, action):\n    return state, True", "score": 1, "fails": []}]
    p = synth._obj_funsearch_prompt(samples, "actions=[4]", failed=["tried Z -> scored 0"])
    assert "predict_v0" in p and "predict_v1" in p and "predict_v2" in p
    assert "tried Z -> scored 0" in p and "do not repeat" in p.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.arcv/bin/python -m pytest tests/test_e125_synth_obj.py -q`
Expected: FAIL (`render_obj_transitions` missing).

- [ ] **Step 3: Write minimal implementation** (append to `experiments/e125/synth.py`)

```python
# --- object-state synthesis path (predict(state,action)->(next_state,level_up)); reuses the FunSearch
#     _Database/_softmax/_rename_fn/failed_summaries machinery, but renders/scoring is over OBJECT state ---
from e125 import objstate as _objstate_s


def _objs(s):
    return (f"bg={s.get('bg')} objects=["
            + ", ".join(f"(c{o['color']} y{o['y']} x{o['x']} s{o['size']})" for o in s.get("objects", []))
            + "]")


def render_obj_transitions(transitions, k=12):
    out = []
    for t in transitions[:k]:
        out.append(f"action={t['action']} level_up={bool(t['level_up'])}\n"
                   f"FROM: {_objs(t['state'])}\nTO:   {_objs(t['next_state'])}")
    return "\n---\n".join(out)


_OBJ_GOAL_INSTR = (
    "\n\nIMPORTANT -- the win condition is NOT given and no observed transition won. HYPOTHESISE the goal from "
    "the object configurations (e.g. a movable object reaching a target object's position). Bake it into "
    "predict()'s level_up (True only when next_state matches your hypothesised win; it must be False on every "
    "observed transition above). Also write goal_score(state) -> float: a SYMBOLIC energy LOWER nearer the goal "
    "(e.g. Manhattan distance between the mover and the target object), varying smoothly. Operate on the object "
    "dict only (state['objects'] = list of {color,size,y,x}); pure Python, no imports, no numpy.")


def _obj_contract(action_api):
    return (f"You are reverse-engineering an unknown grid game's dynamics from observed OBJECT-state transitions. "
            f"Do NOT run shell commands or read files. Write `predict(state, action) -> (next_state, level_up)` "
            f"in pure Python (NO imports, NO numpy), where `state` is a dict "
            f"{{'bg': int, 'objects': [{{'color','size','y','x'}}, ...]}}, `action` is a list like [4] or [6,x,y], "
            f"`next_state` is the predicted next state dict (same shape), `level_up` a bool. Actions: {action_api}")


def _obj_prompt(transitions, action_api, counterexample=None):
    base = (_obj_contract(action_api) + "\n\nObserved transitions:\n"
            + render_obj_transitions(transitions))
    if counterexample is not None:
        base += (f"\n\nYour previous predict() FAILED on:\naction={counterexample['action']} "
                 f"level_up={bool(counterexample['level_up'])}\nFROM: {_objs(counterexample['state'])}\n"
                 f"TO:   {_objs(counterexample['next_state'])}")
    return base + _OBJ_GOAL_INSTR + "\n\nReturn JSON {predict_src, goal_score_src, rationale}."


def _obj_diff(fails, fields=("color", "y", "x"), k=3):
    out = []
    for t, ns in fails[:k]:
        if ns is None:
            out.append(f"action={t['action']}: predict raised/failed"); continue
        pk = _objstate_s.state_key(ns, fields)[1]
        rk = _objstate_s.state_key(t["next_state"], fields)[1]
        msg = f"you->{pk} real->{rk}" if pk != rk else "(objects match -- the level_up flag is wrong)"
        out.append(f"action={t['action']}: {msg}")
    return "\n".join(out)


def _obj_funsearch_prompt(samples, action_api, failed=None):
    progs = sorted(samples, key=lambda p: p["score"])
    blocks = [f"# predict_v{i} (score {p['score']})\n```python\n{_rename_fn(p['src'], 'predict', f'predict_v{i}')}\n```"
              for i, p in enumerate(progs)]
    nextv = len(progs)
    diff = _obj_diff(progs[-1].get("fails") or [])
    fail_block = ""
    if failed:
        fail_block = ("\n\nAlready tried and FAILED -- do not repeat these approaches:\n"
                      + "\n".join(f"- {f}" for f in failed) + "\n")
    return ("These are successive predict() object-state world models, ordered by increasing score:\n\n"
            + "\n\n".join(blocks)
            + f"\n\nWrite an IMPROVED `predict_v{nextv}` scoring HIGHER than all above. predict_v{nextv-1} "
            f"still mispredicts:\n{diff}\n{fail_block}Name the function `predict` (pure Python, no imports). "
            f"Keep/improve the win hypothesis in level_up and the goal_score(state) energy. Actions: {action_api}. "
            f"Return JSON {{predict_src, goal_score_src, rationale}}.")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.arcv/bin/python -m pytest tests/test_e125_synth_obj.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit (authorized)**

```bash
git add experiments/e125/synth.py tests/test_e125_synth_obj.py
git commit -m "E125 P2.5: object-state rendering + FunSearch prompts"
```

---

## Task 2: Ensemble primitives — top-k + disagreement

**Files:**
- Modify: `experiments/e125/synth.py` (add `_Database.top_k`; add module fn `ensemble_disagreement`)
- Test: `tests/test_e125_ensemble.py`

**Interfaces:**
- Consumes: `_Database` (existing), `objstate.state_key`.
- Produces: `_Database.top_k(self, k) -> list[prog]` — up to k distinct-by-`src` programs, highest score first. `synth.ensemble_disagreement(predict_fns, state, action, fields=("color","y","x")) -> float` — in `[0,1]`: run each fn on `(deepcopy state, action)`; a fn that errors counts as a distinct "error" outcome; disagreement = fraction of fns whose `state_key(next_state)` differs from the plurality outcome (0.0 when all agree, 1 fn → 0.0).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_e125_ensemble.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
import numpy as np
from e125 import synth, verify

S0 = {"bg": 0, "objects": [{"color": 3, "size": 1, "y": 1, "x": 1}]}
MOVE = "def predict(state, action):\n    ns={'bg':state['bg'],'objects':[dict(o) for o in state['objects']]}\n    [o.__setitem__('x',o['x']+1) for o in ns['objects']]\n    return ns, False"
STAY = "def predict(state, action):\n    return {'bg':state['bg'],'objects':[dict(o) for o in state['objects']]}, False"
BOOM = "def predict(state, action):\n    raise ValueError('x')"

def test_top_k_returns_highest_scoring_distinct():
    db = synth._Database(rng=np.random.RandomState(0))
    db.register("a", None, 1, (True, False), [], None)
    db.register("b", None, 3, (True, True), [], None)
    db.register("a", None, 1, (True, False), [], None)   # dup src
    top = db.top_k(2)
    assert [p["score"] for p in top] == [3, 1] and len({p["src"] for p in top}) == 2

def test_disagreement_zero_when_all_agree():
    fns = [verify.compile_obj_predict(MOVE), verify.compile_obj_predict(MOVE)]
    assert synth.ensemble_disagreement(fns, S0, [4]) == 0.0

def test_disagreement_positive_when_split():
    fns = [verify.compile_obj_predict(MOVE), verify.compile_obj_predict(STAY), verify.compile_obj_predict(BOOM)]
    d = synth.ensemble_disagreement(fns, S0, [4])
    assert d > 0.0

def test_disagreement_single_fn_is_zero():
    assert synth.ensemble_disagreement([verify.compile_obj_predict(MOVE)], S0, [4]) == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.arcv/bin/python -m pytest tests/test_e125_ensemble.py -q`
Expected: FAIL (`_Database.top_k` / `ensemble_disagreement` missing).

- [ ] **Step 3: Write minimal implementation** (append/extend in `experiments/e125/synth.py`)

Add this method inside `class _Database` (place it after `sample`):

```python
    def top_k(self, k):
        """Up to k distinct-by-src programs, highest score first (the ensemble)."""
        progs = [p for c in self.clusters.values() for p in c["progs"]]
        progs.sort(key=lambda p: -p["score"])
        out, seen = [], set()
        for p in progs:
            if p["src"] in seen:
                continue
            seen.add(p["src"]); out.append(p)
            if len(out) >= k:
                break
        return out
```

Add this module-level function (near the other object helpers):

```python
import copy as _copy_s


def ensemble_disagreement(predict_fns, state, action, fields=("color", "y", "x")):
    """Fraction of programs whose predicted next_state key differs from the plurality (errors are their own
    outcome). 0.0 when all agree or a single program."""
    fns = [f for f in predict_fns if f is not None]
    if len(fns) <= 1:
        return 0.0
    outcomes = []
    for f in fns:
        try:
            ns, _ = f(_copy_s.deepcopy(state), list(action))
            outcomes.append(_objstate_s.state_key(ns, fields))
        except Exception:
            outcomes.append(("__error__",))
    counts = {}
    for o in outcomes:
        counts[o] = counts.get(o, 0) + 1
    plurality = max(counts.values())
    return (len(outcomes) - plurality) / len(outcomes)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.arcv/bin/python -m pytest tests/test_e125_ensemble.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit (authorized)**

```bash
git add experiments/e125/synth.py tests/test_e125_ensemble.py
git commit -m "E125 P2.6: ensemble primitives (top_k + disagreement)"
```

---

## Task 3: `synthesize_obj` — FunSearch over object state, returning a top-k ensemble

**Files:**
- Modify: `experiments/e125/synth.py` (add `synthesize_obj`)
- Test: `tests/test_e125_synth_obj.py` (extend)

**Interfaces:**
- Consumes: `_Database`, `_obj_prompt`, `_obj_funsearch_prompt`, `verify.compile_obj_predict`, `verify.score_obj`, `objstate.state_key`, `_Database.top_k`, `codex_iso.run`, `capture_lib.codex_record`, existing `SCHEMA`.
- Produces: `synth.synthesize_obj(transitions, action_api, game, model="gpt-5.5", n_retries=4, traces_dir=None, _runner=None, functions_per_prompt=2, seed=0, seed_src=None, k_ensemble=3, fields=("color","y","x")) -> (src, predict_fn, goal_fn, ensemble)` — `ensemble` is a list of compiled predict callables (the top-k verified programs, length ≥1; `[predict_fn]` when only one). Returns `(None, None, None, [])` if no program fully passes the gate.

- [ ] **Step 1: Write the failing test** (append to `tests/test_e125_synth_obj.py`)

```python
GOODO = ("def predict(state, action):\n"
         "    ns={'bg':state['bg'],'objects':[dict(o) for o in state['objects']]}\n"
         "    if action==[4]:\n        [o.__setitem__('x', o['x']+1) for o in ns['objects']]\n"
         "    return ns, False")
GOALO = "def goal_score(state):\n    o=state['objects'][0]\n    return float(5 - o['x'])"
S2 = {"bg": 0, "objects": [{"color": 3, "size": 1, "y": 1, "x": 3}]}
TR2 = [_t(S0, [4], S1, False), _t(S1, [4], S2, False)]

def _runner(src, goal):
    def run(prompt, schema, model, game, **kw):
        return {"final": {"predict_src": src, "goal_score_src": goal, "rationale": "x"},
                "events": [], "tainted": False, "raw": "", "model_version": ""}
    return run

def test_synthesize_obj_accepts_and_returns_ensemble(tmp_path):
    src, fn, goal, ens = synth.synthesize_obj(TR2, "actions=[4]", "g", n_retries=1,
                                              traces_dir=str(tmp_path), _runner=_runner(GOODO, GOALO))
    assert fn is not None and callable(goal)
    ns, lu = fn(dict(S0), [4]); assert ns["objects"][0]["x"] == 2
    assert isinstance(ens, list) and len(ens) >= 1 and all(callable(f) for f in ens)

def test_synthesize_obj_rejects_numpy_predict(tmp_path):
    npp = "def predict(state, action):\n    import numpy as np\n    return state, False"
    src, fn, goal, ens = synth.synthesize_obj(TR2, "actions=[4]", "g", n_retries=1,
                                              traces_dir=str(tmp_path), _runner=_runner(npp, GOALO))
    assert fn is None and ens == []      # gate env == sandbox: a numpy predict cannot pass

def test_synthesize_obj_returns_none_when_never_passes(tmp_path):
    bad = "def predict(state, action):\n    return state, False"   # never moves -> mispredicts
    src, fn, goal, ens = synth.synthesize_obj(TR2, "actions=[4]", "g", n_retries=2,
                                              traces_dir=str(tmp_path), _runner=_runner(bad, GOALO))
    assert fn is None and ens == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.arcv/bin/python -m pytest tests/test_e125_synth_obj.py -q`
Expected: FAIL (`synthesize_obj` missing).

- [ ] **Step 3: Write minimal implementation** (append to `experiments/e125/synth.py`)

```python
def _score_obj_program(predict_fn, transitions, fields):
    """(n_matched, signature, fails) over object state -- the clustering signature for the FunSearch DB."""
    n, fails = verify.score_obj(predict_fn, transitions, fields)
    failed_ids = {id(t) for (t, _) in fails}
    sig = tuple(id(t) not in failed_ids for t in transitions)
    return n, sig, fails


def synthesize_obj(transitions, action_api, game, model="gpt-5.5", n_retries=4, traces_dir=None, _runner=None,
                   functions_per_prompt=2, seed=0, seed_src=None, k_ensemble=3, fields=("color", "y", "x")):
    """FunSearch over OBJECT-state predict(); decision-equivalent gate (verify.check_obj); sandbox-env compile
    (verify.compile_obj_predict). Returns (src, predict_fn, goal_fn, ensemble[top-k callables]) on a full
    gate-pass, else (None, None, None, [])."""
    run = _runner or codex_iso.run
    if len(transitions) < 2:
        return None, None, None, []
    split = max(1, min(len(transitions) - 1, int(len(transitions) * 0.7)))
    train, held = transitions[:split], transitions[split:]
    db = _Database(functions_per_prompt=functions_per_prompt, rng=np.random.RandomState(seed))

    def _ensemble():
        return [p["fn"] for p in db.top_k(k_ensemble) if p["fn"] is not None] or ([db.best["fn"]] if db.best and db.best["fn"] else [])

    def _accept(prog):
        goal_fn = verify.compile_goal(prog["goal_src"]) if prog.get("goal_src") else None
        if traces_dir:
            try:
                with open(os.path.join(traces_dir, f"{game}_obj_verified.py"), "w") as fh:
                    fh.write(f"# E125 verified OBJECT predict()+goal_score() for {game}\n{prog['src'] or ''}\n\n{prog.get('goal_src') or ''}\n")
            except Exception:
                pass
        return prog["src"], prog["fn"], goal_fn, _ensemble()

    if seed_src:
        sfn = verify.compile_obj_predict(seed_src)
        if sfn is not None:
            sc, sig, fails = _score_obj_program(sfn, held, fields)
            db.register(seed_src, sfn, sc, sig, fails, None, rationale="carried-forward best")
            if sc == len(held):
                return _accept(db.best)

    for attempt in range(n_retries):
        samples = db.sample()
        prompt = (_obj_prompt(train, action_api, None) if not samples
                  else _obj_funsearch_prompt(samples, action_api, failed=db.failed_summaries()))
        res = run(prompt, SCHEMA, model, game)
        final = res.get("final") or {}
        src = final.get("predict_src")
        goal_src = final.get("goal_score_src")
        rationale = final.get("rationale") or ""
        tainted = bool(res.get("tainted"))
        fn = None if tainted else verify.compile_obj_predict(src or "")
        sc, sig, fails = _score_obj_program(fn, held, fields)
        if src and not tainted:
            db.register(src, fn, sc, sig, fails, goal_src, rationale=rationale)
        best_full = db.best is not None and db.best["score"] == len(held) and db.best["fn"] is not None
        if traces_dir:
            capture_lib.codex_record(traces_dir, {"game": game, "level": 0, "regime": attempt, "model": model,
                "model_version": res.get("model_version", ""), "prompt": prompt, "raw": res.get("raw", ""),
                "events": res.get("events", []), "parsed": {"subgoals": [], "macros": []},
                "decision": ("accept" if best_full else f"evolve {sc}/{len(held)} (best {db.best['score'] if db.best else 0})"),
                "tainted": tainted})
        if best_full:
            return _accept(db.best)
    return None, None, None, []
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.arcv/bin/python -m pytest tests/test_e125_synth_obj.py -q`
Expected: PASS (3 new + 4 from Task 1 = 7).

- [ ] **Step 5: Run the full E125 suite (no regressions)**

Run: `DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib ~/.arcv/bin/python -m pytest tests/test_e125_*.py -q`
Expected: all pass (70 prior + new synth_obj/ensemble tests).

- [ ] **Step 6: Commit (authorized)**

```bash
git add experiments/e125/synth.py tests/test_e125_synth_obj.py
git commit -m "E125 P2.7: synthesize_obj (object-state FunSearch + top-k ensemble)"
```

---

## Self-Review

**Spec coverage (Plan 2's slice of the design):** object-state synthesis (decision 5) → Tasks 1+3; decision-equivalent gate reuse + sandbox-env compile (decisions 1,6) → Task 3 (`compile_obj_predict`, `check_obj`); top-k ensemble + disagreement (decision 7) → Tasks 2+3. Deferred to Plan 3: `plan_in_world`, `traverse.py`, `solve_game`, rule-library transfer, **`goal_score` shaping from grounded outcomes** (decision 9 — needs the live grounding loop), and `to_spec` capturing goal-energy code (a serialize-side fix flagged in Plan 1).

**Placeholder scan:** none — every step has runnable test + implementation code. Live codex validation is intentionally NOT a subagent step (it costs spend and needs the env); it is run by the controller after the plan lands.

**Type consistency:** object transition `{state,action,next_state,level_up}` matches Plan 1's `verify.score_obj`/`check_obj`; `synthesize_obj` returns a 4-tuple `(src, predict_fn, goal_fn, ensemble)` (distinct from the frame `synthesize` 3-tuple — intentional, documented); `_Database.register(src,fn,score,signature,fails,goal_src,rationale="")` and `.sample()`/`.failed_summaries()`/`.top_k()` are the shared FunSearch DB; `compile_obj_predict`/`compile_goal` come from Plan 1's `verify`; `ensemble_disagreement(fns, state, action, fields)` matches its test.

## Live validation (controller-run, after the plan lands — not a subagent task)

Synthesize a verified object-state world on the cleanest pilot (dc22), then assemble + round-trip the World:
`synthesize_obj(obj_transitions, ...)` → `world.build_world(src, goal_src, initial_obj_state, actions, "dc22")` → `world.round_trip_ok(w)`. Report verified-gate %, ensemble size, and disagreement on a few states — honestly, no banked answers.
