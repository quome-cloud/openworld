# E125 Structured Executable-World-Model Agent — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Beat E124 by planning in *simulation* — codex (GPT-5.5) synthesizes a verified `predict()` world model, the harness plans a level in that code model and executes only verified plans, cracking a level real-env search could not.

**Architecture:** A Python harness where codex only *proposes* `predict(frame,action)→(next_frame,level_up)` code; the harness enforces a verifier gate (exact-match on held-out real transitions), plans in the synthesized `World` via `e119/planner`, and executes against the real env step-by-step, halting on any sim-vs-real mismatch. OpenWorld-native, ~80% reuse of E112/E119/E121–124.

**Tech Stack:** Python 3.14 (`~/.arcv/bin/python`), numpy, `arc_agi`/`arcengine`, `openworld`, `codex exec` CLI (`gpt-5.5`), pytest. Reuses `experiments/e124/{codex_iso,sandbox_exec}`, `experiments/e119/{planner,perceive}`, `experiments/e121_surprise_regimes`, `scripts/capture_lib`.

## Global Constraints

- **Source-free (cardinal):** codex sees only collected transitions (frames+actions+level_up), never game source; every codex call is M0-isolation-audited (`codex_iso`); a tainted call is discarded.
- **Invariant:** codex only proposes code. The env decides correctness (replay-verified `levels_completed` bump). A wrong `predict()` is caught by the verifier gate or the executor halt — never by trusting the model.
- **Honesty:** `save_results(...)` BEFORE any assert. Report the loose-agent baseline and the E124 control as-is. If a milestone gate fails, stop and report.
- **Run with** `~/.arcv/bin/python`; codex at `~/.local/bin/codex`; default model `gpt-5.5`.
- **predict contract:** `predict(frame: np.ndarray[64,64], action: list[int]) -> (next_frame: np.ndarray[64,64], level_up: bool)`. Verified on the **masked** frame (status bar zeroed).
- **Zero new core deps;** `experiments/` may use arc-venv packages. Commit only when asked.

---

## File Structure

- Create `experiments/e125/__init__.py` — package marker.
- Create `experiments/e125/verify.py` — the verifier gate (compile + run `predict` on held-out transitions, exact-match).
- Create `experiments/e125/synth.py` — codex synthesizes `predict()` + retry-with-counterexample loop + telemetry.
- Create `experiments/e125/simworld.py` — wrap `predict()` as a `SimGame` (and `World`); plan-in-simulation.
- Create `experiments/e125/execute.py` — execute a plan vs the real env, halt on sim-vs-real mismatch, collect new transitions.
- Create `experiments/e125/explorer.py` — collect `(frame,action,next_frame,level_up)` transitions by change-seeking.
- Create `experiments/e125/agent.py` — the single-level loop (explore→synth→plan→execute→resync).
- Create `experiments/e125_executable_world.py` — entry point.
- Tests: `tests/test_e125_verify.py`, `tests/test_e125_simworld.py`, `tests/test_e125_execute.py`, `tests/test_e125_synth.py`, `tests/test_e125_agent.py`.

A **transition** is a dict `{"frame": np.ndarray, "action": list[int], "next_frame": np.ndarray, "level_up": bool}`.

---

## Task 1: Verifier gate — `predict()` must exact-match held-out transitions

**Files:**
- Create: `experiments/e125/__init__.py` (empty), `experiments/e125/verify.py`
- Test: `tests/test_e125_verify.py`

**Interfaces:**
- Produces: `verify.compile_predict(src: str) -> callable|None`; `verify.check(predict_fn, transitions: list[dict], mask) -> (ok: bool, counterexample: dict|None)` — runs `predict_fn` on each transition; ok iff the masked predicted next_frame equals the masked real next_frame AND predicted level_up equals real level_up, for every transition. Returns the first failing transition as the counterexample.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_e125_verify.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
import numpy as np
from e125 import verify

def _t(f, a, nf, lu): return {"frame": f, "action": a, "next_frame": nf, "level_up": lu}

# a toy world: action [1] sets cell (0,0)=action count (frame[0,0]+1); level_up when it hits 3
F0 = np.zeros((64,64), dtype=int)
F1 = F0.copy(); F1[0,0] = 1
F2 = F1.copy(); F2[0,0] = 2
GOOD = "def predict(frame, action):\n    nf = frame.copy(); nf[0,0] = frame[0,0] + 1\n    return nf, bool(nf[0,0] == 3)"
BAD  = "def predict(frame, action):\n    return frame.copy(), False"   # never changes -> mispredicts

TRANS = [_t(F0,[1],F1,False), _t(F1,[1],F2,False)]

def test_compile_predict_returns_callable():
    fn = verify.compile_predict(GOOD); assert callable(fn)
    nf, lu = fn(F0, [1]); assert nf[0,0] == 1 and lu is False

def test_compile_predict_bad_src_returns_none():
    assert verify.compile_predict("def predict(:\n bad") is None

def test_check_accepts_exact_model():
    ok, ce = verify.check(verify.compile_predict(GOOD), TRANS, mask=None)
    assert ok is True and ce is None

def test_check_rejects_with_counterexample():
    ok, ce = verify.check(verify.compile_predict(BAD), TRANS, mask=None)
    assert ok is False and ce is not None and ce["action"] == [1]

def test_check_masks_status_bar():
    # a status cell at (63,63) flips every step; with it masked, an otherwise-correct model passes
    a = F0.copy(); a[63,63] = 5; b = F1.copy(); b[63,63] = 9
    mask = np.zeros((64,64), dtype=bool); mask[63,63] = True
    ok, _ = verify.check(verify.compile_predict(GOOD), [_t(a,[1],b,False)], mask=mask)
    assert ok is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.arcv/bin/python -m pytest tests/test_e125_verify.py -q`
Expected: FAIL (`e125.verify` missing).

- [ ] **Step 3: Write minimal implementation**

```python
# experiments/e125/verify.py
"""The verifier gate: a synthesized predict(frame,action)->(next_frame,level_up) is ACCEPTED only if it
exact-matches every held-out transition (masked next-frame equality + level_up equality). Predicts are
compiled in-process for speed (codex is not adversarial; a predict that errors fails the gate)."""
import numpy as np


def compile_predict(src):
    ns = {"np": np, "__builtins__": __builtins__}
    try:
        exec(src, ns)
        fn = ns.get("predict")
        return fn if callable(fn) else None
    except Exception:
        return None


def _masked(frame, mask):
    fr = np.asarray(frame)
    return np.where(mask, 0, fr) if mask is not None else fr


def check(predict_fn, transitions, mask):
    """Return (ok, counterexample). ok iff predict_fn reproduces every transition (masked next-frame + level_up)."""
    if predict_fn is None:
        return False, (transitions[0] if transitions else None)
    for t in transitions:
        try:
            nf, lu = predict_fn(np.asarray(t["frame"]), list(t["action"]))
        except Exception:
            return False, t
        if not np.array_equal(_masked(nf, mask), _masked(t["next_frame"], mask)):
            return False, t
        if bool(lu) != bool(t["level_up"]):
            return False, t
    return True, None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.arcv/bin/python -m pytest tests/test_e125_verify.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit (if approved)**

```bash
git add experiments/e125/__init__.py experiments/e125/verify.py tests/test_e125_verify.py
git commit -m "E125 Task 1: verifier gate (predict exact-match on held-out transitions)"
```

---

## Task 2: Synthesizer — codex writes `predict()` with retry-on-counterexample (Milestone 1)

**Files:**
- Create: `experiments/e125/synth.py`
- Test: `tests/test_e125_synth.py`

**Interfaces:**
- Consumes: `verify.compile_predict`, `verify.check`; `e124.codex_iso.run`; `capture_lib.codex_record`.
- Produces: `synth.render_transitions(transitions, mask, k=12) -> str`; `synth.synthesize(transitions, action_api, game, mask, model="gpt-5.5", n_retries=3, traces_dir=None, _runner=None) -> (src: str|None, predict_fn: callable|None)` — splits transitions train/held-out, asks codex for `predict()`, accepts via the gate, retries with the counterexample appended, returns the first model that passes (or None).

- [ ] **Step 1: Write the failing test** (mock codex via `_runner`)

```python
# tests/test_e125_synth.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
import numpy as np
from e125 import synth

F0 = np.zeros((64,64), dtype=int); F1 = F0.copy(); F1[0,0]=1; F2=F1.copy(); F2[0,0]=2
def _t(f,a,nf,lu): return {"frame":f,"action":a,"next_frame":nf,"level_up":lu}
TRANS = [_t(F0,[1],F1,False), _t(F1,[1],F2,False)]
GOOD = "def predict(frame, action):\n    nf=frame.copy(); nf[0,0]=frame[0,0]+1\n    return nf, bool(nf[0,0]==3)"

def _runner_giving(src):
    def run(prompt, schema, model, game, **kw):
        return {"final": {"predict_src": src, "rationale": "x"}, "events": [], "tainted": False,
                "raw": "", "model_version": "gpt-5.5-test"}
    return run

def test_synthesize_accepts_passing_model(tmp_path):
    src, fn = synth.synthesize(TRANS, "actions=[1]", "g", mask=None, n_retries=1,
                               traces_dir=str(tmp_path), _runner=_runner_giving(GOOD))
    assert fn is not None and src is not None
    nf, lu = fn(F0, [1]); assert nf[0,0] == 1

def test_synthesize_returns_none_when_model_never_passes(tmp_path):
    bad = "def predict(frame, action):\n    return frame.copy(), False"
    src, fn = synth.synthesize(TRANS, "actions=[1]", "g", mask=None, n_retries=2,
                               traces_dir=str(tmp_path), _runner=_runner_giving(bad))
    assert fn is None

def test_synthesize_writes_telemetry(tmp_path):
    synth.synthesize(TRANS, "a", "g", mask=None, n_retries=1, traces_dir=str(tmp_path),
                     _runner=_runner_giving(GOOD))
    assert os.path.exists(tmp_path/"calls.jsonl")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.arcv/bin/python -m pytest tests/test_e125_synth.py -q`
Expected: FAIL (`e125.synth` missing).

- [ ] **Step 3: Write minimal implementation**

```python
# experiments/e125/synth.py
"""Codex synthesizes predict(frame,action)->(next_frame,level_up); accepted ONLY via verify.check on a
held-out split. On a miss, the counterexample is appended and codex re-proposes (bounded retries). Source-free
+ telemetry-captured. Codex is a proposal engine inside the verifier loop -- never an authority."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "scripts"))
import numpy as np
from e125 import verify
from e124 import codex_iso
import capture_lib

SCHEMA = {"type": "object", "additionalProperties": False, "required": ["predict_src", "rationale"],
          "properties": {"predict_src": {"type": "string"}, "rationale": {"type": "string"}}}


def _grid(frame, mask):
    fr = verify._masked(frame, mask)
    return "\n".join("".join(f"{int(c):x}" for c in row) for row in np.asarray(fr).reshape(64, 64))


def render_transitions(transitions, mask, k=12):
    out = []
    for t in transitions[:k]:
        out.append(f"action={t['action']} level_up={bool(t['level_up'])}\nFROM:\n{_grid(t['frame'],mask)}\n"
                   f"TO:\n{_grid(t['next_frame'],mask)}")
    return "\n---\n".join(out)


def _prompt(transitions, action_api, mask, counterexample):
    base = (f"You are reverse-engineering an unknown 64x64 grid game's dynamics from observed transitions. "
            f"Do NOT run shell commands or read files. Write a Python function "
            f"`predict(frame, action) -> (next_frame, level_up)` using numpy as np only (no imports/IO), where "
            f"`frame` is a 64x64 int array, `action` is a list like [1] or [6,x,y], `next_frame` is the "
            f"predicted next 64x64 array, and `level_up` is a bool (did the level advance).\n\nActions: "
            f"{action_api}\n\nObserved transitions (hex grids, status bar masked):\n{render_transitions(transitions, mask)}")
    if counterexample is not None:
        base += (f"\n\nYour previous predict() FAILED on this transition (fix it):\naction="
                 f"{counterexample['action']} level_up={bool(counterexample['level_up'])}\nFROM:\n"
                 f"{_grid(counterexample['frame'],mask)}\nTO:\n{_grid(counterexample['next_frame'],mask)}")
    return base + "\n\nReturn JSON {predict_src, rationale}."


def synthesize(transitions, action_api, game, mask, model="gpt-5.5", n_retries=3, traces_dir=None, _runner=None):
    run = _runner or codex_iso.run
    split = max(1, int(len(transitions) * 0.7))
    train, held = transitions[:split], transitions[split:] or transitions
    ce = None
    for attempt in range(n_retries):
        prompt = _prompt(train, action_api, mask, ce)
        res = run(prompt, SCHEMA, model, game)
        final = res.get("final") or {}
        src = final.get("predict_src")
        tainted = bool(res.get("tainted"))
        fn = None if tainted else verify.compile_predict(src or "")
        ok, ce = verify.check(fn, held, mask) if fn else (False, held[0] if held else None)
        if traces_dir:
            capture_lib.codex_record(traces_dir, {"game": game, "level": 0, "regime": attempt, "model": model,
                "model_version": res.get("model_version", ""), "prompt": prompt, "raw": res.get("raw", ""),
                "events": res.get("events", []), "parsed": {"subgoals": [], "macros": []},
                "decision": ("accept" if ok else "reject"), "tainted": tainted})
        if ok:
            return src, fn
    return None, None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.arcv/bin/python -m pytest tests/test_e125_synth.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: LIVE Milestone-1 gate (manual — the make-or-break)**

Collect ~20 real transitions from a pilot game and ask codex for a `predict()` that passes the held-out gate:
```bash
~/.arcv/bin/python - <<'PY'
import sys; sys.path.insert(0,"experiments")
from arc3_sandbox import SandboxGame
from e125 import synth
from e119 import perceive
g=SandboxGame("g50t"); g.reset(); frames=[g.frame]; trans=[]
import random
for _ in range(24):
    a=[random.choice([x for x in g.avail if x in (1,2,3,4,5,7)] or [g.avail[0]])]
    pf=g.frame.copy(); lv=g.levels; g.step(*a)
    trans.append({"frame":pf,"action":a,"next_frame":g.frame.copy(),"level_up":g.levels>lv}); frames.append(g.frame)
mask=perceive.status_mask(frames)
src,fn=synth.synthesize(trans, f"actions={g.avail}", "g50t", mask, model="gpt-5.5", n_retries=3,
                        traces_dir="experiments/results/e125_traces")
print("PREDICT SYNTHESIZED + PASSED GATE:", fn is not None)
print(src[:400] if src else "FAILED gate")
PY
```
**GATE:** does codex produce a `predict()` that exact-matches held-out **real** transitions? If yes → Milestone 2. If no after retries → that is the new bottleneck (codex can't model these dynamics); stop and report honestly.

- [ ] **Step 6: Commit (if approved)**

```bash
git add experiments/e125/synth.py tests/test_e125_synth.py
git commit -m "E125 Task 2: codex predict() synthesizer + verifier-gate retry (Milestone 1)"
```

---

## Task 3: SimWorld — plan in the synthesized model (not the env)

**Files:**
- Create: `experiments/e125/simworld.py`
- Test: `tests/test_e125_simworld.py`

**Interfaces:**
- Consumes: a `predict_fn` (from synth), `e119/planner`.
- Produces: `simworld.SimGame(predict_fn, initial_frame)` with `reset()/frame/levels/done/step(*a)` driven by `predict_fn` (NOT the env); `simworld.plan(predict_fn, initial_frame, candidates_fn, budget, max_depth=40) -> list[action]|None` — BFS in the SimGame for a trajectory whose predicted `level_up` fires.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_e125_simworld.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
import numpy as np
from e125 import simworld

# model: action [1] increments frame[0,0]; level_up when it reaches 5 (depth 5 -> blind-real would be slow)
PRED = "def predict(frame, action):\n    nf=frame.copy()\n    if action==[1]: nf[0,0]=frame[0,0]+1\n    return nf, bool(nf[0,0]==5)"
from e125 import verify
fn = verify.compile_predict(PRED)

def test_simgame_steps_via_predict():
    g = simworld.SimGame(fn, np.zeros((64,64),dtype=int)); g.reset()
    g.step(1); assert g.frame[0,0]==1 and g.levels==0
    for _ in range(4): g.step(1)
    assert g.levels==1 and g.done

def test_plan_finds_winning_trajectory_in_sim():
    plan = simworld.plan(fn, np.zeros((64,64),dtype=int), lambda fr:[[1],[2]], budget=2000)
    assert plan == [[1],[1],[1],[1],[1]]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.arcv/bin/python -m pytest tests/test_e125_simworld.py -q`
Expected: FAIL (`e125.simworld` missing).

- [ ] **Step 3: Write minimal implementation**

```python
# experiments/e125/simworld.py
"""Wrap a synthesized predict() as a SimGame so planning happens IN the code model (free, deep), not the
real env. plan() searches the SimGame for a trajectory whose predicted level_up fires."""
from collections import deque
import numpy as np


class SimGame:
    """A game-shaped wrapper over predict(frame,action)->(next_frame,level_up). reset()/step(*a) only touch
    the code model, never the real env."""
    def __init__(self, predict_fn, initial_frame):
        self.predict_fn = predict_fn
        self._init = np.asarray(initial_frame).copy()
        self.reset()
    def reset(self):
        self.frame = self._init.copy(); self.levels = 0; self.done = False; return self
    def step(self, a, x=None, y=None):
        action = [a] if x is None else [a, x, y]
        try:
            nf, lu = self.predict_fn(self.frame, action)
        except Exception:
            self.done = True; return
        self.frame = np.asarray(nf)
        if lu:
            self.levels += 1; self.done = True


def plan(predict_fn, initial_frame, candidates_fn, budget, max_depth=40):
    """BFS in the SimGame for an action sequence whose predicted level_up fires. Returns the sequence or None."""
    steps = [s if isinstance(s, list) else [s] for s in candidates_fn(initial_frame)]
    frontier = deque([[]]); seen = set(); n = 0
    while frontier and n < budget:
        prefix = frontier.popleft()
        for st in steps:
            cand = prefix + [st]
            key = tuple(map(tuple, cand))
            if key in seen:
                continue
            seen.add(key); n += 1
            g = SimGame(predict_fn, initial_frame)
            for a in cand:
                g.step(*a)
                if g.done:
                    break
            if g.levels > 0:
                return cand
            if len(cand) < max_depth and not g.done:
                frontier.append(cand)
            if n >= budget:
                break
    return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.arcv/bin/python -m pytest tests/test_e125_simworld.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit (if approved)**

```bash
git add experiments/e125/simworld.py tests/test_e125_simworld.py
git commit -m "E125 Task 3: SimGame + plan-in-simulation"
```

---

## Task 4: Executor — execute a plan vs the real env, halt on sim-vs-real mismatch

**Files:**
- Create: `experiments/e125/execute.py`
- Test: `tests/test_e125_execute.py`

**Interfaces:**
- Consumes: a `predict_fn`, `verify._masked`.
- Produces: `execute.execute_plan(real_game, plan, predict_fn, mask) -> dict` with keys `{"solved": bool, "verified_prefix": list, "new_transitions": list[dict], "halt_step": int|None}`. Replays the plan on `real_game` step-by-step; at each step, if the real masked next-frame != predict's masked next-frame, **halt**, record the real transition as a new (model-surprising) transition. A real `levels` bump = solved.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_e125_execute.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
import numpy as np
from e125 import execute, verify

class RealGame:
    """action [1] increments (0,0) until 3 then a level-up; BUT at (0,0)==2 the real board ALSO sets (1,1)=7
    (a dynamic the model below doesn't know) -> a sim-vs-real mismatch the executor must catch."""
    def __init__(self): self.reset()
    def reset(self): self.c=0; self.levels=0; self.done=False; self.frame=np.zeros((64,64),dtype=int)
    def step(self,a,x=None,y=None):
        if a==1: self.c+=1
        self.frame=np.zeros((64,64),dtype=int); self.frame[0,0]=self.c
        if self.c==2: self.frame[1,1]=7
        if self.c==3: self.levels=1; self.done=True

# model: increments (0,0), level_up at 3, but NEVER sets (1,1) -> mismatch at step where c becomes 2
PRED="def predict(frame, action):\n    nf=frame.copy()\n    if action==[1]: nf[0,0]=frame[0,0]+1\n    return nf, bool(nf[0,0]==3)"
fn=verify.compile_predict(PRED)

def test_execute_halts_on_mismatch_and_records_transition():
    r = execute.execute_plan(RealGame(), [[1],[1],[1]], fn, mask=None)
    assert r["solved"] is False
    assert r["halt_step"] == 2                       # mismatch when c becomes 2 (real sets (1,1)=7)
    assert len(r["new_transitions"]) == 1
    assert r["new_transitions"][0]["next_frame"][1,1] == 7

def test_execute_solves_when_model_matches():
    # mask out (1,1) so the unmodeled cell is ignored -> model matches -> plan solves
    mask=np.zeros((64,64),dtype=bool); mask[1,1]=True
    r = execute.execute_plan(RealGame(), [[1],[1],[1]], fn, mask=mask)
    assert r["solved"] is True and r["halt_step"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.arcv/bin/python -m pytest tests/test_e125_execute.py -q`
Expected: FAIL (`e125.execute` missing).

- [ ] **Step 3: Write minimal implementation**

```python
# experiments/e125/execute.py
"""Execute a sim-planned trajectory against the REAL env, step-by-step, halting the instant the real masked
next-frame diverges from predict()'s -- that divergence is the model-surprise signal (E122-style); the real
transition is recorded so the model can be re-synthesized. Only verified plans touch the env (action-efficient)."""
import numpy as np
from e125 import verify


def execute_plan(real_game, plan, predict_fn, mask):
    real_game.reset(); base = real_game.levels
    cur = np.asarray(real_game.frame).copy()
    verified, new_trans = [], []
    for i, a in enumerate(plan):
        try:
            pred_nf, _ = predict_fn(cur, list(a))
        except Exception:
            pred_nf = cur
        real_game.step(*a)
        real_nf = np.asarray(real_game.frame)
        if real_game.levels > base:
            verified.append(a)
            return {"solved": True, "verified_prefix": verified, "new_transitions": new_trans, "halt_step": None}
        if not np.array_equal(verify._masked(pred_nf, mask), verify._masked(real_nf, mask)):
            new_trans.append({"frame": cur.copy(), "action": list(a), "next_frame": real_nf.copy(),
                              "level_up": False})
            return {"solved": False, "verified_prefix": verified, "new_transitions": new_trans, "halt_step": i}
        verified.append(a); cur = real_nf.copy()
    return {"solved": False, "verified_prefix": verified, "new_transitions": new_trans, "halt_step": None}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.arcv/bin/python -m pytest tests/test_e125_execute.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit (if approved)**

```bash
git add experiments/e125/execute.py tests/test_e125_execute.py
git commit -m "E125 Task 4: executor + halt-on-sim-vs-real-mismatch"
```

---

## Task 5: Explorer + single-level agent loop (Milestone 2 — the E124-beating proof)

**Files:**
- Create: `experiments/e125/explorer.py`, `experiments/e125/agent.py`, `experiments/e125_executable_world.py`
- Test: `tests/test_e125_agent.py`

**Interfaces:**
- Consumes: `explorer.collect`, `synth.synthesize`, `simworld.plan`, `execute.execute_plan`, `e119/perceive.status_mask`.
- Produces: `explorer.collect(game_factory, candidates_fn, budget) -> list[dict]` (change-seeking transitions, deduped); `agent.solve_level(game_factory, candidates_fn, action_api, game, mask, synth_fn, budget_explore=60, budget_plan=20000, rounds=6, traces_dir=None) -> dict` with `{"solved","actions","rounds_used","real_actions"}`. `synth_fn` is injectable (mock in tests; `synth.synthesize` live). The loop: explore → synthesize predict() → plan-in-sim → execute-verify → on halt, add the new transition + re-synthesize → repeat.

- [ ] **Step 1: Write the failing test** (synthetic game + a mock `synth_fn` that returns the true model)

```python
# tests/test_e125_agent.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
import numpy as np
from e125 import agent, verify

class Deep:
    """level-up only after [1]x6 (depth 6); frame[0,0]=count of 1s. Real env BFS to depth 6 over {1,2} is
    expensive, but planning in the synthesized model is instant."""
    def __init__(self): self.reset()
    def reset(self): self.c=0; self.levels=0; self.done=False; self.frame=np.zeros((64,64),dtype=int)
    def step(self,a,x=None,y=None):
        if a==1: self.c+=1
        else: self.c=0
        self.frame=np.zeros((64,64),dtype=int); self.frame[0,0]=self.c
        if self.c==6: self.levels=1; self.done=True
        if self.c>8: self.done=True

TRUE="def predict(frame, action):\n    nf=frame.copy()\n    nf[0,0]=frame[0,0]+1 if action==[1] else 0\n    return nf, bool((frame[0,0]+1 if action==[1] else 0)==6)"

def test_solve_level_via_plan_in_sim():
    synth_fn = lambda transitions, action_api, game, mask, **kw: (TRUE, verify.compile_predict(TRUE))
    r = agent.solve_level(Deep, lambda fr:[[1],[2]], "actions=[1,2]", "deep", mask=None,
                          synth_fn=synth_fn, budget_explore=20, budget_plan=5000, rounds=3)
    assert r["solved"] is True
    assert r["actions"] == [[1],[1],[1],[1],[1],[1]]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.arcv/bin/python -m pytest tests/test_e125_agent.py -q`
Expected: FAIL (`e125.agent` missing).

- [ ] **Step 3: Write minimal implementation**

```python
# experiments/e125/explorer.py
"""Change-seeking exploration: collect exact (frame,action,next_frame,level_up) transitions, preferring
actions that change the board (signal). Env = ground truth."""
import numpy as np


def collect(game_factory, candidates_fn, budget):
    g = game_factory(); g.reset()
    trans = []; seen = set()
    for _ in range(budget):
        cands = [s if isinstance(s, list) else [s] for s in candidates_fn(g.frame)]
        if not cands:
            break
        a = cands[len(trans) % len(cands)]               # round-robin (deterministic, covers the action set)
        pf = np.asarray(g.frame).copy(); lv = g.levels
        g.step(*a)
        nf = np.asarray(g.frame).copy()
        key = (pf.tobytes(), tuple(a))
        if key not in seen:
            seen.add(key)
            trans.append({"frame": pf, "action": list(a), "next_frame": nf, "level_up": g.levels > lv})
        if g.done:
            g = game_factory(); g.reset()
    return trans
```

```python
# experiments/e125/agent.py
"""The single-level loop: explore -> synthesize predict() (verifier-gated) -> plan IN SIMULATION -> execute
vs the real env, halting on mismatch -> add the surprising transition + re-synthesize -> repeat. Only verified
plans touch the env. The env decides correctness (a real levels bump)."""
from e125 import explorer, simworld, execute


def solve_level(game_factory, candidates_fn, action_api, game, mask, synth_fn,
                budget_explore=60, budget_plan=20000, rounds=6, traces_dir=None):
    trans = explorer.collect(game_factory, candidates_fn, budget_explore)
    real_actions = budget_explore
    committed = []
    for rnd in range(rounds):
        src, fn = synth_fn(trans, action_api, game, mask, traces_dir=traces_dir)
        if fn is None:
            return {"solved": False, "actions": committed, "rounds_used": rnd, "real_actions": real_actions,
                    "reason": "no verified predict()"}
        init = game_factory(); init.reset()
        for a in committed:
            init.step(*a)
        plan = simworld.plan(fn, init.frame, candidates_fn, budget_plan)
        if plan is None:
            return {"solved": False, "actions": committed, "rounds_used": rnd, "real_actions": real_actions,
                    "reason": "no sim plan"}
        # execute committed+plan against a fresh real game, but verify only the new `plan` segment
        rg = game_factory(); rg.reset()
        for a in committed:
            rg.step(*a)
        res = _exec_from(rg, plan, fn, mask)
        real_actions += len(res["verified_prefix"]) + (1 if res["halt_step"] is not None else 0)
        committed += res["verified_prefix"]
        if res["solved"]:
            return {"solved": True, "actions": committed, "rounds_used": rnd + 1, "real_actions": real_actions}
        if res["new_transitions"]:
            trans = trans + res["new_transitions"]       # add the surprising transition, re-synthesize next round
        else:
            return {"solved": False, "actions": committed, "rounds_used": rnd + 1, "real_actions": real_actions,
                    "reason": "plan exhausted without progress"}
    return {"solved": False, "actions": committed, "rounds_used": rounds, "real_actions": real_actions}


def _exec_from(real_game, plan, predict_fn, mask):
    """execute.execute_plan but the real_game is already advanced to the committed prefix (do not reset)."""
    import numpy as np
    from e125 import verify
    base = real_game.levels; cur = np.asarray(real_game.frame).copy()
    verified, new_trans = [], []
    for i, a in enumerate(plan):
        try:
            pred_nf, _ = predict_fn(cur, list(a))
        except Exception:
            pred_nf = cur
        real_game.step(*a); real_nf = np.asarray(real_game.frame)
        if real_game.levels > base:
            verified.append(a)
            return {"solved": True, "verified_prefix": verified, "new_transitions": new_trans, "halt_step": None}
        if not np.array_equal(verify._masked(pred_nf, mask), verify._masked(real_nf, mask)):
            new_trans.append({"frame": cur.copy(), "action": list(a), "next_frame": real_nf.copy(), "level_up": False})
            return {"solved": False, "verified_prefix": verified, "new_transitions": new_trans, "halt_step": i}
        verified.append(a); cur = real_nf.copy()
    return {"solved": False, "verified_prefix": verified, "new_transitions": new_trans, "halt_step": None}
```

```python
# experiments/e125_executable_world.py
"""E125 entry: structured executable-world-model agent. Solve a level by synthesizing a verified predict(),
planning in simulation, and executing verified plans. save_results before asserts (CLAUDE.md)."""
import os, sys, argparse, json
sys.path.insert(0, os.path.dirname(__file__))
from e125 import agent, synth
from common import save_results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default="g50t")
    ap.add_argument("--model", default="gpt-5.5")
    ap.add_argument("--traces", default="experiments/results/e125_traces")
    a = ap.parse_args()
    from arc3_sandbox import SandboxGame
    from e119 import perceive
    results = {}
    for gid in a.games.split(","):
        g = SandboxGame(gid); g.reset()
        avail = g.avail
        cands = (lambda fr: [[x] for x in avail if x in (1, 2, 3, 4, 5, 7)])
        mask = perceive.status_mask([g.frame])
        sfn = lambda tr, api, game, m, **kw: synth.synthesize(tr, api, game, m, model=a.model, **kw)
        results[gid] = agent.solve_level(lambda: SandboxGame(gid), cands, f"actions={avail}", gid, mask, sfn,
                                         traces_dir=a.traces)
    save_results("e125_executable_world", {"experiment": "e125_executable_world", "games": results})
    print("[e125]", json.dumps({k: {kk: v[kk] for kk in ("solved", "real_actions", "rounds_used")}
                                for k, v in results.items()}))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.arcv/bin/python -m pytest tests/test_e125_agent.py -q`
Expected: PASS.

- [ ] **Step 5: LIVE Milestone-2 run (manual — the E124-beating proof, only if Milestone 1 gate passed)**

```bash
DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib ~/.arcv/bin/python experiments/e125_executable_world.py --games g50t
```
Expected: a result dict per game. **The claim:** E125 solves `g50t` L0 (or reaches deeper than E124's 0) by planning in simulation, using far fewer real-env actions than blind search. Report honestly whichever way it goes.

- [ ] **Step 6: Commit (if approved)**

```bash
git add experiments/e125/explorer.py experiments/e125/agent.py experiments/e125_executable_world.py tests/test_e125_agent.py
git commit -m "E125 Task 5: explorer + single-level agent loop (Milestone 2)"
```

---

## Milestone 3 (deferred — plan after the Milestone-1/2 gates)

If Milestone 1 (codex models the dynamics) and Milestone 2 (plan-in-sim beats E124) pass, write a follow-up plan for: the **head-to-head harness** (E125 structured vs the loose sweep agent on `g50t` + `sp80`/`dc22`, table = levels · real-env actions · verification rate), **cross-level composition** (`PhasedTransition` carry-forward over the compositionality cliff), and **MDL refactoring** (codex simplifies the code on stall, rejected if it breaks the regression set). Do **not** build M3 until the M1/M2 gates are green.

---

## Self-Review

**Spec coverage:** §Thesis/§Architecture → Tasks 1–5 (verifier gate=T1, synth=T2, simworld plan-in-sim=T3, executor halt=T4, explorer+loop=T5). §Verifier gate → T1. §Plan-in-sim → T3. §Executor+halt+resync → T4 + T5 loop. §Source-free/telemetry → T2 (reuses `codex_iso`+`capture_lib`). §Metrics (real_actions/solved) → T5 return + entry. §Milestones → T2 Step 5 (M1 gate), T5 Step 5 (M2 proof), M3 deferred. §Compose/MDL/head-to-head → M3 (deferred per the spec's gating).

**Placeholder scan:** no TBD/TODO; every code step has complete code; the only deferral (M3) is explicit and spec-sanctioned, not a vague placeholder.

**Type consistency:** transition dict keys `{frame,action,next_frame,level_up}` consistent across T1/T2/T4/T5; `predict(frame,action)->(next_frame,level_up)` consistent; `verify.check(fn,transitions,mask)->(ok,ce)` and `verify._masked` reused in T2/T4/T5; `synthesize(transitions,action_api,game,mask,...)->(src,fn)` matches the `synth_fn` injection point in T5; `execute_plan(...)->{solved,verified_prefix,new_transitions,halt_step}` matches the loop's use (T5 re-implements the prefix-aware variant `_exec_from` deliberately, noted in-code).
