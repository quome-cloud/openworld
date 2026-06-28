# E127 Source-Simulated Reconstruction Core — Implementation Plan (Milestone 1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the reconstruction-and-certification core for E127: from sandboxed play of an ARC-AGI-3 game (no source access), reconstruct a *stateful* engine as code via a two-model differential-CEGIS loop, and certify it against the real environment to a bounded error — emitting an equivalence-to-real certificate plus the A-vs-B-vs-real gap.

**Architecture:** A self-contained `experiments/e127/` package. Model-authored engines are compiled in a sandbox and scored by *rollout* against a ground-truth game (a `GameLike` duck type satisfied by both the real `SandboxGame` and a deterministic `ToyGame` test fixture). The CEGIS loop treats the **real env as the convergence oracle** and the **second model as a diversity source** for counterexamples; convergence is a statistical equivalence-to-real claim (Clopper–Pearson lower bound on held-out next-frame accuracy with per-level coverage), never two-model agreement. Everything is offline-testable with `ToyGame` + injected fake model runners; the only real-LLM / real-arc_agi piece (`sandbox.py`) is integration-smoke-tested and gated.

**Tech Stack:** Python 3, `numpy` (allowed in experiments), `openworld.sandbox.SAFE_BUILTINS` (zero-dep core, already on `main`), `pytest`. The arc venv interpreter is `~/.arcv/bin/python` (override via `ARC_VENV`). No new third-party runtime deps.

**Milestone 1 deliverable:** `experiments/e127/reconstruct.py::reconstruct(...)` returns a certified-or-bounded stateful engine source + certificate + A-vs-B-vs-real gap, proven on `ToyGame` with fake runners. **Milestone 2 (separate later plan):** `world127.py` (engine→OpenWorld `World`), `solve.py` (receding-horizon plan-then-verify under ensemble pessimism), `experiments/e127_source_simulated.py` harness, results JSON, and paper integration.

## Global Constraints

- **SOURCE-FREE / SOLUTION-FREE:** no task may read any game's `<game>.py`, use `inspect.getsource` on a game, `importlib.util.spec_from_file_location` a game, reference an `environment_files/` dir, or bank/seed from solution traces. The agent learns dynamics only by acting through `GameLike`.
- **Self-contained off `main`:** E127 vendors its own primitives under `experiments/e127/` (sandbox, model runner, audit). Do NOT import from `experiments/e125/`, `experiments/e124/`, or `experiments/arc3_sandbox.py` — none are on `main`. The ONLY cross-package import allowed is `from openworld.sandbox import SAFE_BUILTINS`.
- **Convergence target is the REAL env, not A≈B.** Two-model agreement is only a heuristic for label-spend and a diversity source. Never make A==B an acceptance gate.
- **Engines are STATEFUL symbolic code** (declared latent `state`), scored on rollouts. No stateless `step(frame,action)` contract. No neural nets.
- **Never mask the correctness comparison.** Frame-correctness scoring compares FULL frames. The identity mask is used only for state-novelty hashing in probes/coverage, and what it hides is reported.
- **Frames are `numpy` 2-D `int` arrays in memory**; lists only at JSON/IO boundaries. Grids are square (ToyGame uses 8×8; real games 64×64) — all code must be size-agnostic (never hard-code 64 or 8).
- **Actions are 3-tuples `(kind:int, x:int|None, y:int|None)`** where `kind` 1–5,7 are simple and 6 is a click using `x,y` (both 0-based, x=col, y=row). `None` x/y for non-click actions.
- **Determinism / honesty (CLAUDE.md):** fixed seeds; `assert` the sign/shape of every claim; call any `save_results` BEFORE asserts. Tests must be deterministic (no `random` without a seed, no wall-clock, no real network).
- **Commit after each task** with the trailing `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>` line. Work happens in the worktree `/Users/jim/Desktop/openworld-e127` on branch `e127-source-simulated`.
- **Run tests with the arc venv:** `~/.arcv/bin/python -m pytest tests/e127/<file> -v`.
- **All perception/interaction modalities are in scope.** Frame perception is the `(H,W)` grid; interaction is directional (`1–5,7`) AND click/mouse (`ACTION6` with `x,y`). Click targets are inferred ONLY from pixels (small connected components + rare-color cells) — never from source or any privileged engine method. Candidate actions are derived from the game's `available_actions`, so a click-only game gets clicks at inferred targets and a directional game gets directional moves. The reconstruct loop (Task 9) must be validated on BOTH a directional fixture (`ToyGame`) and a click fixture (`ToyClickGame`).

## Shared interfaces (every task depends on these — copy exactly)

**`GameLike` duck type** (the ground-truth oracle; satisfied by `ToyGame` and `SandboxGame`):
```
reset() -> frame            # np.ndarray (H,W) int; resets to start, sets attrs
step(a:int, x=None, y=None) -> frame
# attributes after reset/step: .frame (np.ndarray), .levels (int), .win (int), .avail (list[int]), .done (bool)
```

**Engine** (model-authored, compiled by `safe_exec.compile_engine`): source defines a class `Engine` with:
```
__init__(self)              # no required args
reset(self) -> frame        # np.ndarray int; sets self.state (a dict of declared latent vars incl. "levels")
step(self, action) -> frame # action=(kind,x,y); mutates self.state; returns next frame
is_win(self, prev_frame) -> bool   # reads self.state (procedural progress)
# self.state: dict; MUST contain integer key "levels"
```

**Episode** (produced by `engine.rollout` and by `play` over a `GameLike`): a list of step dicts; element 0 is the reset with `action=None`:
```
{"action": (kind,x,y) | None, "frame": np.ndarray, "levels": int}
```

**Module/function signatures fixed across tasks:**
- `safe_exec.compile_engine(src:str) -> (callable() -> Engine) | None`   # a *factory* returning a fresh Engine; None on compile failure
- `engine.rollout(factory, actions:list[tuple]) -> list[np.ndarray]`     # [reset_frame, frame_after_a0, ...]; raises EngineError on runtime fault
- `engine.EngineError`  (exception)
- `engine.play(game:GameLike, actions:list[tuple]) -> Episode`           # drives a GameLike; element 0 = reset
- `engine.score_rollout(factory, episode:Episode) -> dict`               # {"transitions","exact","cell_acc","levelup_match","levelup_total","errored"}
- `engine.identity_mask(episodes:list[Episode], thr=0.95) -> np.ndarray`  # bool (H,W); True = changes on >=thr of steps
- `engine.first_disagreement(factoryA, factoryB, actions) -> int | None`  # first index where full-frame rollouts differ, else None
- `engine.looks_like_lookup_table(src:str) -> bool`                       # static degeneracy heuristic
- `probes.find_counterexamples(factory, real_factory, observed:list[Episode], mask, action_api, budget) -> list[dict]`  # each {"actions":[...], "index":int, "real_frame":np.ndarray, "engine_frame":np.ndarray, "kind":str}
- `probes.property_violations(factory, real_factory, action_api, budget) -> list[dict]`
- `certify.clopper_pearson_lower(k:int, n:int, delta:float) -> float`
- `certify.certify_engine(factory, holdout:list[Episode], n_levels:int, eps=0.01, delta=0.05, coverage_target=0.8) -> dict`  # the certificate
- `iso.run(prompt:str, model:str, game:str="", workdir=None, timeout=600, _exec=None) -> dict`   # {"engine_src":str|None, "rationale":str, "raw":str}
- `audit.audit_dir(wd:str) -> list[str]`  ;  `audit.audit_clean(wd:str) -> bool`
- `sandbox.SandboxGame(gid, venv=ARC_VENV)`  (implements GameLike)
- `reconstruct.reconstruct(real_factory, action_api, n_levels, models=("claude","codex"), max_rounds=4, budget=None, _runners=None, seed=0) -> dict`

---

### Task 1: Package scaffold + sandboxed engine compiler (`safe_exec.py`)

**Files:**
- Create: `experiments/e127/__init__.py`
- Create: `experiments/e127/safe_exec.py`
- Create: `tests/e127/__init__.py`
- Test: `tests/e127/test_safe_exec.py`

**Interfaces:**
- Produces: `compile_engine(src) -> factory|None`. The factory called with no args returns a fresh object exposing `reset/step/is_win/state`.
- Consumes: `from openworld.sandbox import SAFE_BUILTINS`.

- [ ] **Step 1: Write the failing test**
```python
# tests/e127/test_safe_exec.py
import numpy as np
from experiments.e127.safe_exec import compile_engine

GOOD = '''
class Engine:
    def __init__(self):
        self.state = {"levels": 0, "t": 0}
    def reset(self):
        self.state = {"levels": 0, "t": 0}
        return np.zeros((4, 4), dtype=int)
    def step(self, action):
        self.state["t"] += 1
        f = np.zeros((4, 4), dtype=int); f[0, 0] = self.state["t"] % 16
        return f
    def is_win(self, prev_frame):
        return self.state["levels"] >= 1
'''

def test_compiles_and_runs():
    factory = compile_engine(GOOD)
    assert factory is not None
    e = factory()
    f0 = e.reset()
    assert f0.shape == (4, 4) and f0.sum() == 0
    f1 = e.step((7, None, None))
    assert f1[0, 0] == 1 and e.state["t"] == 1

def test_fresh_instances_are_independent():
    factory = compile_engine(GOOD)
    a, b = factory(), factory()
    a.reset(); a.step((7, None, None))
    b.reset()
    assert a.state["t"] == 1 and b.state["t"] == 0

def test_syntax_error_returns_none():
    assert compile_engine("class Engine(:\n  pass") is None

def test_missing_engine_class_returns_none():
    assert compile_engine("x = 1") is None

def test_numpy_available_but_imports_blocked():
    # numpy usable via the injected `np`; but `import os` must fail at runtime
    bad = "class Engine:\n    def reset(self):\n        import os\n        return np.zeros((2,2),dtype=int)\n"
    factory = compile_engine(bad)
    # compiles (def body not executed yet) but reset() raises -> factory ok, reset raises
    e = factory()
    raised = False
    try:
        e.reset()
    except Exception:
        raised = True
    assert raised
```

- [ ] **Step 2: Run test to verify it fails**
Run: `~/.arcv/bin/python -m pytest tests/e127/test_safe_exec.py -v`
Expected: FAIL — `ModuleNotFoundError: experiments.e127.safe_exec`.

- [ ] **Step 3: Write minimal implementation**
```python
# experiments/e127/__init__.py
# (empty package marker)
```
```python
# tests/e127/__init__.py
# (empty package marker)
```
```python
# experiments/e127/safe_exec.py
"""Compile a model-authored stateful Engine into a fresh-instance factory, sandboxed.

Engines run inside openworld's SAFE_BUILTINS (no __import__, no open/exec) PLUS a numpy handle
`np`, because pixel/grid math legitimately needs arrays. The gate environment is therefore a
SUBSET of what a World transition would later get (numpy + safe builtins), so a compiling engine
never fails for a missing name at search time. A source that fails to define `Engine` or raises at
class-definition time returns None; runtime faults inside reset/step surface to the caller."""
import numpy as np
from openworld.sandbox import SAFE_BUILTINS


def compile_engine(src):
    """Return a zero-arg factory producing fresh Engine instances, or None on compile failure."""
    ns = {"np": np, "__builtins__": SAFE_BUILTINS}
    try:
        exec(src, ns)
    except Exception:
        return None
    cls = ns.get("Engine")
    if not isinstance(cls, type):
        return None

    def factory():
        return cls()

    # Smoke: construct once so an Engine whose __init__ explodes is rejected early.
    try:
        factory()
    except Exception:
        return None
    return factory
```

- [ ] **Step 4: Run test to verify it passes**
Run: `~/.arcv/bin/python -m pytest tests/e127/test_safe_exec.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**
```bash
git add experiments/e127/__init__.py experiments/e127/safe_exec.py tests/e127/__init__.py tests/e127/test_safe_exec.py
git commit -m "E127 T1: sandboxed stateful-engine compiler (safe_exec)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: ToyGame fixture — deterministic hidden-state ground truth (`tests/e127/toy.py`)

A small deterministic game that is the REAL-env stand-in for all offline tests. It has the three properties that make E127 non-trivial: (1) **hidden state** (a collected-gems counter not directly shown), (2) a **status-bar cell** that changes every step (the masking/identity target — but is deterministically predictable from a latent step counter, so an engine MUST model it), and (3) a **procedural level-up** (`levels` rises only after the hidden counter reaches a threshold). It satisfies `GameLike`. The task also ships `TOY_ENGINE_SRC`: a *faithful* Engine reimplementation used as the "correct reconstruction" in later tests, and `TOY_WRONG_SRC`: a plausible-but-wrong engine (ignores collection) used to exercise counterexample/ gap machinery.

**Files:**
- Create: `tests/e127/toy.py`
- Test: `tests/e127/test_toy.py`

**Interfaces:**
- Produces: `ToyGame()` (GameLike); `toy_factory() -> ToyGame`; `TOY_ENGINE_SRC` (str, a faithful Engine); `TOY_WRONG_SRC` (str); `ACTION_API` (str description) and `TOY_ACTIONS=[1,2,3,4,5,7]`.
- Consumes: nothing.

ToyGame rules (8×8, colors 0–15), **fixed and exact** (engines must rediscover these by acting):
- Layout: cursor at (y,x)=(4,4) shown as color 8. Three gems (color 4) at fixed cells (1,1),(1,6),(6,3). Background 0. Row 0 is the status bar.
- Status bar: cell (0,0) = `(t % 15) + 1` where `t` = number of steps since reset (so it changes every step, never 0; cells (0,1..7) stay 0).
- Actions: 1=up,2=down,3=left,4=right move the cursor one cell (clamped to rows 1..7, cols 0..7; row 0 is reserved for the status bar so the cursor never enters it). 5=interact and 7=noop do not move. 6 is not in `avail`.
- Collection: if after a move the cursor lands on a gem cell, the gem is removed (set to bg) and hidden `collected += 1`.
- Level-up: when `collected == 3`, set `levels += 1`, `collected = 0`, and reload the three gems (cursor stays). `win = 1` (one level to clear). `done = (levels >= win)`.
- Determinism: identical action sequences from `reset()` produce identical frames.

- [ ] **Step 1: Write the failing test**
```python
# tests/e127/test_toy.py
import numpy as np
from tests.e127.toy import ToyGame, toy_factory, TOY_ENGINE_SRC, TOY_WRONG_SRC, TOY_ACTIONS
from experiments.e127.safe_exec import compile_engine

def test_reset_layout():
    g = ToyGame(); f = g.reset()
    assert f.shape == (8, 8)
    assert f[4, 4] == 8                       # cursor
    assert f[1, 1] == 4 and f[1, 6] == 4 and f[6, 3] == 4   # gems
    assert f[0, 0] == 1                        # status bar t=0 -> (0%15)+1
    assert g.levels == 0 and g.win == 1 and g.done is False

def test_status_bar_changes_every_step():
    g = ToyGame(); g.reset()
    g.step(7); assert g.frame[0, 0] == 2
    g.step(7); assert g.frame[0, 0] == 3

def test_cursor_moves_and_clamps():
    g = ToyGame(); g.reset()
    g.step(1); assert g.frame[4, 4] == 0 and g.frame[3, 4] == 8   # moved up
    for _ in range(10):
        g.step(1)
    assert np.argwhere(g.frame == 8)[0][0] == 1                   # clamped at row 1

def test_collection_and_levelup_is_procedural():
    g = ToyGame(); g.reset()
    # path cursor (4,4) -> collect (1,1): up x3 to row1, left x3 to col1
    for a in (1, 1, 1, 3, 3, 3):
        g.step(a)
    assert g.levels == 0                       # only 1 gem collected, no level yet
    # collect (1,6): right x5 to col6
    for a in (4, 4, 4, 4, 4):
        g.step(a)
    # collect (6,3): down x5 to row6, then to col3 (already? col6->col3 left x3)
    for a in (2, 2, 2, 2, 2, 3, 3, 3):
        g.step(a)
    assert g.levels == 1 and g.done is True    # third gem -> level up, game done

def test_determinism():
    g1, g2 = ToyGame(), ToyGame()
    g1.reset(); g2.reset()
    seq = [1, 3, 7, 4, 2, 5]
    for a in seq:
        g1.step(a); g2.step(a)
    assert np.array_equal(g1.frame, g2.frame)

def test_faithful_engine_matches_toygame():
    # The reference reconstruction reproduces ToyGame frame-for-frame over a long sequence.
    factory = compile_engine(TOY_ENGINE_SRC); assert factory is not None
    e = factory(); g = ToyGame()
    ef = e.reset(); gf = g.reset()
    assert np.array_equal(ef, gf)
    rng = np.random.default_rng(0)
    for _ in range(60):
        a = int(rng.choice(TOY_ACTIONS))
        ef = e.step((a, None, None)); gf = g.step(a)
        assert np.array_equal(ef, gf), f"mismatch at action {a}"

def test_wrong_engine_diverges_from_toygame():
    factory = compile_engine(TOY_WRONG_SRC); assert factory is not None
    e = factory(); g = ToyGame()
    e.reset(); g.reset()
    diverged = False
    for a in (1, 1, 1, 3, 3, 3, 7, 7):
        ef = e.step((a, None, None)); gf = g.step(a)
        if not np.array_equal(ef, gf):
            diverged = True
    assert diverged
```

- [ ] **Step 2: Run test to verify it fails**
Run: `~/.arcv/bin/python -m pytest tests/e127/test_toy.py -v`
Expected: FAIL — `ModuleNotFoundError: tests.e127.toy`.

- [ ] **Step 3: Write minimal implementation**
```python
# tests/e127/toy.py
"""Deterministic hidden-state ground-truth game for E127 offline tests (GameLike)."""
import numpy as np

_GEMS = [(1, 1), (1, 6), (6, 3)]
TOY_ACTIONS = [1, 2, 3, 4, 5, 7]
ACTION_API = ("Actions: 1=up,2=down,3=left,4=right move a cursor (color 8) one cell, clamped to "
              "rows 1..7 / cols 0..7. 5=interact and 7=noop do not move. Row 0 is a status bar. "
              "Grid 8x8, colors 0-15. No clicks (action 6 unavailable).")


def _draw(cursor, gems, t):
    f = np.zeros((8, 8), dtype=int)
    for (gy, gx) in gems:
        f[gy, gx] = 4
    f[cursor[0], cursor[1]] = 8
    f[0, 0] = (t % 15) + 1
    return f


class ToyGame:
    def __init__(self):
        self.win = 1
        self._reset_fields()

    def _reset_fields(self):
        self.cursor = [4, 4]
        self.gems = list(_GEMS)
        self.collected = 0          # HIDDEN
        self.t = 0
        self.levels = 0
        self.done = False
        self.avail = [1, 2, 3, 4, 5, 7]

    def reset(self):
        self._reset_fields()
        self.frame = _draw(self.cursor, self.gems, self.t)
        return self.frame

    def step(self, a, x=None, y=None):
        if not self.done:
            self.t += 1
            ny, nx = self.cursor
            if a == 1:
                ny = max(1, ny - 1)
            elif a == 2:
                ny = min(7, ny + 1)
            elif a == 3:
                nx = max(0, nx - 1)
            elif a == 4:
                nx = min(7, nx + 1)
            self.cursor = [ny, nx]
            if (ny, nx) in self.gems:
                self.gems.remove((ny, nx))
                self.collected += 1
                if self.collected == 3:
                    self.levels += 1
                    self.collected = 0
                    self.gems = list(_GEMS)
                    self.done = self.levels >= self.win
        self.frame = _draw(self.cursor, self.gems, self.t)
        return self.frame


def toy_factory():
    return ToyGame()


# A FAITHFUL reconstruction (what a perfect model would author). Mirrors ToyGame exactly.
TOY_ENGINE_SRC = '''
_GEMS = [(1, 1), (1, 6), (6, 3)]
class Engine:
    def __init__(self):
        self.state = {"levels": 0, "collected": 0, "t": 0, "cursor": [4, 4], "gems": list(_GEMS), "done": False}
    def _draw(self):
        f = np.zeros((8, 8), dtype=int)
        for (gy, gx) in self.state["gems"]:
            f[gy, gx] = 4
        c = self.state["cursor"]; f[c[0], c[1]] = 8
        f[0, 0] = (self.state["t"] % 15) + 1
        return f
    def reset(self):
        self.state = {"levels": 0, "collected": 0, "t": 0, "cursor": [4, 4], "gems": list(_GEMS), "done": False}
        return self._draw()
    def step(self, action):
        a = action[0]; s = self.state
        if not s["done"]:
            s["t"] += 1
            ny, nx = s["cursor"]
            if a == 1: ny = max(1, ny - 1)
            elif a == 2: ny = min(7, ny + 1)
            elif a == 3: nx = max(0, nx - 1)
            elif a == 4: nx = min(7, nx + 1)
            s["cursor"] = [ny, nx]
            if (ny, nx) in s["gems"]:
                s["gems"].remove((ny, nx)); s["collected"] += 1
                if s["collected"] == 3:
                    s["levels"] += 1; s["collected"] = 0; s["gems"] = list(_GEMS)
                    s["done"] = s["levels"] >= 1
        return self._draw()
    def is_win(self, prev_frame):
        return self.state["levels"] >= 1
'''

# A PLAUSIBLE-BUT-WRONG reconstruction: moves the cursor correctly but never collects gems / levels up.
TOY_WRONG_SRC = '''
_GEMS = [(1, 1), (1, 6), (6, 3)]
class Engine:
    def __init__(self):
        self.state = {"levels": 0, "t": 0, "cursor": [4, 4]}
    def _draw(self):
        f = np.zeros((8, 8), dtype=int)
        for (gy, gx) in _GEMS:
            f[gy, gx] = 4
        c = self.state["cursor"]; f[c[0], c[1]] = 8
        f[0, 0] = (self.state["t"] % 15) + 1
        return f
    def reset(self):
        self.state = {"levels": 0, "t": 0, "cursor": [4, 4]}
        return self._draw()
    def step(self, action):
        a = action[0]; s = self.state; s["t"] += 1
        ny, nx = s["cursor"]
        if a == 1: ny = max(1, ny - 1)
        elif a == 2: ny = min(7, ny + 1)
        elif a == 3: nx = max(0, nx - 1)
        elif a == 4: nx = min(7, nx + 1)
        s["cursor"] = [ny, nx]
        return self._draw()
    def is_win(self, prev_frame):
        return False
'''
```

- [ ] **Step 4: Run test to verify it passes**
Run: `~/.arcv/bin/python -m pytest tests/e127/test_toy.py -v`
Expected: PASS (7 tests). If `test_collection_and_levelup_is_procedural` fails, fix the ToyGame movement math until the documented gem-collection path levels up exactly once — the test path is the contract.

- [ ] **Step 5: Commit**
```bash
git add tests/e127/toy.py tests/e127/test_toy.py
git commit -m "E127 T2: ToyGame deterministic hidden-state ground truth + reference engines

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Engine rollout, scoring, mask, disagreement (`engine.py`)

**Files:**
- Create: `experiments/e127/engine.py`
- Test: `tests/e127/test_engine.py`

**Interfaces:**
- Consumes: `safe_exec.compile_engine`; `tests.e127.toy` (in tests only); `GameLike`.
- Produces: `EngineError`, `rollout`, `play`, `score_rollout`, `identity_mask`, `first_disagreement`, `looks_like_lookup_table` (signatures in Shared interfaces). `score_rollout` compares FULL frames for `exact`/`cell_acc` (never masked); `levelup_*` compares the engine's `state["levels"]` increments against the episode's `levels` increments at each transition.

- [ ] **Step 1: Write the failing test**
```python
# tests/e127/test_engine.py
import numpy as np
from experiments.e127 import engine
from experiments.e127.safe_exec import compile_engine
from tests.e127.toy import ToyGame, toy_factory, TOY_ENGINE_SRC, TOY_WRONG_SRC

def _acts(seq):
    return [(a, None, None) for a in seq]

def test_play_then_score_faithful_is_perfect():
    g = ToyGame()
    ep = engine.play(g, _acts([1, 1, 1, 3, 3, 3, 4, 4]))
    assert ep[0]["action"] is None and ep[0]["frame"].shape == (8, 8)
    factory = compile_engine(TOY_ENGINE_SRC)
    s = engine.score_rollout(factory, ep)
    assert s["errored"] is False
    assert s["transitions"] == 8
    assert s["exact"] == 8 and s["cell_acc"] == 1.0

def test_score_wrong_engine_imperfect():
    g = ToyGame()
    ep = engine.play(g, _acts([1, 1, 1, 3, 3, 3, 7, 7]))   # collects gem (1,1); wrong engine won't
    factory = compile_engine(TOY_WRONG_SRC)
    s = engine.score_rollout(factory, ep)
    assert s["exact"] < s["transitions"]                    # diverges once gem is collected

def test_levelup_accounting():
    g = ToyGame()
    seq = [1, 1, 1, 3, 3, 3, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 3, 3, 3]   # full clear -> 1 levelup
    ep = engine.play(g, _acts(seq))
    assert ep[-1]["levels"] == 1
    faithful = compile_engine(TOY_ENGINE_SRC)
    s = engine.score_rollout(faithful, ep)
    assert s["levelup_total"] == 1 and s["levelup_match"] == 1

def test_rollout_runtime_fault_is_errored_not_raised_in_score():
    bad = "class Engine:\n    def reset(self): return np.zeros((8,8),dtype=int)\n    def step(self, a): raise ValueError('boom')\n    def is_win(self, p): return False\n"
    factory = compile_engine(bad)
    g = ToyGame(); ep = engine.play(g, _acts([7, 7]))
    s = engine.score_rollout(factory, ep)
    assert s["errored"] is True and s["exact"] == 0

def test_identity_mask_flags_status_bar_only():
    g = ToyGame()
    ep = engine.play(g, _acts([7] * 30))     # noop: only the status bar (0,0) changes every step
    mask = engine.identity_mask([ep], thr=0.95)
    assert mask.shape == (8, 8)
    assert mask[0, 0] == True
    assert mask.sum() == 1                    # nothing else changes under pure noop

def test_first_disagreement():
    a = compile_engine(TOY_ENGINE_SRC); b = compile_engine(TOY_WRONG_SRC)
    acts = _acts([1, 1, 1, 3, 3, 3, 7])       # gem collected at the 6th action (index 5)
    idx = engine.first_disagreement(a, b, acts)
    assert idx is not None and idx == 5
    same = engine.first_disagreement(a, a, acts)
    assert same is None

def test_lookup_table_heuristic():
    assert engine.looks_like_lookup_table("class Engine:\n    TABLE = {" + ",".join(f"{i}:{i}" for i in range(200)) + "}\n") is True
    assert engine.looks_like_lookup_table(TOY_ENGINE_SRC) is False
```

- [ ] **Step 2: Run test to verify it fails**
Run: `~/.arcv/bin/python -m pytest tests/e127/test_engine.py -v`
Expected: FAIL — `ModuleNotFoundError: experiments.e127.engine`.

- [ ] **Step 3: Write minimal implementation**
```python
# experiments/e127/engine.py
"""Rollout, scoring, identity-masking and disagreement for stateful reconstructed engines.

Correctness scoring compares FULL frames (the engine must predict every cell, including the
status bar, from its own latent state). The identity mask is computed separately and used ONLY
for state-novelty hashing in probes — never to relax a correctness comparison."""
import numpy as np


class EngineError(Exception):
    pass


def rollout(factory, actions):
    """Return [reset_frame, frame_after_action0, ...]; raises EngineError on any runtime fault."""
    try:
        e = factory()
        frames = [np.asarray(e.reset())]
        for a in actions:
            frames.append(np.asarray(e.step(a)))
        return frames
    except Exception as ex:
        raise EngineError(str(ex))


def play(game, actions):
    """Drive a GameLike with `actions` (list of (kind,x,y)); return an Episode (elem 0 = reset)."""
    f0 = np.asarray(game.reset())
    ep = [{"action": None, "frame": f0, "levels": int(game.levels)}]
    for a in actions:
        kind, x, y = a
        f = np.asarray(game.step(kind, x, y))
        ep.append({"action": a, "frame": f, "levels": int(game.levels)})
    return ep


def score_rollout(factory, episode):
    """Score a factory's rollout against an observed Episode. FULL-frame correctness."""
    actions = [s["action"] for s in episode[1:]]
    try:
        e = factory()
        pred = [np.asarray(e.reset())]
        levels_pred = [int(e.state.get("levels", 0))]
        for a in actions:
            pred.append(np.asarray(e.step(a)))
            levels_pred.append(int(e.state.get("levels", 0)))
    except Exception:
        n = len(actions)
        return {"transitions": n, "exact": 0, "cell_acc": 0.0,
                "levelup_match": 0, "levelup_total": 0, "errored": True}
    exact = cell_sum = cell_tot = lv_match = lv_tot = 0
    for i in range(1, len(episode)):
        real = episode[i]["frame"]; pf = pred[i]
        if pf.shape != real.shape:
            cell_tot += real.size
            continue
        eq = (pf == real)
        if eq.all():
            exact += 1
        cell_sum += int(eq.sum()); cell_tot += real.size
        real_up = episode[i]["levels"] - episode[i - 1]["levels"]
        pred_up = levels_pred[i] - levels_pred[i - 1]
        if real_up > 0:
            lv_tot += 1
            if pred_up == real_up:
                lv_match += 1
    return {"transitions": len(episode) - 1, "exact": exact,
            "cell_acc": (cell_sum / cell_tot) if cell_tot else 0.0,
            "levelup_match": lv_match, "levelup_total": lv_tot, "errored": False}


def identity_mask(episodes, thr=0.95):
    """Bool mask (H,W): True where a cell changes between consecutive frames on >= thr of steps,
    aggregated across episodes. For state-IDENTITY hashing only."""
    H = W = None
    changed = total = None
    for ep in episodes:
        for i in range(1, len(ep)):
            a, b = ep[i - 1]["frame"], ep[i]["frame"]
            if changed is None:
                H, W = a.shape
                changed = np.zeros((H, W), dtype=float)
                total = 0
            changed += (a != b).astype(float)
            total += 1
    if total == 0:
        return np.zeros((1, 1), dtype=bool)
    return (changed / total) >= thr


def first_disagreement(factoryA, factoryB, actions):
    """First index in the rollout where A and B produce different full frames, else None.
    Index 0 is the reset frame; transitions are indices 1..len(actions)."""
    try:
        fa = rollout(factoryA, actions); fb = rollout(factoryB, actions)
    except EngineError:
        return 0
    n = min(len(fa), len(fb))
    for i in range(n):
        if fa[i].shape != fb[i].shape or not np.array_equal(fa[i], fb[i]):
            return i
    return None


def looks_like_lookup_table(src, max_int_literals=120):
    """Static degeneracy heuristic: an engine that memorizes observed frames as a big literal table.
    Flags sources with an excessive count of integer literals (a frame->frame dict). The PRIMARY
    defense against memorization is the disjoint held-out set in certify; this is a cheap pre-filter."""
    import re
    ints = re.findall(r"(?<![\w.])\d+", src)
    return len(ints) > max_int_literals
```

- [ ] **Step 4: Run test to verify it passes**
Run: `~/.arcv/bin/python -m pytest tests/e127/test_engine.py -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**
```bash
git add experiments/e127/engine.py tests/e127/test_engine.py
git commit -m "E127 T3: engine rollout/scoring/identity-mask/disagreement

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Statistical certificate (`certify.py`)

**Files:**
- Create: `experiments/e127/certify.py`
- Test: `tests/e127/test_certify.py`

**Interfaces:**
- Consumes: `engine.score_rollout`.
- Produces: `clopper_pearson_lower(k, n, delta)`, `betai(a, b, x)` (regularized incomplete beta, exposed for testing), `certify_engine(factory, holdout, n_levels, eps, delta, coverage_target)`.
- Certificate dict keys: `{"pass":bool, "acc":float, "acc_lower":float, "n":int, "exact":int, "eps":float, "delta":float, "coverage":float, "coverage_target":float, "levelup_match":int, "levelup_total":int, "errored":bool}`. `pass` is True iff `acc_lower >= 1-eps` AND `coverage >= coverage_target` AND not errored. `coverage` = fraction of `n_levels` that appear (as a `levels` value) among the holdout episodes' steps.

- [ ] **Step 1: Write the failing test**
```python
# tests/e127/test_certify.py
import math
import numpy as np
from experiments.e127 import certify, engine
from experiments.e127.safe_exec import compile_engine
from tests.e127.toy import ToyGame, TOY_ENGINE_SRC, TOY_WRONG_SRC

def _acts(seq):
    return [(a, None, None) for a in seq]

def test_betai_known_values():
    # I_x(1,1) = x ; symmetry I_x(a,b) = 1 - I_{1-x}(b,a)
    assert abs(certify.betai(1, 1, 0.37) - 0.37) < 1e-9
    assert abs(certify.betai(2, 3, 0.5) - (1 - certify.betai(3, 2, 0.5))) < 1e-9

def test_clopper_pearson_closed_form_k_equals_n():
    # For k=n successes, CP lower bound = delta**(1/n)
    for n, delta in [(10, 0.05), (50, 0.05), (300, 0.05)]:
        assert abs(certify.clopper_pearson_lower(n, n, delta) - delta ** (1.0 / n)) < 1e-6

def test_clopper_pearson_monotone():
    a = certify.clopper_pearson_lower(95, 100, 0.05)
    b = certify.clopper_pearson_lower(99, 100, 0.05)
    assert 0.0 < a < b < 1.0

def _holdout(n_eps=40, seed=1):
    rng = np.random.default_rng(seed)
    eps = []
    for _ in range(n_eps):
        g = ToyGame()
        k = int(rng.integers(6, 20))
        seq = [int(rng.choice([1, 2, 3, 4, 5, 7])) for _ in range(k)]
        eps.append(engine.play(g, _acts(seq)))
    return eps

def test_faithful_engine_certifies():
    factory = compile_engine(TOY_ENGINE_SRC)
    cert = certify.certify_engine(factory, _holdout(), n_levels=1, eps=0.01, delta=0.05, coverage_target=0.0)
    assert cert["errored"] is False
    assert cert["exact"] == cert["n"]          # perfect reproduction
    assert cert["acc_lower"] >= 0.99
    assert cert["pass"] is True

def test_wrong_engine_fails_certificate():
    factory = compile_engine(TOY_WRONG_SRC)
    cert = certify.certify_engine(factory, _holdout(), n_levels=1, eps=0.01, delta=0.05, coverage_target=0.0)
    assert cert["pass"] is False
    assert cert["acc"] < 1.0
```

- [ ] **Step 2: Run test to verify it fails**
Run: `~/.arcv/bin/python -m pytest tests/e127/test_certify.py -v`
Expected: FAIL — `ModuleNotFoundError: experiments.e127.certify`.

- [ ] **Step 3: Write minimal implementation**
```python
# experiments/e127/certify.py
"""Equivalence-to-real certificate: a Clopper-Pearson lower bound on held-out next-frame accuracy,
plus per-level coverage. This is the acceptance gate -- NOT two-model agreement. A certificate that
fails still returns its measured numbers (a bound, never a binary 'unified')."""
import math
from experiments.e127 import engine as _engine


def betai(a, b, x):
    """Regularized incomplete beta I_x(a,b) via the Lentz continued fraction (Numerical Recipes)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(a * math.log(x) + b * math.log(1.0 - x) - lbeta) / a
    # continued fraction (betacf)
    tiny = 1e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-12:
            break
    cf = front * h
    if x < (a + 1.0) / (a + b + 2.0):
        return cf
    return 1.0 - cf  # use symmetry was applied implicitly; handled by front choice
    # NOTE: front uses x directly; for x beyond the pivot the standard trick is to compute the
    # complement. The test_betai_known_values cases stay on the convergent side; for the CP solve
    # below we always evaluate via bisection on p, where this form is well-behaved on (0,1).


def clopper_pearson_lower(k, n, delta):
    """Lower (1-delta) confidence bound on a binomial proportion with k successes in n trials.
    Defined by P[Bin(n,L) >= k] = delta  <=>  I_L(k, n-k+1) = delta. Solve by bisection in L."""
    if n == 0:
        return 0.0
    if k == 0:
        return 0.0
    if k == n:
        return delta ** (1.0 / n)
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if betai(k, n - k + 1, mid) > delta:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def certify_engine(factory, holdout, n_levels, eps=0.01, delta=0.05, coverage_target=0.8):
    """Score `factory` on a disjoint held-out set of Episodes; emit the certificate dict."""
    n = exact = lv_match = lv_tot = 0
    errored = False
    seen_levels = set()
    for ep in holdout:
        for s in ep:
            seen_levels.add(int(s["levels"]))
        sc = _engine.score_rollout(factory, ep)
        if sc["errored"]:
            errored = True
        n += sc["transitions"]; exact += sc["exact"]
        lv_match += sc["levelup_match"]; lv_tot += sc["levelup_total"]
    acc = (exact / n) if n else 0.0
    acc_lower = clopper_pearson_lower(exact, n, delta) if n else 0.0
    coverage = (len(seen_levels & set(range(n_levels + 1))) / max(1, n_levels + 1))
    passed = (not errored) and (acc_lower >= 1.0 - eps) and (coverage >= coverage_target)
    return {"pass": bool(passed), "acc": acc, "acc_lower": acc_lower, "n": n, "exact": exact,
            "eps": eps, "delta": delta, "coverage": coverage, "coverage_target": coverage_target,
            "levelup_match": lv_match, "levelup_total": lv_tot, "errored": errored}
```
Note for the implementer: if `test_betai_known_values` reveals the pivot/complement branch is wrong for `I_x(2,3)` vs `I_x(3,2)`, implement the standard Numerical-Recipes `betai` that swaps to the complement when `x >= (a+1)/(a+b+2)` by computing `1 - betacf(b,a,1-x)*...`. The `clopper_pearson_lower` bisection only needs `betai` correct and monotone on `(0,1)`, which the closed-form `k==n` test and the monotonicity test pin down.

- [ ] **Step 4: Run test to verify it passes**
Run: `~/.arcv/bin/python -m pytest tests/e127/test_certify.py -v`
Expected: PASS (5 tests). Fix `betai` per the note until `test_betai_known_values` passes.

- [ ] **Step 5: Commit**
```bash
git add experiments/e127/certify.py tests/e127/test_certify.py
git commit -m "E127 T4: equivalence-to-real certificate (Clopper-Pearson + coverage)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Counterexample search & property falsifiers (`probes.py`)

**Files:**
- Create: `experiments/e127/probes.py`
- Test: `tests/e127/test_probes.py`

**Interfaces:**
- Consumes: `engine.rollout/play/first_disagreement/identity_mask`; a `real_factory() -> GameLike`.
- Produces:
  - `find_counterexamples(factory, real_factory, observed, mask, action_api, budget) -> list[cex]` where each `cex = {"actions":[...], "index":int, "real_frame":np.ndarray, "engine_frame":np.ndarray, "kind":str}`. Strategy: from prefixes of `observed` episodes plus novelty-guided random extensions, roll out the engine and the real env (real via prefix-shared replay from a fresh `real_factory()`); return the first transition where the engine's frame != the real frame. `budget` caps real-env steps.
  - `property_violations(factory, real_factory, action_api, budget) -> list[cex]` with `kind` in {"determinism","levelup_delta","color_range"}: checks engine predictions against universal properties verified on the real env (determinism: same actions twice → same frames; levelup_delta: when real `levels` rises, the board changes a lot; color_range: predicted cells stay 0..15).
- Real-env step accounting: both functions return `(list, real_steps_used)`? No — keep return = list; track steps via a passed-in mutable `budget` object: `budget = {"limit":int, "used":int}`; functions increment `budget["used"]` and stop when `used >= limit`.

- [ ] **Step 1: Write the failing test**
```python
# tests/e127/test_probes.py
import numpy as np
from experiments.e127 import probes, engine
from experiments.e127.safe_exec import compile_engine
from tests.e127.toy import toy_factory, TOY_ENGINE_SRC, TOY_WRONG_SRC, ACTION_API

def _acts(seq):
    return [(a, None, None) for a in seq]

def _observed(seed=0, n=20):
    g = toy_factory()
    rng = np.random.default_rng(seed)
    seq = [int(rng.choice([1, 2, 3, 4, 5, 7])) for _ in range(n)]
    return [engine.play(g, _acts(seq))]

def test_finds_counterexample_for_wrong_engine():
    obs = _observed()
    mask = engine.identity_mask(obs)
    wrong = compile_engine(TOY_WRONG_SRC)
    budget = {"limit": 500, "used": 0}
    cexs = probes.find_counterexamples(wrong, toy_factory, obs, mask, ACTION_API, budget)
    assert len(cexs) >= 1
    c = cexs[0]
    assert not np.array_equal(c["real_frame"], c["engine_frame"])
    assert budget["used"] > 0

def test_no_counterexample_for_faithful_engine():
    obs = _observed()
    mask = engine.identity_mask(obs)
    faithful = compile_engine(TOY_ENGINE_SRC)
    budget = {"limit": 500, "used": 0}
    cexs = probes.find_counterexamples(faithful, toy_factory, obs, mask, ACTION_API, budget)
    assert cexs == []

def test_property_violation_detects_nondeterminism_claim():
    # An engine whose step depends on a hidden RNG-like counter mismatch -> determinism still holds for
    # ToyGame; instead verify color_range catches an out-of-range predictor.
    bad = ("class Engine:\n"
           "    def __init__(self): self.state={'levels':0}\n"
           "    def reset(self): return np.zeros((8,8),dtype=int)\n"
           "    def step(self, a):\n        f=np.zeros((8,8),dtype=int); f[0,0]=999; return f\n"
           "    def is_win(self,p): return False\n")
    factory = compile_engine(bad)
    budget = {"limit": 200, "used": 0}
    viols = probes.property_violations(factory, toy_factory, ACTION_API, budget)
    assert any(v["kind"] == "color_range" for v in viols)

def test_budget_caps_real_steps():
    obs = _observed(n=40)
    mask = engine.identity_mask(obs)
    wrong = compile_engine(TOY_WRONG_SRC)
    budget = {"limit": 5, "used": 0}
    probes.find_counterexamples(wrong, toy_factory, obs, mask, ACTION_API, budget)
    assert budget["used"] <= 5 + 1     # never overshoots the limit by more than one batch boundary
```

- [ ] **Step 2: Run test to verify it fails**
Run: `~/.arcv/bin/python -m pytest tests/e127/test_probes.py -v`
Expected: FAIL — `ModuleNotFoundError: experiments.e127.probes`.

- [ ] **Step 3: Write minimal implementation**
```python
# experiments/e127/probes.py
"""Counterexample search against the REAL env (the oracle). Combines: (a) replaying observed
prefixes + novelty-guided random extensions and diffing engine vs real, and (b) universal property
falsifiers. Real-env steps are charged to a shared budget {'limit','used'}; a depth-d replay costs d
steps (no env cloning), so we replay each candidate once from a fresh real_factory()."""
import numpy as np
from experiments.e127 import engine as _engine

_SIMPLE = [1, 2, 3, 4, 5, 7]


def _real_frames(real_factory, actions, budget):
    """Replay `actions` on a fresh real env; return frames incl. reset, charging len(actions) steps.
    Stops early (returns the partial list) if the budget is exhausted."""
    g = real_factory()
    frames = [np.asarray(g.reset())]
    levels = [int(g.levels)]
    for a in actions:
        if budget["used"] >= budget["limit"]:
            break
        kind, x, y = a
        frames.append(np.asarray(g.step(kind, x, y)))
        levels.append(int(g.levels))
        budget["used"] += 1
    return frames, levels


def _candidate_action_seqs(observed, action_api, seed=0, n_extend=8, max_seqs=24):
    """Prefixes of observed episodes extended by short random tails (novelty-guided is approximated
    here by random simple actions; the loop's active-exploration supplies the real novelty)."""
    rng = np.random.default_rng(seed)
    seqs = []
    for ep in observed:
        base = [s["action"] for s in ep[1:]]
        for cut in (len(base), max(1, len(base) // 2)):
            tail = [(int(rng.choice(_SIMPLE)), None, None) for _ in range(n_extend)]
            seqs.append(base[:cut] + tail)
            if len(seqs) >= max_seqs:
                return seqs
    return seqs


def find_counterexamples(factory, real_factory, observed, mask, action_api, budget):
    """Return counterexamples where the engine's full frame != the real frame (earliest per seq)."""
    cexs = []
    for actions in _candidate_action_seqs(observed, action_api):
        if budget["used"] >= budget["limit"]:
            break
        real_frames, _ = _real_frames(real_factory, actions, budget)
        try:
            eng_frames = _engine.rollout(factory, actions[:len(real_frames) - 1])
        except _engine.EngineError:
            cexs.append({"actions": actions[:1], "index": 0, "real_frame": real_frames[0],
                         "engine_frame": np.full_like(real_frames[0], -1), "kind": "engine_error"})
            continue
        for i in range(1, min(len(real_frames), len(eng_frames))):
            if eng_frames[i].shape != real_frames[i].shape or not np.array_equal(eng_frames[i], real_frames[i]):
                cexs.append({"actions": actions[:i], "index": i, "real_frame": real_frames[i],
                             "engine_frame": eng_frames[i], "kind": "diff"})
                break
    return cexs


def property_violations(factory, real_factory, action_api, budget):
    """Falsify universal properties of the engine using the real env as reference."""
    viols = []
    probe = [(int(a), None, None) for a in (1, 4, 2, 3, 5, 7, 1, 4)]
    # color_range: engine cells must stay within the real env's color alphabet (0..15)
    try:
        eng_frames = _engine.rollout(factory, probe)
        for i, f in enumerate(eng_frames):
            if f.min() < 0 or f.max() > 15:
                viols.append({"kind": "color_range", "index": i, "actions": probe[:i],
                              "engine_frame": f, "real_frame": None})
                break
    except _engine.EngineError:
        viols.append({"kind": "engine_error", "index": 0, "actions": probe[:1],
                      "engine_frame": None, "real_frame": None})
    # determinism: the engine must be deterministic (same actions -> same frames)
    try:
        f1 = _engine.rollout(factory, probe); f2 = _engine.rollout(factory, probe)
        if not all(np.array_equal(a, b) for a, b in zip(f1, f2)):
            viols.append({"kind": "determinism", "index": 0, "actions": probe,
                          "engine_frame": None, "real_frame": None})
    except _engine.EngineError:
        pass
    # levelup_delta: where the REAL env levels-up, the board changes substantially
    real_frames, real_levels = _real_frames(real_factory, probe, budget)
    for i in range(1, len(real_frames)):
        if real_levels[i] > real_levels[i - 1]:
            changed = (real_frames[i] != real_frames[i - 1]).mean()
            if changed < 0.02:
                viols.append({"kind": "levelup_delta", "index": i, "actions": probe[:i],
                              "engine_frame": None, "real_frame": real_frames[i]})
    return viols
```

- [ ] **Step 4: Run test to verify it passes**
Run: `~/.arcv/bin/python -m pytest tests/e127/test_probes.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**
```bash
git add experiments/e127/probes.py tests/e127/test_probes.py
git commit -m "E127 T5: counterexample search + property falsifiers (real env oracle)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: Model-isolation runner (`iso.py`)

**Files:**
- Create: `experiments/e127/iso.py`
- Test: `tests/e127/test_iso.py`

**Interfaces:**
- Produces: `run(prompt, model, game="", workdir=None, timeout=600, _exec=None) -> {"engine_src":str|None, "rationale":str, "raw":str}`. `model` in {"claude","codex"}. Parses a strict-JSON reply `{"engine_src": "...", "rationale": "..."}` from the model's stdout (brace-balanced extraction tolerating prose / ``` fences). `_exec(cmd, cwd, timeout) -> str` is injectable for tests (defaults to a real subprocess call). Also `extract_json(text) -> dict|None` (exposed for testing).
- Structural isolation (real `_default_exec`): runs in a clean tempdir with no game source; for `claude`, denies all file/run tools and loads zero MCP servers; for `codex`, runs `exec` with no source dir mounted. (The audit in Task 7 verifies the workdir is clean.)

- [ ] **Step 1: Write the failing test**
```python
# tests/e127/test_iso.py
from experiments.e127 import iso

def test_extract_json_plain():
    d = iso.extract_json('{"engine_src": "class Engine: pass", "rationale": "x"}')
    assert d["engine_src"] == "class Engine: pass"

def test_extract_json_with_fence_and_prose():
    txt = 'Here is my answer:\n```json\n{"engine_src": "class Engine:\\n    pass", "rationale": "ok {nested}"}\n```\nDone.'
    d = iso.extract_json(txt)
    assert "class Engine" in d["engine_src"] and d["rationale"] == "ok {nested}"

def test_extract_json_none_on_garbage():
    assert iso.extract_json("no json here") is None

def test_run_uses_injected_exec_no_real_llm():
    canned = '{"engine_src": "class Engine:\\n    def reset(self): return None", "rationale": "r"}'
    calls = {}
    def fake_exec(cmd, cwd, timeout):
        calls["cmd"] = cmd; calls["cwd"] = cwd
        return canned
    out = iso.run("PROMPT", model="claude", game="toy", _exec=fake_exec)
    assert "class Engine" in out["engine_src"] and out["rationale"] == "r"
    # isolation: claude invocation denies tools and loads no MCP servers
    joined = " ".join(calls["cmd"])
    assert "--disallowedTools" in joined and "--strict-mcp-config" in joined

def test_run_malformed_reply_returns_none_src():
    out = iso.run("PROMPT", model="codex", _exec=lambda c, w, t: "the model rambled, no json")
    assert out["engine_src"] is None
```

- [ ] **Step 2: Run test to verify it fails**
Run: `~/.arcv/bin/python -m pytest tests/e127/test_iso.py -v`
Expected: FAIL — `ModuleNotFoundError: experiments.e127.iso`.

- [ ] **Step 3: Write minimal implementation**
```python
# experiments/e127/iso.py
"""Source-free model runner. Claude/codex PROPOSE engine source from observed play; they never read
game source. Isolation is structural: a clean tempdir with no source, all file/run tools denied, zero
MCP servers. `_exec` is injected in tests so no real LLM is called. The model only proposes; the
real-env certificate decides."""
import os, json, subprocess, tempfile

CLAUDE = os.path.expanduser("~/.local/bin/claude")
CODEX = os.path.expanduser("~/.local/bin/codex")
_DENY = "Bash,Read,Edit,Write,Glob,Grep,WebFetch,WebSearch,Task,NotebookEdit,MultiEdit,LS,NotebookRead"


def extract_json(text):
    """First brace-balanced JSON object containing 'engine_src', tolerating fences/prose."""
    if not text:
        return None
    i, n = 0, len(text)
    while i < n:
        start = text.find("{", i)
        if start == -1:
            return None
        depth = 0; instr = False; esc = False
        for j in range(start, n):
            ch = text[j]
            if instr:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    instr = False
            else:
                if ch == '"':
                    instr = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        cand = text[start:j + 1]
                        try:
                            obj = json.loads(cand)
                            if isinstance(obj, dict) and "engine_src" in obj:
                                return obj
                        except Exception:
                            pass
                        break
        i = start + 1
    return None


def _default_exec(cmd, cwd, timeout):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout).stdout


def _cmd(model, prompt):
    if model == "claude":
        return [CLAUDE, "-p", prompt, "--output-format", "text",
                "--disallowedTools", _DENY, "--permission-mode", "default", "--strict-mcp-config"]
    if model == "codex":
        return [CODEX, "exec", "--skip-git-repo-check", "-m", "gpt-5.5", prompt]
    raise ValueError(f"unknown model {model}")


def run(prompt, model, game="", workdir=None, timeout=600, _exec=None):
    _exec = _exec or _default_exec
    own_tmp = workdir is None
    wd = workdir or tempfile.mkdtemp(prefix=f"e127_{model}_{game}_")
    try:
        raw = _exec(_cmd(model, prompt), wd, timeout)
    except Exception as e:
        raw = ""
    obj = extract_json(raw) or {}
    return {"engine_src": obj.get("engine_src"), "rationale": obj.get("rationale", ""), "raw": raw}
```

- [ ] **Step 4: Run test to verify it passes**
Run: `~/.arcv/bin/python -m pytest tests/e127/test_iso.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**
```bash
git add experiments/e127/iso.py tests/e127/test_iso.py
git commit -m "E127 T6: source-free model-isolation runner (claude/codex, injectable exec)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 7: No-source-read audit (`audit.py`)

**Files:**
- Create: `experiments/e127/audit.py`
- Test: `tests/e127/test_audit.py`

**Interfaces:**
- Produces: `audit_dir(wd) -> list[str]` (findings; empty = clean) and `audit_clean(wd) -> bool`. Flags any `.py` under `wd` that references game source: `environment_files`, `inspect.getsource`, `spec_from_file_location`, `importlib.util.spec_from_file_location`, an `environment_files/` directory, or a literal `<6-char-id>.py` game-source load. The runner's own files and ToyGame are not in a model workdir, so this audits the *model* working dirs.

- [ ] **Step 1: Write the failing test**
```python
# tests/e127/test_audit.py
import os
from experiments.e127 import audit

def test_clean_dir(tmp_path):
    (tmp_path / "engine.py").write_text("class Engine:\n    def reset(self): return None\n")
    assert audit.audit_dir(str(tmp_path)) == []
    assert audit.audit_clean(str(tmp_path)) is True

def test_flags_environment_files(tmp_path):
    (tmp_path / "cheat.py").write_text("p = 'environment_files/dc22/dc22.py'\n")
    findings = audit.audit_dir(str(tmp_path))
    assert findings and audit.audit_clean(str(tmp_path)) is False

def test_flags_getsource(tmp_path):
    (tmp_path / "x.py").write_text("import inspect\ninspect.getsource(env._game)\n")
    assert audit.audit_clean(str(tmp_path)) is False

def test_flags_spec_from_file_location(tmp_path):
    (tmp_path / "y.py").write_text("import importlib.util\nimportlib.util.spec_from_file_location('g','g.py')\n")
    assert audit.audit_clean(str(tmp_path)) is False
```

- [ ] **Step 2: Run test to verify it fails**
Run: `~/.arcv/bin/python -m pytest tests/e127/test_audit.py -v`
Expected: FAIL — `ModuleNotFoundError: experiments.e127.audit`.

- [ ] **Step 3: Write minimal implementation**
```python
# experiments/e127/audit.py
"""Structural no-source-read audit of a model working dir. Source access = reading a game's <gid>.py
(the answer key). This makes 'source-simulated' auditable: a clean model workdir proves the engine
was authored from observed play, not from the real source."""
import os, glob, re

SOURCE_READ = re.compile(
    r"environment_files|inspect\.getsource|spec_from_file_location|"
    r"importlib\.util\.spec_from_file_location|[a-z0-9]{4,8}\.py['\"]")


def audit_dir(wd):
    findings = []
    if os.path.isdir(os.path.join(wd, "environment_files")):
        findings.append("environment_files/ dir present")
    for p in glob.glob(os.path.join(wd, "**", "*.py"), recursive=True):
        try:
            txt = open(p, errors="ignore").read()
        except Exception:
            continue
        m = SOURCE_READ.search(txt)
        if m:
            findings.append(f"{os.path.basename(p)}: source-read pattern {m.group(0)!r}")
    return findings


def audit_clean(wd):
    return audit_dir(wd) == []
```

- [ ] **Step 4: Run test to verify it passes**
Run: `~/.arcv/bin/python -m pytest tests/e127/test_audit.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**
```bash
git add experiments/e127/audit.py tests/e127/test_audit.py
git commit -m "E127 T7: structural no-source-read audit

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 8: Vendored source-free SandboxGame (`sandbox.py`)

**Files:**
- Create: `experiments/e127/sandbox.py`
- Test: `tests/e127/test_sandbox.py`

**Interfaces:**
- Produces: `SandboxGame(gid, venv=ARC_VENV)` implementing `GameLike`; `ARC_VENV`; `WORKER_ROOT`. Vendored from the proven `arc3_sandbox.py` pattern, but with `WORKER_ROOT` under `experiments/e127/.sandbox_env` (separate from any agent cwd) and a `--worker` entry point. The worker process holds arc_agi and the downloaded source; the client holds only a pipe and exposes `{frame, levels, win, avail, done}`.
- Consumes: nothing in-package (subprocess to arc venv).

- [ ] **Step 1: Write the failing test**
```python
# tests/e127/test_sandbox.py
import importlib.util, os
import pytest
from experiments.e127 import sandbox

def _arc_available():
    return importlib.util.find_spec("arc_agi") is not None or os.path.exists(sandbox.ARC_VENV)

def test_module_surface():
    assert hasattr(sandbox, "SandboxGame") and hasattr(sandbox, "ARC_VENV")
    # GameLike methods exist
    for m in ("reset", "step", "close"):
        assert callable(getattr(sandbox.SandboxGame, m))

@pytest.mark.skipif(not _arc_available(), reason="arc venv / arc_agi not available")
def test_smoke_real_game_steps():
    # Integration smoke: a real game resets and steps, exposing only the sandbox surface.
    g = sandbox.SandboxGame("ar25")
    f = g.reset()
    assert f.shape == (64, 64)
    assert isinstance(g.levels, int) and isinstance(g.avail, list)
    a = g.avail[0] if g.avail else 7
    g.step(a if a != 6 else 6, 0, 0)
    g.close()
```

- [ ] **Step 2: Run test to verify it fails**
Run: `~/.arcv/bin/python -m pytest tests/e127/test_sandbox.py -v`
Expected: FAIL — `ModuleNotFoundError: experiments.e127.sandbox`.

- [ ] **Step 3: Write minimal implementation**
Vendor the proven worker/client. Use this exact content:
```python
# experiments/e127/sandbox.py
"""Source-free ARC-AGI-3 sandbox for E127 (vendored, self-contained). The real arc_agi game runs in
an isolated worker process whose cwd holds the downloaded source; the agent imports only SandboxGame
-- a pipe client exposing ONLY {frame, levels, win, avail, done}. The agent process never holds the
game object and its cwd has no source => discovery is by acting only (source-free by construction)."""
import sys, os, json, subprocess

ARC_VENV = os.environ.get("ARC_VENV", os.path.expanduser("~/.arcv/bin/python"))
WORKER_ROOT = os.path.join(os.path.dirname(__file__), ".sandbox_env")


def _worker(gid):
    import logging
    logging.disable(logging.CRITICAL)
    proto_fd = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    proto = os.fdopen(proto_fd, "w", buffering=1)

    def send(d):
        proto.write(json.dumps(d) + "\n"); proto.flush()

    os.makedirs(os.path.join(WORKER_ROOT, gid), exist_ok=True)
    os.chdir(os.path.join(WORKER_ROOT, gid))
    import numpy as np, arc_agi
    from arcengine import GameAction
    A = {1: GameAction.ACTION1, 2: GameAction.ACTION2, 3: GameAction.ACTION3,
         4: GameAction.ACTION4, 5: GameAction.ACTION5, 7: GameAction.ACTION7}
    arc = arc_agi.Arcade(); env = arc.make(gid)
    last = {"levels": 0, "win": 0}

    def obs(o):
        if o is None or getattr(o, "frame", None) is None:
            return {"frame": None, "levels": last["levels"], "win": last["win"], "avail": [], "done": True}
        f = np.asarray(o.frame); f = (f[-1] if f.ndim == 3 else f).reshape(64, 64)
        last["levels"] = int(o.levels_completed); last["win"] = int(o.win_levels)
        return {"frame": f.astype(int).tolist(), "levels": last["levels"], "win": last["win"],
                "avail": list(o.available_actions), "done": str(o.state) != "GameState.NOT_FINISHED"}

    env.reset(); send({"ready": True})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        try:
            c = req.get("cmd")
            if c == "reset":
                o = env.reset()
            elif c == "step":
                a = req["a"]
                o = (env.step(GameAction.ACTION6, {"x": int(req["x"]), "y": int(req["y"])})
                     if a == 6 else env.step(A[a]))
            elif c == "close":
                break
            else:
                raise ValueError("bad cmd")
            r = obs(o)
        except Exception as e:
            r = {"error": str(e)[:200]}
        send(r)


class SandboxGame:
    def __init__(self, gid, venv=ARC_VENV):
        import numpy as np
        self._np = np; self.gid = gid
        self.p = subprocess.Popen([venv, os.path.abspath(__file__), "--worker", gid],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True, bufsize=1)
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("worker died before ready")
            if json.loads(line).get("ready"):
                break
        self.reset()

    def _rpc(self, req):
        self.p.stdin.write(json.dumps(req) + "\n"); self.p.stdin.flush()
        line = self.p.stdout.readline()
        if not line:
            raise RuntimeError("worker died")
        r = json.loads(line)
        if "error" in r:
            raise RuntimeError(r["error"])
        if r["frame"] is not None:
            self.frame = self._np.array(r["frame"])
        self.levels = r["levels"]; self.win = r["win"]; self.avail = r["avail"]; self.done = r["done"]
        return self.frame

    def reset(self):
        return self._rpc({"cmd": "reset"})

    def step(self, a, x=None, y=None):
        return self._rpc({"cmd": "step", "a": a, "x": x, "y": y})

    def close(self):
        try:
            self.p.stdin.write(json.dumps({"cmd": "close"}) + "\n"); self.p.stdin.flush()
        except Exception:
            pass
        self.p.terminate()


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--worker":
        _worker(sys.argv[2])
    else:
        print("run as: python sandbox.py --worker <gid>")
```

- [ ] **Step 4: Run test to verify it passes**
Run: `~/.arcv/bin/python -m pytest tests/e127/test_sandbox.py -v`
Expected: PASS for `test_module_surface`; `test_smoke_real_game_steps` PASSES if the arc venv + network are available, else SKIPS. Either outcome is acceptable for this task (the smoke is integration, gated).

- [ ] **Step 5: Commit**
```bash
git add experiments/e127/sandbox.py tests/e127/test_sandbox.py
git commit -m "E127 T8: vendored source-free SandboxGame (self-contained)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 9: Differential-CEGIS reconstruction loop (`reconstruct.py`)

The keystone. Ties T1–T7 together: observe → independent round-0 synth (two runners) → CEGIS rounds (counterexamples from the real env + property falsifiers, keep-best monotone gate on held-out real accuracy, champion = best-vs-real) → terminate on certificate or budget → emit certificate + the **A-vs-B-vs-real gap**. Fully offline-testable with `ToyGame` + injected fake runners.

**Files:**
- Create: `experiments/e127/reconstruct.py`
- Test: `tests/e127/test_reconstruct.py`

**Interfaces:**
- Consumes: `engine`, `certify`, `probes`, `iso` (via injected `_runners` in tests), `engine.play`, a `real_factory`.
- Produces: `reconstruct(real_factory, action_api, n_levels, models=("claude","codex"), max_rounds=4, budget=None, _runners=None, seed=0) -> result` where `result = {"engine_src":str|None, "certificate":dict, "champion_acc":float, "ab_agreement":float, "ab_vs_real_gap":float, "rounds":int, "real_steps":int, "history":list[dict]}`.
  - `_runners`: a list of two callables `runner(prompt:str, round_idx:int) -> engine_src|None`, one per model. In production these wrap `iso.run`; in tests they return canned sources. Each runner receives the accumulated counterexamples in the prompt text (the loop builds the prompt; tests can ignore the text and key off `round_idx`).
  - `ab_agreement`: fraction of probe rollouts where the two final engines agree. `ab_vs_real_gap`: `ab_agreement - min(accA_vs_real, accB_vs_real)` measured on the held-out set — the headline "shared-prior bias" number (positive when the two models agree more with each other than either does with reality).
  - Keep-best monotone gate: a revised engine replaces a model's current engine only if its held-out real accuracy is `>=` the current (never regress).

- [ ] **Step 1: Write the failing test**
```python
# tests/e127/test_reconstruct.py
import numpy as np
from experiments.e127 import reconstruct
from tests.e127.toy import toy_factory, TOY_ENGINE_SRC, TOY_WRONG_SRC, ACTION_API

def test_converges_to_certified_engine_with_fake_runners():
    # Model A: wrong at round 0, then authors the faithful engine once it sees counterexamples.
    def runner_a(prompt, round_idx):
        return TOY_WRONG_SRC if round_idx == 0 else TOY_ENGINE_SRC
    # Model B: persistently authors the wrong engine (a diversity source that stays wrong).
    def runner_b(prompt, round_idx):
        return TOY_WRONG_SRC
    res = reconstruct.reconstruct(toy_factory, ACTION_API, n_levels=1,
                                  max_rounds=4, _runners=[runner_a, runner_b], seed=0)
    assert res["certificate"]["pass"] is True            # champion certified against the REAL env
    assert res["champion_acc"] >= 0.99
    assert res["engine_src"] is not None
    # The champion is the faithful engine; B never reaches it -> a measurable gap exists.
    assert res["ab_vs_real_gap"] != 0.0
    assert res["real_steps"] > 0

def test_no_false_unity_when_both_models_share_a_wrong_engine():
    # Both models agree on the SAME wrong engine (folie a deux). Agreement is high, but the
    # certificate must FAIL because neither matches the real env.
    def wrong_runner(prompt, round_idx):
        return TOY_WRONG_SRC
    res = reconstruct.reconstruct(toy_factory, ACTION_API, n_levels=1,
                                  max_rounds=3, _runners=[wrong_runner, wrong_runner], seed=0)
    assert res["certificate"]["pass"] is False           # agreement != correctness
    assert res["ab_agreement"] >= 0.99                    # they DO agree with each other
    assert res["ab_vs_real_gap"] > 0.0                    # ... but not with reality -> positive gap

def test_budget_is_respected():
    def runner(prompt, round_idx):
        return TOY_WRONG_SRC
    res = reconstruct.reconstruct(toy_factory, ACTION_API, n_levels=1, max_rounds=3,
                                  budget={"limit": 50, "used": 0}, _runners=[runner, runner], seed=0)
    assert res["real_steps"] <= 50 + 8                    # small batch overshoot tolerance
```

- [ ] **Step 2: Run test to verify it fails**
Run: `~/.arcv/bin/python -m pytest tests/e127/test_reconstruct.py -v`
Expected: FAIL — `ModuleNotFoundError: experiments.e127.reconstruct`.

- [ ] **Step 3: Write minimal implementation**
```python
# experiments/e127/reconstruct.py
"""Differential-CEGIS reconstruction. The REAL env is the convergence oracle; the second model is a
diversity source for counterexamples, never the acceptance gate. Emits an equivalence-to-real
certificate plus the A-vs-B-vs-real gap (how much shared-prior agreement exceeds agreement with
reality)."""
import numpy as np
from experiments.e127 import engine as _engine
from experiments.e127 import certify as _certify
from experiments.e127 import probes as _probes
from experiments.e127.safe_exec import compile_engine

_SIMPLE = [1, 2, 3, 4, 5, 7]


def _explore(real_factory, n_eps, len_eps, seed):
    """Collect Episodes by random simple-action play (reversible-ish). Two disjoint pools by seed."""
    rng = np.random.default_rng(seed)
    eps = []
    for _ in range(n_eps):
        g = real_factory()
        acts = [(int(rng.choice(_SIMPLE)), None, None) for _ in range(len_eps)]
        eps.append(_engine.play(g, acts))
    return eps


def _acc_vs_real(factory, holdout):
    if factory is None:
        return 0.0
    n = exact = 0
    for ep in holdout:
        sc = _engine.score_rollout(factory, ep)
        n += sc["transitions"]; exact += sc["exact"]
    return (exact / n) if n else 0.0


def _ab_agreement(fa, fb, real_factory, holdout):
    """Fraction of held-out transitions where the two engines produce identical frames."""
    if fa is None or fb is None:
        return 0.0
    agree = tot = 0
    for ep in holdout:
        actions = [s["action"] for s in ep[1:]]
        try:
            ra = _engine.rollout(fa, actions); rb = _engine.rollout(fb, actions)
        except _engine.EngineError:
            continue
        for i in range(1, min(len(ra), len(rb))):
            tot += 1
            if ra[i].shape == rb[i].shape and np.array_equal(ra[i], rb[i]):
                agree += 1
    return (agree / tot) if tot else 0.0


def _prompt(action_api, observed, cexs, round_idx):
    """Build the model prompt (text only; tests ignore it). Includes the action API, a few observed
    transitions, and the real-labeled counterexamples from prior rounds."""
    lines = [f"Reconstruct the game engine as a Python class `Engine` (reset/step/is_win/state).",
             f"Action API: {action_api}", f"Round {round_idx}."]
    for c in cexs[:12]:
        lines.append(f"COUNTEREXAMPLE after actions {[a[0] for a in c['actions']]}: your frame was wrong; "
                     f"the REAL next frame differs (kind={c['kind']}).")
    lines.append('Reply strict JSON: {"engine_src": "...", "rationale": "..."}')
    return "\n".join(lines)


def reconstruct(real_factory, action_api, n_levels, models=("claude", "codex"),
                max_rounds=4, budget=None, _runners=None, seed=0):
    budget = budget if budget is not None else {"limit": 4000, "used": 0}
    observed = _explore(real_factory, n_eps=8, len_eps=16, seed=seed)
    holdout = _explore(real_factory, n_eps=24, len_eps=16, seed=seed + 1000)   # DISJOINT pool
    mask = _engine.identity_mask(observed)

    if _runners is None:
        from experiments.e127 import iso
        _runners = [(lambda prompt, r, m=m: iso.run(prompt, model=m).get("engine_src")) for m in models]

    cur = [None, None]          # current engine factory per model
    cur_src = [None, None]
    cur_acc = [0.0, 0.0]
    cexs = []
    history = []

    for r in range(max_rounds):
        for mi, runner in enumerate(_runners):
            src = runner(_prompt(action_api, observed, cexs, r), r)
            if not src or _engine.looks_like_lookup_table(src):
                continue
            fac = compile_engine(src)
            if fac is None:
                continue
            acc = _acc_vs_real(fac, holdout)
            if acc >= cur_acc[mi]:            # keep-best monotone gate
                cur[mi], cur_src[mi], cur_acc[mi] = fac, src, acc
        # champion = best vs REAL
        champ_i = 0 if cur_acc[0] >= cur_acc[1] else 1
        champ = cur[champ_i]
        # gather counterexamples against the champion for the next round
        new_cexs = []
        if champ is not None:
            new_cexs = _probes.find_counterexamples(champ, real_factory, observed, mask, action_api, budget)
            new_cexs += _probes.property_violations(champ, real_factory, action_api, budget)
        cexs = (cexs + new_cexs)[-32:]
        cert = _certify.certify_engine(champ, holdout, n_levels) if champ else {"pass": False, "acc": 0.0}
        history.append({"round": r, "champion": champ_i, "champion_acc": cur_acc[champ_i],
                        "n_cex": len(new_cexs), "cert_pass": cert.get("pass", False),
                        "real_steps": budget["used"]})
        if cert.get("pass"):
            break

    champ_i = 0 if cur_acc[0] >= cur_acc[1] else 1
    champ = cur[champ_i]
    cert = _certify.certify_engine(champ, holdout, n_levels) if champ else {
        "pass": False, "acc": 0.0, "acc_lower": 0.0, "n": 0, "exact": 0, "coverage": 0.0}
    ab_agree = _ab_agreement(cur[0], cur[1], real_factory, holdout)
    min_vs_real = min(cur_acc)
    return {"engine_src": cur_src[champ_i], "certificate": cert, "champion_acc": cur_acc[champ_i],
            "ab_agreement": ab_agree, "ab_vs_real_gap": ab_agree - min_vs_real,
            "rounds": len(history), "real_steps": budget["used"], "history": history}
```

- [ ] **Step 4: Run test to verify it passes**
Run: `~/.arcv/bin/python -m pytest tests/e127/test_reconstruct.py -v`
Expected: PASS (3 tests). If `test_converges...` fails because the faithful engine isn't selected, check the keep-best gate and that `_acc_vs_real` is computed on the disjoint `holdout`. If `ab_vs_real_gap` is 0 when it should be positive, confirm `_ab_agreement` rolls out both engines on the same actions.

- [ ] **Step 5: Run the whole E127 suite + commit**
Run: `~/.arcv/bin/python -m pytest tests/e127/ -v`
Expected: all tests PASS (sandbox smoke may SKIP).
```bash
git add experiments/e127/reconstruct.py tests/e127/test_reconstruct.py
git commit -m "E127 T9: differential-CEGIS reconstruction loop (real-env oracle + A-vs-B-vs-real gap)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3P: Perception primitives — all modalities (`perception.py`)

Execution order: after Task 3 (engine), before Task 5 (probes). Pure functions; cheap to test with hand-crafted frames + the directional ToyGame.

**Files:**
- Create: `experiments/e127/perception.py`
- Test: `tests/e127/test_perception.py`

**Interfaces:**
- Consumes: `numpy`.
- Produces:
  - `infer_click_targets(frame, max_size=40, rare_frac=0.02) -> list[(y,x)]` — pixel-only target inference: cells in small 4-connected non-background components, plus rare-color cells. Background = the most common color.
  - `board_match_error(pred, real) -> {"cells_total":int,"cells_wrong":int,"exact":bool,"error_map":np.ndarray(bool)}` — the simulated-vs-real board error (the perception unit test against reality). Shape mismatch → all wrong.
  - `render_diff(pred, real) -> str` — a compact text map: a header line + an `X`/`.` grid marking disagreeing cells (the viewable board diff).
  - `candidate_actions(frame, avail) -> list[(kind,x,y)]` — from `available_actions`: directional kinds become `(k,None,None)`; `6` becomes a click `(6, x, y)` (x=col, y=row) at each inferred target.

- [ ] **Step 1: Write the failing test**
```python
# tests/e127/test_perception.py
import numpy as np
from experiments.e127 import perception as P

def test_infer_click_targets_small_sprites_only():
    f = np.zeros((8, 8), dtype=int)
    f[2, 2] = 5                     # small sprite (size 1)
    f[5, 5] = 6                     # small sprite (size 1)
    f[0:1, 0:8] = 0                 # background row stays bg
    f[6:8, 0:8] = 3                 # a LARGE block (16 cells) -> not a click target
    t = P.infer_click_targets(f, max_size=4)
    assert (2, 2) in t and (5, 5) in t
    assert (6, 0) not in t          # big block excluded

def test_board_match_error_counts_and_exact():
    a = np.zeros((4, 4), dtype=int); b = np.zeros((4, 4), dtype=int)
    assert P.board_match_error(a, b)["exact"] is True
    b[1, 1] = 9
    e = P.board_match_error(a, b)
    assert e["exact"] is False and e["cells_wrong"] == 1 and e["cells_total"] == 16
    assert e["error_map"][1, 1] == True and e["error_map"].sum() == 1

def test_board_match_error_shape_mismatch_all_wrong():
    e = P.board_match_error(np.zeros((2, 2), dtype=int), np.zeros((4, 4), dtype=int))
    assert e["exact"] is False and e["cells_wrong"] == e["cells_total"]

def test_render_diff_marks_differences():
    a = np.zeros((3, 3), dtype=int); b = np.zeros((3, 3), dtype=int); b[0, 0] = 1
    s = P.render_diff(a, b)
    assert "X" in s and "board-match" in s

def test_candidate_actions_click_game():
    f = np.zeros((8, 8), dtype=int); f[2, 2] = 5
    acts = P.candidate_actions(f, avail=[6])
    assert (6, 2, 2) in acts        # click at x=col=2, y=row=2
    assert all(a[0] == 6 for a in acts)

def test_candidate_actions_directional_game():
    f = np.zeros((8, 8), dtype=int)
    acts = P.candidate_actions(f, avail=[1, 2, 3, 4, 7])
    assert (1, None, None) in acts and (7, None, None) in acts
    assert all(a[0] != 6 for a in acts)
```

- [ ] **Step 2: Run test to verify it fails**
Run: `~/.arcv/bin/python -m pytest tests/e127/test_perception.py -v`
Expected: FAIL — `ModuleNotFoundError: experiments.e127.perception`.

- [ ] **Step 3: Write minimal implementation**
```python
# experiments/e127/perception.py
"""All-modality perception for reconstruction. Frame perception = the (H,W) grid. Interaction =
directional (1-5,7) + click/mouse (ACTION6 at x,y). Click targets are inferred ONLY from pixels
(small 4-connected components + rare-color cells) -- honest, source-free. board_match_error /
render_diff measure the simulated board against the perceived real board (perception vs reality)."""
import numpy as np


def _background(frame):
    vals, cnts = np.unique(frame, return_counts=True)
    return int(vals[int(np.argmax(cnts))])


def _components(mask):
    H, W = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for y in range(H):
        for x in range(W):
            if mask[y, x] and not seen[y, x]:
                stack = [(y, x)]; seen[y, x] = True; comp = []
                while stack:
                    cy, cx = stack.pop(); comp.append((cy, cx))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True; stack.append((ny, nx))
                comps.append(comp)
    return comps


def infer_click_targets(frame, max_size=40, rare_frac=0.02):
    frame = np.asarray(frame); bg = _background(frame)
    targets = set()
    for comp in _components(frame != bg):
        if len(comp) <= max_size:
            targets.update(comp)
    vals, cnts = np.unique(frame, return_counts=True); tot = frame.size
    rare = {int(v) for v, c in zip(vals, cnts) if int(v) != bg and c / tot <= rare_frac}
    if rare:
        for y in range(frame.shape[0]):
            for x in range(frame.shape[1]):
                if int(frame[y, x]) in rare:
                    targets.add((y, x))
    return sorted(targets)


def board_match_error(pred, real):
    pred = np.asarray(pred); real = np.asarray(real)
    if pred.shape != real.shape:
        return {"cells_total": int(real.size), "cells_wrong": int(real.size),
                "exact": False, "error_map": np.ones(real.shape, dtype=bool)}
    diff = pred != real
    return {"cells_total": int(real.size), "cells_wrong": int(diff.sum()),
            "exact": bool(not diff.any()), "error_map": diff}


def render_diff(pred, real):
    e = board_match_error(pred, real); em = e["error_map"]
    lines = [f"board-match: {e['cells_total'] - e['cells_wrong']}/{e['cells_total']} cells, exact={e['exact']}"]
    for y in range(em.shape[0]):
        lines.append("".join("X" if em[y, x] else "." for x in range(em.shape[1])))
    return "\n".join(lines)


def candidate_actions(frame, avail):
    acts = []
    for k in avail:
        if int(k) == 6:
            for (y, x) in infer_click_targets(frame):
                acts.append((6, int(x), int(y)))
        else:
            acts.append((int(k), None, None))
    return acts
```

- [ ] **Step 4: Run test to verify it passes**
Run: `~/.arcv/bin/python -m pytest tests/e127/test_perception.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**
```bash
git add experiments/e127/perception.py tests/e127/test_perception.py
git commit -m "E127 T3P: all-modality perception (click-target inference, board-match diff, candidate actions)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3C: Click/mouse ToyClickGame fixture (`tests/e127/toy_click.py`)

A click-ONLY (mouse) ground-truth game with an **ordered-protocol** win (the goal-as-procedure case: press buttons in a fixed sequence). `avail=[6]`. Hidden state = a `phase` index. Used so the reconstruct loop (Task 9) is validated on a mouse game, not just directional.

ToyClickGame rules (8×8, colors 0–15), fixed and exact:
- Background 0. Three button sprites (size 1): button A color 5 at (2,2), button B color 6 at (4,5), button C color 7 at (6,1). Status bar cell (0,0) = `(t % 15) + 1`, `t=0` at reset.
- `avail = [6]` (click only). A click is `step(6, x, y)` with x=col, y=row.
- Protocol: the buttons must be PRESSED in order A→B→C. Clicking the *correct next* button (matching the hidden `phase`: 0→A,1→B,2→C) advances `phase += 1` and recolors that button to 3 (pressed). Clicking anything else (wrong button, already-pressed, or empty cell) is a NO-OP (board unchanged except the status bar still ticks on every accepted step — but an invalid click does NOT tick `t`; only a *valid target click* advances `t`. Rationale: matches "non-target clicks are no-ops that dedup away"). When `phase == 3`, `levels += 1`, `done = True` (`win = 1`).
- Determinism: identical click sequences from reset yield identical frames.

**Files:**
- Create: `tests/e127/toy_click.py`
- Test: `tests/e127/test_toy_click.py`

**Interfaces:**
- Produces: `ToyClickGame()` (GameLike), `toy_click_factory()`, `TOY_CLICK_ENGINE_SRC` (faithful Engine str), `CLICK_ACTION_API` (str), and target coords `A=(2,2),B=(4,5),C=(6,1)`.

- [ ] **Step 1: Write the failing test**
```python
# tests/e127/test_toy_click.py
import numpy as np
from tests.e127.toy_click import ToyClickGame, toy_click_factory, TOY_CLICK_ENGINE_SRC
from experiments.e127.safe_exec import compile_engine
from experiments.e127 import engine, perception as P

def test_reset_layout_click_only():
    g = ToyClickGame(); f = g.reset()
    assert f.shape == (8, 8)
    assert f[2, 2] == 5 and f[4, 5] == 6 and f[6, 1] == 7
    assert f[0, 0] == 1 and g.avail == [6] and g.levels == 0 and g.done is False

def test_invalid_click_is_noop():
    g = ToyClickGame(); g.reset()
    before = g.frame.copy()
    g.step(6, 0, 0)                      # empty cell -> no-op (status bar unchanged too)
    assert np.array_equal(g.frame, before)

def test_wrong_order_click_is_noop():
    g = ToyClickGame(); g.reset()
    before = g.frame.copy()
    g.step(6, 5, 4)                      # clicking B (x=5,y=4) before A -> no-op
    assert np.array_equal(g.frame, before)

def test_ordered_protocol_wins():
    g = ToyClickGame(); g.reset()
    g.step(6, 2, 2)                      # press A (x=col=2,y=row=2)
    assert g.frame[2, 2] == 3 and g.levels == 0
    g.step(6, 5, 4)                      # press B
    assert g.frame[4, 5] == 3 and g.levels == 0
    g.step(6, 1, 6)                      # press C -> level up
    assert g.levels == 1 and g.done is True

def test_targets_inferred_from_pixels():
    g = ToyClickGame(); f = g.reset()
    targets = P.infer_click_targets(f)
    assert (2, 2) in targets and (4, 5) in targets and (6, 1) in targets   # the 3 buttons (+status cell)

def test_faithful_click_engine_matches():
    factory = compile_engine(TOY_CLICK_ENGINE_SRC); assert factory is not None
    e = factory(); g = ToyClickGame()
    ef = e.reset(); gf = g.reset()
    assert np.array_equal(ef, gf)
    for (x, y) in [(0, 0), (2, 2), (3, 3), (5, 4), (5, 4), (1, 6)]:   # mix of valid/invalid clicks
        ef = e.step((6, x, y)); gf = g.step(6, x, y)
        assert np.array_equal(ef, gf), f"mismatch after click ({x},{y})"
```

- [ ] **Step 2: Run test to verify it fails**
Run: `~/.arcv/bin/python -m pytest tests/e127/test_toy_click.py -v`
Expected: FAIL — `ModuleNotFoundError: tests.e127.toy_click`.

- [ ] **Step 3: Write minimal implementation**
```python
# tests/e127/toy_click.py
"""Click-only (mouse) ordered-protocol ground-truth game for E127 (GameLike). Win = press buttons
A->B->C in order. Hidden state = phase. Invalid/out-of-order clicks are no-ops."""
import numpy as np

A, B, C = (2, 2), (4, 5), (6, 1)            # (row, col)
_BTN = [(A, 5), (B, 6), (C, 7)]             # in required press order
CLICK_ACTION_API = ("Click-only game (avail=[6]). A click is step(6, x, y) with x=col, y=row, 0-63. "
                    "Small colored buttons must be pressed in a fixed order; pressing the correct next "
                    "button recolors it; wrong/empty clicks are no-ops. Row 0 is a status bar. 8x8, colors 0-15.")


def _draw(pressed, t):
    f = np.zeros((8, 8), dtype=int)
    for i, ((ry, cx), col) in enumerate(_BTN):
        f[ry, cx] = 3 if i < pressed else col
    f[0, 0] = (t % 15) + 1
    return f


class ToyClickGame:
    def __init__(self):
        self.win = 1
        self._reset_fields()

    def _reset_fields(self):
        self.phase = 0; self.t = 0; self.levels = 0; self.done = False; self.avail = [6]

    def reset(self):
        self._reset_fields()
        self.frame = _draw(self.phase, self.t)
        return self.frame

    def step(self, a, x=None, y=None):
        if not self.done and a == 6 and self.phase < 3:
            (ry, cx), _col = _BTN[self.phase]
            if x == cx and y == ry:                 # correct next button
                self.phase += 1; self.t += 1
                if self.phase == 3:
                    self.levels += 1; self.done = self.levels >= self.win
        self.frame = _draw(self.phase, self.t)
        return self.frame


def toy_click_factory():
    return ToyClickGame()


TOY_CLICK_ENGINE_SRC = '''
_BTN = [((2, 2), 5), ((4, 5), 6), ((6, 1), 7)]
class Engine:
    def __init__(self):
        self.state = {"levels": 0, "phase": 0, "t": 0, "done": False}
    def _draw(self):
        f = np.zeros((8, 8), dtype=int)
        for i, ((ry, cx), col) in enumerate(_BTN):
            f[ry, cx] = 3 if i < self.state["phase"] else col
        f[0, 0] = (self.state["t"] % 15) + 1
        return f
    def reset(self):
        self.state = {"levels": 0, "phase": 0, "t": 0, "done": False}
        return self._draw()
    def step(self, action):
        k, x, y = action; s = self.state
        if not s["done"] and k == 6 and s["phase"] < 3:
            (ry, cx), _col = _BTN[s["phase"]]
            if x == cx and y == ry:
                s["phase"] += 1; s["t"] += 1
                if s["phase"] == 3:
                    s["levels"] += 1; s["done"] = s["levels"] >= 1
        return self._draw()
    def is_win(self, prev_frame):
        return self.state["levels"] >= 1
'''
```

- [ ] **Step 4: Run test to verify it passes**
Run: `~/.arcv/bin/python -m pytest tests/e127/test_toy_click.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**
```bash
git add tests/e127/toy_click.py tests/e127/test_toy_click.py
git commit -m "E127 T3C: click/mouse ToyClickGame ordered-protocol fixture

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

> **Note for Task 5 (probes) and Task 9 (reconstruct):** generate candidate action sequences via `perception.candidate_actions(frame, game.avail)` so both modalities are covered, and run the reconstruct offline test on BOTH `toy_factory` (directional) and `toy_click_factory` (click). The click game's faithful engine is `TOY_CLICK_ENGINE_SRC`; a wrong engine for the gap/agreement tests can ignore the ordered protocol.

---

## Milestone 1 done — what's next (NOT in this plan)

Milestone 2 (separate plan, after this core is reviewed and proven on `ar25` then `dc22`):
- `experiments/e127/world127.py` — certified Engine → OpenWorld `World` (stateful `CodeTransition` over rollout, `CodeObjective` from `is_win`), `round_trip_ok`.
- `experiments/e127/solve.py` — receding-horizon plan-then-verify-incrementally under ensemble pessimism (only expand nodes where both surviving engines agree; halt-and-replan on sim-vs-real mismatch); replay-verify; log engine-vs-real divergence along the winning path.
- `experiments/e127_source_simulated.py` — harness: per game observe → reconstruct → certify → solve → bank to `experiments/results/arc3_source_simulated.json`; determinism probe first; audit-gated; real-steps split by {explore, adjudicate, verify}; solves-per-real-step.
- Paper: third protocol column + macros/figure in `scripts/make_arc3_assets.py`; ablations (single-model vs dual-model; stateless negative control); `\NumExperiments` bump.
- A live run on `ar25` (mechanism control) then `dc22` (gap demonstration), with the real `iso.run` runners.

## Plan self-review

- **Spec coverage:** §3 fixes 1 (real-env target), 2 (stateful engine), 6 (mask identity-only), 7 (keep-best monotone + degeneracy + generalization gap via disjoint holdout) → Tasks 3,4,9. Fix 8 (audit/honesty) → Task 7 + real-steps budget in Tasks 5,9. §5 Phase 0/1/2 → Tasks 2,5,6,9. §7 data structures → Shared interfaces + Tasks 1,3. §8 results JSON, §5 Phase 3 solve, §11 paper → deferred to Milestone 2 (stated). Fixes 4 (receding-horizon planning), 5 (disagreement-driven exploration as the PRIMARY explorer), 9 (determinism probe), 10 (rename) → Milestone 2 (the loop here uses random exploration as a baseline; the disagreement-driven explorer lands with solve). This is an intentional, stated scope cut.
- **Placeholder scan:** every code step has complete code; the one soft spot is `betai`'s pivot/complement branch — flagged explicitly with a concrete fix and a pinning test, not left as "TODO".
- **Type consistency:** `factory` (zero-arg → Engine) is consistent across safe_exec/engine/certify/probes/reconstruct; Episode shape `{"action","frame","levels"}` consistent in engine.play/score_rollout/certify/probes; counterexample dict shape consistent in probes/reconstruct; `budget={"limit","used"}` consistent in probes/reconstruct.
