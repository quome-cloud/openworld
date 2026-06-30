# E134 — Composite Multi-Perception World for the EWM Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Replace the single object-state perceptor with an OpenWorld **composite multi-perception world** (several `CodePerceptor` lenses fused, combined by `ConsensusTransition` SELECT), and feed its non-aliasing multi-modal state to the EWM reasoning agent (E133). Targets the measured wall — perception aliasing at deep levels — and showcases composite > single perception (the arc-3 "winning method": *the leverage is in perception and synthesis*).

**Architecture:** `perceptors.py` holds K self-contained perception lenses (object, salience, **timer/status as a value not a mask**, symmetry, color-algebra, region). `composite.py` fuses them into a combined state key (so whichever modality carries the win-relevant feature, it's captured) and implements **fidelity-SELECT** (pick the modality whose discovered `(key,action)→key'` table is most Markov/consistent on observed transitions — the verification-based selection the papers require; *averaging hurts*, E95). It drops into the E133 `ewm_toolkit` so the agent perceives via the composite. Honest scope: this is the **perception** half — full-game cracking still needs the agent's reasoning; the composite removes the aliasing that limits it and adds coverage.

**Tech Stack:** Python 3.9 (`~/.arcv/bin/python`), stdlib + numpy. Reuses `experiments/e125/objstate.py`, `experiments/e133/ewm_toolkit.py`. Self-contained (drops into an audited source-free workspace — reads no game code).

## Global Constraints

- **Source-free**; perceptors read only the frame, never game code.
- **SELECT, not average** (E95: per-cell majority hurt ka59 0.13 vs 0.27 best-single). The combiner picks the highest-fidelity modality; `vote` only where members are env-verified + id-aligned.
- New code under `experiments/e134/` + `tests/e134/`; numpy allowed; no `openworld/` core changes.
- Run tests: `~/.arcv/bin/python -m pytest tests/e134/ -v`. Commit messages end with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

### Task 1: The K perception lenses

**Files:** Create `experiments/e134/__init__.py`, `experiments/e134/perceptors.py`; Test `tests/e134/__init__.py`, `tests/e134/test_perceptors.py`.

**Interfaces:** each lens is `perceive(frame) -> dict` returning a small JSON-able state; `LENSES = {name: perceive_fn}`:
- `objects` — connected components `[{color,size,y,x}]` (reuse e125.objstate).
- `salience` — small/rare-color components first (the click-target ranking, E107).
- `meter` — cells that change on ~every step (status bar / **timer/counter**) reported as their VALUES (a counter channel), NOT masked away — so a timer-driven win is visible.
- `symmetry` — horizontal/vertical/diagonal symmetry booleans + count of symmetry-breaking cells.
- `palette` — per-color cell counts (color-algebra).
- `regions` — coarse `G×G` (G=8) occupancy grid of the dominant non-bg color per cell.
- `key_of(state) -> tuple` — canonical hashable key for any lens output.

- [ ] **Step 1: Write the failing test**

```python
# tests/e134/test_perceptors.py
import numpy as np
from experiments.e134.perceptors import LENSES, key_of


def _frame():
    f = np.zeros((16, 16), dtype=int)
    f[1, 1] = 3                      # rare singleton
    f[0, :] = (np.arange(16) % 9)    # a changing top "status bar"
    f[8:12, 4:8] = 5                 # a block (for symmetry/region)
    return f


def test_all_lenses_produce_keys():
    f = _frame()
    for name, fn in LENSES.items():
        s = fn(f)
        assert isinstance(s, dict)
        k = key_of(s)
        assert isinstance(k, tuple) and k == key_of(fn(f))   # deterministic


def test_meter_lens_captures_the_status_row_values():
    # the 'meter' lens must expose the top-row values (a timer/counter) -- the feature object-state drops
    f = _frame()
    m = LENSES['meter'](f)
    assert 'meter' in m and len(m['meter']) > 0


def test_salience_ranks_small_rare_first():
    f = _frame()
    s = LENSES['salience'](f)
    assert s['targets'] and s['targets'][0][:2] == [1, 1]   # the rare singleton at (y,x)=(1,1)
```

- [ ] **Step 2: Run** → `ModuleNotFoundError`.
- [ ] **Step 3: Implement** the package markers + `perceptors.py` (self-contained; `objects`/`salience` reuse `experiments.e125.objstate` via a sys.path insert; `meter` detects high-variance cells over a single frame by the top status region OR is documented to need a probe — for the single-frame test, expose the top row's distinct values; `key_of` sorts/rounds to a tuple).
- [ ] **Step 4: Run** → 3 pass.
- [ ] **Step 5: Commit.**

---

### Task 2: Composite fuse + fidelity-SELECT (ConsensusTransition)

**Files:** Create `experiments/e134/composite.py`; Test `tests/e134/test_composite.py`.

**Interfaces:**
- `composite_key(frame, lenses=LENSES) -> tuple` — concatenate every lens's `key_of` into one tuple (the non-aliasing fused key: two frames differ in the composite key iff they differ under ANY lens).
- `fidelity(transitions, fn) -> float` — given observed `[(frame,action,next_frame)]`, build the `(key_of(fn(frame)), action) -> key_of(fn(next_frame))` table and return the fraction of `(key,action)` pairs that map **consistently** to a single next-key (Markov-ness of that lens) AND that are non-degenerate (the lens distinguishes states: penalize a lens that collapses everything to one key).
- `select_lens(transitions, lenses=LENSES) -> (name, fn, score)` — the **SELECT** combiner: return the highest-fidelity lens (ties → broader state space). This is `ConsensusTransition(mode="select")`; never average.

- [ ] **Step 1: Write the failing test**

```python
# tests/e134/test_composite.py
import numpy as np
from experiments.e134.composite import composite_key, fidelity, select_lens


def test_composite_key_distinguishes_when_any_lens_does():
    a = np.zeros((8, 8), dtype=int); b = a.copy(); b[0, 0] = 7   # differs only in one cell
    assert composite_key(a) != composite_key(b)                  # some lens catches it


def test_select_prefers_the_markov_nonaliasing_lens():
    # synthetic: 'objects' aliases (drops the deciding top-row counter) so its transitions are
    # inconsistent; 'meter' is Markov. SELECT must pick the consistent, non-degenerate lens.
    frames = []
    for t in range(4):
        f = np.zeros((8, 8), dtype=int); f[4, 4] = 5; f[0, 0] = t   # only the meter cell advances
        frames.append(f)
    trans = [(frames[i], 1, frames[i + 1]) for i in range(3)]
    name, fn, score = select_lens(trans)
    assert name == 'meter' and score > 0.0
```

- [ ] **Step 2: Run** → fail.
- [ ] **Step 3: Implement** `composite_key`, `fidelity`, `select_lens` per the Interfaces (consistency = 1 − fraction of (key,action) pairs with >1 distinct next-key; degenerate-penalty if the lens yields <2 distinct keys over the transitions).
- [ ] **Step 4: Run** `~/.arcv/bin/python -m pytest tests/e134/ -v` → all pass.
- [ ] **Step 5: Commit.**

---

### Task 3: Wire the composite into the EWM toolkit + the agent harness

**Files:** Modify `experiments/e133/ewm_toolkit.py` (ADD `composite_key`, `select_lens`, `LENSES` re-export — backward-compatible); Modify `scripts/run_arc_agent_ewm_toolkit.sh` (copy e134 perceptors+composite into the workspace; TASK.md tells the agent to perceive via the **composite** and to `select_lens` the planning modality); Test `tests/e134/test_toolkit_wire.py`.

**Interfaces:** the workspace gets `perceptors.py` + `composite.py` alongside `ewm_toolkit.py`; the agent's recipe becomes: perceive with `composite_key` (so the win-relevant feature is never aliased away), `select_lens` on observed transitions to pick the highest-fidelity modality, build that modality's `WorldSim`, then `plan_in_model` + verify (unchanged). Add one sentence to the TASK.md: *"Your state key MUST be composite_key(frame) — a single object lens silently drops timers/animation/1-cell indicators that decide the win; select_lens tells you which modality to PLAN in."*

- [ ] **Step 1: Write the failing test** (`tests/e134/test_toolkit_wire.py`): assert `from experiments.e134.composite import composite_key, select_lens` and `from experiments.e134.perceptors import LENSES` all import and that `composite_key` of two differing frames differ; assert the harness file contains `composite.py` in its copy list.
- [ ] **Step 2: Run** → fail.
- [ ] **Step 3: Implement** the re-exports + harness edits.
- [ ] **Step 4: Run** the whole `tests/e134/` suite green; smoke `~/.arcv/bin/python -c "import experiments.e134.composite, experiments.e134.perceptors"`.
- [ ] **Step 5: Commit.**

---

## Self-Review

**Spec coverage:** K lenses → Task 1; composite fuse + fidelity-SELECT (ConsensusTransition, the E95 *select-not-average* rule) → Task 2; wired into the EWM agent (the full-game cracker) → Task 3. The real-game run is via the existing `scripts/sweep_ewm_toolkit.sh` after Task 3.

**Honest risk (on record):** per the arc-3 paper, multi-perception consensus is **coverage/breadth, not a deep mechanism** — it raised *reach*, while full-game cracking is the reasoning agent's. The composite's contribution to the 11→15 gap is (a) removing perception aliasing that limits the agent on deep levels and (b) modality coverage for games one lens aliases (cn04/lf52/m0r0/s5i5 precedent). It is not guaranteed to crack the deepest procedural walls alone; reported honestly either way.
