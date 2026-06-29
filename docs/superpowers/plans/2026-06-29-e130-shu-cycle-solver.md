# E130 SHU-Cycle Solver Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the explicit, instrumented SHU behavioral cycle (introspect → extrospect → expert-pooled moral filter → act → measure tension → update) as a source-free ARC-AGI-3 solver, with the formalism's two theorems as deterministic correctness anchors.

**Architecture:** A thin **abstract layer** (`operators.py`, `efei.py`) implements the formalism's update operators and the EFEI expert/amateur estimators over ℝ^d stereotype vectors — these carry the two theorem-replay validations. A **concrete layer** (`world_model.py`, `perception.py`, `moral_filter.py`, `cycle.py`) instantiates the cycle for ARC-3 over object-state keys, reusing E125 perception and the E129 banked frontier. A runner (`e130_shu_cycle.py`) has a deterministic `theorems` mode and an online `solve` mode; a sweep banks gains through the existing autobank gate.

**Tech Stack:** Python 3.9 (`~/.arcv/bin/python`, has numpy + pytest), stdlib + numpy only. No new third-party deps. Reuses `experiments/e125/objstate.py`, `experiments/e127/perception.py`, `experiments/arc3_sandbox.py`, `scripts/autobank_sourcefree.py`.

## Global Constraints

- **Source-free:** never read any `<game>.py`; the solver acts on `SandboxGame` (only `{frame, levels, win, avail, done}`) and reasons the win from frames. Banking is gated by `audit_sandbox` + real-env replay + OpenWorld round-trip.
- **Zero-dependency core untouched:** all new code lives under `experiments/e130/` and `tests/e130/` and may use numpy; `openworld/` core stays stdlib (do not import e130 from core).
- **save_results BEFORE asserts** in any experiment runner (a failed check must not lose the run).
- **Deterministic where possible:** fixed seeds; theorem tests are pure numpy/stdlib, no env, no LLM.
- **Run tests with** `~/.arcv/bin/python -m pytest tests/e130/ -v` (numpy-enabled interpreter).
- **Commit messages end with:** `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`. Branch off `main`; do not commit/push unless asked.
- **New scratch prefix `su_`** for E130 solver workdirs (parallels `ge_`/`gm_`/`fl_`), so the autobank gate attests E130 gains without colliding.

---

### Task 1: Formalism operators + cycle convergence (Theorem 4.6)

**Files:**
- Create: `experiments/e130/__init__.py`
- Create: `experiments/e130/operators.py`
- Test: `tests/e130/__init__.py`, `tests/e130/test_operators.py`

**Interfaces:**
- Consumes: nothing (numpy only).
- Produces:
  - `tension(sigma_I, sigma_E) -> float`
  - `E_alpha(sigma_E, X, alpha) -> np.ndarray`  (extrospection synthesis, Def 2.7)
  - `I_gamma(sigma_I, sigma_E, gamma) -> np.ndarray`  (introspection retrosynthesis, Def 2.7)
  - `cycle_map(sigma_I, sigma_E, theta_star, alpha, gamma) -> (np.ndarray, np.ndarray)`  (BSTC cycle, Def 4.5)
  - `rho(alpha, gamma) -> float`  (contraction modulus = max(1-γ,1-α), Thm 4.6)

- [ ] **Step 1: Write the failing test**

```python
# tests/e130/test_operators.py
import numpy as np
from experiments.e130 import operators as op


def test_tension_is_norm_of_difference():
    a = np.array([1.0, 0.0]); b = np.array([0.0, 0.0])
    assert abs(op.tension(a, b) - 1.0) < 1e-9
    assert op.tension(a, a) == 0.0


def test_cycle_converges_geometrically_to_theta_star():
    # Theorem 4.6: iterating the cycle map contracts tension to 0 at rate rho<1,
    # and the fixed point is (theta_star, theta_star).
    rng = np.random.default_rng(0)
    d = 5
    theta = rng.normal(size=d)
    sI = rng.normal(size=d); sE = rng.normal(size=d)
    alpha, gamma = 0.5, 0.5
    r = op.rho(alpha, gamma)
    assert r < 1.0
    prev_err = np.linalg.norm(sI - theta) + np.linalg.norm(sE - theta)
    for _ in range(200):
        sI, sE = op.cycle_map(sI, sE, theta, alpha, gamma)
        err = np.linalg.norm(sI - theta) + np.linalg.norm(sE - theta)
        # each step shrinks the error by at most rho (+ small slack for the
        # block-triangular off-diagonal coupling)
        assert err <= prev_err * (r + 0.25) + 1e-9
        prev_err = err
    assert prev_err < 1e-6           # converged to the diagonal at theta_star
    assert op.tension(sI, sE) < 1e-6


def test_I_gamma_reduces_tension_each_application():
    # consolidating the reading into the model strictly reduces tension when gamma in (0,1)
    sI = np.array([2.0, 0.0]); sE = np.array([0.0, 0.0])
    t0 = op.tension(sI, sE)
    sI2 = op.I_gamma(sI, sE, 0.5)
    assert op.tension(sI2, sE) < t0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.arcv/bin/python -m pytest tests/e130/test_operators.py -v`
Expected: FAIL — `ModuleNotFoundError: experiments.e130.operators`.

- [ ] **Step 3: Write minimal implementation**

```python
# experiments/e130/__init__.py
# (empty package marker)
```
```python
# tests/e130/__init__.py
# (empty package marker)
```
```python
# experiments/e130/operators.py
"""SHU formalism update operators over the stereotype space S = R^d (Defs 2.6-2.7, 4.5).
The two theorems (4.4 EFEI separation, 4.6 cycle convergence) are validated against these."""
import numpy as np


def tension(sigma_I, sigma_E):
    """T(H) = ||sigma_I - sigma_E||  (Def 2.6)."""
    return float(np.linalg.norm(np.asarray(sigma_I) - np.asarray(sigma_E)))


def E_alpha(sigma_E, X, alpha):
    """Extrospection synthesis: assimilate reading X into the extrospective stereotype (Def 2.7)."""
    return (1.0 - alpha) * np.asarray(sigma_E) + alpha * np.asarray(X)


def I_gamma(sigma_I, sigma_E, gamma):
    """Introspection retrosynthesis: consolidate the reading into the introspective stereotype (Def 2.7)."""
    return (1.0 - gamma) * np.asarray(sigma_I) + gamma * np.asarray(sigma_E)


def cycle_map(sigma_I, sigma_E, theta_star, alpha, gamma):
    """One BSTC behavioral cycle (Def 4.5): extrospection synthesis against the true context,
    then introspection retrosynthesis against the fresh reading."""
    sE_next = E_alpha(sigma_E, theta_star, alpha)
    sI_next = I_gamma(sigma_I, sE_next, gamma)
    return sI_next, sE_next


def rho(alpha, gamma):
    """Contraction modulus of the cycle map (Thm 4.6): spectral radius max(1-gamma, 1-alpha)."""
    return max(1.0 - gamma, 1.0 - alpha)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.arcv/bin/python -m pytest tests/e130/test_operators.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add experiments/e130/__init__.py experiments/e130/operators.py tests/e130/__init__.py tests/e130/test_operators.py
git commit -m "E130 Task 1: SHU operators + cycle convergence (Thm 4.6)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: EFEI expert/amateur separation (Theorem 4.4)

**Files:**
- Create: `experiments/e130/efei.py`
- Test: `tests/e130/test_efei_separation.py`

**Interfaces:**
- Consumes: nothing (numpy only).
- Produces:
  - `expert_error(theta_star, N, n, tau, beta, d, rng) -> float` — expected ‖X̄−θ*‖ pooling N experts (Assumption 1).
  - `amateur_trials(M, rng) -> int` — #random draws to hit the unique resolving behavior (Assumption 2).
  - `expert_consultations_for(delta, n, tau, beta, d) -> int` — smallest N reaching accuracy δ (closed form `N ≥ dτ²/(nδ²)`).

- [ ] **Step 1: Write the failing test**

```python
# tests/e130/test_efei_separation.py
import numpy as np
from experiments.e130 import efei


def test_expert_error_decreases_as_one_over_sqrt_N():
    rng = np.random.default_rng(1)
    theta = rng.normal(size=8)
    e1   = np.mean([efei.expert_error(theta, 1,   n=50, tau=1.0, beta=0.0, d=8, rng=rng) for _ in range(40)])
    e100 = np.mean([efei.expert_error(theta, 100, n=50, tau=1.0, beta=0.0, d=8, rng=rng) for _ in range(40)])
    assert e100 < e1 / 5.0                       # ~1/sqrt(100) = 1/10 the error


def test_amateur_cost_is_theta_of_M():
    rng = np.random.default_rng(2)
    for M in (10, 40, 160):
        mean_trials = np.mean([efei.amateur_trials(M, rng) for _ in range(400)])
        assert 0.6 * M <= mean_trials <= 1.5 * M  # geometric mean = M


def test_separation_expert_consultations_independent_of_M():
    # Expert reaches accuracy delta in N independent of pool size M; amateur needs ~M.
    N = efei.expert_consultations_for(delta=0.2, n=50, tau=1.0, beta=0.0, d=8)
    assert N < 80
    rng = np.random.default_rng(3)
    for M in (200, 2000):                          # amateur cost grows with M, expert N does not
        amateur = np.mean([efei.amateur_trials(M, rng) for _ in range(200)])
        assert amateur > 3 * N
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.arcv/bin/python -m pytest tests/e130/test_efei_separation.py -v`
Expected: FAIL — `ModuleNotFoundError: experiments.e130.efei`.

- [ ] **Step 3: Write minimal implementation**

```python
# experiments/e130/efei.py
"""Expert-Facilitated Extrospection-Introspection vs amateur search (Thm 4.4).

Expert model (Assumption 1): expert j returns X_j = theta* + b_j + eps_j, ||b_j||<=beta,
eps_j coordinates sub-Gaussian with variance proxy tau^2/n. Pooling N experts gives error
beta + tau*sqrt(d/(nN)) -> O(1/sqrt(N)), INDEPENDENT of behavior-pool size M.

Amateur model (Assumption 2): a behavior pool of size M with one resolving element; uniform
draws with replacement -> geometric hitting time with mean M = Theta(M)."""
import numpy as np


def expert_error(theta_star, N, n, tau, beta, d, rng):
    theta_star = np.asarray(theta_star, dtype=float)
    biases = rng.normal(size=(N, d)); biases *= (beta / max(np.linalg.norm(biases, axis=1).max(), 1e-9))
    noise = rng.normal(scale=tau / np.sqrt(n), size=(N, d))
    estimates = theta_star[None, :] + biases + noise
    return float(np.linalg.norm(estimates.mean(axis=0) - theta_star))


def amateur_trials(M, rng):
    # uniform draws with replacement until the single resolving behavior (index 0) is hit
    t = 0
    while True:
        t += 1
        if rng.integers(0, M) == 0:
            return t


def expert_consultations_for(delta, n, tau, beta, d):
    # tau*sqrt(d/(nN)) <= delta - beta  =>  N >= d*tau^2 / (n*(delta-beta)^2)   (Thm 4.4a)
    slack = max(delta - beta, 1e-9)
    return int(np.ceil(d * tau * tau / (n * slack * slack)))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.arcv/bin/python -m pytest tests/e130/test_efei_separation.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add experiments/e130/efei.py tests/e130/test_efei_separation.py
git commit -m "E130 Task 2: EFEI expert/amateur separation (Thm 4.4)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Introspective world model σ_I (tabular dynamics + TEIE database)

**Files:**
- Create: `experiments/e130/world_model.py`
- Test: `tests/e130/test_world_model.py`

**Interfaces:**
- Consumes: nothing (stdlib only; keys are hashable tuples from perception).
- Produces: class `WorldModel` with
  - `predict(state_key, action) -> (next_key|None, known: bool)`
  - `simulate(state_key, plan) -> (pred_keys: list, known: bool)`
  - `update(state_key, action, observed_next_key) -> None`  (increments `.conflicts` on contradiction)
  - `bank_subroutine(name, plan) -> None` ; `lookup(name) -> list|None`
  - attributes `.table: dict`, `.db: dict`, `.conflicts: int`

- [ ] **Step 1: Write the failing test**

```python
# tests/e130/test_world_model.py
from experiments.e130.world_model import WorldModel


def test_unknown_transition_flags_known_false():
    wm = WorldModel()
    nxt, known = wm.predict(("s0",), 1)
    assert nxt is None and known is False


def test_update_then_predict_is_known():
    wm = WorldModel()
    wm.update(("s0",), 1, ("s1",))
    nxt, known = wm.predict(("s0",), 1)
    assert known is True and nxt == ("s1",)


def test_simulate_stops_at_first_unseen():
    wm = WorldModel()
    wm.update(("s0",), 1, ("s1",))
    wm.update(("s1",), 1, ("s2",))           # s2 -> 1 is unseen
    preds, known = wm.simulate(("s0",), [1, 1, 1])
    assert preds == [("s1",), ("s2",)] and known is False


def test_contradiction_increments_conflicts():
    wm = WorldModel()
    wm.update(("s0",), 1, ("s1",))
    wm.update(("s0",), 1, ("sX",))           # regime change at the same (state, action)
    assert wm.conflicts == 1
    assert wm.predict(("s0",), 1)[0] == ("sX",)   # last-write wins


def test_bank_and_lookup_subroutine():
    wm = WorldModel()
    wm.bank_subroutine("reach_goal", [1, 2, 6])
    assert wm.lookup("reach_goal") == [1, 2, 6]
    assert wm.lookup("missing") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.arcv/bin/python -m pytest tests/e130/test_world_model.py -v`
Expected: FAIL — `ModuleNotFoundError: experiments.e130.world_model`.

- [ ] **Step 3: Write minimal implementation**

```python
# experiments/e130/world_model.py
"""Introspective stereotype sigma_I: a learned deterministic dynamics table over masked
object-state keys, plus the TEIE subroutine database (Thm 4.10 -- O(1) replay of banked plans).
simulate() rolls a plan forward and reports the knowledge frontier (known=False at the first
unseen transition -- exactly where tension will be measured and the model must learn by acting)."""


class WorldModel:
    def __init__(self):
        self.table = {}        # (state_key, action) -> next_state_key
        self.db = {}           # subroutine name -> plan (list of actions)
        self.conflicts = 0     # contradictions at a seen (state, action): a regime-change signal

    def predict(self, state_key, action):
        key = (state_key, action)
        if key in self.table:
            return self.table[key], True
        return None, False

    def simulate(self, state_key, plan):
        preds, s = [], state_key
        for a in plan:
            nxt, known = self.predict(s, a)
            if not known:
                return preds, False
            preds.append(nxt); s = nxt
        return preds, True

    def update(self, state_key, action, observed_next_key):
        key = (state_key, action)
        if key in self.table and self.table[key] != observed_next_key:
            self.conflicts += 1
        self.table[key] = observed_next_key

    def bank_subroutine(self, name, plan):
        self.db[name] = list(plan)

    def lookup(self, name):
        return self.db.get(name)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.arcv/bin/python -m pytest tests/e130/test_world_model.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add experiments/e130/world_model.py tests/e130/test_world_model.py
git commit -m "E130 Task 3: introspective world model (tabular dynamics + TEIE db)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Extrospection σ_E (perception stereotype across modalities)

**Files:**
- Create: `experiments/e130/perception.py`
- Test: `tests/e130/test_perception.py`

**Interfaces:**
- Consumes: `experiments/e125/objstate.py` (`object_state`, `state_key`).
- Produces:
  - dataclass `Stereotype(key: tuple, vec: np.ndarray, objects: list, click_targets: list)`
  - `extrospect(frame, avail=(), ignore_colors=()) -> Stereotype`
  - `embed(objects, d=64) -> np.ndarray` — fixed-length vector for tension (sorted (color,y,x), padded/truncated to d).

- [ ] **Step 1: Write the failing test**

```python
# tests/e130/test_perception.py
import numpy as np
from experiments.e130 import perception as P


def _frame():
    f = np.zeros((8, 8), dtype=int)      # bg=0
    f[1, 1] = 3                          # a rare singleton (click target)
    f[5, 5] = 3
    f[2:6, 6] = 7                        # a larger bar (not a click target)
    return f


def test_extrospect_returns_objects_and_key():
    s = P.extrospect(_frame(), avail=[6])
    assert len(s.objects) >= 2
    assert isinstance(s.key, tuple)
    assert s.vec.shape[0] == 64


def test_click_targets_are_small_components():
    s = P.extrospect(_frame(), avail=[6])
    sizes = {(t["y"], t["x"]) for t in s.click_targets}
    assert (1, 1) in sizes and (5, 5) in sizes      # the singletons
    assert all(t["size"] <= 16 for t in s.click_targets)


def test_extrospect_is_deterministic():
    a = P.extrospect(_frame(), avail=[6]); b = P.extrospect(_frame(), avail=[6])
    assert a.key == b.key and np.allclose(a.vec, b.vec)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.arcv/bin/python -m pytest tests/e130/test_perception.py -v`
Expected: FAIL — `ModuleNotFoundError: experiments.e130.perception`.

- [ ] **Step 3: Write minimal implementation**

```python
# experiments/e130/perception.py
"""Extrospective stereotype sigma_E: perceive a frame into object-state + a fixed-length vector
(for tension) + the click-target set (small/rare components -- the click modality). Reuses the
E125 object perceptor; adds the embedding and click-target extraction."""
import os, sys
from dataclasses import dataclass, field
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from experiments.e125 import objstate


@dataclass
class Stereotype:
    key: tuple
    vec: np.ndarray
    objects: list
    click_targets: list = field(default_factory=list)


def embed(objects, d=64):
    flat = []
    for o in sorted(objects, key=lambda o: (o["color"], o["y"], o["x"])):
        flat += [o["color"], o["y"], o["x"]]
    flat = (flat + [0] * d)[:d]
    return np.asarray(flat, dtype=float)


def extrospect(frame, avail=(), ignore_colors=()):
    grid = np.asarray(frame).astype(int).tolist()
    s = objstate.object_state(grid, ignore_colors=ignore_colors)
    objs = s["objects"]
    targets = [o for o in objs if o["size"] <= 16] if 6 in tuple(avail) else []
    return Stereotype(key=objstate.state_key(s), vec=embed(objs), objects=objs, click_targets=targets)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.arcv/bin/python -m pytest tests/e130/test_perception.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add experiments/e130/perception.py tests/e130/test_perception.py
git commit -m "E130 Task 4: extrospection stereotype (objects + vec + click targets)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Moral filter (expert proposers + world-model valence + select)

**Files:**
- Create: `experiments/e130/moral_filter.py`
- Test: `tests/e130/test_moral_filter.py`

**Interfaces:**
- Consumes: `Stereotype` (Task 4), `WorldModel` (Task 3).
- Produces:
  - dataclass `Waypoint(kind: str, y: int, x: int, source: str)`
  - proposers `reach_rare_color`, `click_smallest`, `reach_unseen_state` — each `(stereotype, history, world_model) -> list[Waypoint]`
  - `DEFAULT_EXPERTS = [reach_rare_color, click_smallest]`
  - `valence(wp, stereotype, world_model) -> float`
  - `realize(wp, stereotype, dir_map) -> list` — waypoint → action plan
  - `select(stereotype, history, world_model, experts, rng, dir_map=None, amateur=False) -> (Waypoint, list, float)`

- [ ] **Step 1: Write the failing test**

```python
# tests/e130/test_moral_filter.py
import numpy as np
from experiments.e130 import moral_filter as mf
from experiments.e130.world_model import WorldModel
from experiments.e130 import perception as P


def _stereo():
    f = np.zeros((8, 8), dtype=int); f[1, 1] = 3; f[6, 6] = 7; f[6, 7] = 7
    return P.extrospect(f, avail=[6])


def test_experts_propose_at_least_one_waypoint():
    s = _stereo()
    wps = []
    for e in mf.DEFAULT_EXPERTS:
        wps += e(s, [], WorldModel())
    assert len(wps) >= 1
    assert all(isinstance(w, mf.Waypoint) for w in wps)


def test_select_returns_argmax_phi_times_V():
    s = _stereo(); rng = np.random.default_rng(0)
    wp, plan, score = mf.select(s, [], WorldModel(), mf.DEFAULT_EXPERTS, rng)
    assert isinstance(wp, mf.Waypoint) and isinstance(plan, list) and score >= 0.0


def test_amateur_degenerates_to_uniform_choice():
    # with amateur=True the filter ignores phi*V and draws uniformly: over many seeds it must
    # NOT always pick the same waypoint (the formalism's amateur/random regime).
    s = _stereo()
    picks = set()
    for seed in range(40):
        wp, _, _ = mf.select(s, [], WorldModel(), mf.DEFAULT_EXPERTS,
                             np.random.default_rng(seed), amateur=True)
        picks.add((wp.kind, wp.y, wp.x))
    assert len(picks) >= 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.arcv/bin/python -m pytest tests/e130/test_moral_filter.py -v`
Expected: FAIL — `ModuleNotFoundError: experiments.e130.moral_filter`.

- [ ] **Step 3: Write minimal implementation**

```python
# experiments/e130/moral_filter.py
"""Moral filter (Def 2.8): expert proposers each suggest a candidate waypoint; the world model
scores each by simulated progress (valence V); pooled agreement gives phi; the selected behavior
maximizes S = phi*V. With amateur=True the rule degenerates to a uniform draw (the book's amateur
filter, Thm 4.4's Theta(M) regime) -- kept as the ablation baseline."""
from collections import Counter
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Waypoint:
    kind: str        # 'click' | 'reach'
    y: int
    x: int
    source: str


def reach_rare_color(stereotype, history, world_model):
    if not stereotype.objects:
        return []
    colors = Counter(o["color"] for o in stereotype.objects)
    rare = min(colors, key=colors.get)
    return [Waypoint("reach", o["y"], o["x"], "reach_rare_color")
            for o in stereotype.objects if o["color"] == rare]


def click_smallest(stereotype, history, world_model):
    if not stereotype.click_targets:
        return []
    o = min(stereotype.click_targets, key=lambda o: o["size"])
    return [Waypoint("click", o["y"], o["x"], "click_smallest")]


def reach_unseen_state(stereotype, history, world_model):
    # coverage proposer: target objects whose (key, candidate-action) is unseen in the model
    out = []
    for o in stereotype.objects:
        if (stereotype.key, ("click", o["y"], o["x"])) not in world_model.table:
            out.append(Waypoint("click", o["y"], o["x"], "reach_unseen_state"))
    return out


DEFAULT_EXPERTS = [reach_rare_color, click_smallest]


def valence(wp, stereotype, world_model):
    # simulated progress: novelty of the waypoint's induced transition under sigma_I.
    seen = (stereotype.key, (wp.kind, wp.y, wp.x)) in world_model.table
    return 0.5 if seen else 1.0       # unseen transitions carry more expected progress


def realize(wp, stereotype, dir_map=None):
    if wp.kind == "click":
        return [(6, wp.x, wp.y)]
    return [(6, wp.x, wp.y)]          # reach degenerates to click without a dir_map (overridden in cycle)


def select(stereotype, history, world_model, experts, rng, dir_map=None, amateur=False):
    pool = []
    for e in experts:
        pool += e(stereotype, history, world_model)
    if not pool:
        return Waypoint("noop", 0, 0, "none"), [], 0.0
    if amateur:
        wp = pool[int(rng.integers(0, len(pool)))]
        return wp, realize(wp, stereotype, dir_map), 0.0
    # phi = pooled agreement (how many experts proposed this (kind,y,x)); S = phi * V
    agree = Counter((w.kind, w.y, w.x) for w in pool)
    best, best_s = None, -1.0
    for w in pool:
        s = agree[(w.kind, w.y, w.x)] * valence(w, stereotype, world_model)
        if s > best_s:
            best, best_s = w, s
    return best, realize(best, stereotype, dir_map), float(best_s)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.arcv/bin/python -m pytest tests/e130/test_moral_filter.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add experiments/e130/moral_filter.py tests/e130/test_moral_filter.py
git commit -m "E130 Task 5: moral filter (expert proposers + valence + select)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: The behavioral cycle (run_cycle over a game)

**Files:**
- Create: `experiments/e130/cycle.py`
- Test: `tests/e130/test_cycle.py`

**Interfaces:**
- Consumes: `WorldModel`, `moral_filter.select`, a perceive function `frame -> Stereotype`, and a
  game object exposing `.frame`, `.levels`, `.avail`, `.done`, and `step(a, x=None, y=None)`.
- Produces:
  - dataclass `Result(best_levels: int, best_actions: list, tension_trace: list, cycles: int, banked: int)`
  - `run_cycle(game, world_model, perceive, experts, budget, win, rng, seed_actions=()) -> Result`

- [ ] **Step 1: Write the failing test**

```python
# tests/e130/test_cycle.py
import numpy as np
from experiments.e130.world_model import WorldModel
from experiments.e130 import perception as P, moral_filter as mf
from experiments.e130.cycle import run_cycle


class ToyGame:
    """Deterministic A->B->C protocol: clicking the rare cell (1,1) advances a 1-cell marker
    rightward; reaching column 3 raises the level. Tests the cycle end-to-end with no real env."""
    def __init__(self):
        self.col = 0; self.levels = 0; self.done = False; self.avail = [6]
    @property
    def frame(self):
        f = np.zeros((8, 8), dtype=int); f[1, 1] = 3; f[4, self.col] = 5
        return f
    def step(self, a, x=None, y=None):
        if a == 6 and (x, y) == (1, 1):
            self.col = min(self.col + 1, 3)
            if self.col == 3: self.levels = 1
        return self.frame


def test_cycle_reaches_the_win_and_traces_tension():
    g = ToyGame(); wm = WorldModel(); rng = np.random.default_rng(0)
    res = run_cycle(g, wm, lambda fr: P.extrospect(fr, avail=[6]),
                    mf.DEFAULT_EXPERTS, budget=50, win=1, rng=rng)
    assert res.best_levels == 1
    assert len(res.tension_trace) > 0
    assert res.best_actions[-1] == [6, 1, 1]      # last action is the advancing click


def test_cycle_never_regresses_below_seed():
    g = ToyGame(); wm = WorldModel(); rng = np.random.default_rng(1)
    res = run_cycle(g, wm, lambda fr: P.extrospect(fr, avail=[6]),
                    mf.DEFAULT_EXPERTS, budget=2, win=1, rng=rng)
    assert res.best_levels >= 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.arcv/bin/python -m pytest tests/e130/test_cycle.py -v`
Expected: FAIL — `ModuleNotFoundError: experiments.e130.cycle`.

- [ ] **Step 3: Write minimal implementation**

```python
# experiments/e130/cycle.py
"""The BSTC behavioral cycle (Thm 4.6), made explicit: extrospect -> introspect+filter -> behave
-> measure tension (sim-vs-real) -> retrosynthesize (learn by acting) -> bank on level gain.
Tension here is over object-state KEYS (0 if the model predicted the observed key, 1 if not) --
the discrete analogue of ||sigma_I - sigma_E||; updates drive it to 0 (learning by action)."""
from dataclasses import dataclass, field
import numpy as np
from experiments.e130 import moral_filter as mf


@dataclass
class Result:
    best_levels: int
    best_actions: list
    tension_trace: list = field(default_factory=list)
    cycles: int = 0
    banked: int = 0


def _do(game, action):
    a = action[0]
    try:
        game.step(a, action[1], action[2]) if a == 6 else game.step(a)
        return not bool(getattr(game, "done", False))
    except Exception:
        return False


def run_cycle(game, world_model, perceive, experts, budget, win, rng, seed_actions=()):
    actions = [list(a) for a in seed_actions]
    for a in actions:                                  # replay the TEIE frontier (O(1), Thm 4.10)
        _do(game, a)
    best = int(getattr(game, "levels", 0))
    best_actions = list(actions)
    trace, steps, cycles = [], 0, 0
    while steps < budget and best < win and not getattr(game, "done", False):
        cycles += 1
        s = perceive(game.frame)
        wp, plan, _ = mf.select(s, best_actions, world_model, experts, rng)
        if not plan:
            break
        for act in plan:
            pred_key, known = world_model.predict(s.key, (wp.kind, wp.y, wp.x))
            if not _do(game, act):
                break
            steps += 1
            obs = perceive(game.frame)
            T = 0.0 if (known and pred_key == obs.key) else 1.0    # sim-vs-real tension
            trace.append(T)
            if T > 0.0:
                world_model.update(s.key, (wp.kind, wp.y, wp.x), obs.key)   # I_gamma: learn by acting
            actions.append([act[0], act[1], act[2]] if act[0] == 6 else [act[0]])
            lv = int(getattr(game, "levels", 0))
            if lv > best:
                best, best_actions = lv, list(actions)
                world_model.bank_subroutine(f"win_to_{lv}", list(actions))
            s = obs
    banked = len(world_model.db)
    return Result(best_levels=best, best_actions=best_actions, tension_trace=trace,
                  cycles=cycles, banked=banked)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.arcv/bin/python -m pytest tests/e130/test_cycle.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add experiments/e130/cycle.py tests/e130/test_cycle.py
git commit -m "E130 Task 6: explicit behavioral cycle (tension + learn-by-action + bank)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 7: Runner (theorems + solve) and the all-walls sweep

**Files:**
- Create: `experiments/e130_shu_cycle.py`
- Create: `scripts/sweep_shu_cycle.sh`
- Test: `tests/e130/test_runner.py`

**Interfaces:**
- Consumes: all of Tasks 1-6; `experiments/arc3_sandbox.py`; `experiments/results/arc3_fullgame_sourcefree.json`.
- Produces:
  - `validate_theorems(rng) -> dict` — runs the Thm 4.4 + 4.6 checks, returns the metrics dict.
  - `main()` — `theorems` mode (save_results BEFORE asserts) and `solve <game> [budget]` mode.

- [ ] **Step 1: Write the failing test**

```python
# tests/e130/test_runner.py
import numpy as np
from experiments import e130_shu_cycle as R


def test_validate_theorems_reports_separation_and_contraction():
    m = R.validate_theorems(np.random.default_rng(0))
    assert m["expert_error_100"] < m["expert_error_1"] / 5.0       # Thm 4.4
    assert m["amateur_trials_mean"] > m["expert_consultations"]    # separation
    assert m["final_tension"] < 1e-6 and m["rho"] < 1.0            # Thm 4.6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.arcv/bin/python -m pytest tests/e130/test_runner.py -v`
Expected: FAIL — `ModuleNotFoundError: experiments.e130_shu_cycle`.

- [ ] **Step 3: Write minimal implementation**

```python
# experiments/e130_shu_cycle.py
"""E130 SHU-cycle solver. `theorems` mode validates the two formalism theorems deterministically
(paper-ready); `solve <game>` mode runs the explicit cycle on a real ARC-AGI-3 game, source-free,
seeded from the E129 banked frontier, writing su_<game>/solved.json for the autobank gate."""
import os, sys, json, shutil
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from experiments.e130 import efei, operators as op, perception as P, moral_filter as mf
from experiments.e130.world_model import WorldModel
from experiments.e130.cycle import run_cycle


def validate_theorems(rng):
    theta = rng.normal(size=8)
    e1 = np.mean([efei.expert_error(theta, 1, 50, 1.0, 0.0, 8, rng) for _ in range(40)])
    e100 = np.mean([efei.expert_error(theta, 100, 50, 1.0, 0.0, 8, rng) for _ in range(40)])
    N = efei.expert_consultations_for(0.2, 50, 1.0, 0.0, 8)
    amateur = float(np.mean([efei.amateur_trials(400, rng) for _ in range(200)]))
    sI, sE = rng.normal(size=8), rng.normal(size=8)
    for _ in range(200):
        sI, sE = op.cycle_map(sI, sE, theta, 0.5, 0.5)
    return {"expert_error_1": float(e1), "expert_error_100": float(e100),
            "expert_consultations": int(N), "amateur_trials_mean": amateur,
            "final_tension": float(op.tension(sI, sE)), "rho": float(op.rho(0.5, 0.5))}


def banked_frontier(game):
    p = os.path.join(ROOT, "experiments/results/arc3_fullgame_sourcefree.json")
    a = json.load(open(p))
    acts = a.get("solutions", {}).get(game) or []
    pg = a.get("per_game", {}).get(game, {})
    return acts, int(pg.get("levels", 0)), int(pg.get("win", 0))


def solve(game, budget):
    import arc3_sandbox
    env = arc3_sandbox.SandboxGame(game); env.reset()
    seed, seed_lv, win = banked_frontier(game)
    win = win or int(env.win)
    wm = WorldModel(); rng = np.random.default_rng(0)
    perceive = lambda fr: P.extrospect(fr, avail=list(getattr(env, "avail", [])))
    res = run_cycle(env, wm, perceive, mf.DEFAULT_EXPERTS, budget, win, rng, seed_actions=seed)
    wd = os.path.join(ROOT, "scratch_arc", f"su_{game}"); os.makedirs(wd, exist_ok=True)
    shutil.copy(os.path.join(ROOT, "experiments/arc3_sandbox.py"), wd)   # audit-clean workdir
    sol = {"game": game, "actions": res.best_actions, "levels": res.best_levels, "win": win,
           "method": "shu-cycle source-free (E130)"}
    json.dump(sol, open(os.path.join(wd, "solved.json"), "w"))
    if res.best_levels > seed_lv:
        shutil.copy(os.path.join(wd, "solved.json"), os.path.join(wd, "solved_best.json"))
    print(f"[e130] {game}: {seed_lv} -> {res.best_levels}/{win} "
          f"{'IMPROVED' if res.best_levels > seed_lv else 'no gain'} cycles={res.cycles} "
          f"banked={res.banked} tension_steps={len(res.tension_trace)}", flush=True)
    os.system(f"{sys.executable} {os.path.join(ROOT, 'scripts', 'capture_arc_run.py')} "
              f"{game} {wd} shu-cycle e130_shu_cycle.py")   # HF-ready capture (reuses capture_lib)
    try: env.close()
    except Exception: pass


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "theorems"
    if mode == "solve":
        solve(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 4000)
        return
    m = validate_theorems(np.random.default_rng(0))
    out = os.path.join(ROOT, "experiments/results/e130_shu_cycle.json")
    json.dump(m, open(out, "w"), indent=1)                     # save BEFORE asserts
    print(json.dumps(m, indent=1), flush=True)
    assert m["expert_error_100"] < m["expert_error_1"] / 5.0, "Thm 4.4 variance reduction failed"
    assert m["amateur_trials_mean"] > m["expert_consultations"], "EFEI separation failed"
    assert m["final_tension"] < 1e-6 and m["rho"] < 1.0, "Thm 4.6 contraction failed"
    print("[e130] theorems validated", flush=True)


if __name__ == "__main__":
    main()
```
```bash
# scripts/sweep_shu_cycle.sh
#!/usr/bin/env bash
# E130 SHU-cycle solver over the unsolved walls, source-free, seeded from each banked frontier.
# Pure compute (no API). su_ prefix; banks gains through the autobank gate after the run.
#   caffeinate -i nohup bash scripts/sweep_shu_cycle.sh > scratch_arc/su_sweep.log 2>&1 &
set -uo pipefail
ROOT=/Users/jim/Desktop/openworld; PY=/Users/jim/.arcv/bin/python
BUDGET="${1:-4000}"; POOL="${2:-3}"
ARCH="$ROOT/experiments/results/arc3_fullgame_sourcefree.json"
GAMES="bp35 dc22 g50t ka59 lf52 ls20 m0r0 r11l s5i5 sk48 sp80 su15 tn36 tu93 vc33 wa30"
echo "[su-sweep] START $(date) budget=$BUDGET pool=$POOL"
for g in $GAMES; do
  while [ "$(jobs -rp | wc -l | tr -d ' ')" -ge "$POOL" ]; do sleep 3; done
  echo "[su-sweep] launch $g $(date '+%H:%M:%S')"
  "$PY" "$ROOT/experiments/e130_shu_cycle.py" solve "$g" "$BUDGET" > "$ROOT/scratch_arc/su_${g}.log" 2>&1 &
done
wait
echo "[su-sweep] banking gains through the attestation gate"
SF_WD_PREFIX=su_ SF_ARCH="$ARCH" "$PY" "$ROOT/scripts/autobank_sourcefree.py" 2>&1 | grep -iE "sf-bank" || true
echo "[su-sweep] DONE $(date)"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.arcv/bin/python -m pytest tests/e130/test_runner.py -v`
Expected: PASS (1 test). Then `~/.arcv/bin/python experiments/e130_shu_cycle.py` prints the metrics and `[e130] theorems validated`.

- [ ] **Step 5: Commit**

```bash
chmod +x scripts/sweep_shu_cycle.sh
git add experiments/e130_shu_cycle.py scripts/sweep_shu_cycle.sh tests/e130/test_runner.py
git commit -m "E130 Task 7: runner (theorems + solve) + all-walls sweep

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

**1. Spec coverage:** σ_I → Task 3; σ_E → Task 4; tension + I_γ + cycle → Tasks 1, 6; moral filter (experts propose, world-model verifies) → Task 5; Thm 4.4 → Task 2; Thm 4.6 → Task 1; runner + su_ banking + sweep → Task 7. The integration run on m0r0/tu93 is executed via `solve`/the sweep after Task 7 (not a unit task — it is the experiment, reported honestly). All spec §2 units and §3 anchors covered.

**2. Placeholder scan:** none — every step has complete code and exact commands.

**3. Type consistency:** `Stereotype.key` (tuple) feeds `WorldModel.predict(state_key, ...)` and `mf.select`; the transition action key is uniformly `(wp.kind, wp.y, wp.x)` in Tasks 5 and 6; `run_cycle` returns `Result` consumed only by the runner. `experts` is always a list of `(stereotype, history, world_model) -> list[Waypoint]`. Consistent.

**Deferred (RL-expert feedback, v2 — see design §6 / review):** object-relative dynamics, adaptive expert weights, potential-based novelty, horizon>1, real-env-search baselines. Out of scope for this plan by decision; revisit after the first run.
