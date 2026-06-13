# E36 Representations Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (inline). Steps use checkbox (`- [ ]`) syntax.

**Goal:** `experiments/e36_representations.py` — a deterministic, offline experiment showing a composite-of-small-worlds representation beats a monolithic learner on compositional generalization, interference, and sample efficiency.

**Architecture:** Pure-numpy parametric K-sector world + hand oracle; a generic MLP and 1-NN (E12-style, sized by in/out dims); composite conditions build a real `CompositeWorld` with learned/exact per-sector `Transition`s. Three legs, fixed seeds. Spec: `docs/superpowers/specs/2026-06-12-e36-representations-design.md`.

**Tech Stack:** numpy + stdlib + `openworld.compose` / `openworld.transition` / `openworld.state`. No Ollama.

**The crux (read first):** the generalization claim is about *novel combinations of seen per-part values*, NOT novel per-part values (MLPs can't extrapolate a value range, and that would sink the composite too). So the **training joint distribution must cover each sector's full marginal range while covering only a thin slice of the joint product**; the test set is the full joint product. Construction below makes this exact.

---

### Task 1: World, oracle, and the composite-symbolic ceiling

**Files:** Create `experiments/e36_representations.py`

- [ ] **Step 1: sector world + oracle.**

```python
"""E36 - Composition yields better representations than monolithic learners.

A factored economy of K sectors (the E30 substrate), but here we sample
transitions and LEARN, comparing a monolithic MLP / 1-NN on the joint state
against a composite of small per-sector learners (and the exact symbolic
composite). Three legs: compositional generalization (novel combinations of
seen per-part values), interference (sequential forgetting), and sample
efficiency. Offline, deterministic, pure numpy - no Ollama.

The generalization split is the careful part: training covers each sector's
full MARGINAL value range but only a thin slice of the JOINT product, so the
composite (which needs only marginals) generalizes and the monolith (which
needs the joint) does not.
"""
import json
import numpy as np

from openworld import Action
from openworld.compose import Aggregator, CompositeWorld
from openworld.transition import Transition
from openworld.world import World
from common import save_results

G = 6                       # per-field grid 0..G
FIELDS = ["stock", "output", "waste"]
SECTOR_PARAMS = [           # distinct per-sector coefficients (cost,gain,rec,thresh,amt)
    dict(cost=1, gain=2, rec=1, thresh=3, amt=1),
    dict(cost=2, gain=1, rec=2, thresh=2, amt=2),
    dict(cost=1, gain=3, rec=1, thresh=4, amt=1),
    dict(cost=2, gain=2, rec=1, thresh=3, amt=2),
    dict(cost=1, gain=1, rec=2, thresh=2, amt=1),
]
ACTIONS = ["produce", "recycle", "wait"]


def clamp(v):
    return max(0, min(G, v))


def sector_step(s, action, p):
    """Deterministic branchy per-sector update, clamped to 0..G."""
    stock, output, waste = s["stock"], s["output"], s["waste"]
    if action == "produce" and stock >= p["cost"]:
        stock -= p["cost"]; output += p["gain"]; waste += 1
    elif action == "recycle" and waste >= p["rec"]:
        waste -= p["rec"]; stock += 1
    if output > p["thresh"]:                 # decay branch
        waste += p["amt"]
    return {"stock": clamp(stock), "output": clamp(output), "waste": clamp(waste)}
```

- [ ] **Step 2: symbolic per-sector Transition + composite builder.**

```python
class SectorTransition(Transition):
    """Exact dynamics for one sector (the symbolic ceiling)."""
    def __init__(self, params):
        self.params = params
    def step(self, state, action):
        if action.name in ACTIONS:
            return state.__class__(sector_step(dict(state), action.name, self.params))
        return state.copy()


def make_sector_world(i):
    return World(name=f"sector{i}", description="one economic sector",
                 initial_state={f: 0 for f in FIELDS}, actions=ACTIONS,
                 transition=SectorTransition(SECTOR_PARAMS[i]))


def build_composite(k, child_transitions=None):
    """K-sector composite. child_transitions[i] overrides sector i's dynamics
    (used to plug in LEARNED transitions); default = exact symbolic."""
    children = {}
    for i in range(k):
        w = make_sector_world(i)
        if child_transitions is not None:
            w.transition = child_transitions[i]
        children[f"s{i}"] = w
    return CompositeWorld(name=f"econ{k}", children=children,
        aggregators=[Aggregator("total_output",
                     lambda kids: sum(c["output"] for c in kids.values()))])
```

- [ ] **Step 3: oracle next-state for a joint (state, sector, action)** and a quick self-test:

```python
def joint_oracle(joint, active, action, k):
    """Next joint state: update sector `active`, pass the rest through."""
    out = {ns: dict(slice_) for ns, slice_ in joint.items() if ns.startswith("s")}
    out[f"s{active}"] = sector_step(out[f"s{active}"], action, SECTOR_PARAMS[active])
    return out
```

Run an inline assert in `__main__` (Step added in Task 4): stepping
`build_composite(3)` with `Action("s1:produce")` equals `joint_oracle` for a
few random joints. This guards the symbolic ceiling = 1.0.

- [ ] **Step 4: commit** `git add experiments/e36_representations.py && git commit -m "e36: parametric sector world, exact composite, oracle"`

---

### Task 2: learners (MLP + 1-NN) and the data split

**Files:** Modify `experiments/e36_representations.py`

- [ ] **Step 1: generic numpy MLP + 1-NN** (E12 algorithm, generic dims):

```python
class MLP:
    def __init__(self, n_in, n_out, hidden, seed=0):
        rng = np.random.RandomState(seed)
        self.w1 = rng.randn(n_in, hidden) * 0.1; self.b1 = np.zeros(hidden)
        self.w2 = rng.randn(hidden, hidden) * 0.1; self.b2 = np.zeros(hidden)
        self.w3 = rng.randn(hidden, n_out) * 0.1; self.b3 = np.zeros(n_out)
    def forward(self, x):
        self.h1 = np.maximum(0, x @ self.w1 + self.b1)
        self.h2 = np.maximum(0, self.h1 @ self.w2 + self.b2)
        return self.h2 @ self.w3 + self.b3
    def train(self, x, y, epochs=2000, lr=1e-2):
        for _ in range(epochs):
            p = self.forward(x); g = 2*(p-y)/len(x)
            gw3 = self.h2.T@g; gh2 = (g@self.w3.T)*(self.h2>0)
            gw2 = self.h1.T@gh2; gh1 = (gh2@self.w2.T)*(self.h1>0); gw1 = x.T@gh1
            self.w3-=lr*gw3; self.b3-=lr*g.sum(0)
            self.w2-=lr*gw2; self.b2-=lr*gh2.sum(0)
            self.w1-=lr*gw1; self.b1-=lr*gh1.sum(0)
    def n_params(self):
        return sum(a.size for a in (self.w1,self.b1,self.w2,self.b2,self.w3,self.b3))

def knn_predict(train_x, train_y, q):
    d = ((train_x - q)**2).sum(1); return train_y[d.argmin()]
```

- [ ] **Step 2: encoders + data construction.** Joint encode = concat of each
  sector's `[stock,output,waste]/G` + action onehot + active-sector onehot.
  Sector encode = `[stock,output,waste]/G` + action onehot. Labels = next
  slice values (not normalized; round predictions to int and clamp for exact
  match).

  **Training joint set (the crux):** thin diagonal band. For each base value
  `v in 0..G` and each sector-perturbation, build joints where every sector's
  fields are within ±1 of `v` (so marginally each sector hits 0..G across the
  set, but the joint stays near the diagonal). For each such joint, for each
  `active in range(k)` and each action, emit a transition. This is the data
  ALL learned conditions train on.

  **Test joint set:** uniformly random joints over the full product (each
  field iid uniform 0..G), `active`/action random — overwhelmingly
  off-diagonal (unseen combinations). Fixed seed; ~400 test transitions.

  Provide `make_train(k, n_cap=None)` (optionally subsample to `n_cap` for the
  sample-efficiency leg) and `make_test(k)`.

- [ ] **Step 3: commit** `git add -A && git commit -m "e36: numpy MLP/1-NN learners and the marginal-cover/joint-novel data split"`

---

### Task 3: the three legs + main

**Files:** Modify `experiments/e36_representations.py`

- [ ] **Step 1: condition runners.** Each returns exact-match accuracy on the
  joint test set.
  - `eval_monolith(train, test, k, hidden)`: train one MLP on joint
    transitions; predict next active-slice (monolith predicts the WHOLE next
    joint = active updated + others copied; simplest: predict only the active
    sector's next slice given the joint encode, then exact-match that slice —
    document this scoring choice in a comment, it is the update the action
    actually performs).
  - `eval_knn(train, test, k)`: 1-NN on the same joint encode/label.
  - `eval_composite_learned(train, test, k, hidden_each)`: split `train` by
    active sector; train K per-sector MLPs on sector-encoded transitions;
    build the composite with learned `Transition`s that round/clamp the net's
    output; score by stepping the composite on each test case and exact-match
    the active slice (equivalently call the active child net).
  - `eval_composite_symbolic(test, k)`: build exact composite, step, exact
    match — must be 1.0 (assert).
  - Capacity fairness: set monolith `hidden` so its `n_params()` >= sum of the
    K child nets' params; record both counts.

- [ ] **Step 2: legs.**
  - `leg_generalization()`: for `k in (2,3,4,5)`, all four conditions on the
    marginal-cover/joint-novel split; record accuracy per condition per k.
  - `leg_interference()`: k=4. Monolith trained sequentially — for each sector
    in order, train the SAME net only on that sector's transitions; after the
    last, measure retained accuracy on sector 0's test transitions; also
    record a jointly-trained monolith and the composite (per-child, isolated)
    on the same sector-0 test. Report retained accuracy each.
  - `leg_sample_efficiency()`: k=3. For `n_cap in (100, 1000, 10000)`,
    composite-learned vs monolith accuracy on the joint test set; symbolic
    line at 1.0 / zero data for reference.

- [ ] **Step 3: main** — run all legs, `save_results("e36_representations", {...})`
  with per-leg detail + param counts + a `"sanity_symbolic_exact": bool`.
  Print three compact tables. Include the Task-1 oracle assert at startup.

- [ ] **Step 4: run it.** `python experiments/e36_representations.py`
  Expected: `composite_symbolic` = 1.00 everywhere (assert holds);
  `composite_learned` >> `monolith` on generalization with the gap widening
  in k; monolith shows interference (low retained sector-0 accuracy) while
  composite retains; composite reaches high accuracy at smaller `n_cap`.
  **Honest-results rule:** record whatever happens. If composite does NOT
  win a leg, keep the result and note it — do not tune toward a conclusion
  beyond making the symbolic sanity pass and the learners actually train
  (loss decreasing; if a net doesn't fit even its own training data, fix
  epochs/lr/hidden, that's a training bug not a result).

- [ ] **Step 5: commit** `git add -A && git commit -m "e36: three legs (generalization, interference, sample efficiency) + results"`

---

### Task 4: sanity test

**Files:** Create `tests/test_e36.py` (offline, fast — uses tiny k)

- [ ] **Step 1:** a fast test that imports the module, builds `build_composite(2)`,
  steps it, and asserts it matches `joint_oracle`; asserts an MLP's training
  loss decreases on a trivial fit. Keep k and epochs tiny so it runs in <2s.
  Run `python -m pytest tests/test_e36.py -q`; commit.
