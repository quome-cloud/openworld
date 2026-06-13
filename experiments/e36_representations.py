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
import numpy as np

# --- Guarded sklearn import -------------------------------------------------
# sklearn is OPTIONAL. The zero-dependency framework claim must hold: the
# experiment runs on numpy alone. If sklearn is absent, the sklearn-backed
# monolithic zoo arms (ridge/knn5/svr/gp/random_forest/hist_grad_boost) and
# the composite_hgb arm are skipped and recorded as null/absent. The pure-numpy
# "koopman" arm (EDMD, deg-2 lift) and the existing MLP/1-NN/composite arms
# always run.
try:
    from sklearn.ensemble import (
        HistGradientBoostingRegressor,
        RandomForestRegressor,
    )
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF
    from sklearn.linear_model import Ridge
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.svm import SVR
    SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when sklearn absent
    SKLEARN_AVAILABLE = False
    print("[e36] sklearn not installed; skipping sklearn zoo arms "
          "(ridge/knn5/svr/gp/random_forest/hist_grad_boost, composite_hgb). "
          "Pure-numpy koopman + MLP/1-NN/composite arms still run.")

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


# ---------------------------------------------------------------------------
# Symbolic per-sector Transition + composite builder (the framework machinery)
# ---------------------------------------------------------------------------

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


def joint_oracle(joint, active, action, k):
    """Next joint state: update sector `active`, pass the rest through."""
    out = {ns: dict(slice_) for ns, slice_ in joint.items() if ns.startswith("s")}
    out[f"s{active}"] = sector_step(out[f"s{active}"], action, SECTOR_PARAMS[active])
    return out


# ---------------------------------------------------------------------------
# Learners: generic numpy MLP (E12 algorithm) + 1-NN
# ---------------------------------------------------------------------------

class MLP:
    """Two hidden layers, ReLU, MSE on next-state; trained full-batch (E12)."""

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
        """Full-batch gradient descent. Returns (first_loss, last_loss)."""
        first = last = None
        for ep in range(epochs):
            p = self.forward(x); g = 2 * (p - y) / len(x)
            gw3 = self.h2.T @ g; gh2 = (g @ self.w3.T) * (self.h2 > 0)
            gw2 = self.h1.T @ gh2; gh1 = (gh2 @ self.w2.T) * (self.h1 > 0); gw1 = x.T @ gh1
            self.w3 -= lr * gw3; self.b3 -= lr * g.sum(0)
            self.w2 -= lr * gw2; self.b2 -= lr * gh2.sum(0)
            self.w1 -= lr * gw1; self.b1 -= lr * gh1.sum(0)
            loss = float(((p - y) ** 2).mean())
            if ep == 0:
                first = loss
            last = loss
        return first, last

    def n_params(self):
        return sum(a.size for a in (self.w1, self.b1, self.w2, self.b2, self.w3, self.b3))


def knn_predict(train_x, train_y, q):
    d = ((train_x - q) ** 2).sum(1)
    return train_y[d.argmin()]


# ---------------------------------------------------------------------------
# Encoders. A "transition" is (joint, active, action, next_active_slice).
#   joint encode  = concat per-sector [stock,output,waste]/G + action onehot
#                   + active-sector onehot          (the MONOLITH's input)
#   sector encode = [stock,output,waste]/G + action onehot   (a CHILD's input)
# Labels = next active-slice values (raw ints; predictions round+clamp for
# exact match). The update the action actually performs is to the active
# sector; every condition is scored on exact-matching that next slice.
# ---------------------------------------------------------------------------

def action_onehot(action):
    return [1.0 if action == a else 0.0 for a in ACTIONS]


def sector_encode(slice_, action):
    return [slice_[f] / G for f in FIELDS] + action_onehot(action)


def joint_encode(joint, active, action, k):
    vec = []
    for i in range(k):
        vec += [joint[f"s{i}"][f] / G for f in FIELDS]
    vec += action_onehot(action)
    vec += [1.0 if i == active else 0.0 for i in range(k)]
    return vec


def slice_label(next_slice):
    return [float(next_slice[f]) for f in FIELDS]


# ---------------------------------------------------------------------------
# Data split (THE CRUX). The "joint product" whose novel combinations we test
# is the CROSS-SECTOR product: which sector-states co-occur. Training is a thin
# diagonal slice of THAT product - all sectors hold the SAME local slice - so
# every CHILD individually sees its FULL local space (all stock/output/waste
# combinations, hence the rule is fully coverable and the composite reaches the
# ceiling) while the MONOLITH only ever sees joints where the sectors are in
# identical local states. The test draws each sector's local slice iid uniform
# over the full grid, so cross-sector combinations are overwhelmingly novel.
# (Restricting a single sector's local field RANGES instead would sink the
# child nets too and make the experiment meaningless - we do NOT do that.)
# ---------------------------------------------------------------------------

def _local_slices(seed):
    """All single-sector local slices over the full local grid 0..G^3,
    shuffled. This is the FULL marginal coverage every child gets."""
    slices = [{"stock": a, "output": b, "waste": c}
              for a in range(G + 1) for b in range(G + 1) for c in range(G + 1)]
    np.random.RandomState(seed).shuffle(slices)
    return slices


def _band_joints(k, seed):
    """Diagonal slice of the cross-sector product: every sector holds the SAME
    local slice, so each sector marginally covers its FULL local space while the
    joint stays on the cross-sector diagonal (sectors never differ)."""
    return [{f"s{i}": dict(sl) for i in range(k)} for sl in _local_slices(seed)]


def make_train(k, n_cap=None, seed=11):
    """Marginal-cover / joint-novel training transitions.

    Each transition: (joint, active, action, next_active_slice). Returned as a
    list of dicts so condition runners can build their own encodings.
    """
    joints = _band_joints(k, seed)
    data = []
    for joint in joints:
        for active in range(k):
            for action in ACTIONS:
                nxt = sector_step(dict(joint[f"s{active}"]), action, SECTOR_PARAMS[active])
                data.append({"joint": {f"s{i}": dict(joint[f"s{i}"]) for i in range(k)},
                             "active": active, "action": action, "next_slice": nxt})
    rng = np.random.RandomState(seed + 1)
    rng.shuffle(data)
    if n_cap is not None:
        data = data[:n_cap]
    return data


def make_test(k, n=400, seed=99):
    """Uniformly random joints over the full product (mostly off-diagonal)."""
    rng = np.random.RandomState(seed)
    data = []
    for _ in range(n):
        joint = {f"s{i}": {f: int(rng.randint(0, G + 1)) for f in FIELDS} for i in range(k)}
        active = int(rng.randint(0, k))
        action = ACTIONS[int(rng.randint(0, len(ACTIONS)))]
        nxt = sector_step(dict(joint[f"s{active}"]), action, SECTOR_PARAMS[active])
        data.append({"joint": joint, "active": active, "action": action, "next_slice": nxt})
    return data


def fraction_off_diagonal(data):
    """Diagnostic: fraction of joints that are OFF the cross-sector diagonal,
    i.e. where the sectors do NOT all hold the same local slice. The training
    band is on-diagonal (all sectors identical); the test is iid-per-sector, so
    almost all of it is off-diagonal (novel cross-sector combinations)."""
    off = 0
    for tx in data:
        slices = [tuple(tx["joint"][ns][f] for f in FIELDS) for ns in sorted(tx["joint"])]
        if len(set(slices)) > 1:   # sectors differ => off the diagonal
            off += 1
    return off / len(data)


# ---------------------------------------------------------------------------
# Learned per-sector Transition: wraps a trained MLP, rounds+clamps to a slice
# ---------------------------------------------------------------------------

class LearnedSectorTransition(Transition):
    """Plugs a trained per-sector MLP into a CompositeWorld child."""
    def __init__(self, mlp):
        self.mlp = mlp

    def step(self, state, action):
        if action.name not in ACTIONS:
            return state.copy()
        x = np.array([sector_encode(dict(state), action.name)])
        y = self.mlp.forward(x)[0]
        return state.__class__({f: clamp(int(round(v))) for f, v in zip(FIELDS, y)})


def _pred_slice(y):
    return {f: clamp(int(round(v))) for f, v in zip(FIELDS, y)}


def _child_hidden_and_params(hidden_each):
    """Params of one child MLP with the given hidden width (sector dims)."""
    n_in = len(FIELDS) + len(ACTIONS)
    n_out = len(FIELDS)
    return MLP(n_in, n_out, hidden_each).n_params()


def _monolith_hidden_for_capacity(k, hidden_each, child_total_params):
    """Smallest hidden width whose monolith n_params() >= child_total_params."""
    n_in = k * len(FIELDS) + len(ACTIONS) + k
    n_out = len(FIELDS)
    h = 2
    while MLP(n_in, n_out, h).n_params() < child_total_params:
        h += 2
    return h


# ---------------------------------------------------------------------------
# Condition runners. Each returns exact-match accuracy on the joint test set.
# Scoring is always on the NEXT ACTIVE SLICE (the update the action performs).
# ---------------------------------------------------------------------------

def eval_monolith(train, test, k, hidden, seed=0, epochs=2500, lr=1e-2):
    """One MLP on joint (state, active, action) -> next active slice."""
    x = np.array([joint_encode(tx["joint"], tx["active"], tx["action"], k) for tx in train])
    y = np.array([slice_label(tx["next_slice"]) for tx in train])
    mlp = MLP(x.shape[1], y.shape[1], hidden, seed=seed)
    first, last = mlp.train(x, y, epochs=epochs, lr=lr)
    hits = 0
    for tx in test:
        q = np.array([joint_encode(tx["joint"], tx["active"], tx["action"], k)])
        if _pred_slice(mlp.forward(q)[0]) == tx["next_slice"]:
            hits += 1
    return {"acc": hits / len(test), "train_loss_first": first, "train_loss_last": last,
            "n_params": mlp.n_params(), "hidden": hidden}


def eval_knn(train, test, k):
    """1-NN on the joint encode/label."""
    tx_x = np.array([joint_encode(tx["joint"], tx["active"], tx["action"], k) for tx in train])
    tx_y = [tx["next_slice"] for tx in train]
    hits = 0
    for tx in test:
        q = np.array(joint_encode(tx["joint"], tx["active"], tx["action"], k))
        nxt = tx_y[int(((tx_x - q) ** 2).sum(1).argmin())]
        if nxt == tx["next_slice"]:
            hits += 1
    return {"acc": hits / len(test)}


# ---------------------------------------------------------------------------
# Monolithic learner ZOO (generalization leg only). Every arm trains on the
# JOINT (state, active, action) encode -> next-active-slice label, and is scored
# by the IDENTICAL round+clamp+exact-full-slice-match protocol as eval_monolith
# (via _pred_slice). sklearn arms are guarded; "koopman" is pure numpy.
# ---------------------------------------------------------------------------

def _joint_xy(train, k):
    x = np.array([joint_encode(tx["joint"], tx["active"], tx["action"], k) for tx in train])
    y = np.array([slice_label(tx["next_slice"]) for tx in train])
    return x, y


def _score_regressor(model, train, test, k, extra=None):
    """Fit a multi-output regressor on the joint encode, score by exact match."""
    x, y = _joint_xy(train, k)
    model.fit(x, y)
    xt = np.array([joint_encode(tx["joint"], tx["active"], tx["action"], k) for tx in test])
    preds = model.predict(xt)
    preds = np.asarray(preds)
    if preds.ndim == 1:                 # single-output safety (not expected here)
        preds = preds[:, None]
    hits = 0
    for p, tx in zip(preds, test):
        if _pred_slice(p) == tx["next_slice"]:
            hits += 1
    out = {"acc": hits / len(test)}
    if extra:
        out.update(extra)
    return out


def _absent(reason="sklearn not installed"):
    return {"acc": None, "absent": True, "note": reason}


def eval_ridge(train, test, k):
    if not SKLEARN_AVAILABLE:
        return _absent()
    return _score_regressor(Ridge(), train, test, k)


def eval_knn5(train, test, k):
    if not SKLEARN_AVAILABLE:
        return _absent()
    return _score_regressor(KNeighborsRegressor(n_neighbors=5), train, test, k,
                            extra={"n_neighbors": 5})


def eval_svr(train, test, k):
    if not SKLEARN_AVAILABLE:
        return _absent()
    # SVR is single-output; wrap per-output.
    return _score_regressor(MultiOutputRegressor(SVR()), train, test, k)


def eval_gp(train, test, k):
    if not SKLEARN_AVAILABLE:
        return _absent()
    # GaussianProcessRegressor natively supports multi-output y (no wrapper).
    gp = GaussianProcessRegressor(kernel=RBF(), random_state=0, normalize_y=True)
    return _score_regressor(gp, train, test, k)


def eval_random_forest(train, test, k):
    if not SKLEARN_AVAILABLE:
        return _absent()
    rf = RandomForestRegressor(n_estimators=100, random_state=0)
    return _score_regressor(rf, train, test, k, extra={"n_estimators": 100})


def eval_hist_grad_boost(train, test, k):
    if not SKLEARN_AVAILABLE:
        return _absent()
    # HistGradientBoostingRegressor is single-output; wrap per-output.
    hgb = MultiOutputRegressor(HistGradientBoostingRegressor(random_state=0))
    return _score_regressor(hgb, train, test, k)


# --- Pure-numpy Koopman / Extended DMD (deg-2 lift) -------------------------

def _poly2_lift(x):
    """Degree-2 polynomial feature lift in pure numpy: a bias column, the raw
    features x, and all pairwise products (incl. squares) x_i*x_j (i<=j).
    Shape (n, 1 + d + d(d+1)/2). This is the EDMD observable dictionary."""
    x = np.asarray(x, dtype=float)
    n, d = x.shape
    cols = [np.ones((n, 1)), x]
    for i in range(d):
        cols.append(x[:, i:i + 1] * x[:, i:])     # x_i * x_j for j>=i (incl. square)
    return np.hstack(cols)


def eval_koopman(train, test, k):
    """Koopman / EDMD (deg-2). Lift the joint encode with degree-2 polynomial
    observables, fit a linear operator K by least squares lift(X) @ K ~= Y, then
    predict lift(Xtest) @ K and round+clamp. Pure numpy - the no-sklearn arm."""
    x, y = _joint_xy(train, k)
    phi = _poly2_lift(x)
    K, *_ = np.linalg.lstsq(phi, y, rcond=None)   # (n_lift, n_out)
    xt = np.array([joint_encode(tx["joint"], tx["active"], tx["action"], k) for tx in test])
    preds = _poly2_lift(xt) @ K
    hits = sum(1 for p, tx in zip(preds, test) if _pred_slice(p) == tx["next_slice"])
    return {"acc": hits / len(test), "lift": "poly_deg2", "n_lift": phi.shape[1]}


# --- Composite with a non-MLP (HGB) child -----------------------------------

class LearnedSectorRegressor(Transition):
    """Plugs a fitted multi-output sklearn regressor into a CompositeWorld child,
    rounding+clamping its prediction to an integer slice (same protocol as the
    MLP-backed LearnedSectorTransition)."""
    def __init__(self, model):
        self.model = model

    def step(self, state, action):
        if action.name not in ACTIONS:
            return state.copy()
        x = np.array([sector_encode(dict(state), action.name)])
        y = np.asarray(self.model.predict(x))[0]
        return state.__class__({f: clamp(int(round(v))) for f, v in zip(FIELDS, y)})


def eval_composite_hgb(train, test, k):
    """Composite whose per-sector child is a HistGradientBoostingRegressor
    (wrapped multi-output) instead of an MLP. Same per-sector training split and
    same compose-and-score protocol as eval_composite_learned. Guarded."""
    if not SKLEARN_AVAILABLE:
        return _absent()
    children = []
    for i in range(k):
        sub = [tx for tx in train if tx["active"] == i]
        model = MultiOutputRegressor(HistGradientBoostingRegressor(random_state=0))
        if sub:
            x = np.array([sector_encode(tx["joint"][f"s{i}"], tx["action"]) for tx in sub])
            yy = np.array([slice_label(tx["next_slice"]) for tx in sub])
            model.fit(x, yy)
        children.append(LearnedSectorRegressor(model))
    comp = build_composite(k, child_transitions=children)
    hits = 0
    for tx in test:
        st = comp.initial_state.copy()
        for i in range(k):
            st[f"s{i}"] = dict(tx["joint"][f"s{i}"])
        out = comp.transition.step(st, Action(f's{tx["active"]}:{tx["action"]}'))
        if dict(out[f's{tx["active"]}']) == tx["next_slice"]:
            hits += 1
    return {"acc": hits / len(test), "child": "hist_grad_boost"}


def train_child_mlps(train, k, hidden_each, seed=0, epochs=2500, lr=1e-2):
    """Split train by active sector; train one small MLP per sector on its own
    sector-local transitions. Returns (mlps, per_child_loss, total_params)."""
    mlps, losses = [], []
    for i in range(k):
        sub = [tx for tx in train if tx["active"] == i]
        mlp = MLP(len(FIELDS) + len(ACTIONS), len(FIELDS), hidden_each, seed=seed + i)
        if sub:
            x = np.array([sector_encode(tx["joint"][f"s{i}"], tx["action"]) for tx in sub])
            y = np.array([slice_label(tx["next_slice"]) for tx in sub])
            first, last = mlp.train(x, y, epochs=epochs, lr=lr)
            losses.append({"sector": i, "n": len(sub), "loss_first": first, "loss_last": last})
        else:
            losses.append({"sector": i, "n": 0, "loss_first": None, "loss_last": None})
        mlps.append(mlp)
    total = sum(m.n_params() for m in mlps)
    return mlps, losses, total


def eval_composite_learned(train, test, k, hidden_each, seed=0, epochs=2500, lr=1e-2):
    """K per-sector MLPs wired through a real CompositeWorld; score the active
    slice after stepping the composite (exercises the actual machinery)."""
    mlps, losses, total = train_child_mlps(train, k, hidden_each, seed, epochs, lr)
    comp = build_composite(k, child_transitions=[LearnedSectorTransition(m) for m in mlps])
    hits = 0
    for tx in test:
        st = comp.initial_state.copy()
        for i in range(k):
            st[f"s{i}"] = dict(tx["joint"][f"s{i}"])
        out = comp.transition.step(st, Action(f's{tx["active"]}:{tx["action"]}'))
        if dict(out[f's{tx["active"]}']) == tx["next_slice"]:
            hits += 1
    return {"acc": hits / len(test), "child_losses": losses, "n_params": total,
            "hidden_each": hidden_each}


def eval_composite_symbolic(test, k):
    """Exact composite; must be 1.0 (oracle ceiling)."""
    comp = build_composite(k)
    hits = 0
    for tx in test:
        st = comp.initial_state.copy()
        for i in range(k):
            st[f"s{i}"] = dict(tx["joint"][f"s{i}"])
        out = comp.transition.step(st, Action(f's{tx["active"]}:{tx["action"]}'))
        if dict(out[f's{tx["active"]}']) == tx["next_slice"]:
            hits += 1
    return {"acc": hits / len(test)}


# ---------------------------------------------------------------------------
# Sizing: pick child hidden, then size monolith for >= capacity fairness.
# ---------------------------------------------------------------------------

HIDDEN_EACH = 32


def sizing(k, hidden_each=HIDDEN_EACH):
    child_total = k * _child_hidden_and_params(hidden_each)
    mono_hidden = _monolith_hidden_for_capacity(k, hidden_each, child_total)
    mono_params = MLP(k * len(FIELDS) + len(ACTIONS) + k, len(FIELDS), mono_hidden).n_params()
    return {"hidden_each": hidden_each, "child_total_params": child_total,
            "monolith_hidden": mono_hidden, "monolith_params": mono_params}


# ---------------------------------------------------------------------------
# Leg 1: compositional generalization over k = 2..5
# ---------------------------------------------------------------------------

def leg_generalization():
    rows = []
    symbolic_exact = True
    for k in (2, 3, 4, 5):
        train, test = make_train(k), make_test(k)
        sz = sizing(k)
        mono = eval_monolith(train, test, k, sz["monolith_hidden"])
        knn = eval_knn(train, test, k)
        comp = eval_composite_learned(train, test, k, sz["hidden_each"])
        sym = eval_composite_symbolic(test, k)
        symbolic_exact = symbolic_exact and (sym["acc"] == 1.0)
        # Monolithic learner ZOO (joint-input, identical scoring as monolith).
        ridge = eval_ridge(train, test, k)
        knn5 = eval_knn5(train, test, k)
        svr = eval_svr(train, test, k)
        gp = eval_gp(train, test, k)
        rf = eval_random_forest(train, test, k)
        hgb = eval_hist_grad_boost(train, test, k)
        koopman = eval_koopman(train, test, k)
        comp_hgb = eval_composite_hgb(train, test, k)
        rows.append({
            "k": k, "n_train": len(train), "n_test": len(test),
            "test_off_diagonal": fraction_off_diagonal(test),
            "sizing": sz,
            # existing arms
            "monolith": mono, "knn1": knn,
            "composite_learned": comp, "composite_symbolic": sym,
            # monolithic zoo (sklearn-guarded except koopman which is pure numpy)
            "ridge": ridge, "knn5": knn5, "svr": svr, "gp": gp,
            "random_forest": rf, "hist_grad_boost": hgb, "koopman": koopman,
            # non-MLP composite (sklearn-guarded)
            "composite_hgb": comp_hgb,
        })
    return rows, symbolic_exact


# ---------------------------------------------------------------------------
# Leg 2: interference at k=4 (catastrophic forgetting in shared weights)
# ---------------------------------------------------------------------------

def leg_interference():
    k = 4
    train, test = make_train(k), make_test(k)
    sz = sizing(k)
    # test transitions that act on sector 0 (the "first task" we measure retention on)
    test0 = [tx for tx in test if tx["active"] == 0]
    if not test0:
        test0 = test  # safety; shouldn't happen with n=400

    # (a) Monolith trained SEQUENTIALLY: same net, one sector at a time, in order.
    n_in = k * len(FIELDS) + len(ACTIONS) + k
    seq = MLP(n_in, len(FIELDS), sz["monolith_hidden"], seed=0)
    per_stage = []
    for i in range(k):
        sub = [tx for tx in train if tx["active"] == i]
        x = np.array([joint_encode(tx["joint"], tx["active"], tx["action"], k) for tx in sub])
        y = np.array([slice_label(tx["next_slice"]) for tx in sub])
        first, last = seq.train(x, y, epochs=2500, lr=1e-2)
        # retained accuracy on sector-0 test after this stage
        h = sum(1 for tx in test0
                if _pred_slice(seq.forward(np.array([joint_encode(
                    tx["joint"], tx["active"], tx["action"], k)]))[0]) == tx["next_slice"])
        per_stage.append({"after_sector": i, "sector0_acc": h / len(test0),
                          "loss_first": first, "loss_last": last})
    seq_retained = per_stage[-1]["sector0_acc"]

    # (b) Jointly-trained monolith (all sectors at once) on the same sector-0 test.
    x = np.array([joint_encode(tx["joint"], tx["active"], tx["action"], k) for tx in train])
    y = np.array([slice_label(tx["next_slice"]) for tx in train])
    joint_mono = MLP(n_in, len(FIELDS), sz["monolith_hidden"], seed=0)
    joint_mono.train(x, y, epochs=2500, lr=1e-2)
    joint_mono_retained = sum(
        1 for tx in test0 if _pred_slice(joint_mono.forward(np.array([joint_encode(
            tx["joint"], tx["active"], tx["action"], k)]))[0]) == tx["next_slice"]) / len(test0)

    # (c) Composite: per-child isolated nets; sector-0 child is never touched by
    # later sectors' training, so retention is structural.
    comp_mlps, _, _ = train_child_mlps(train, k, sz["hidden_each"], seed=0)
    comp_child0 = comp_mlps[0]
    comp_retained = sum(
        1 for tx in test0
        if _pred_slice(comp_child0.forward(np.array([sector_encode(
            tx["joint"]["s0"], tx["action"])]))[0]) == tx["next_slice"]) / len(test0)

    # symbolic composite child-0 retention = 1.0 (no weights to overwrite)
    sym = eval_composite_symbolic(test0, k)

    return {
        "k": k, "n_test_sector0": len(test0), "sizing": sz,
        "monolith_sequential_retained": seq_retained,
        "monolith_sequential_stages": per_stage,
        "monolith_joint_retained": joint_mono_retained,
        "composite_learned_retained": comp_retained,
        "composite_symbolic_retained": sym["acc"],
    }


# ---------------------------------------------------------------------------
# Leg 3: sample efficiency at k=3
# ---------------------------------------------------------------------------

def leg_sample_efficiency():
    k = 3
    test = make_test(k)
    sz = sizing(k)
    full_train = make_train(k)  # used to know the max available
    rows = []
    for n_cap in (100, 1000, 10000):
        train = make_train(k, n_cap=n_cap)
        mono = eval_monolith(train, test, k, sz["monolith_hidden"])
        comp = eval_composite_learned(train, test, k, sz["hidden_each"])
        rows.append({
            "n_cap": n_cap, "n_train_actual": len(train),
            "monolith_acc": mono["acc"], "monolith_loss_last": mono["train_loss_last"],
            "composite_learned_acc": comp["acc"],
            "composite_symbolic_acc": 1.0,  # zero data; ceiling
        })
    return {"k": k, "n_test": len(test), "max_train_available": len(full_train),
            "sizing": sz, "rows": rows}


# ---------------------------------------------------------------------------
# Oracle self-test + main
# ---------------------------------------------------------------------------

def assert_symbolic_ceiling():
    """Guard: composite step == joint_oracle on random joints for k=2..5."""
    for k in (2, 3, 4, 5):
        rng = np.random.RandomState(7)
        comp = build_composite(k)
        for _ in range(40):
            joint = {f"s{i}": {f: int(rng.randint(0, G + 1)) for f in FIELDS} for i in range(k)}
            active = int(rng.randint(0, k))
            action = ACTIONS[int(rng.randint(0, len(ACTIONS)))]
            st = comp.initial_state.copy()
            for i in range(k):
                st[f"s{i}"] = dict(joint[f"s{i}"])
            out = comp.transition.step(st, Action(f"s{active}:{action}"))
            got = {f"s{i}": dict(out[f"s{i}"]) for i in range(k)}
            assert got == joint_oracle(joint, active, action, k), (k, active, action)


def main():
    assert_symbolic_ceiling()
    print("[oracle] symbolic composite == joint_oracle for k=2..5  OK\n")

    gen_rows, gen_symbolic_exact = leg_generalization()
    interference = leg_interference()
    efficiency = leg_sample_efficiency()

    sanity_symbolic_exact = bool(
        gen_symbolic_exact
        and interference["composite_symbolic_retained"] == 1.0
        and all(r["composite_symbolic_acc"] == 1.0 for r in efficiency["rows"])
    )

    # ---- tables ----
    def _fmt(arm):
        a = arm.get("acc")
        return f"{a:>7.3f}" if isinstance(a, (int, float)) else f"{'absent':>7}"

    zoo_arms = ["monolith", "ridge", "knn1", "knn5", "svr", "gp",
                "random_forest", "hist_grad_boost", "koopman"]
    print("LEG 1 - Compositional generalization (exact next-slice acc, joint-novel test)")
    print("  Monolithic ZOO (joint input) + composite arms. acc='absent' => sklearn missing.")
    header = "  " + f"{'k':>2} " + " ".join(f"{a[:8]:>8}" for a in zoo_arms)
    header += f"  {'cmp-mlp':>8} {'cmp-hgb':>8} {'cmp-sym':>8}"
    print(header)
    for r in gen_rows:
        line = "  " + f"{r['k']:>2} " + " ".join(_fmt(r[a]) for a in zoo_arms)
        line += f"  {_fmt(r['composite_learned']):>8} {_fmt(r['composite_hgb']):>8} "
        line += f"{_fmt(r['composite_symbolic']):>8}"
        print(line)

    print("\nLEG 2 - Interference at k=4 (retained sector-0 acc)")
    print(f"  monolith sequential : {interference['monolith_sequential_retained']:.3f}")
    print(f"  monolith joint      : {interference['monolith_joint_retained']:.3f}")
    print(f"  composite learned   : {interference['composite_learned_retained']:.3f}")
    print(f"  composite symbolic  : {interference['composite_symbolic_retained']:.3f}")
    print("  sequential stages   : " + ", ".join(
        f"after s{s['after_sector']}={s['sector0_acc']:.2f}"
        for s in interference["monolith_sequential_stages"]))

    print("\nLEG 3 - Sample efficiency at k=3 (exact next-slice acc)")
    print(f"  {'n_cap':>6} {'n_actual':>9} {'monolith':>9} {'comp-learn':>11} {'comp-sym':>9}")
    for r in efficiency["rows"]:
        print(f"  {r['n_cap']:>6} {r['n_train_actual']:>9} {r['monolith_acc']:>9.3f} "
              f"{r['composite_learned_acc']:>11.3f} {r['composite_symbolic_acc']:>9.3f}")

    print(f"\nsanity_symbolic_exact = {sanity_symbolic_exact}")

    save_results("e36_representations", {
        "grid_G": G, "fields": FIELDS, "actions": ACTIONS,
        "sector_params": SECTOR_PARAMS,
        "hidden_each": HIDDEN_EACH,
        "leg_generalization": gen_rows,
        "leg_interference": interference,
        "leg_sample_efficiency": efficiency,
        "sanity_symbolic_exact": sanity_symbolic_exact,
        "learner_families": {
            "sklearn_available": SKLEARN_AVAILABLE,
            "scope": ("The monolithic learner zoo + composite_hgb are added to the "
                      "generalization leg ONLY; interference and sample_efficiency "
                      "legs remain monolith-MLP vs composite."),
            "sklearn_optional_arms": [
                "ridge", "knn5", "svr", "gp", "random_forest",
                "hist_grad_boost", "composite_hgb",
            ],
            "pure_numpy_arms": ["monolith", "knn1", "koopman",
                                "composite_learned", "composite_symbolic"],
            "koopman_note": ("koopman is pure-numpy Extended DMD: degree-2 polynomial "
                             "lift (bias + raw + all pairwise products incl. squares), "
                             "linear operator fit by np.linalg.lstsq, predict+round+clamp."),
            "scoring": ("ALL monolithic arms (incl. continuous regressors) use the "
                        "IDENTICAL protocol: round-to-int, clamp to [0,G], exact "
                        "full-slice match - the same _pred_slice used by the MLP arm."),
            "absent_handling": ("If sklearn is missing, optional arms record "
                                "{'acc': null, 'absent': true}; the experiment still "
                                "runs on numpy alone (zero-dependency claim holds)."),
        },
        "note": (
            "Training covers each sector's full MARGINAL value range but only a "
            "thin diagonal slice of the JOINT product; test is the full random "
            "joint product (novel combinations of seen per-part values). "
            "Composite needs only marginals; monolith needs the joint."
        ),
    })


if __name__ == "__main__":
    main()
