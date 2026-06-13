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
# Data split (THE CRUX). Training = thin diagonal band: every sector's fields
# within +/-1 of a shared base value v in 0..G, so each sector MARGINALLY
# covers 0..G but the joint stays near the diagonal. Test = full random joint
# product (each field iid uniform 0..G), overwhelmingly off-diagonal => novel
# COMBINATIONS of per-sector values that were each individually seen.
# ---------------------------------------------------------------------------

def _band_joints(k, seed):
    """Joints near the diagonal: every field within +/-1 of a base value v."""
    rng = np.random.RandomState(seed)
    joints = []
    # For each base value v and a set of small perturbations, build joints
    # where every sector's every field is clamp(v + delta), delta in {-1,0,1}.
    for v in range(0, G + 1):
        for _ in range(8):  # several perturbed joints per base value
            joint = {}
            for i in range(k):
                joint[f"s{i}"] = {
                    f: clamp(v + int(rng.randint(-1, 2))) for f in FIELDS
                }
            joints.append(joint)
    return joints


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
    """Diagnostic: fraction of test joints whose sectors are NOT all near a
    common base value (i.e. genuinely novel combinations vs the train band)."""
    off = 0
    for tx in data:
        vals = [tx["joint"][ns][f] for ns in tx["joint"] for f in FIELDS]
        if max(vals) - min(vals) > 2:
            off += 1
    return off / len(data)
