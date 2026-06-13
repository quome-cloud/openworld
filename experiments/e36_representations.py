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
