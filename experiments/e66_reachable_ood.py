"""E65 - Stronger, on-manifold OOD probe for the world-model bake-off.

Fixes the methodological weakness in E63's OOD probe (scale every state field x10),
which produces OFF-MANIFOLD states the world never visits (e.g. sprint violates its
own invariant shipped+backlog=12). A learned model "failing" on impossible states is
uninformative.

Here OOD = a held-out region of the REACHABLE state space. We roll out the true
oracle from the initial state to collect the on-manifold transition pool, then split
DISTINCT source states two ways, train each learned model only on the train-region
transitions, and probe on the held-out regions:

  in-region   : held-out train-region states (generalization within the seen region)
  interp-OOD  : hash-held-out reachable states, scattered through the manifold
  extrap-OOD  : reachable states beyond the median of a progress coordinate
                (sprint: shipped, triage: tick) -- a contiguous unseen region

All three probe sets are states the oracle actually produces, so a model that LEARNED
the rule generalizes; a memorizer/local fitter fails. Verified code (CWM) is the
reference (exact by construction), not a contestant. 5 seeds, sprint + triage.
We also keep the old x10 column for direct comparison.
"""

import json
import random
from pathlib import Path
from statistics import mean

import numpy as np

from common import WORLD_SPECS
from e63_world_model_bakeoff import (CodeWM, KoopmanWM, LinearWM, MLPWM, NNWM,
                                     TabularWM, fields_of, mk_encode)

SEEDS = [0, 1, 2, 3, 4]
K = 10000
DOMAINS = ["sprint", "triage"]
PROGRESS = {"sprint": "shipped", "triage": "tick"}   # monotone coordinate per world
TRAJ, TRAJ_LEN = 4000, 30
RESULTS_DIR = Path(__file__).resolve().parent / "results"
LEARNED = {"1-NN": NNWM, "tabular": TabularWM, "linear": LinearWM,
           "koopman": KoopmanWM, "MLP": MLPWM}


def reachable_transitions(oracle, initial, actions, rng, n_traj=TRAJ, length=TRAJ_LEN):
    """Collect on-manifold (s, a, ns) by rolling the true oracle from init."""
    trans = []
    for _ in range(n_traj):
        s = dict(initial)
        for _ in range(length):
            a = rng.choice(actions)
            ns = oracle(dict(s), {"name": a, "params": {}, "agent": None})
            trans.append((dict(s), a, dict(ns)))
            s = ns
    return trans


def skey(state, fields):
    return tuple(state[f] for f in fields)


def region_split(trans, fields, domain, rng):
    """Partition DISTINCT source states into train / interp-OOD / extrap-OOD.

    interp: deterministic per-state hash, 70/30 (scattered, on-manifold interpolation).
    extrap: progress coordinate > median -> held out (contiguous unseen region).
    A state can be in both OOD sets; train-region = in NEITHER OOD set.
    """
    states = {skey(s, fields): s for s, _, _ in trans}
    coord = PROGRESS[domain]
    vals = sorted(state[coord] for state in states.values())
    med = vals[len(vals) // 2]

    interp_ood, extrap_ood = set(), set()
    for k, state in states.items():
        # stable hash independent of run-to-run salting
        h = int.from_bytes(__import__("hashlib").sha1(repr(k).encode()).digest()[:4], "big")
        if h % 10 >= 7:
            interp_ood.add(k)
        if state[coord] > med:
            extrap_ood.add(k)
    return interp_ood, extrap_ood, coord, med


def probe_acc_on(model, probes, oracle):
    if not probes:
        return None
    hits = 0
    for s, a in probes:
        exp = oracle(dict(s), {"name": a, "params": {}, "agent": None})
        if model.predict_state(dict(s), a) == exp:
            hits += 1
    return round(hits / len(probes), 3)


def x10_probes(oracle, initial, actions, fields, rng, n=24):
    """The OLD off-manifold probe, for direct comparison."""
    probes, s = [], dict(initial)
    for _ in range(n):
        a = rng.choice(actions)
        probes.append(({f: s[f] * 10 for f in fields}, a))
        s = oracle(dict(s), {"name": a, "params": {}, "agent": None})
        if rng.random() < 0.3:
            s = dict(initial)
    return probes


def run_domain(domain):
    spec = WORLD_SPECS[domain]
    oracle, initial, actions = spec["oracle"], spec["initial"], spec["actions"]
    fields = fields_of(initial)
    enc = mk_encode(fields, actions)

    rng0 = random.Random(12345)
    pool = reachable_transitions(oracle, initial, actions, rng0)
    interp_ood, extrap_ood, coord, med = region_split(pool, fields, domain, rng0)

    # bucket transitions by region of their SOURCE state
    train_tr, interp_tr, extrap_tr, inregion_tr = [], [], [], []
    for t in pool:
        k = skey(t[0], fields)
        in_i, in_e = k in interp_ood, k in extrap_ood
        if not in_i and not in_e:
            train_tr.append(t)
        if in_i:
            interp_tr.append(t)
        if in_e:
            extrap_tr.append(t)
    # in-region probe: held-out train-region transitions (disjoint from training rows)

    out = {"coord": coord, "median": med, "n_distinct": len({skey(t[0], fields) for t in pool}),
           "n_train_region_tr": len(train_tr), "n_interp_ood_tr": len(interp_tr),
           "n_extrap_ood_tr": len(extrap_tr), "models": {}}

    for name, ctor in LEARNED.items():
        ins, interp, extrap, x10 = [], [], [], []
        for seed in SEEDS:
            rng = random.Random(seed)
            tr = list(train_tr)
            rng.shuffle(tr)
            split = int(0.85 * len(tr))
            fit, heldin = tr[:split][:K], tr[split:]
            model = (ctor(fit, enc, fields, actions, seed) if name == "MLP"
                     else ctor(fit, enc, fields, actions))
            ins.append(probe_acc_on(model, [(s, a) for s, a, _ in heldin[:200]], oracle))
            interp.append(probe_acc_on(model, [(s, a) for s, a, _ in interp_tr[:200]], oracle))
            extrap.append(probe_acc_on(model, [(s, a) for s, a, _ in extrap_tr[:200]], oracle))
            x10.append(probe_acc_on(model, x10_probes(oracle, initial, actions, fields, random.Random(2)), oracle))
        out["models"][name] = {
            "in_region": round(mean(ins), 3),
            "interp_ood": round(mean(interp), 3),
            "extrap_ood": round(mean(extrap), 3),
            "x10_old": round(mean(x10), 3),
        }
    # reference: verified code is exact by construction on every set
    code = CodeWM(oracle)
    out["models"]["verified code (reference)"] = {
        "in_region": probe_acc_on(code, [(s, a) for s, a, _ in train_tr[:200]], oracle),
        "interp_ood": probe_acc_on(code, [(s, a) for s, a, _ in interp_tr[:200]], oracle),
        "extrap_ood": probe_acc_on(code, [(s, a) for s, a, _ in extrap_tr[:200]], oracle),
        "x10_old": probe_acc_on(code, x10_probes(oracle, initial, actions, fields, random.Random(2)), oracle),
        "note": "exact by construction (oracle); reference, not a contestant",
    }
    return out


def main():
    results = {"experiment": "e66_reachable_ood", "k_trained": K, "seeds": SEEDS,
               "traj": TRAJ, "traj_len": TRAJ_LEN, "domains": {}}
    for dom in DOMAINS:
        results["domains"][dom] = run_domain(dom)

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "e66_reachable_ood.json").write_text(json.dumps(results, indent=2))

    order = list(LEARNED) + ["verified code (reference)"]
    for dom in DOMAINS:
        d = results["domains"][dom]
        print(f"\n[{dom}]  coord={d['coord']} median={d['median']} "
              f"distinct={d['n_distinct']}  (train {d['n_train_region_tr']} / "
              f"interp {d['n_interp_ood_tr']} / extrap {d['n_extrap_ood_tr']} tr)")
        print(f"  {'model':<26}{'in':>7}{'interp':>9}{'extrap':>9}{'x10(old)':>10}")
        for name in order:
            m = d["models"][name]
            print(f"  {name:<26}{str(m['in_region']):>7}{str(m['interp_ood']):>9}"
                  f"{str(m['extrap_ood']):>9}{str(m['x10_old']):>10}")
    print("\nOOD now = held-out REACHABLE region (on-manifold), not x10 fantasy states.")


if __name__ == "__main__":
    main()
