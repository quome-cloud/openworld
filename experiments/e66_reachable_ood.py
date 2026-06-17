"""E66 - Stronger, on-manifold OOD probe for the world-model bake-off.

Fixes the methodological weakness in E63's OOD probe (scale every state field x10),
which produces OFF-MANIFOLD states the world never visits (e.g. sprint violates its
own invariant shipped+backlog=12). A learned model "failing" on impossible states is
uninformative.

Here OOD = a held-out region of the REACHABLE state space. We roll out the true
oracle from the initial state to collect the on-manifold transition pool, then split
DISTINCT source states into three DISJOINT regions using a progress coordinate
(sprint: shipped, triage: tick) and a per-state hash, train each learned model only
on the train-region transitions, and probe on the held-out regions:

  train region : coord <= median AND hash%10 <  7  -- the model only sees these
  in-region    : held-out train-region transitions (generalization within the seen region)
  interp-OOD   : coord <= median AND hash%10 >= 7  -- scattered holdout WITHIN the
                 seen coordinate range (on-manifold interpolation)
  extrap-OOD   : coord >  median                   -- a contiguous region beyond the
                 seen coordinate range (on-manifold extrapolation)

The three regions are DISJOINT by construction: extrap-OOD is exactly coord > median;
interp-OOD and the train region partition coord <= median by the hash bit, so no state
is ever in both interp-OOD and extrap-OOD (the earlier overlapping split contaminated
"interp" with extrapolation states). All three probe sets are states the oracle
actually produces, so a model that LEARNED the rule generalizes; a memorizer/local
fitter fails. Verified code (CWM) is the reference (exact by construction), not a
contestant. 5 seeds, sprint + triage. We also keep the old x10 column for comparison.

Caveat: triage's progress coordinate is `tick`, a MONOTONIC counter, so it has no
valid interpolation regime (distinct source states have distinct tick; 1-NN interp~0).
The triage domain output carries a machine-readable `interp_caveat`; sprint carries
the interpolation claim.
"""

import hashlib
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

TRIAGE_INTERP_CAVEAT = ("tick is a monotonic counter; triage has no valid "
                        "interpolation regime -- sprint carries the interpolation claim")


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
    """Partition DISTINCT source states into three DISJOINT regions.

    Let coord = the progress coordinate (sprint: shipped, triage: tick) and
    med = its median over the distinct source states.

      extrap_ood : coord >  med                   (contiguous unseen region)
      interp_ood : coord <= med AND hash%10 >= 7   (scattered on-manifold holdout)
      train      : coord <= med AND hash%10 <  7   (everything else; the only data
                                                    the learned models ever see)

    The hash is a deterministic per-state SHA-1 (independent of run-to-run salting),
    so interp_ood is a stable 30% holdout WITHIN the seen coordinate range. Because
    extrap_ood is defined purely by coord > med and interp_ood lives entirely in
    coord <= med, the three sets are pairwise disjoint -- "interp" can no longer be
    contaminated by extrapolation states.
    """
    states = {skey(s, fields): s for s, _, _ in trans}
    coord = PROGRESS[domain]
    vals = sorted(state[coord] for state in states.values())
    med = vals[len(vals) // 2]

    interp_ood, extrap_ood = set(), set()
    for k, state in states.items():
        if state[coord] > med:
            extrap_ood.add(k)
            continue  # extrapolation region -- never also interp
        # within the seen coordinate range: scatter a stable 30% holdout
        h = int.from_bytes(hashlib.sha1(repr(k).encode()).digest()[:4], "big")
        if h % 10 >= 7:
            interp_ood.add(k)
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


def shuffled_head(rows, n=200, seed=0):
    """Deterministically shuffle (fixed seed) then take the first n rows.

    The reachable pool is in trajectory order, so a raw [:n] slice has an
    early-rollout bias. A fixed-seed shuffle de-biases the probe sample while
    keeping the result reproducible."""
    r = list(rows)
    random.Random(seed).shuffle(r)
    return r[:n]


def run_domain(domain):
    spec = WORLD_SPECS[domain]
    oracle, initial, actions = spec["oracle"], spec["initial"], spec["actions"]
    fields = fields_of(initial)
    enc = mk_encode(fields, actions)

    rng0 = random.Random(12345)
    pool = reachable_transitions(oracle, initial, actions, rng0)
    interp_ood, extrap_ood, coord, med = region_split(pool, fields, domain, rng0)

    # bucket transitions by region of their SOURCE state (regions are disjoint)
    train_tr, interp_tr, extrap_tr = [], [], []
    for t in pool:
        k = skey(t[0], fields)
        if k in extrap_ood:
            extrap_tr.append(t)
        elif k in interp_ood:
            interp_tr.append(t)
        else:
            train_tr.append(t)
    # in-region probe: held-out train-region transitions (disjoint from training rows)

    # x10 probes are model-independent: compute once per domain, reuse for every
    # seed and for the verified-code row.
    x10_set = x10_probes(oracle, initial, actions, fields, random.Random(2))

    # probe samples are fixed (shuffled, de-biased) per region: shared across seeds
    interp_probe = [(s, a) for s, a, _ in shuffled_head(interp_tr, 200, seed=101)]
    extrap_probe = [(s, a) for s, a, _ in shuffled_head(extrap_tr, 200, seed=102)]

    out = {"coord": coord, "median": med, "n_distinct": len({skey(t[0], fields) for t in pool}),
           "n_train_region_tr": len(train_tr), "n_interp_ood_tr": len(interp_tr),
           "n_extrap_ood_tr": len(extrap_tr), "models": {}}
    if domain == "triage":
        out["interp_caveat"] = TRIAGE_INTERP_CAVEAT

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
            in_probe = [(s, a) for s, a, _ in shuffled_head(heldin, 200, seed=100)]
            ins.append(probe_acc_on(model, in_probe, oracle))
            interp.append(probe_acc_on(model, interp_probe, oracle))
            extrap.append(probe_acc_on(model, extrap_probe, oracle))
            x10.append(probe_acc_on(model, x10_set, oracle))
        out["models"][name] = {
            "in_region": round(mean(ins), 3),
            "interp_ood": round(mean(interp), 3),
            "extrap_ood": round(mean(extrap), 3),
            "x10_old": round(mean(x10), 3),
        }
    # reference: verified code is exact by construction on every set
    code = CodeWM(oracle)
    out["models"]["verified code (reference)"] = {
        "in_region": probe_acc_on(code, [(s, a) for s, a, _ in shuffled_head(train_tr, 200, seed=103)], oracle),
        "interp_ood": probe_acc_on(code, interp_probe, oracle),
        "extrap_ood": probe_acc_on(code, extrap_probe, oracle),
        "x10_old": probe_acc_on(code, x10_set, oracle),
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
        if "interp_caveat" in d:
            print(f"  caveat: {d['interp_caveat']}")
        print(f"  {'model':<26}{'in':>7}{'interp':>9}{'extrap':>9}{'x10(old)':>10}")
        for name in order:
            m = d["models"][name]
            print(f"  {name:<26}{str(m['in_region']):>7}{str(m['interp_ood']):>9}"
                  f"{str(m['extrap_ood']):>9}{str(m['x10_old']):>10}")
    print("\nOOD now = held-out REACHABLE region (on-manifold), not x10 fantasy states.")

    # ---- self-checks (convention: save_results BEFORE asserts) ----
    sprint = results["domains"]["sprint"]
    ref = sprint["models"]["verified code (reference)"]
    assert ref["in_region"] == 1.0 and ref["interp_ood"] == 1.0 and ref["extrap_ood"] == 1.0, \
        f"verified code must be exact on every reachable region: {ref}"
    mlp = sprint["models"]["MLP"]
    assert mlp["interp_ood"] > mlp["extrap_ood"], \
        f"sprint MLP should interpolate better than it extrapolates: {mlp}"
    nn_extrap = sprint["models"]["1-NN"]["extrap_ood"]
    assert nn_extrap <= 0.1, f"sprint 1-NN extrapolation should be ~0: {nn_extrap}"
    print("self-checks passed.")


if __name__ == "__main__":
    main()
