"""E37b - Induction OOD recheck on a REACHABLE held-out region (fixes the x10 probe).

E37/E38 claim code-induction beats statistical learning OOD because the learned
baselines "collapse to zero OOD on every replicate" -- but OOD there = SPRINT_PROBES_SCALED,
the x10 magnitude probe, whose states are OFF the reachable manifold (e.g. shipped=70,
backlog=50 violates sprint's shipped+backlog=12 invariant). E64 showed that on a held-out
REACHABLE region the learned dynamics models do NOT uniformly collapse.

This reruns the SAME induction pipeline (e37's induce_from_traces + e12 MLP/1-NN) but:
  - trains every model on transitions from a held-out TRAIN-REGION of the reachable space
  - probes on: in-region (held out) / interp-OOD / extrap-OOD -- all ON-MANIFOLD
  - keeps the old x10 column for direct contrast
The original in-dist/x10 numbers are already in results/e37_induction.json; this adds the
reachable columns. Question: does the induced CODE stay exact on reachable OOD (i.e. did it
induce the true rules), and do MLP/1-NN still collapse, or was the "wide margin" a x10 artifact?

Uses local Ollama qwen2.5:7b. Launch via bgjob --notify quome.
"""

import hashlib
import json
import random
from pathlib import Path
from statistics import mean

import numpy as np

from openworld.state import Action
from common import (SPRINT_ACTIONS, SPRINT_INITIAL, SPRINT_PROBES, SPRINT_PROBES_SCALED,
                    require_ollama, sprint_ground_truth)
from e12_learned_baseline import FIELDS
from e37_induction import (MODEL, collect as e37_collect, induce_from_traces,
                           probe_acc_code, probe_acc_knn, probe_acc_mlp, train_mlp)
from openworld import OllamaLLM

KS = [100, 1000]
REPLICATES = 2
SEED = 371
COORD = "shipped"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def reachable_pool(rng, n_traj=4000, length=30):
    trans, acts = [], SPRINT_ACTIONS + ["noop"]
    for _ in range(n_traj):
        s = dict(SPRINT_INITIAL)
        for _ in range(length):
            a = rng.choice(acts)
            ns = sprint_ground_truth(dict(s), {"name": a, "params": {}, "agent": None})
            trans.append((dict(s), a, dict(ns)))
            s = ns
            if s["backlog"] == 0 and rng.random() < 0.3:
                s = dict(SPRINT_INITIAL)
    return trans


def skey(s):
    return tuple(s[f] for f in FIELDS)


def region_split(pool):
    states = {skey(s): s for s, _, _ in pool}
    med = sorted(st[COORD] for st in states.values())[len(states) // 2]
    interp, extrap = set(), set()
    for k, st in states.items():
        h = int.from_bytes(hashlib.sha1(repr(k).encode()).digest()[:4], "big")
        if h % 10 >= 7:
            interp.add(k)
        if st[COORD] > med:
            extrap.add(k)
    return interp, extrap, med


def as_probes(trans, cap=120):
    return [(s, Action(a)) for s, a, _ in trans[:cap]]


def main():
    require_ollama(MODEL)
    rng = random.Random(SEED)
    pool = reachable_pool(rng)
    interp, extrap, med = region_split(pool)

    train_tr, in_held, interp_tr, extrap_tr = [], [], [], []
    for t in pool:
        k = skey(t[0])
        in_i, in_e = k in interp, k in extrap
        if not in_i and not in_e:
            train_tr.append(t)
        if in_i:
            interp_tr.append(t)
        if in_e:
            extrap_tr.append(t)
    random.Random(SEED).shuffle(train_tr)
    split = int(0.85 * len(train_tr))
    fit_pool, in_held = train_tr[:split], train_tr[split:]

    probes = {
        "in_region": as_probes(in_held),
        "interp_ood": as_probes(interp_tr),
        "extrap_ood": as_probes(extrap_tr),
        "in_dist_orig": SPRINT_PROBES,        # paper's hand-picked in-dist
        "x10_old": SPRINT_PROBES_SCALED,      # paper's x10 OOD (off-manifold)
    }
    print(f"[e37b] reachable: {len(set(skey(t[0]) for t in pool))} distinct, "
          f"{COORD} median={med}; fit_pool={len(fit_pool)} in_held={len(in_held)} "
          f"interp={len(interp_tr)} extrap={len(extrap_tr)}")

    def eval_all(code, net, traces):
        return {
            "code": {p: round(probe_acc_code(code, probes[p]), 3) for p in probes},
            "mlp": {p: round(probe_acc_mlp(net, probes[p]), 3) for p in probes},
            "knn1": {p: round(probe_acc_knn(traces, probes[p]), 3) for p in probes},
        }

    rows = []
    for k in KS:
        for rep in range(REPLICATES):
            r = random.Random(SEED + rep)
            traces = list(fit_pool)
            r.shuffle(traces)
            traces = traces[:k]
            llm = OllamaLLM(model=MODEL, temperature=0.4, options={"seed": SEED + rep})
            code, repro = induce_from_traces(llm, traces)
            net = train_mlp(traces, seed=rep)
            row = {"k": k, "rep": rep, "induce_repro": round(repro, 3), **eval_all(code, net, traces)}
            rows.append(row)
            c, m, n = row["code"], row["mlp"], row["knn1"]
            print(f"  k={k} rep={rep} repro={repro:.2f} | "
                  f"CODE in={c['in_region']} interp={c['interp_ood']} extrap={c['extrap_ood']} x10={c['x10_old']} | "
                  f"MLP in={m['in_region']} interp={m['interp_ood']} extrap={m['extrap_ood']} x10={m['x10_old']} | "
                  f"KNN interp={n['interp_ood']} extrap={n['extrap_ood']} x10={n['x10_old']}")

    # aggregate means per model per probe
    agg = {}
    for model in ("code", "mlp", "knn1"):
        agg[model] = {p: round(mean(rw[model][p] for rw in rows), 3) for p in probes}
    out = {"experiment": "e37b_reachable_induction", "model": MODEL, "ks": KS,
           "replicates": REPLICATES, "coord": COORD, "median": med,
           "rows": rows, "aggregate": agg}
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "e37b_reachable_induction.json").write_text(json.dumps(out, indent=2))

    print("\n[e37b] AGGREGATE (mean over k x rep):")
    print(f"  {'model':<8}{'in_reg':>8}{'interp':>8}{'extrap':>8}{'in_dist':>9}{'x10_old':>9}")
    for model in ("code", "mlp", "knn1"):
        a = agg[model]
        print(f"  {model:<8}{a['in_region']:>8}{a['interp_ood']:>8}{a['extrap_ood']:>8}"
              f"{a['in_dist_orig']:>9}{a['x10_old']:>9}")
    print("\nIf MLP/KNN interp/extrap >> 0 while x10=0, the 'collapse to zero OOD' claim was a x10 artifact.")


if __name__ == "__main__":
    main()
