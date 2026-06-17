"""E37c - CLEAN induction OOD recheck: gate on train-reproduction = 1.0.

E37b's induced-code numbers were unreliable because the LLM only reproduced 0.55-0.81 of its
training traces (restricting training to the held-out region thinned + skewed the examples, so
qwen2.5:7b couldn't recover the full rule set -- notably the post-ship `bugs += debt//4` branch).
That degraded induction, not the method.

This run controls induction quality:
  - branch-covering example selection: stratify shown examples by action and span the debt range
    (so the debt//4 branch is visible), instead of the first-40-distinct.
  - repro=1.0 gate: keep re-inducing (more attempts + seed retries) until the induced program
    reproduces ALL its training traces, or the budget is exhausted; only repro=1.0 programs are
    scored for OOD. We report how many replicates reached the gate.
Training data is still the held-out reachable TRAIN-REGION (extrap region excluded), so the
extrapolation test is honest. MLP/1-NN train on the identical train-region traces.

Question: does CORRECTLY-induced code (repro=1.0) extrapolate to the unseen reachable region while
the learned baselines collapse? Uses local Ollama qwen2.5:7b. Launch via bgjob --notify quome.
"""

import hashlib
import json
import random
from pathlib import Path
from statistics import mean

import numpy as np

from openworld import OllamaLLM
from openworld.parsing import extract_code
from openworld.state import Action
from common import (SPRINT_ACTIONS, SPRINT_INITIAL, SPRINT_PROBES, SPRINT_PROBES_SCALED,
                    require_ollama, sprint_ground_truth)
from e12_learned_baseline import FIELDS
from e37_induction import (INDUCE_SYSTEM, MODEL, probe_acc_code, probe_acc_knn,
                           probe_acc_mlp, reproduces, train_mlp)

KS = [1000]
REPLICATES = 3
SEED = 372
COORD = "shipped"
SHOW = 60               # distinct examples placed in the prompt (was 40)
SYNTH_ATTEMPTS = 8      # per induction call (was 4)
SEED_RETRIES = 4        # extra seeds to chase repro=1.0
RESULTS_DIR = Path(__file__).resolve().parent / "results"
ACTS = SPRINT_ACTIONS + ["noop"]


def reachable_pool(rng, n_traj=5000, length=30):
    trans = []
    for _ in range(n_traj):
        s = dict(SPRINT_INITIAL)
        for _ in range(length):
            a = rng.choice(ACTS)
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


def branch_covering(traces, n=SHOW):
    """Distinct (s,a) examples stratified by action, spanning the debt range so the
    post-ship bugs+=debt//4 branch is visible."""
    by_act = {}
    seen = set()
    for s, a, ns in traces:
        k = (skey(s), a)
        if k in seen:
            continue
        seen.add(k)
        by_act.setdefault(a, []).append((s, a, ns))
    # within each action, sort by debt so we sample low..high (debt//4 thresholds)
    for a in by_act:
        by_act[a].sort(key=lambda t: t[0]["debt"])
    out, ai = [], {a: 0 for a in by_act}
    acts = list(by_act)
    while len(out) < n and any(ai[a] < len(by_act[a]) for a in acts):
        for a in acts:
            if ai[a] < len(by_act[a]):
                # spread across the sorted list, not just the front
                idx = (ai[a] * 7) % len(by_act[a])
                out.append(by_act[a][idx])
                ai[a] += 1
                if len(out) >= n:
                    break
    return out


def induce_prompt(shown):
    lines = [f"State fields: {FIELDS}. Actions: {ACTS}.", "", "Observed transitions:"]
    for s, a, n in shown:
        lines.append(f"  state={s}, action={{'name': {a!r}}} -> {n}")
    lines += ["", "Write transition(state, action) consistent with ALL of these."]
    return "\n".join(lines)


def induce_gated(traces, base_seed):
    """Re-induce until repro=1.0 or budget exhausted; return (code, repro, attempts)."""
    from openworld.sandbox import run_transition_code  # noqa: F401 (used via reproduces)
    shown = branch_covering(traces)
    prompt = induce_prompt(shown)
    best_code, best_repro, total = None, -1.0, 0
    for retry in range(SEED_RETRIES + 1):
        llm = OllamaLLM(model=MODEL, temperature=0.4, options={"seed": base_seed + 1000 * retry})
        for _ in range(SYNTH_ATTEMPTS):
            total += 1
            code = extract_code(llm.ask(prompt, system=INDUCE_SYSTEM))
            repro = reproduces(code, traces)
            if repro > best_repro:
                best_code, best_repro = code, repro
            if repro == 1.0:
                return best_code, best_repro, total
    return best_code, best_repro, total


def main():
    require_ollama(MODEL)
    pool = reachable_pool(random.Random(SEED))
    interp, extrap, med = region_split(pool)

    train_tr, interp_tr, extrap_tr = [], [], []
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

    def P(trans, cap=120):
        return [(s, Action(a)) for s, a, _ in trans[:cap]]
    probes = {"in_region": P(in_held), "interp_ood": P(interp_tr), "extrap_ood": P(extrap_tr),
              "in_dist_orig": SPRINT_PROBES, "x10_old": SPRINT_PROBES_SCALED}
    print(f"[e37c] reachable distinct={len(set(skey(t[0]) for t in pool))} {COORD}-median={med}; "
          f"fit={len(fit_pool)} in_held={len(in_held)} interp={len(interp_tr)} extrap={len(extrap_tr)}")

    rows = []
    for k in KS:
        for rep in range(REPLICATES):
            r = random.Random(SEED + rep)
            traces = list(fit_pool); r.shuffle(traces); traces = traces[:k]
            code, repro, attempts = induce_gated(traces, SEED + 100 * rep)
            net = train_mlp(traces, seed=rep)
            row = {"k": k, "rep": rep, "induce_repro": round(repro, 3), "induce_attempts": attempts,
                   "gated_ok": repro == 1.0,
                   "code": {p: round(probe_acc_code(code, probes[p]), 3) for p in probes},
                   "mlp": {p: round(probe_acc_mlp(net, probes[p]), 3) for p in probes},
                   "knn1": {p: round(probe_acc_knn(traces, probes[p]), 3) for p in probes}}
            rows.append(row)
            c, m = row["code"], row["mlp"]
            print(f"  k={k} rep={rep} repro={repro:.2f} ({attempts} att, gate={'Y' if row['gated_ok'] else 'N'}) | "
                  f"CODE in={c['in_region']} interp={c['interp_ood']} extrap={c['extrap_ood']} x10={c['x10_old']} | "
                  f"MLP in={m['in_region']} interp={m['interp_ood']} extrap={m['extrap_ood']} x10={m['x10_old']}")

    gated = [rw for rw in rows if rw["gated_ok"]]
    pset = list(probes)
    agg_all = {mdl: {p: round(mean(rw[mdl][p] for rw in rows), 3) for p in pset} for mdl in ("code", "mlp", "knn1")}
    agg_gated = ({p: round(mean(rw["code"][p] for rw in gated), 3) for p in pset} if gated else None)
    out = {"experiment": "e37c_clean_induction", "model": MODEL, "ks": KS, "replicates": REPLICATES,
           "coord": COORD, "median": med, "show": SHOW, "rows": rows,
           "n_gated_repro1": len(gated), "aggregate_all": agg_all,
           "code_aggregate_gated_only": agg_gated}
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "e37c_clean_induction.json").write_text(json.dumps(out, indent=2))

    print(f"\n[e37c] {len(gated)}/{len(rows)} replicates hit repro=1.0.")
    print(f"  {'model':<14}{'in_reg':>8}{'interp':>8}{'extrap':>8}{'in_dist':>9}{'x10':>7}")
    for mdl in ("code", "mlp", "knn1"):
        a = agg_all[mdl]
        print(f"  {mdl+' (all)':<14}{a['in_region']:>8}{a['interp_ood']:>8}{a['extrap_ood']:>8}{a['in_dist_orig']:>9}{a['x10_old']:>7}")
    if agg_gated:
        a = agg_gated
        print(f"  {'code (repro=1)':<14}{a['in_region']:>8}{a['interp_ood']:>8}{a['extrap_ood']:>8}{a['in_dist_orig']:>9}{a['x10_old']:>7}")
    print("\nCLEAN TEST: if code(repro=1) extrap ~ 1.0 while MLP/KNN extrap ~ 0, the extrapolation claim holds for correctly-induced code.")


if __name__ == "__main__":
    main()
