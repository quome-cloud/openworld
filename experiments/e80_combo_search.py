"""E80 combinatorial world search: among the 100 heterogeneous worlds (E68) -- which globally do
NOT transfer (E71: same-sector competence ~0.22) -- find a SUBSET that DOES generalize, using the
scaling-law coherence proxy as the search heuristic.

Finding such a subset is combinatorial (2^N subsets, NP-hard to brute force). The law makes it
tractable: world->world transfer competence is a cheap pairwise coherence signal; a coherent
subset (high mutual transfer) is predicted to generalize, so we grow subsets greedily by
coherence and CONFIRM by training a shared abstract policy on the subset minus one world and
evaluating the held-out world. A while loop tries coherence-ranked seeds and stops at the first
subset whose held-out competence clears a bar well above the heterogeneous baseline.

Offline, deterministic, CPU-only (reuses E71's abstract world interface). save_results before
asserts. Run: python3 e80_combo_search.py
"""

import json
import random
from pathlib import Path

import numpy as np

import e71_generalization as G
from common import save_results

RESULTS = G.RESULTS
THRESHOLD = 0.55      # held-out competence that counts as "generalizes" (baseline ~0.22)
MIN_SIZE, SIZE_CAP = 4, 9
MAX_SEEDS = 60        # coherence-ranked seed pairs to try before giving up
SEED = 808


def load_worlds():
    """Replicate E71's steerable-world loading (abstract interface per world)."""
    e70 = json.loads((RESULTS / "e70_world_bench.json").read_text())
    steer = {(r["sector"], r["world"]) for r in e70["per_world"] if r.get("controllable")}
    flat = []
    for sec in G.SECTORS:
        for f in sorted((G.RECIPES / sec).glob("*.json")):
            if (sec, f.stem) not in steer:
                continue
            spec = json.loads(f.read_text())
            step, s0, acts, _ = G.load_world(spec)
            if len(acts) < 2:
                continue
            target, direction, named = G.parse_task(spec, s0)
            if target is None:
                target, direction = G.most_variable_field(step, s0, acts, G.SEED), 1
            if target is None:
                continue
            idx = len(flat)
            g_rand, lohi = G.random_stats(step, s0, acts, target, direction, G.SEED + idx)
            g_plan = G.planner_return(step, s0, acts, target, direction, G.SEED + idx)
            if g_plan - g_rand < 1e-9:
                continue
            flat.append(dict(world=f.stem, sector=sec, step=step, s0=s0, acts=acts,
                             target=target, dir=direction, lohi=lohi,
                             g_rand=g_rand, g_plan=g_plan, n_actions=len(acts)))
    return flat


def train_q_multi(subset, max_a, episodes, seed):
    """A SHARED tabular Q over (target-bin x action-index), trained across all worlds in the
    subset (the abstract interface is shared; each world keeps its own binning)."""
    q = np.zeros((G.N_BINS, max_a))
    rng = random.Random(seed)
    alpha, gamma = 0.3, 0.9
    for ep in range(episodes):
        w = subset[ep % len(subset)]
        eps = max(0.05, 1.0 - ep / max(1, episodes * 0.7))
        s = dict(w["s0"])
        b = G._bin(float(s[w["target"]]), w["lohi"])
        for _ in range(G.H):
            valid = list(range(len(w["acts"])))
            ai = rng.choice(valid) if rng.random() < eps else max(valid, key=lambda i: q[b, i])
            ns = w["step"](s, w["acts"][ai])
            r = w["dir"] * (float(ns[w["target"]]) - float(s[w["target"]]))
            nb = G._bin(float(ns[w["target"]]), w["lohi"])
            q[b, ai] += alpha * (r + gamma * q[nb].max() - q[b, ai])
            s, b = ns, nb
    return q


def held_out_competence(subset, max_a, seed):
    """Train shared policy on subset minus each member; evaluate the held-out member. Mean
    competence = does a policy learned on the OTHER worlds generalize to an unseen member?"""
    comps = []
    for h in range(len(subset)):
        train = subset[:h] + subset[h + 1:]
        q = train_q_multi(train, max_a, G.Q_EPISODES, seed + h)
        u = subset[h]
        g = G.eval_q(u["step"], u["s0"], u["acts"], u["target"], u["dir"], u["lohi"], q)
        c = G.competence(g, u["g_rand"], u["g_plan"])
        if c is not None:
            comps.append(c)
    return float(np.mean(comps)) if comps else None


def main():
    flat = load_worlds()
    n = len(flat)
    max_a = max(w["n_actions"] for w in flat)
    print(f"[combo] {n} steerable heterogeneous worlds", flush=True)

    # ---- pairwise transfer matrix T[i,j] = competence of world-i's specialist on world-j ----
    for i, w in enumerate(flat):
        w["q"] = G.train_q(w["step"], w["s0"], w["acts"], w["target"], w["dir"], w["lohi"],
                           G.SEED + i, G.Q_EPISODES, max_a=max_a)
    T = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            u = flat[j]
            c = G.competence(G.eval_q(u["step"], u["s0"], u["acts"], u["target"], u["dir"],
                                      u["lohi"], flat[i]["q"]), u["g_rand"], u["g_plan"])
            if c is not None:
                T[i, j] = c
    sym = np.nan_to_num((T + T.T) / 2.0, nan=-1.0)   # symmetric coherence, missing -> -1

    def coherence(S):
        if len(S) < 2:
            return -1.0
        vals = [sym[i, j] for a, i in enumerate(S) for j in S[a + 1:]]
        return float(np.mean(vals))

    # global heterogeneous baseline: random subsets shouldn't generalize
    rng = random.Random(SEED)
    rand_comps = []
    for _ in range(8):
        S = rng.sample(range(n), 6)
        rand_comps.append(held_out_competence([flat[i] for i in S], max_a, SEED + sum(S)))
    rand_baseline = float(np.nanmean([c for c in rand_comps if c is not None]))
    print(f"[combo] random-subset held-out competence (baseline): {rand_baseline:.3f}", flush=True)

    # ---- coherence-guided combinatorial search: grow from high-coherence seed pairs ----
    pairs = sorted(((sym[i, j], i, j) for i in range(n) for j in range(i + 1, n)),
                   reverse=True)[:MAX_SEEDS]
    res = {"task": "combinatorial search for a generalizing subset of heterogeneous worlds",
           "n_worlds": n, "threshold": THRESHOLD, "random_baseline": round(rand_baseline, 3),
           "seeds_tried": 0, "found": None, "trace": []}

    found = None
    for s_i, (score, i0, j0) in enumerate(pairs):
        if found:
            break
        res["seeds_tried"] = s_i + 1
        S = [i0, j0]
        # greedily add the world that maximizes subset coherence
        while len(S) < SIZE_CAP:
            cand = max((c for c in range(n) if c not in S),
                       key=lambda c: coherence(S + [c]))
            if coherence(S + [cand]) < coherence(S) - 0.05:
                break
            S.append(cand)
            if len(S) >= MIN_SIZE:
                comp = held_out_competence([flat[k] for k in S], max_a, SEED + s_i)
                res["trace"].append({"size": len(S), "coherence": round(coherence(S), 3),
                                     "held_out_competence": round(comp, 3) if comp else None})
                if comp is not None and comp >= THRESHOLD:
                    found = {"worlds": [flat[k]["world"] for k in S],
                             "sectors": sorted({flat[k]["sector"] for k in S}),
                             "size": len(S), "coherence": round(coherence(S), 3),
                             "held_out_competence": round(comp, 3)}
                    break
    res["found"] = found
    save_results("e80_combo_search", res)

    if found:
        print(f"[combo] FOUND generalizing subset (n={found['size']}): {found['worlds']}\n"
              f"  sectors {found['sectors']} | coherence {found['coherence']} | "
              f"held-out competence {found['held_out_competence']} "
              f"(vs random baseline {rand_baseline:.3f})", flush=True)
        # self-check: the law-found subset generalizes well above random subsets
        assert found["held_out_competence"] >= THRESHOLD
        assert found["held_out_competence"] > rand_baseline + 0.15
    else:
        print(f"[combo] no generalizing subset found in {res['seeds_tried']} seeds "
              f"(baseline {rand_baseline:.3f})", flush=True)


if __name__ == "__main__":
    main()
