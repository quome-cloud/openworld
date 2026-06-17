"""E64 - Bayesian causal-assumption testing on the factored many-worlds store.

A causal DAG is a set of assumptions (which edges exist, the sign of an effect).
Treat each uncertain edge as a parameter with candidate values, let Mechanisms
predict binarised (cut-point) outcomes from their parents under those parameters,
and feed observed transitions to a WorldStore (E46). The factored version space
prunes every assumption-combination inconsistent with the data without ever
enumerating the joint -- so the surviving posterior is the set of causal
assumptions the data supports, and the combinations it cannot separate are an
explicit read-out of (non)identifiability.

Grounded in the FELICITy perinatal DAG (Income -> Prenatal Stress -> Preterm
gestational age -> low Bayley developmental score), the Shapiro et al. (2013)
psychosocial-stress -> preterm-birth chain. Outcomes are cut-points, as in the
FELICITy CUT_* variables.

Deterministic and offline (no LLM): a fixed-seed synthetic cohort drawn from a
known structure. Two paths:
  (A) exact version-space pruning collapses the candidate models and surfaces an
      unidentifiable edge as an even posterior;
  (B) an additive soft-Bayesian scorer over the SAME Mechanisms (flip-noise
      likelihood) stays robust where hard pruning over-prunes under measurement
      noise.
"""

import math
from itertools import product

from openworld import COUNTING, Mechanism, WorldStore

from common import save_results

# --- uncertain assumptions: each questioned edge is a parameter --------------
PARAMS = {
    "A_income_stress":  ["causal", "null"],   # Income_low -> Prenatal_Stress ?
    "A_stress_preterm": ["causal", "null"],   # Prenatal_Stress -> Preterm ?
    "A_preterm_bayley": ["causal", "null"],   # Preterm -> Low_Bayley ?
    "A_stress_direct":  ["yes", "no"],        # direct Prenatal_Stress -> Low_Bayley ?
}
TRUTH = {"A_income_stress": "causal", "A_stress_preterm": "causal",
         "A_preterm_bayley": "causal", "A_stress_direct": "no"}
IDENTIFIED = ["A_income_stress", "A_stress_preterm", "A_preterm_bayley"]


# --- mechanisms: one discretised observable per node; scopes PARTITION params -
def stress_fn(state, action, p):
    return state["income_low"] if p["A_income_stress"] == "causal" else 0


def preterm_fn(state, action, p):
    if p["A_stress_preterm"] == "causal":
        return 1 if (state["stress_hi"] or state["smoking"]) else 0
    return state["smoking"]


def bayley_fn(state, action, p):
    pt = state["preterm"] if p["A_preterm_bayley"] == "causal" else 0
    st = state["stress_hi"] if p["A_stress_direct"] == "yes" else 0
    return 1 if (pt or st) else 0


MECHANISMS = [
    Mechanism("stress",  "stress_hi", ("A_income_stress",), stress_fn),
    Mechanism("preterm", "preterm",   ("A_stress_preterm",), preterm_fn),
    Mechanism("bayley",  "low_bayley", ("A_preterm_bayley", "A_stress_direct"), bayley_fn),
]


def cohort(n, flip=0.0, seed=1234):
    """Observed transitions under TRUTH; optionally flip each outcome w.p. flip."""
    s = seed

    def rnd():
        nonlocal s
        s = (1103515245 * s + 12345) & 0x7FFFFFFF
        return s / 0x7FFFFFFF

    rows = []
    for _ in range(n):
        st = {"income_low": 1 if rnd() < 0.5 else 0,
              "smoking": 1 if rnd() < 0.4 else 0}
        st["stress_hi"] = stress_fn(st, None, TRUTH)
        st["preterm"] = preterm_fn(st, None, TRUTH)
        nxt = {"stress_hi": st["stress_hi"], "preterm": st["preterm"],
               "low_bayley": bayley_fn(st, None, TRUTH)}
        if flip:
            for k in nxt:
                if rnd() < flip:
                    nxt[k] = 1 - nxt[k]
        rows.append((dict(st), {"name": "observe"}, nxt))
    return rows


def soft_posterior(rows, eps):
    """Posterior over assumption-combos under a per-outcome flip likelihood."""
    keys = list(PARAMS)
    combos = [dict(zip(keys, vals)) for vals in product(*[PARAMS[k] for k in keys])]
    mism = []
    for c in combos:
        m = 0
        for state, action, nxt in rows:
            for mech in MECHANISMS:
                pred = mech.fn(state, action, c)
                if pred is not None and pred != nxt.get(mech.observable):
                    m += 1
        mism.append(m)
    raw = [math.exp(k * math.log(eps / (1 - eps))) for k in mism]
    z = sum(raw) or 1.0
    w = [r / z for r in raw]

    def marg(param):
        out = {v: 0.0 for v in PARAMS[param]}
        for c, wi in zip(combos, w):
            out[c[param]] += wi
        return out
    return marg


def main():
    n_models = 1
    for dom in PARAMS.values():
        n_models *= len(dom)

    # (A) exact version-space pruning on clean data
    clean_n, noise_flip, noisy_n = 200, 0.12, 400
    store = WorldStore(PARAMS, MECHANISMS, semiring=COUNTING)
    for state, action, nxt in cohort(clean_n, flip=0.0):
        store.observe(state, action, nxt)
    exact = {p: store.marginal(p) for p in PARAMS}
    survivors = store.count()

    # (B) noisy data: hard over-prunes, soft stays robust
    noisy = cohort(noisy_n, flip=noise_flip, seed=99)
    hard = WorldStore(PARAMS, MECHANISMS, semiring=COUNTING)
    for state, action, nxt in noisy:
        hard.observe(state, action, nxt)
    hard_survivors = hard.count()
    soft_marg = soft_posterior(noisy, eps=noise_flip)
    soft = {p: soft_marg(p) for p in PARAMS}

    soft_min_identified = min(soft[p][TRUTH[p]] for p in IDENTIFIED)
    direct_exact = exact["A_stress_direct"]
    direct_soft = soft["A_stress_direct"]

    payload = {
        "dag": "Income->Stress->Preterm->Bayley (+Smoking->Preterm, +direct Stress->Bayley?)",
        "params": {p: {"candidates": PARAMS[p], "truth": TRUTH[p],
                       "exact_posterior": exact[p], "soft_posterior": soft[p],
                       "identified": p in IDENTIFIED} for p in PARAMS},
        "n_candidate_models": n_models,
        "clean_cohort_n": clean_n,
        "survivors_clean": survivors,
        "n_identified_edges": len(IDENTIFIED),
        "noise_flip": noise_flip,
        "noisy_cohort_n": noisy_n,
        "hard_survivors_noisy": hard_survivors,
        "soft_min_identified_truth": round(soft_min_identified, 4),
        "direct_edge_exact_even": direct_exact,
        "direct_edge_soft_even": direct_soft,
    }
    save_results("e64_causal_assumptions", payload)

    print("E64 - Bayesian causal-assumption testing\n")
    print(f"  candidate causal models: {n_models}")
    print(f"  [A] exact version-space survivors (clean n={clean_n}): {survivors}")
    for p in PARAMS:
        tag = "identified" if p in IDENTIFIED else "UNIDENTIFIED"
        print(f"      {p:<18} exact={exact[p]}  ({tag})")
    print(f"  [B] hard survivors under {noise_flip:.0%} noise (n={noisy_n}): {hard_survivors}")
    print(f"      soft posterior min over identified edges = {soft_min_identified:.3f}")
    print(f"      direct edge soft posterior = {direct_soft}")

    # --- honest self-checks (save first) ---
    assert n_models == 16, n_models
    assert survivors == 2, f"clean pruning should leave the 2 models differing only in the unidentified edge, got {survivors}"
    for p in IDENTIFIED:
        assert exact[p][TRUTH[p]] == 1.0, f"{p} should be pinned to truth under exact pruning"
    assert abs(direct_exact["yes"] - 0.5) < 1e-9 and abs(direct_exact["no"] - 0.5) < 1e-9, \
        "the direct stress->bayley edge is non-identifiable -> even posterior"
    assert hard_survivors == 0, "hard version-space over-prunes under noise (truth eliminated)"
    assert soft_min_identified > 0.9, "soft posterior should still recover the identified edges under noise"
    assert abs(direct_soft["yes"] - 0.5) < 0.05, "soft posterior leaves the unidentifiable edge ~even"
    print("\nchecks pass: exact pruning recovers the identified edges and flags the "
          "non-identifiable one; the soft scorer stays robust where hard pruning fails.")


if __name__ == "__main__":
    main()
