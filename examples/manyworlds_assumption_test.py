"""Bayesian causal-assumption testing on OpenWorld's many-worlds store.

Idea: a causal DAG is a set of assumptions (which edges exist, their sign). Treat
each UNCERTAIN assumption as a parameter with a small set of candidate values, let
`Mechanism`s predict (discretized) outcomes from the data under those assumptions,
and feed observed transitions to a `WorldStore`. The store keeps the FACTORED
posterior over assumption-combinations without enumerating the joint, pruning every
combination that contradicts the data. The surviving posterior IS the set of causal
assumptions the data supports -- and the combinations it CANNOT separate are an
explicit read-out of (non)identifiability.

Grounded in the user's FELICITy causal-discovery DAG (dagitty format), whose key
perinatal chain is exactly Shapiro et al. (2013):

    Income(SDOH) -> Prenatal Stress -> Preterm(gest. age) -> low Bayley score
                    Smoking ----------^   (a second cause of preterm; decorrelator)
                    Prenatal Stress ----------------------> low Bayley? (direct?)

We test four uncertain edges and recover which the data supports. Outcomes are
binarised (cut-point) exactly like FELICITy's CUT_* developmental variables, which
is what makes the exact, discrete `Mechanism` matching the right tool.

Two paths are shown:
  (A) NATIVE exact version-space pruning (WorldStore) -- zero-noise data.
  (B) An additive SOFT-Bayesian scorer over the same Mechanisms with a flip-noise
      likelihood -- robust where hard pruning over-prunes. This extends the
      many-worlds idea (a likelihood weight per observation) WITHOUT touching core.

Run:  python examples/manyworlds_assumption_test.py
"""
from __future__ import annotations

import math
from itertools import product

from openworld import COUNTING, Mechanism, WorldStore

# ---------------------------------------------------------------------------
# 0. A compact dagitty parser -- DAGs really are the input (FELICITy uses this
#    exact .txt/.dot syntax: NODE [tags] and "A" -> "B").
# ---------------------------------------------------------------------------
def parse_dagitty(text):
    """Return (nodes: {name: [tags]}, edges: [(src, dst)]) from a dagitty block."""
    nodes, edges = {}, []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(("dag", "bb=", "}", "{")):
            continue
        if "->" in line:
            a, b = line.split("->", 1)
            edges.append((a.strip().strip('"'), b.strip().strip('"')))
        elif "[" in line:
            name = line[: line.index("[")].strip().strip('"')
            tags = line[line.index("[") + 1: line.rindex("]")]
            nodes[name] = [t.split("=")[0].strip() for t in tags.split(",")]
    return nodes, edges


# A small, on-point subgraph of the FELICITy stress DAG (dagitty syntax).
FELICITY_SUBDAG = """
dag {
"Income_low" [exposure]
"Prenatal_Stress" [exposure]
"Smoking" [exposure]
"Preterm" []
"Low_Bayley" [outcome]
"Income_low" -> "Prenatal_Stress"
"Prenatal_Stress" -> "Preterm"
"Smoking" -> "Preterm"
"Preterm" -> "Low_Bayley"
"Prenatal_Stress" -> "Low_Bayley"
}
"""

# ---------------------------------------------------------------------------
# 1. Uncertain assumptions = parameters with candidate values. Each questioned
#    edge from the DAG becomes a parameter the data will adjudicate.
# ---------------------------------------------------------------------------
PARAMS = {
    "A_income_stress":  ["causal", "null"],   # Income_low -> Prenatal_Stress ?
    "A_stress_preterm": ["causal", "null"],   # Prenatal_Stress -> Preterm ?
    "A_preterm_bayley": ["causal", "null"],   # Preterm -> Low_Bayley ?
    "A_stress_direct":  ["yes", "no"],        # direct Prenatal_Stress -> Low_Bayley ?
}
TRUTH = {"A_income_stress": "causal", "A_stress_preterm": "causal",
         "A_preterm_bayley": "causal", "A_stress_direct": "no"}


# ---------------------------------------------------------------------------
# 2. Mechanisms: each predicts ONE discretised observable from the current state
#    + the params in its scope. Scopes PARTITION the params (WorldStore's
#    factored-exact requirement: no parameter shared across scopes).
# ---------------------------------------------------------------------------
def stress_fn(state, action, p):
    return state["income_low"] if p["A_income_stress"] == "causal" else 0


def preterm_fn(state, action, p):
    if p["A_stress_preterm"] == "causal":
        return 1 if (state["stress_hi"] or state["smoking"]) else 0
    return state["smoking"]                       # stress has no effect; smoking does


def bayley_fn(state, action, p):
    pt = state["preterm"] if p["A_preterm_bayley"] == "causal" else 0
    st = state["stress_hi"] if p["A_stress_direct"] == "yes" else 0
    return 1 if (pt or st) else 0


MECHANISMS = [
    Mechanism("stress",  "stress_hi", ("A_income_stress",), stress_fn),
    Mechanism("preterm", "preterm",   ("A_stress_preterm",), preterm_fn),
    Mechanism("bayley",  "low_bayley", ("A_preterm_bayley", "A_stress_direct"), bayley_fn),
]


# ---------------------------------------------------------------------------
# 3. Ground-truth cohort generator (a tiny deterministic PRNG -> reproducible).
# ---------------------------------------------------------------------------
def cohort(n, flip=0.0, seed=1234):
    """Yield observed transitions under TRUTH; optionally flip each outcome w.p. flip."""
    s = seed
    def rnd():                                    # LCG -> [0,1)
        nonlocal s
        s = (1103515245 * s + 12345) & 0x7FFFFFFF
        return s / 0x7FFFFFFF
    rows = []
    for _ in range(n):
        income_low = 1 if rnd() < 0.5 else 0
        smoking = 1 if rnd() < 0.4 else 0
        st = {"income_low": income_low, "smoking": smoking}
        st["stress_hi"] = stress_fn(st, None, TRUTH)
        st["preterm"] = preterm_fn(st, None, TRUTH)
        low_bayley = bayley_fn(st, None, TRUTH)
        nxt = {"stress_hi": st["stress_hi"], "preterm": st["preterm"],
               "low_bayley": low_bayley}
        if flip:                                   # measurement noise on outcomes
            for k in nxt:
                if rnd() < flip:
                    nxt[k] = 1 - nxt[k]
        rows.append((dict(st), {"name": "observe"}, nxt))
    return rows


def show_marginals(title, marg):
    print(f"\n{title}")
    for k in PARAMS:
        post = marg(k)
        star = " <- TRUTH" if max(post, key=post.get) == TRUTH[k] and \
            post[TRUTH[k]] > 0.5 else ""
        cells = "  ".join(f"{v}:{post[v]:.2f}" for v in PARAMS[k])
        print(f"  {k:<18} {cells}{star}")


# ---------------------------------------------------------------------------
# (B) Additive soft-Bayesian scorer over the SAME mechanisms (noise-robust).
# ---------------------------------------------------------------------------
def soft_posterior(rows, eps=0.1):
    """Posterior over full assumption-combos under a per-outcome flip likelihood.

    P(combo | data) ~ prod_rows prod_obs [ (1-eps) if predicted==observed else eps ].
    Enumerates the 16-combo joint (small) -- a graded alternative to hard pruning.
    """
    keys = list(PARAMS)
    combos = [dict(zip(keys, vals)) for vals in product(*[PARAMS[k] for k in keys])]
    logw = []
    for c in combos:
        lw = 0.0
        for state, action, nxt in rows:
            for m in MECHANISMS:
                pred = m.fn(state, action, c)
                if pred is None:
                    continue
                match = (pred == nxt.get(m.observable))
                lw += (0.0 if match else 1.0) * 1.0  # accumulate mismatches
        logw.append(lw)
    # convert mismatch counts -> unnormalised posterior under flip prob eps
    raw = [math.exp(mismatch * math.log(eps / (1 - eps))) for mismatch in logw]
    z = sum(raw) or 1.0
    weights = [r / z for r in raw]

    def marg(param):
        i = keys.index(param)
        out = {v: 0.0 for v in PARAMS[param]}
        for c, w in zip(combos, weights):
            out[c[param]] += w
        return out
    return marg


if __name__ == "__main__":
    nodes, edges = parse_dagitty(FELICITY_SUBDAG)
    print("Parsed FELICITy sub-DAG (dagitty input):")
    print(f"  exposures: {[n for n, t in nodes.items() if 'exposure' in t]}")
    print(f"  outcome:   {[n for n, t in nodes.items() if 'outcome' in t]}")
    print(f"  {len(edges)} edges; {len(PARAMS)} of them treated as UNCERTAIN assumptions")
    print(f"  joint assumption space: {2*2*2*2} candidate causal models")

    rows = cohort(n=200, flip=0.0)

    # (A) NATIVE exact path: factored version-space pruning.
    store = WorldStore(PARAMS, MECHANISMS, semiring=COUNTING)
    print(f"\n[A] Exact version-space  (clean data, n={len(rows)})")
    print(f"  candidate models before data: {store.total_worlds()}")
    for state, action, nxt in rows:
        store.observe(state, action, nxt)
    print(f"  models still consistent after data: {store.count()}")
    show_marginals("  posterior marginals (uniform over survivors):", store.marginal)
    print("\n  Identifiability read-out: the surviving models differ ONLY in "
          "A_stress_direct\n  -> a direct Prenatal_Stress->Low_Bayley edge is NOT "
          "identifiable here, because\n     stress fully determines preterm (no "
          "stress-high / non-preterm stratum to\n     separate the direct path from "
          "the preterm-mediated one). This is the real\n     mediator-vs-direct "
          "problem FELICITy faces -- surfaced explicitly, not hidden.")

    # (B) Soft path on NOISY data, where hard pruning over-prunes.
    noisy = cohort(n=400, flip=0.12, seed=99)
    hard = WorldStore(PARAMS, MECHANISMS, semiring=COUNTING)
    for state, action, nxt in noisy:
        hard.observe(state, action, nxt)
    print(f"\n[B] Noisy data (flip=0.12, n={len(noisy)})")
    print(f"  exact version-space survivors: {hard.count()}  "
          f"(hard matching over-prunes -> truth can be wrongly eliminated)")
    show_marginals("  SOFT Bayesian posterior over the SAME mechanisms "
                   "(noise-robust):", soft_posterior(noisy, eps=0.12))
    print("\n  The soft posterior still concentrates on the true assumptions under "
          "noise,\n  recovering income->stress, stress->preterm, preterm->bayley -- "
          "while honestly\n  leaving the unidentifiable direct edge near 50/50. "
          "This is the natural\n  extension point: a likelihood-weighted semiring "
          "for WorldStore.")
