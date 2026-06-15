"""E51 - Startup growth world model: a YC-style batch and what drives growth.

Like the corporate world (E48) but for a cohort of startups: a batch is a
CompositeWorld of startup child-worlds, each evolving from its factors (team,
market/TAM, product-market fit, grit, capital). Growth is multiplicative in PMF
(no fit -> no growth); revenue compounds; burn depletes runway; traction-gated
fundraising extends it; a startup dies when it runs out of cash without revenue.
Batch quantities (total value created, survival rate, the top-decile share of
value) are Aggregators.

Four results, deterministic/offline:
  1. factors leading to growth (causal value-of-factor): lift each factor and
     measure the change in batch value/survival -> PMF dominates; capital alone
     barely moves it.
  2. power law of returns: a few startups make most of the value (the venture
     home-run dynamic).
  3. counterfactual attribution: a no-PMF batch collapses; doubling capital at
     low PMF does not - separating true drivers from spend.
  4. honest predictability: can month-6 traction pick the eventual biggest
     winner? Only weakly - the tail is luck-dominated (cf. E50).
"""

import numpy as np

from openworld import Aggregator, CompositeWorld, World
from openworld.transition import FunctionTransition

from common import save_results

N = 200                 # startups in the batch (a large YC cohort)
T = 36                  # months
SEED = 51
GMAX = 0.38            # max monthly growth rate
BURN = 20.0           # monthly burn ($k)
CASH0 = 150.0         # seed cash ($k)
REV0 = 2.0            # starting monthly revenue ($k)
MULT = 24            # exit multiple on monthly revenue
FACTORS = ["team", "market", "pmf", "grit", "capital"]


def sample_batch(seed=SEED):
    rng = np.random.RandomState(seed)
    return {
        "team": rng.uniform(0, 1, N),
        "market": rng.uniform(0, 1, N),
        "pmf": rng.uniform(0, 1, N),
        "grit": rng.uniform(0, 1, N),
        "capital": np.full(N, CASH0),          # seed cash; varied in counterfactuals
    }


def simulate(f, seed=SEED):
    """Vectorized monthly evolution of a batch (any size = len of the factor
    arrays). Returns final value, alive mask, and month-6 revenue."""
    n = len(f["pmf"])
    rng = np.random.RandomState(seed + 1)
    rev = np.full(n, REV0)
    cash = f["capital"].astype(float).copy()
    alive = np.ones(n, bool)
    rev6 = np.zeros(n)
    # growth driven by PMF (multiplicative gate) x market x team x grit
    g_base = GMAX * f["pmf"] * (0.35 + 0.65 * f["market"]) * \
        (0.5 + 0.5 * f["team"]) * (0.7 + 0.3 * f["grit"])
    for m in range(T):
        noise = 1 + rng.normal(0, 0.22, n)              # idiosyncratic execution luck
        g = np.clip(g_base * noise, -0.1, None)
        rev = np.where(alive, rev * (1 + g), rev)
        cash = np.where(alive, cash - BURN, cash)
        # traction-gated fundraise: real growth + fit buys a runway extension
        raise_mask = alive & (cash < 2 * BURN) & (rev > REV0 * 1.5) & (f["pmf"] > 0.45)
        cash = np.where(raise_mask, cash + 12 * rev, cash)
        # idiosyncratic shock: key-person loss / competitor / black swan, ~1%/mo,
        # independent of factors -> real tail luck (winners aren't obvious early)
        shock = alive & (rng.random(n) < 0.010)
        # death: out of cash and revenue can't cover burn, or a shock
        dead = (alive & (cash < 0) & (rev < BURN)) | shock
        rev = np.where(dead, 0.0, rev)
        alive = alive & ~dead
        # ramen-profitable: revenue covers burn -> survive on revenue
        cash = np.where(alive & (cash < 0) & (rev >= BURN), 0.0, cash)
        if m == 5:
            rev6 = rev.copy()
    value = np.where(alive, rev * MULT, 0.0)
    return value, alive, rev6


def gini(x):
    x = np.sort(np.maximum(0, x))
    n = len(x)
    s = x.sum()
    return float((2 * np.sum((np.arange(1, n + 1)) * x)) / (n * s) - (n + 1) / n) if s else 0.0


# --- structural showcase: the batch as a CompositeWorld ---------------------
def build_batch(value):
    kids = {f"s{i}": World(name=f"s{i}", description="startup",
                           initial_state={"value": float(value[i])}, actions=["tick"],
                           transition=FunctionTransition(lambda s, a: s))
            for i in range(N)}
    return CompositeWorld(
        name="batch", children=kids,
        aggregators=[Aggregator("total_value",
                                lambda k: sum(v["value"] for v in k.values()))])


def main():
    f = sample_batch()
    value, alive, rev6 = simulate(f)
    total = float(value.sum())

    # structural self-check: the value aggregator equals the explicit sum
    comp = build_batch(value)
    assert abs(comp.state["_agg"]["total_value"] - total) < 1e-6, "aggregator must match leaves"

    # 1. value-of-factor (causal): lift each factor to its 90th percentile for ALL
    #    startups, measure the change in total batch value.
    voa = {}
    for fac in FACTORS:
        g = {k: v.copy() for k, v in f.items()}
        if fac == "capital":
            g["capital"] = g["capital"] * 2.0           # double the money
        else:
            g[fac] = np.maximum(g[fac], np.quantile(g[fac], 0.9))
        v2, _, _ = simulate(g)
        voa[fac] = {"delta_value_pct": round(100 * (v2.sum() - total) / total, 1)}

    # 2. power law of returns
    sv = np.sort(value)[::-1]
    power = {"survival_rate": round(float(alive.mean()), 3),
             "top1_share": round(float(sv[0] / total), 3),
             "top_decile_share": round(float(sv[:N // 10].sum() / total), 3),
             "value_gini": round(gini(value), 3)}

    # 3. counterfactual attribution
    no_pmf = {**{k: v.copy() for k, v in f.items()}, "pmf": np.zeros(N)}
    cap2_lowpmf = {**{k: v.copy() for k, v in f.items()}, "capital": f["capital"] * 2}
    cap2_lowpmf["pmf"] = np.minimum(f["pmf"], 0.3)       # cap PMF low, throw money
    v_nopmf = simulate(no_pmf)[0].sum()
    v_cap = simulate(cap2_lowpmf)[0].sum()
    counterfactual = {
        "no_pmf_value_pct_of_base": round(100 * v_nopmf / total, 1),
        "double_capital_lowpmf_pct_of_base": round(100 * v_cap / total, 1)}

    # 4. honest predictability: does month-6 revenue rank predict the final winner?
    def spearman(a, b):
        ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
        return float(np.corrcoef(ra, rb)[0, 1])
    rho = spearman(rev6, value)
    top10_final = set(np.argsort(value)[::-1][:N // 10])
    top10_m6 = set(np.argsort(rev6)[::-1][:N // 10])
    winner_in_m6_top10 = int(np.argmax(value)) in top10_m6
    predict = {"spearman_m6_vs_final": round(rho, 3),
               "top_decile_overlap": round(len(top10_final & top10_m6) / (N // 10), 3),
               "winner_in_month6_top_decile": winner_in_m6_top10}

    cum = np.cumsum(sv) / total                          # Lorenz-style value curve
    results = {"n": N, "months": T, "total_value": round(total, 1),
               "value_of_factor": voa, "power_law": power,
               "counterfactual": counterfactual, "predictability": predict,
               "top_values": [round(float(v), 1) for v in sv[:10]],
               "cum_value_share": [round(float(c), 4) for c in cum],
               "scatter": {"rev6": [round(float(x), 2) for x in rev6],
                           "value": [round(float(x), 1) for x in value]}}
    save_results("e51_startups", results)

    print(f"E51 - startup growth world model ({N} startups, {T} months)\n")
    print(f"  1. value-of-factor (Δ batch value when lifted): " +
          ", ".join(f"{k} {voa[k]['delta_value_pct']:+.0f}%" for k in FACTORS))
    print(f"  2. power law: survival {power['survival_rate']:.0%}, top-1 "
          f"{power['top1_share']:.0%} of value, top-decile {power['top_decile_share']:.0%}, "
          f"Gini {power['value_gini']}")
    print(f"  3. counterfactual: no-PMF batch = {counterfactual['no_pmf_value_pct_of_base']}% "
          f"of base; double-capital-at-low-PMF = "
          f"{counterfactual['double_capital_lowpmf_pct_of_base']}% of base")
    print(f"  4. predictability: month-6 vs final Spearman {predict['spearman_m6_vs_final']}, "
          f"top-decile overlap {predict['top_decile_overlap']:.0%}, "
          f"winner in m6 top-decile: {predict['winner_in_month6_top_decile']}")

    # --- self-checks ---
    assert voa["pmf"]["delta_value_pct"] > voa["capital"]["delta_value_pct"], \
        "PMF should drive growth more than capital"
    assert voa["pmf"]["delta_value_pct"] == max(v["delta_value_pct"] for v in voa.values()), \
        "PMF should be the top factor"
    assert power["top_decile_share"] > 0.6, "returns should follow a power law (few winners)"
    assert counterfactual["no_pmf_value_pct_of_base"] < 5, "no PMF should collapse the batch"
    assert counterfactual["double_capital_lowpmf_pct_of_base"] < 60, \
        "money without PMF should not rescue the batch"
    assert predict["spearman_m6_vs_final"] < 0.95, "early traction shouldn't perfectly predict winners"
    print("\nall checks pass; batch value aggregator never drifts from the leaves.")


if __name__ == "__main__":
    main()
