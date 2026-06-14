"""E48 - Composite corporate world: individual, division, and company goals.

A DigitalOcean-style PaaS company as a nested CompositeWorld (company -> divisions
-> individuals): division revenue and company revenue are Aggregators (derived
from the leaves, never drift), and agents at different org levels (senior SWE,
director, CEO) act with different scope. The scientific core is the tension
between optimizing at the individual, division, and company scales.

Revenue has DIMINISHING RETURNS in effort per division (concave), so total
company growth is maximized by spreading effort across divisions by marginal
return (proportional to productivity^2). Concentrating effort - what individuals
chasing visibility/promotion do - is individually tempting but collectively
worse (the E44/E08 Pareto, now on an org chart).

Four agent-level experiments, deterministic/offline:
  1. individual vs collective: a selfishness dial trades company growth for
     concentrated promotions.
  2. value-of-action by level: whose decisions move aggregate growth most
     (CEO budget reallocation vs director vs IC), by causal toggle.
  3. perception cost: acting on coarse transcript-derived state (all-hands) vs
     ground truth - worst for the CEO.
  4. optimal navigation policy: a principled per-role policy beats a greedy one.
"""

import math

from openworld import Aggregator, CompositeWorld, World
from openworld.transition import FunctionTransition

from common import save_results

# division: productivity a (market multiplier), current revenue R0 ($M), headcount
DIVISIONS = {
    "database":   {"a": 2.0, "R0": 200, "hc": 10},
    "serverless": {"a": 1.0, "R0": 250, "hc": 8},
    "storage":    {"a": 0.6, "R0": 400, "hc": 12},   # legacy: big revenue, low growth
    "compute":    {"a": 1.5, "R0": 300, "hc": 10},
    "networking": {"a": 0.4, "R0": 350, "hc": 9},
}
NAMES = list(DIVISIONS)
A = [DIVISIONS[d]["a"] for d in NAMES]
R0 = [DIVISIONS[d]["R0"] for d in NAMES]
HC = [DIVISIONS[d]["hc"] for d in NAMES]
TOTAL_R0 = sum(R0)
B = 60.0                       # company-wide effort/investment budget
SCALE = 8.0


# --- structural showcase: the org as a real CompositeWorld ------------------
def build_company():
    children = {}
    for d in NAMES:
        children[d] = World(
            name=d, description=f"{d} division",
            initial_state={"revenue": float(DIVISIONS[d]["R0"]), "a": DIVISIONS[d]["a"]},
            actions=["tick"], transition=FunctionTransition(lambda s, a: s))
    return CompositeWorld(
        name="company", children=children,
        aggregators=[Aggregator("total_revenue",
                                lambda kids: sum(k["revenue"] for k in kids.values()))])


# --- core economics ---------------------------------------------------------
def normalize(w):
    t = sum(w)
    return [x / t for x in w] if t else [1.0 / len(w)] * len(w)


def eff_q(q):
    return 0.8 + 0.2 * q          # director effectiveness band


def eff_e(e):
    return 0.9 + 0.1 * e          # IC effectiveness band


def delta_revenue(x, q=1.0, e=1.0):
    """Per-division revenue gain under effort allocation x (concave in effort)."""
    f = eff_q(q) * eff_e(e) * SCALE
    return [A[i] * math.sqrt(x[i] * B) * f for i in range(len(NAMES))]


def gini(xs):
    xs = sorted(max(0.0, v) for v in xs)
    n, tot = len(xs), sum(xs)
    if tot == 0:
        return 0.0
    cum = sum((i + 1) * v for i, v in enumerate(xs))
    return (2 * cum) / (n * tot) - (n + 1) / n


def per_ic_impact(dR):
    """Spread each division's revenue gain across its individuals."""
    out = []
    for i in range(len(NAMES)):
        out += [dR[i] / HC[i]] * HC[i]
    return out


X_ALIGNED = normalize([a ** 2 for a in A])           # marginal-return optimal
X_REVENUE = normalize(R0)                             # greedy: fund the big
X_VISIBLE = [1.0 if i == R0.index(max(R0)) else 0.0   # all-in on most visible
             for i in range(len(NAMES))]


# --- claim 1: individual vs collective --------------------------------------
def pareto():
    rows = []
    for j in range(11):
        rho = j / 10.0
        x = [(1 - rho) * X_ALIGNED[i] + rho * X_VISIBLE[i] for i in range(len(NAMES))]
        dR = delta_revenue(x)
        impacts = per_ic_impact(dR)
        rows.append({
            "rho": rho,
            "company_growth": sum(dR) / TOTAL_R0,
            "promo_gini": gini(impacts),
            "top_impact": max(impacts),
        })
    return rows


# --- claim 2: value-of-action by hierarchy level ----------------------------
def value_of_action():
    base = sum(delta_revenue(X_REVENUE, q=0.5, e=0.5))      # greedy everything
    ceo = sum(delta_revenue(X_ALIGNED, q=0.5, e=0.5)) - base   # reallocate budget
    director = sum(delta_revenue(X_REVENUE, q=1.0, e=0.5)) - base  # raise team eff
    ic = sum(delta_revenue(X_REVENUE, q=0.5, e=1.0)) - base        # raise own eff
    return {"baseline": base, "ceo_gain": ceo, "director_gain": director,
            "ic_gain": ic,
            "ceo_pct": 100 * ceo / base, "director_pct": 100 * director / base,
            "ic_pct": 100 * ic / base}


# --- claim 3: perception cost (coarse transcripts) --------------------------
def _num(x):
    """Pull a number out of an extracted value ('42', '42%', 0.42, etc.)."""
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x) * 100 if 0 < x <= 1.5 else float(x)   # 0.42 -> 42
    s = "".join(c for c in str(x) if c.isdigit() or c in ".-")
    try:
        v = float(s)
    except ValueError:
        return None
    return v * 100 if 0 < v <= 1.5 else v


def perception():
    """Real perception: each role extracts division growth from the prose it
    actually sees (CEO <- the all-hands; director <- its own review) via an LLM
    TextPerceptor over the openworld-corp transcript corpus, then acts on the
    perceived state. The all-hands is one transcript summarizing five divisions,
    so it is lossier than a division's detailed review - the CEO pays more."""
    import json
    import sys
    from pathlib import Path
    from openworld import Observation, OllamaLLM, TextPerceptor

    corp_dir = Path(__file__).resolve().parents[1] / "datasets" / "openworld-corp"
    sys.path.insert(0, str(corp_dir))
    import org as corp                                       # the canonical metrics
    corpus = json.loads((corp_dir / "corpus.json").read_text())
    period = corpus["periods"][-1]
    true_g = {d: corp.DIVISIONS[d]["growth"] * 100 for d in NAMES}

    llm = OllamaLLM(model="qwen2.5:7b", temperature=0.0, timeout=240,
                    options={"num_ctx": 8192})

    def rec(kind, scope=None):
        return next((r for r in corpus["records"] if r["type"] == kind
                     and r["period"] == period
                     and (scope is None or r["scope"] == scope)), None)

    # CEO reads ONE all-hands covering all five divisions
    ah = rec("all_hands")
    ceo_fields = [f"{d}_growth_percent" for d in NAMES]
    try:
        raw = TextPerceptor(llm, produces=ceo_fields).perceive(
            Observation("text", ah["transcript"], t=0))
    except Exception:
        raw = {}
    perceived_ceo = {d: _num(raw.get(f"{d}_growth_percent")) for d in NAMES}

    # each director reads its own detailed division review
    perceived_dir = {}
    for d in NAMES:
        r = rec("division_review", d)
        try:
            dr = TextPerceptor(llm, produces=["growth_percent"]).perceive(
                Observation("text", r["transcript"], t=0)) if r else {}
        except Exception:
            dr = {}
        perceived_dir[d] = _num(dr.get("growth_percent"))

    mean_true = sum(true_g.values()) / len(true_g)

    def err(p):                                             # mean abs growth error
        es = [abs((p[d] if p[d] is not None else mean_true) - true_g[d]) for d in NAMES]
        return sum(es) / len(es)

    def growth_from(p):                                     # allocate by perceived growth^2
        w = normalize([max(1.0, (p[d] if p[d] is not None else mean_true)) ** 2
                       for d in NAMES])
        return sum(delta_revenue(w)) / TOTAL_R0

    x_true = normalize([true_g[d] ** 2 for d in NAMES])
    g_true = sum(delta_revenue(x_true)) / TOTAL_R0

    # Individual-level signal: a promotion recommendation lives in a 1:1, not in
    # the all-hands. Measure whether each is RECOVERABLE from each transcript -
    # perception is granularity-bound: aggregate meetings lose individual state.
    sample = [(d, corp.DIVISIONS[d]["ics"][0]["name"]) for d in NAMES]
    rec_1on1 = rec_allhands = 0
    for d, name in sample:
        oo = next((r for r in corpus["records"] if r["type"] == "one_on_one"
                   and r.get("subject") == name and r["period"] == period), None)
        for src, txt, key in (("oo", oo["transcript"] if oo else "", "rec_1on1"),
                              ("ah", ah["transcript"], "rec_allhands")):
            try:
                v = TextPerceptor(llm, produces=["promotion_recommendation"]).perceive(
                    Observation("text", f"Regarding {name}: {txt}", t=0))
                got = bool(str(v.get("promotion_recommendation") or "").strip()
                           not in ("", "none", "unknown", "n/a", "not mentioned"))
            except Exception:
                got = False
            if key == "rec_1on1":
                rec_1on1 += got and src == "oo"
            else:
                rec_allhands += got and src == "ah"
    n = len(sample)
    return {
        "period": period, "true_growth": true_g,
        "division_signal": {
            "ceo_extract_err": err(perceived_ceo), "ceo_growth_gap": g_true - growth_from(perceived_ceo),
            "director_extract_err": err(perceived_dir),
            "ceo_perceived": perceived_ceo, "director_perceived": perceived_dir},
        "individual_signal": {
            "recover_from_one_on_one": rec_1on1 / n,
            "recover_from_all_hands": rec_allhands / n, "n_sampled": n},
        "ground_truth_growth": g_true,
        "corpus": {"n_records": corpus["n_records"], "periods": corpus["periods"]},
    }


# --- claim 4: optimal navigation policy per role ----------------------------
def policies():
    # CEO: principled (marginal-return) vs greedy (fund the big)
    ceo_principled = sum(delta_revenue(X_ALIGNED)) / TOTAL_R0
    ceo_greedy = sum(delta_revenue(X_REVENUE)) / TOTAL_R0
    # director: optimize team effectiveness vs leave it middling
    dir_principled = sum(delta_revenue(X_REVENUE, q=1.0)) / TOTAL_R0
    dir_greedy = sum(delta_revenue(X_REVENUE, q=0.5)) / TOTAL_R0
    # IC: focus vs coast
    ic_principled = sum(delta_revenue(X_REVENUE, e=1.0)) / TOTAL_R0
    ic_greedy = sum(delta_revenue(X_REVENUE, e=0.5)) / TOTAL_R0
    return {
        "ceo": {"principled": ceo_principled, "greedy": ceo_greedy},
        "director": {"principled": dir_principled, "greedy": dir_greedy},
        "ic": {"principled": ic_principled, "greedy": ic_greedy},
    }


def main():
    comp = build_company()
    # structural self-check: the revenue Aggregator equals the explicit sum
    agg = comp.state["_agg"]["total_revenue"]
    assert abs(agg - TOTAL_R0) < 1e-9, "revenue aggregator must equal the leaf sum"

    par = pareto()
    voa = value_of_action()
    perc = perception()
    pol = policies()

    save_results("e48_corporate_world", {
        "divisions": DIVISIONS, "budget": B, "total_revenue": TOTAL_R0,
        "pareto": par, "value_of_action": voa, "perception": perc, "policies": pol,
        "x_aligned": dict(zip(NAMES, X_ALIGNED)),
    })

    aligned, selfish = par[0], par[-1]
    print("E48 - composite corporate world "
          f"({len(NAMES)} divisions, ${TOTAL_R0}M revenue, {sum(HC)} ICs)\n")
    print(f"1. individual vs collective: aligned growth {aligned['company_growth']:.1%} "
          f"(promo Gini {aligned['promo_gini']:.2f}) vs selfish {selfish['company_growth']:.1%} "
          f"(Gini {selfish['promo_gini']:.2f})")
    print(f"2. value-of-action: CEO +{voa['ceo_pct']:.0f}%  director +{voa['director_pct']:.0f}%  "
          f"IC +{voa['ic_pct']:.0f}%  company growth")
    ds, isig = perc["division_signal"], perc["individual_signal"]
    print(f"3. perception ({perc['corpus']['n_records']} real transcripts): "
          f"division growth recovered from all-hands AND reviews "
          f"(err {ds['ceo_extract_err']:.1f}/{ds['director_extract_err']:.1f}pp); but "
          f"individual promo signal recovers {isig['recover_from_one_on_one']:.0%} from 1:1s "
          f"vs {isig['recover_from_all_hands']:.0%} from the all-hands")
    print(f"4. policy (company growth): CEO principled {pol['ceo']['principled']:.1%} vs "
          f"greedy {pol['ceo']['greedy']:.1%}")

    # --- self-checks ---
    assert aligned["company_growth"] > selfish["company_growth"], \
        "aligned (collective) should grow the company more than selfish concentration"
    assert selfish["promo_gini"] > aligned["promo_gini"], \
        "selfish concentration should make promotions more unequal"
    assert voa["ceo_gain"] > voa["director_gain"] > voa["ic_gain"], \
        "leverage hierarchy: CEO budget reallocation > director > IC"
    assert perc["individual_signal"]["recover_from_one_on_one"] > \
        perc["individual_signal"]["recover_from_all_hands"], \
        "individual promo signal lives in 1:1s, not the all-hands (granularity-bound perception)"
    for role in pol.values():
        assert role["principled"] >= role["greedy"], "principled policy beats greedy"
    print("\nall checks pass; composite revenue never drifts from the leaves.")


if __name__ == "__main__":
    main()
