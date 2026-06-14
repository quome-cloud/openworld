"""E44 - Emergent economy capstone: macro phenomena from verified micro rules.

A multi-agent economy assembled entirely from small, separable, verified rules
and composed with the framework's CompositeWorld + Aggregator machinery. Each
agent is a real child World that PRODUCES goods on its tick; the macro
quantities the paper reports (money supply, market price, wealth Gini) are
Aggregators - derived from the leaves, so a summary can never drift from the
micro state. The market clearing, taxation, and redistribution are each their
own toggleable rule.

The point is two-fold:
  (1) EMERGENCE - price formation, inflation, and inequality are nowhere coded
      as targets; they fall out of the agents' selfish micro incentives.
  (2) CAUSAL ATTRIBUTION - because every rule is a separate verified component,
      we can switch one on/off and read its macro effect as a clean
      counterfactual, which a black-box simulator cannot do.

Four claims, each tested by a toggle:
  1. Price formation       - converged price is higher when supply is scarcer.
  2. Inflation             - faucet>sink (no burn tax) inflates money & price;
                             turning the verified burn rule on bends it down.
  3. Inequality            - selfish accumulation raises Gini; the verified
                             redistribution rule lowers it.
  4. Selfish vs cooperative- a policy dial: cooperation raises TOTAL welfare
                             (capital has diminishing returns, so equalizing
                             produces more) while the greediest selfish agent
                             still ends richest. The E08 Pareto tension,
                             emerging from a composed verified economy.

Deterministic and offline (fixed agent endowments, no run-time randomness),
self-checking (conservation of gold + the sign of every claim asserted).
"""

from openworld import Action, Aggregator, CompositeWorld, World
from openworld.transition import FunctionTransition

from common import save_results

N = 6
T = 80
BASES = [1, 2, 4, 8, 12, 18]        # heterogeneous per-agent production endowments
EXP = 0.7                           # capital exponent: <1 is concave (diminishing
                                    # returns), so equalizing wealth raises output
P0 = 8.0
P_MIN = 0.5


# --- the macro aggregators (derived, never simulated) -----------------------
def gini(xs):
    xs = sorted(max(0.0, x) for x in xs)
    n = len(xs)
    tot = sum(xs)
    if tot == 0:
        return 0.0
    cum = sum((i + 1) * x for i, x in enumerate(xs))
    return (2 * cum) / (n * tot) - (n + 1) / n


def agent_keys():
    return [f"a{i}" for i in range(N)]


def make_agent(base, gamma):
    """A real child World: on its tick it gathers goods. Production has a
    capital term with DIMINISHING returns (exponent EXP < 1) - the concavity is
    what makes equalizing wealth raise total output in claim 4."""
    def produce(state, action):
        s = dict(state)
        s["planks"] += base + gamma * (max(0.0, s["gold"]) ** EXP)
        return s
    return World(name="agent", description="a producer",
                 initial_state={"gold": 0.0, "planks": 0.0, "base": base},
                 actions=["produce"], transition=FunctionTransition(produce))


def build_economy(gamma):
    children = {k: make_agent(BASES[i], gamma) for i, k in enumerate(agent_keys())}
    children["market"] = World(
        name="market", description="posts a price",
        initial_state={"price": P0}, actions=["tick"],
        transition=FunctionTransition(lambda s, a: s))   # price set by clearing
    aggs = [
        Aggregator("money", lambda kids: sum(kids[k]["gold"] for k in agent_keys())),
        Aggregator("supply", lambda kids: sum(kids[k]["planks"] for k in agent_keys())),
        Aggregator("mean_price", lambda kids: kids["market"]["price"]),
        Aggregator("gini", lambda kids: gini([kids[k]["gold"] for k in agent_keys()])),
    ]
    return CompositeWorld(
        name="economy", children=children, aggregators=aggs,
        default_actions={k: "produce" for k in agent_keys()})


# --- the separable, toggleable economic rules -------------------------------
def clear_price(state, cfg, money, supply):
    """Tatonnement: downward-sloping demand (rises with money in circulation,
    falls with price) meets inelastic supply; price moves toward clearing."""
    price = state["market"]["price"]
    demand = cfg["D0"] + cfg["beta"] * money - cfg["delta"] * price
    price = max(P_MIN, price + cfg["alpha"] * (demand - supply))
    state["market"]["price"] = price
    return price


def trade_and_tax(state, cfg, price):
    """Agents sell all goods at the posted price; an optional burn tax is the
    SINK that removes gold from circulation. Returns (injected, burned)."""
    injected = burned = 0.0
    for k in agent_keys():
        rev = price * state[k]["planks"]
        burn = cfg["tau"] * rev
        state[k]["gold"] += rev - burn
        state[k]["planks"] = 0.0
        injected += rev
        burned += burn
    return injected, burned


def transfer_equalize(state, frac):
    """A pure transfer: skim `frac` of every agent's gold into a pot and pay it
    back as an equal dividend. Conserves gold; compresses the distribution.
    Used for both redistribution (claim 3) and cooperative pooling (claim 4)."""
    pot = sum(state[k]["gold"] for k in agent_keys()) * frac
    div = pot / N
    for k in agent_keys():
        state[k]["gold"] = state[k]["gold"] * (1 - frac) + div


# --- one economy run --------------------------------------------------------
def run(cfg):
    comp = build_economy(cfg.get("gamma", 0.0))
    money0 = sum(comp.state[k]["gold"] for k in agent_keys())
    injected_cum = burned_cum = 0.0
    traj = {"price": [], "money": [], "gini": []}
    for _ in range(T):
        comp.step(Action("tick"))                          # agents produce (child ticks)
        agg = comp.state["_agg"]
        price = clear_price(comp.state, cfg, agg["money"], agg["supply"])
        inj, brn = trade_and_tax(comp.state, cfg, price)
        injected_cum += inj
        burned_cum += brn
        if cfg.get("redistribute"):
            transfer_equalize(comp.state, cfg["rho_r"])
        if cfg.get("cooperative"):
            transfer_equalize(comp.state, cfg["rho_c"])
        comp.state["_agg"] = comp._aggregates(comp.state)   # re-derive macro readout
        a = comp.state["_agg"]
        traj["price"].append(round(a["mean_price"], 4))
        traj["money"].append(round(a["money"], 4))
        traj["gini"].append(round(a["gini"], 4))
        # conservation: gold changes only by faucet inflow minus sink outflow
        money = a["money"]
        expected = money0 + injected_cum - burned_cum
        assert abs(money - expected) < 1e-6 * max(1.0, abs(expected)), \
            "gold not conserved (rule silently created/destroyed money)"
    golds = [comp.state[k]["gold"] for k in agent_keys()]
    return {"traj": traj, "golds": golds,
            "final_price": traj["price"][-1], "final_gini": traj["gini"][-1],
            "total_welfare": sum(golds), "max_gold": max(golds)}


def slope(xs):
    """Least-squares slope over the second half (after transients)."""
    half = xs[len(xs) // 2:]
    n = len(half)
    mx = (n - 1) / 2
    my = sum(half) / n
    num = sum((i - mx) * (y - my) for i, y in enumerate(half))
    den = sum((i - mx) ** 2 for i in range(n))
    return num / den if den else 0.0


def main():
    base = {"alpha": 0.1, "D0": 60.0, "beta": 0.0, "delta": 1.0, "gamma": 0.0,
            "tau": 0.0, "redistribute": False, "cooperative": False,
            "rho_r": 0.3, "rho_c": 0.3}

    def cfg(**kw):
        return {**base, **kw}

    # Claim 1: price formation - scarce supply -> higher converged price.
    # Same fixed demand; the only change is doubling every agent's output.
    # D0 high enough that both supply levels clear above the price floor.
    global BASES
    saved = BASES
    scarce = run(cfg(D0=120.0))                             # supply sum = 45
    BASES = [2 * b for b in saved]
    abundant = run(cfg(D0=120.0))                           # supply sum = 90
    BASES = saved

    # Claim 2: inflation - money chasing FIXED goods (gamma=0). The faucet is
    # sale revenue; the burn tax is the sink. Toggle the sink off vs on.
    infl_off = run(cfg(beta=0.004, gamma=0.0, tau=0.0))
    infl_on = run(cfg(beta=0.004, gamma=0.0, tau=0.35))

    # Claim 3: inequality - compounding capital (gamma>0), stable price (beta=0
    # so no inflation feedback); redistribution off vs on.
    ineq_off = run(cfg(beta=0.0, gamma=0.6, redistribute=False))
    ineq_on = run(cfg(beta=0.0, gamma=0.6, redistribute=True, rho_r=0.3))

    # Claim 4: dial - selfish vs cooperative (concave capital rewards equality).
    selfish = run(cfg(beta=0.0, gamma=0.6, cooperative=False))
    cooperative = run(cfg(beta=0.0, gamma=0.6, cooperative=True, rho_c=0.35))

    results = {
        "n_agents": N, "horizon": T, "bases": saved,
        "claim1_price_formation": {
            "scarce_supply_price": scarce["final_price"],
            "abundant_supply_price": abundant["final_price"],
            "scarce_traj": scarce["traj"]["price"],
            "abundant_traj": abundant["traj"]["price"]},
        "claim2_inflation": {
            "burn_off_price_slope": round(slope(infl_off["traj"]["price"]), 4),
            "burn_on_price_slope": round(slope(infl_on["traj"]["price"]), 4),
            "burn_off_money_slope": round(slope(infl_off["traj"]["money"]), 4),
            "burn_on_money_slope": round(slope(infl_on["traj"]["money"]), 4),
            "off_price_traj": infl_off["traj"]["price"],
            "on_price_traj": infl_on["traj"]["price"],
            "off_money_traj": infl_off["traj"]["money"],
            "on_money_traj": infl_on["traj"]["money"]},
        "claim3_inequality": {
            "redist_off_gini": ineq_off["final_gini"],
            "redist_on_gini": ineq_on["final_gini"],
            "off_gini_traj": ineq_off["traj"]["gini"],
            "on_gini_traj": ineq_on["traj"]["gini"]},
        "claim4_dial": {
            "selfish_welfare": round(selfish["total_welfare"], 2),
            "cooperative_welfare": round(cooperative["total_welfare"], 2),
            "selfish_max_gold": round(selfish["max_gold"], 2),
            "cooperative_max_gold": round(cooperative["max_gold"], 2),
            "selfish_gini": selfish["final_gini"],
            "cooperative_gini": cooperative["final_gini"]},
    }
    save_results("e44_emergent_economy", results)

    c1, c2, c3, c4 = (results["claim1_price_formation"], results["claim2_inflation"],
                      results["claim3_inequality"], results["claim4_dial"])
    print("E44 - emergent economy from composed verified rules "
          f"({N} agents, {T} ticks).\n")
    print(f"  1. price formation   scarce p*={c1['scarce_supply_price']:.2f}  "
          f"abundant p*={c1['abundant_supply_price']:.2f}  (scarce > abundant)")
    print(f"  2. inflation         price slope  burn off={c2['burn_off_price_slope']:+.3f}  "
          f"on={c2['burn_on_price_slope']:+.3f}  (off > on)")
    print(f"  3. inequality        final Gini   redist off={c3['redist_off_gini']:.3f}  "
          f"on={c3['redist_on_gini']:.3f}  (off > on)")
    print(f"  4. selfish/coop      welfare  selfish={c4['selfish_welfare']:.0f}  "
          f"coop={c4['cooperative_welfare']:.0f}   |   "
          f"max gold  selfish={c4['selfish_max_gold']:.0f}  coop={c4['cooperative_max_gold']:.0f}")

    # self-checks: every claim's sign
    assert c1["scarce_supply_price"] > c1["abundant_supply_price"], "scarcity should raise price"
    assert c2["burn_off_price_slope"] > c2["burn_on_price_slope"], "burn sink should curb inflation"
    assert c2["burn_off_money_slope"] > c2["burn_on_money_slope"], "burn sink should slow money growth"
    assert c3["redist_off_gini"] > c3["redist_on_gini"], "redistribution should lower Gini"
    assert c4["cooperative_welfare"] > c4["selfish_welfare"], "cooperation should raise total welfare"
    assert c4["selfish_max_gold"] > c4["cooperative_max_gold"], "selfish should produce the richest agent"
    print("\n  all four emergent claims hold; gold conserved every tick.")


if __name__ == "__main__":
    main()
