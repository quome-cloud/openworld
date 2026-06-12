"""E31 - Nested + traversal fidelity (offline, no LLM).

A 3-level composite (region > 2 countries > 2 cities each) with hand-written
leaf dynamics, country-level aggregators (gdp, treasury), region aggregators
that chain the inner _agg, one inter-country goods bridge, and one toll Route
between two cities in different countries. A 20-step scripted run (leaf
actions, ticks, two travel attempts - one paid, one denied by the toll) is
replayed by an INDEPENDENT flat-dict oracle that imports nothing from
openworld.compose, so per-step agreement is evidence, not tautology.

Metrics per step: exact equality of every leaf field vs the oracle,
aggregator consistency (stored _agg == recomputation from leaves, at both the
country and region levels), conservation of total money (city treasuries +
agent coins), and registry-location/coins agreement with the oracle.
"""

from openworld.compose import (
    AGENTS_KEY, AGG_KEY, Aggregator, Bridge, CompositeWorld, Route,
    legal_actions, observe,
)
from openworld.state import Action
from openworld.transition import FunctionTransition
from openworld.world import World

from common import save_results

COUNTRIES = ("c0", "c1")
CITIES = ("a", "b")
TOLL = 2
BRIDGE_THRESHOLD = 6   # if c0:a goods exceed this ...
BRIDGE_AMOUNT = 2      # ... move this many goods to c1:a

# Structurally distinct leaf dynamics per city.
CITY_PARAMS = {
    ("c0", "a"): dict(work_goods=2, work_gdp=1, trade_in=3, trade_gdp=2),
    ("c0", "b"): dict(work_goods=1, work_gdp=1, trade_in=2, trade_gdp=3),
    ("c1", "a"): dict(work_goods=3, work_gdp=2, trade_in=4, trade_gdp=3),
    ("c1", "b"): dict(work_goods=2, work_gdp=2, trade_in=2, trade_gdp=1),
}
CITY_INITIAL = {
    ("c0", "a"): {"treasury": 10, "goods": 4, "gdp": 0},
    ("c0", "b"): {"treasury": 8, "goods": 2, "gdp": 0},
    ("c1", "a"): {"treasury": 12, "goods": 5, "gdp": 0},
    ("c1", "b"): {"treasury": 6, "goods": 3, "gdp": 0},
}
DEFAULT_ACTIONS = {
    ("c0", "a"): "work", ("c0", "b"): "work",
    ("c1", "a"): "work", ("c1", "b"): "trade",
}
TIMESCALES = {("c0", "a"): 2}  # c0:a works twice per country tick
AGENT_INITIAL = {"at": "c0:b", "coins": 3}
TOTAL_MONEY = sum(v["treasury"] for v in CITY_INITIAL.values()) + AGENT_INITIAL["coins"]

# 20-step script. Step 4 travel pays the toll (coins 3 -> 1); step 9 travel
# is denied (coins 1 < TOLL), a visa-style veto by on_cross.
SCRIPT = [
    {"name": "c0:b:work"},
    {"name": "c0:b:trade"},
    {"name": "tick"},
    {"name": "travel", "params": {"agent": "trader", "to": "c1:b"}},
    {"name": "c1:b:work"},
    {"name": "c1:b:work"},
    {"name": "c1:b:trade"},
    {"name": "tick"},
    {"name": "travel", "params": {"agent": "trader", "to": "c0:b"}},  # denied
    {"name": "c1:a:work"},
    {"name": "tick"},
    {"name": "c0:a:work"},
    {"name": "c0:a:trade"},
    {"name": "c1:a:trade"},
    {"name": "tick"},
    {"name": "c0:b:work"},
    {"name": "c1:b:trade"},
    {"name": "c0:a:work"},
    {"name": "tick"},
    {"name": "c0:b:trade"},
]


# ---------------------------------------------------------------------------
# Composite world (the system under test)
# ---------------------------------------------------------------------------

def make_city_fn(params):
    def fn(state, action):
        s = dict(state)
        if action["name"] == "work":
            s["goods"] += params["work_goods"]
            s["gdp"] += params["work_gdp"]
        elif action["name"] == "trade" and s["goods"] >= params["trade_in"]:
            s["goods"] -= params["trade_in"]
            s["gdp"] += params["trade_gdp"]
        return s
    return fn


def make_city(country, city):
    return World(
        name=f"{country}:{city}",
        description=f"City {city} of country {country}.",
        initial_state=dict(CITY_INITIAL[(country, city)]),
        actions=["work", "trade", "wait"],
        transition=FunctionTransition(make_city_fn(CITY_PARAMS[(country, city)])),
    )


def make_country(country):
    return CompositeWorld(
        name=country,
        children={city: make_city(country, city) for city in CITIES},
        aggregators=[
            Aggregator("gdp", lambda kids: kids["a"]["gdp"] + kids["b"]["gdp"]),
            Aggregator("treasury",
                       lambda kids: kids["a"]["treasury"] + kids["b"]["treasury"]),
        ],
        default_actions={city: DEFAULT_ACTIONS[(country, city)] for city in CITIES},
        timescales={city: TIMESCALES[(country, city)]
                    for city in CITIES if (country, city) in TIMESCALES},
    )


def goods_bridge_fn(state, action):
    a, b = dict(state["a"]), dict(state["b"])
    if a["goods"] > BRIDGE_THRESHOLD:
        a["goods"] -= BRIDGE_AMOUNT
        b["goods"] += BRIDGE_AMOUNT
    return {"a": a, "b": b}


def toll_fn(state, action):
    agent = dict(state["agent"])
    a, b = dict(state["a"]), dict(state["b"])
    if agent["coins"] >= TOLL:
        agent["coins"] -= TOLL
        b["treasury"] += TOLL  # toll is paid into the destination treasury
    else:
        agent["denied"] = True
    return {"agent": agent, "a": a, "b": b}


def make_region():
    return CompositeWorld(
        name="region",
        children={c: make_country(c) for c in COUNTRIES},
        bridges=[
            Bridge(name="goods_flow", a="c0:a", b="c1:a",
                   transition=FunctionTransition(goods_bridge_fn),
                   description="Surplus goods flow from c0:a to c1:a."),
            Route(name="toll_road", a="c0:b", b="c1:b", transition=None,
                  on_cross=FunctionTransition(toll_fn),
                  description="A toll road between the two border cities."),
        ],
        aggregators=[
            Aggregator("gdp_total",
                       lambda kids: kids["c0"][AGG_KEY]["gdp"] + kids["c1"][AGG_KEY]["gdp"]),
            Aggregator("treasury_total",
                       lambda kids: kids["c0"][AGG_KEY]["treasury"]
                       + kids["c1"][AGG_KEY]["treasury"]),
        ],
        default_actions={c: "tick" for c in COUNTRIES},
        agents={"trader": dict(AGENT_INITIAL)},
    )


# ---------------------------------------------------------------------------
# Independent flat oracle (no imports from openworld.compose; plain dicts)
# ---------------------------------------------------------------------------

def oracle_initial():
    flat = {}
    for (c, k), fields in CITY_INITIAL.items():
        for f, v in fields.items():
            flat[f"{c}_{k}_{f}"] = v
    flat["trader_at"] = AGENT_INITIAL["at"]
    flat["trader_coins"] = AGENT_INITIAL["coins"]
    return flat


def oracle_city_act(flat, c, k, act):
    p = CITY_PARAMS[(c, k)]
    if act == "work":
        flat[f"{c}_{k}_goods"] += p["work_goods"]
        flat[f"{c}_{k}_gdp"] += p["work_gdp"]
    elif act == "trade" and flat[f"{c}_{k}_goods"] >= p["trade_in"]:
        flat[f"{c}_{k}_goods"] -= p["trade_in"]
        flat[f"{c}_{k}_gdp"] += p["trade_gdp"]


def oracle_bridge(flat):
    if flat["c0_a_goods"] > BRIDGE_THRESHOLD:
        flat["c0_a_goods"] -= BRIDGE_AMOUNT
        flat["c1_a_goods"] += BRIDGE_AMOUNT


def oracle_step(flat, entry):
    name = entry["name"]
    if name == "tick":
        for c in COUNTRIES:
            for k in CITIES:
                for _ in range(TIMESCALES.get((c, k), 1)):
                    oracle_city_act(flat, c, k, DEFAULT_ACTIONS[(c, k)])
        oracle_bridge(flat)
    elif name == "travel":
        here, dest = flat["trader_at"], entry["params"]["to"]
        if {here, dest} == {"c0:b", "c1:b"} and dest != here:
            if flat["trader_coins"] >= TOLL:
                flat["trader_coins"] -= TOLL
                dc, dk = dest.split(":")
                flat[f"{dc}_{dk}_treasury"] += TOLL
                flat["trader_at"] = dest
        # bridges do not fire on travel
    else:
        c, k, act = name.split(":")
        oracle_city_act(flat, c, k, act)
        oracle_bridge(flat)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def flatten_leaves(state):
    return {f"{c}_{k}_{f}": state[c][k][f]
            for c in COUNTRIES for k in CITIES
            for f in ("treasury", "goods", "gdp")}


def check_step(state, flat):
    leaves = flatten_leaves(state)
    leaf_exact = leaves == {key: flat[key] for key in leaves}
    registry = state[AGENTS_KEY]["trader"]
    registry_match = (registry["at"] == flat["trader_at"]
                      and registry["coins"] == flat["trader_coins"])
    agg_ok = True
    for c in COUNTRIES:
        stored = state[c][AGG_KEY]
        agg_ok &= stored["gdp"] == state[c]["a"]["gdp"] + state[c]["b"]["gdp"]
        agg_ok &= stored["treasury"] == (
            state[c]["a"]["treasury"] + state[c]["b"]["treasury"])
    recomputed_gdp = sum(state[c][k]["gdp"] for c in COUNTRIES for k in CITIES)
    recomputed_treasury = sum(
        state[c][k]["treasury"] for c in COUNTRIES for k in CITIES)
    agg_ok &= state[AGG_KEY]["gdp_total"] == recomputed_gdp
    agg_ok &= state[AGG_KEY]["treasury_total"] == recomputed_treasury
    money = recomputed_treasury + registry["coins"]
    return {
        "leaf_exact": leaf_exact,
        "registry_match": registry_match,
        "agg_consistent": bool(agg_ok),
        "money_conserved": money == TOTAL_MONEY,
        "total_money": money,
        "leaves": leaves,
        "agent": dict(registry),
    }


def check_observation(region, state, flat):
    """Scoped-view sanity after travel: local slice matches the oracle's
    leaves, ancestors expose aggregates, the route neighbor is visible, and
    legal_actions offers the leaf actions plus the travel edge."""
    view = observe(region, dict(state), "trader")
    here = flat["trader_at"]
    c, k = here.split(":")
    local_ok = all(view["local"][f] == flat[f"{c}_{k}_{f}"]
                   for f in ("treasury", "goods", "gdp"))
    ancestors_ok = "<root>" in view["ancestors"] and c in view["ancestors"]
    other = "c1:b" if here == "c0:b" else "c0:b"
    neighbor_ok = other in view["neighbors"]
    acts = legal_actions(region, dict(state), "trader")
    actions_ok = (f"{here}:work" in acts and f"{here}:trade" in acts
                  and f"travel:{other}" in acts)
    return {"location": here, "local_ok": local_ok, "ancestors_ok": ancestors_ok,
            "neighbor_ok": neighbor_ok, "legal_actions_ok": actions_ok,
            "legal_actions": acts}


def main():
    region = make_region()
    flat = oracle_initial()
    steps = []
    observation_checks = []
    for i, entry in enumerate(SCRIPT, start=1):
        state = dict(region.step(
            Action(entry["name"], params=entry.get("params", {}))))
        oracle_step(flat, entry)
        record = {"step": i, "action": entry["name"], **check_step(state, flat)}
        steps.append(record)
        if entry["name"] == "travel":
            obs = check_observation(region, region.state, flat)
            obs["step"] = i
            observation_checks.append(obs)

    n = len(steps)
    counts = {
        "steps": n,
        "leaf_exact": sum(s["leaf_exact"] for s in steps),
        "registry_match": sum(s["registry_match"] for s in steps),
        "agg_consistent": sum(s["agg_consistent"] for s in steps),
        "money_conserved": sum(s["money_conserved"] for s in steps),
        "observation_checks_passed": sum(
            o["local_ok"] and o["ancestors_ok"] and o["neighbor_ok"]
            and o["legal_actions_ok"] for o in observation_checks),
        "observation_checks": len(observation_checks),
    }
    all_pass = all(
        counts[k] == n for k in
        ("leaf_exact", "registry_match", "agg_consistent", "money_conserved")
    ) and counts["observation_checks_passed"] == counts["observation_checks"]

    save_results("e31_nested_fidelity", {
        "structure": "region > {c0, c1} > {a, b}",
        "total_money": TOTAL_MONEY,
        "script": SCRIPT,
        "counts": counts,
        "all_pass": all_pass,
        "per_step": steps,
        "observation_checks": observation_checks,
        "oracle_final": flat,
    })

    print(f"E31 nested fidelity over {n} scripted steps:")
    print(f"  exact leaf match     : {counts['leaf_exact']}/{n}")
    print(f"  registry matches     : {counts['registry_match']}/{n}")
    print(f"  aggregator consistent: {counts['agg_consistent']}/{n}")
    print(f"  money conserved      : {counts['money_conserved']}/{n} "
          f"(total = {TOTAL_MONEY})")
    print(f"  scoped-view checks   : {counts['observation_checks_passed']}"
          f"/{counts['observation_checks']} (after each travel attempt)")
    print(f"  ALL PASS: {all_pass}")


if __name__ == "__main__":
    main()
