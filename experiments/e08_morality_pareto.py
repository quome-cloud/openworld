"""E8 - Tunable morality: dial sweep, Pareto frontier, interior optimum.

Replicates the tunable-morality protocol on the commons world: sweep the
moral weight lambda over [0, 1], trace the welfare/fairness frontier, and
locate the interior net-tension optimum. Deterministic; no LLM calls.
"""

from openworld import Action, Agent, Dial, FunctionTransition, Objective, Simulation, World, sweep

from common import save_results

LAMBDAS = [round(0.1 * i, 1) for i in range(11)]
STEPS = 8


def commons_dynamics(state, action):
    s = dict(state)
    s["harvested"] = dict(s["harvested"])
    agent = action.get("agent")
    if action["name"] == "pick" and s["apples"] > 0 and agent:
        s["apples"] -= 1
        s["harvested"][agent] += 1
    return s


def main():
    morality = Dial("morality", value=0.0)

    def considerate(state, actions):
        lead = state["harvested"]["alice"] - state["harvested"]["bob"]
        allowed = round((1.0 - morality.value) * 5)
        if state["apples"] > 0 and lead <= allowed:
            return Action("pick")
        return Action("wait")

    bob_turn = {"n": 0}

    def lazy_bob(state, actions):
        bob_turn["n"] += 1
        if state["apples"] > 0 and bob_turn["n"] % 2 == 0:
            return Action("pick")
        return Action("wait")

    world = World(
        name="commons", description="A shared commons of apples.",
        initial_state={"apples": 12, "harvested": {"alice": 0, "bob": 0}},
        actions=["pick", "wait"],
        transition=FunctionTransition(commons_dynamics),
    )
    sim = Simulation(
        world,
        agents=[Agent(name="alice", policy=considerate), Agent(name="bob", policy=lazy_bob)],
        objectives=[
            Objective("welfare", fn=lambda s, a, ns: s["apples"] - ns["apples"], weight=1.0),
            Objective("fairness",
                      fn=lambda s, a, ns: -(max(ns["harvested"].values()) - min(ns["harvested"].values())),
                      weight=morality),
        ],
    )

    result = sweep(sim, dial="morality", values=LAMBDAS, steps=STEPS)
    points = [{"lambda": p.dial_value,
               "welfare": p.mean_totals["welfare"],
               "fairness": p.mean_totals["fairness"]} for p in result.points]
    frontier = result.pareto(["welfare", "fairness"])

    # Net-tension optima. Two standard aggregations over the frontier:
    #  - equal-weight sum of normalized objectives (reveals frontier shape:
    #    a corner winner means the normalized frontier is convex)
    #  - Nash bargaining product of gains over the disagreement point (the
    #    standard balanced operating point; interior whenever both objectives
    #    can gain simultaneously)
    welfare_vals = [p["welfare"] for p in points]
    fairness_vals = [p["fairness"] for p in points]

    def norm(v, vals):
        lo, hi = min(vals), max(vals)
        return (v - lo) / (hi - lo) if hi > lo else 0.0

    sums = [norm(p["welfare"], welfare_vals) + norm(p["fairness"], fairness_vals)
            for p in points]
    sum_idx = max(range(len(points)), key=lambda i: sums[i])
    nash = [norm(p["welfare"], welfare_vals) * norm(p["fairness"], fairness_vals)
            for p in points]
    nash_idx = max(range(len(points)), key=lambda i: nash[i])
    best_idx = nash_idx

    # Monotone controllability: fairness should not decrease as lambda rises.
    fairness_monotone = all(
        fairness_vals[i + 1] >= fairness_vals[i] - 1e-9 for i in range(len(points) - 1)
    )

    save_results("e08_morality_pareto", {
        "lambdas": LAMBDAS, "steps": STEPS, "points": points,
        "pareto_frontier_size": len(frontier),
        "pareto_lambdas": [p.dial_value for p in frontier],
        "equal_weight_optimum_lambda": points[sum_idx]["lambda"],
        "nash_optimum_lambda": points[nash_idx]["lambda"],
        "nash_optimum_is_interior": 0.0 < points[nash_idx]["lambda"] < 1.0,
        "fairness_monotone_in_lambda": fairness_monotone,
        "welfare_range": [min(welfare_vals), max(welfare_vals)],
        "fairness_range": [min(fairness_vals), max(fairness_vals)],
    })
    print(f"frontier size {len(frontier)}/{len(LAMBDAS)}, "
          f"Nash optimum at lambda={points[best_idx]['lambda']} "
          f"(equal-weight at {points[sum_idx]['lambda']}), "
          f"fairness monotone: {fairness_monotone}")


if __name__ == "__main__":
    main()
