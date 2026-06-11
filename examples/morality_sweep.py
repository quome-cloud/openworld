"""Morality-dial sweep: trace the Pareto frontier between individual welfare
and collective fairness, in the spirit of tunable-morality world models.

Two agents share a commons. A morality dial (lambda) weights a fairness
objective and throttles how far ahead an agent lets itself get. Sweeping
lambda traces the trade-off between total harvest and equal distribution -
no retraining, just turning the dial.

    python examples/morality_sweep.py
"""

from openworld import (
    Action,
    Agent,
    Dial,
    FunctionTransition,
    Objective,
    Simulation,
    World,
    sweep,
)


def commons_dynamics(state, action):
    next_state = dict(state)
    next_state["harvested"] = dict(next_state["harvested"])
    agent = action.get("agent")
    if action["name"] == "pick" and next_state["apples"] > 0 and agent:
        next_state["apples"] -= 1
        next_state["harvested"][agent] += 1
    return next_state


def fairness(state, action, next_state):
    counts = list(next_state["harvested"].values())
    return -(max(counts) - min(counts))


def welfare(state, action, next_state):
    return state["apples"] - next_state["apples"]


def main():
    morality = Dial("morality", value=0.0)

    def considerate(name):
        """Pick unless ahead of the others by more than the dial tolerates."""

        def policy(state, actions):
            mine = state["harvested"][name]
            others = min(v for k, v in state["harvested"].items() if k != name)
            allowed_lead = round((1.0 - morality.value) * 5)
            if state["apples"] > 0 and mine - others <= allowed_lead:
                return Action("pick")
            return Action("wait")

        return policy

    world = World(
        name="commons",
        description="A shared commons of apples.",
        initial_state={"apples": 12, "harvested": {"alice": 0, "bob": 0}},
        actions=["pick", "wait"],
        transition=FunctionTransition(commons_dynamics),
    )
    # Bob is lazy: he only picks every other opportunity, so unequal outcomes
    # appear unless Alice restrains herself.
    bob_turn = {"n": 0}

    def lazy_bob(state, actions):
        bob_turn["n"] += 1
        if state["apples"] > 0 and bob_turn["n"] % 2 == 0:
            return Action("pick")
        return Action("wait")

    sim = Simulation(
        world,
        agents=[
            Agent(name="alice", policy=considerate("alice")),
            Agent(name="bob", policy=lazy_bob),
        ],
        objectives=[
            Objective("welfare", fn=welfare, weight=1.0,
                      description="total apples harvested"),
            Objective("fairness", fn=fairness, weight=morality,
                      description="negative harvest gap between agents"),
        ],
    )

    result = sweep(sim, dial="morality", values=[0.0, 0.1, 0.25, 0.5, 0.75, 1.0], steps=8)

    print("Morality sweep (totals per episode):\n")
    print(result.table())

    frontier = result.pareto(["welfare", "fairness"])
    print("\nPareto frontier (welfare vs fairness):")
    for point in frontier:
        print(
            f"  lambda={point.dial_value:<5} "
            f"welfare={point.mean_totals['welfare']:<6.1f} "
            f"fairness={point.mean_totals['fairness']:.1f}"
        )

    best = result.best("aggregate")
    print(f"\nBest aggregate at lambda={best.dial_value}")


if __name__ == "__main__":
    main()
