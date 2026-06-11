"""Legaltech tutorial: settlement negotiation between two counsel agents.

Plaintiff and defense counsel negotiate a settlement. Every turn of posturing
burns legal fees. The defense follows a fixed playbook; the plaintiff's
'cooperativeness' dial controls concession size and willingness to accept,
sweeping the trade-off between recovery amount and litigation cost: hold out
for more, or settle early and cheap.

Demonstrates: multi-agent simulation, asymmetric policies, and objectives
that fire on discrete events (the settlement step).

    python tutorials/legaltech_settlement.py
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


def negotiation_dynamics(state, action):
    """Amounts in $k. Once settled, the world is inert."""
    s = dict(state)
    if s["settled"]:
        return s
    name, agent = action["name"], action.get("agent")
    if name == "concede":
        amount = int(action["params"].get("amount", 5))
        if agent == "plaintiff_counsel":
            s["demand"] = max(s["offer"], s["demand"] - amount)
        elif agent == "defense_counsel":
            s["offer"] = min(s["demand"], s["offer"] + amount)
    elif name == "accept":
        s["settled"] = True
        # Accepting takes the other side's current number.
        s["amount"] = s["offer"] if agent == "plaintiff_counsel" else s["demand"]
    # Every un-settled turn bills both sides.
    if not s["settled"]:
        s["fees"] += 2
        s["round"] += 1
    return s


def main():
    cooperativeness = Dial("cooperativeness", value=0.0)

    def plaintiff(state, actions):
        gap = state["demand"] - state["offer"]
        tolerance = round(cooperativeness.value * 30)  # acceptable gap in $k
        if gap <= tolerance:
            return Action("accept")
        concession = 1 + round(cooperativeness.value * 9)
        return Action("concede", params={"amount": concession})

    def defense(state, actions):
        # Fixed playbook: raise the offer steadily, take any near deal.
        if state["demand"] - state["offer"] <= 5:
            return Action("accept")
        return Action("concede", params={"amount": 4})

    world = World(
        name="settlement",
        description="A two-party settlement negotiation over a $90k demand.",
        initial_state={
            "round": 0, "demand": 90, "offer": 10,
            "settled": False, "amount": 0, "fees": 0,
        },
        actions=["concede", "accept", "hold"],
        transition=FunctionTransition(negotiation_dynamics),
    )

    sim = Simulation(
        world,
        agents=[
            Agent(name="plaintiff_counsel", policy=plaintiff),
            Agent(name="defense_counsel", policy=defense),
        ],
        objectives=[
            Objective(
                "recovery",
                fn=lambda s, a, ns: float(ns["amount"]) if ns["settled"] and not s["settled"] else 0.0,
                weight=1.0,
                description="settlement amount, scored once on the settling step",
            ),
            Objective(
                "cost_control",
                fn=lambda s, a, ns: -float(ns["fees"] - s["fees"]),
                weight=cooperativeness,
                description="negative legal fees billed this turn",
            ),
        ],
    )

    result = sweep(
        sim, dial="cooperativeness", values=[0.0, 0.25, 0.5, 0.75, 1.0], steps=20
    )
    print("Cooperativeness sweep (totals per matter):\n")
    print(result.table())

    print("\nOutcome detail:")
    for point in result.points:
        final = point.trajectories[0].final_state
        status = f"settled at ${final['amount']}k in round {final['round']}" if final["settled"] else "NO SETTLEMENT"
        print(f"  lambda={point.dial_value:<5} {status:<34} total fees ${final['fees']}k")

    print("\nPareto frontier (recovery vs cost_control):")
    for point in result.pareto(["recovery", "cost_control"]):
        print(
            f"  lambda={point.dial_value:<5} recovery={point.mean_totals['recovery']:<6.1f} "
            f"fees={-point.mean_totals['cost_control']:.1f}"
        )


if __name__ == "__main__":
    main()
