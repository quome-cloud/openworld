"""Finance tutorial: portfolio rebalancing under a caution dial.

A trader rebalances one position against a deterministic price path embedded
in the world state. A 'caution' dial sets the target market exposure and
weights a safety objective, sweeping the frontier between growth and risk.

Demonstrates: deterministic schedules carried inside the state (so dynamics
stay a pure function), float-valued state, and best-setting selection.

    python tutorials/finance_portfolio.py
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

# A choppy but upward price path: rallies punctuated by drawdowns.
PRICES = [100, 104, 101, 107, 112, 106, 113, 119, 111, 118, 124, 117, 125]


def equity(state):
    return state["cash"] + state["shares"] * state["price"]


def market_dynamics(state, action):
    s = dict(state)
    name = action["name"]
    if name == "buy" and s["cash"] >= s["price"]:
        s["cash"] -= s["price"]
        s["shares"] += 1
    elif name == "sell" and s["shares"] > 0:
        s["cash"] += s["price"]
        s["shares"] -= 1
    # The market ticks after the order: next price from the embedded path.
    if s["t"] + 1 < len(s["prices"]):
        s["t"] += 1
        s["price"] = s["prices"][s["t"]]
    return s


def main():
    caution = Dial("caution", value=0.0)

    def rebalancer(state, actions):
        """Trade toward a target exposure of (1 - caution) of total equity."""
        total = equity(state)
        exposure = (state["shares"] * state["price"]) / total if total else 0.0
        target = 1.0 - caution.value
        if exposure < target and state["cash"] >= state["price"]:
            return Action("buy")
        if exposure > target + 0.10 and state["shares"] > 0:
            return Action("sell")
        return Action("hold")

    world = World(
        name="portfolio",
        description="A single-asset portfolio rebalanced against a price path.",
        initial_state={
            "t": 0, "price": PRICES[0], "prices": list(PRICES),
            "cash": 1000.0, "shares": 0,
        },
        actions=["buy", "sell", "hold"],
        transition=FunctionTransition(market_dynamics),
    )

    sim = Simulation(
        world,
        agents=[Agent(name="trader", policy=rebalancer)],
        objectives=[
            Objective("growth", fn=lambda s, a, ns: equity(ns) - equity(s), weight=1.0,
                      description="change in total equity this step"),
            Objective("safety", fn=lambda s, a, ns: -(ns["shares"] * ns["price"]) / equity(ns),
                      weight=caution,
                      description="negative market exposure after the step"),
        ],
    )

    result = sweep(sim, dial="caution", values=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                   steps=len(PRICES) - 1)
    print("Caution sweep (totals per run):\n")
    print(result.table())

    print("\nFinal books:")
    for point in result.points:
        final = point.trajectories[0].final_state
        print(
            f"  caution={point.dial_value:<4} equity=${equity(final):<8.0f} "
            f"cash=${final['cash']:<7.0f} shares={final['shares']}"
        )

    print("\nPareto frontier (growth vs safety):")
    for point in result.pareto(["growth", "safety"]):
        print(
            f"  caution={point.dial_value:<4} growth={point.mean_totals['growth']:<7.1f} "
            f"avg-exposure={-point.mean_totals['safety'] / (len(PRICES) - 1):.2f}"
        )

    best = result.best("aggregate")
    print(f"\nBest dial-weighted aggregate at caution={best.dial_value}")


if __name__ == "__main__":
    main()
