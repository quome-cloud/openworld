"""E33 - Dynamic rules x composition x traversal: when the rules change, agents move.

A deterministic, offline demonstration (no LLM, like E31). Two city economies
compose into a world joined by a toll route. City c0 runs a PhasedTransition:
after its 8th step it enters austerity - work yields nothing and the observed
`rate` drops to zero. A greedy policy agent works wherever its scoped
observe() view shows the best rate, so when c0's regime turns, the agent pays
the toll and re-locates to c1.

The same 20-step scenario runs twice: WITH the route and WITHOUT it (the
stranded counterfactual). The gap between the two world-GDP trajectories is
what mobility is worth under a regime shock: composition provides somewhere
to go, dynamic rules provide the reason, traversal is how behavior adapts.

Every number is deterministic; results feed fig_dynamic_traversal in
scripts/make_paper_assets.py.
"""

from openworld import Action, World
from openworld.compose import AGENTS_KEY, Aggregator, CompositeWorld, Route, legal_actions, observe
from openworld.transition import PhasedTransition, Transition

from common import save_results

SWITCH_AFTER = 8   # c0 enters austerity after its 8th step
STEPS = 20
TOLL = 2


class WorkTransition(Transition):
    """'work' adds the city's rate to its gdp."""

    def step(self, state, action):
        s = state.copy()
        if action.name == "work":
            s["gdp"] += s["rate"]
        return s


class AusterityTransition(Transition):
    """Austerity regime: the observable rate collapses; work yields nothing."""

    def step(self, state, action):
        s = state.copy()
        s["rate"] = 0
        return s


class TollTransition(Transition):
    """Crossing pays TOLL coins into the destination treasury."""

    def step(self, state, action):
        s = state.copy()
        if action.name == "cross" and s["agent"].get("coins", 0) >= TOLL:
            s["agent"]["coins"] -= TOLL
            s["b"]["treasury"] = s["b"].get("treasury", 0) + TOLL
        return s


def make_city(rate, phased=False):
    transition = WorkTransition()
    if phased:
        transition = PhasedTransition([(0, WorkTransition()),
                                       (SWITCH_AFTER, AusterityTransition())])
    return World(
        name="city",
        description="a city whose work action yields its current rate",
        initial_state={"gdp": 0, "rate": rate, "treasury": 0},
        actions=["work", "wait"],
        transition=transition,
    )


def make_world(with_route):
    bridges = [Route("toll-road", "c0", "c1", transition=None,
                     on_cross=TollTransition())] if with_route else []
    return CompositeWorld(
        name="economy",
        children={"c0": make_city(rate=3, phased=True), "c1": make_city(rate=2)},
        agents={"alice": {"at": "c0", "coins": 5}},
        bridges=bridges,
        aggregators=[Aggregator("world_gdp",
                                lambda kids: sum(c["gdp"] for c in kids.values()))],
    )


def greedy_policy(comp, state):
    """Work where the observed rate is best; cross when a neighbor beats home."""
    view = observe(comp, state, "alice")
    here = view["location"]
    local_rate = view["local"]["rate"]
    for dest, summary in view["neighbors"].items():
        if summary.get("rate", 0) > local_rate:
            if f"travel:{dest}" in legal_actions(comp, state, "alice"):
                return Action("travel", params={"agent": "alice", "to": dest})
    return Action(f"{here}:work", agent="alice")


def run(with_route):
    comp = make_world(with_route)
    records = []
    for step in range(1, STEPS + 1):
        action = greedy_policy(comp, comp.state)
        state = comp.step(action)
        records.append({
            "step": step,
            "action": action.name,
            "agent_at": state[AGENTS_KEY]["alice"]["at"],
            "coins": state[AGENTS_KEY]["alice"]["coins"],
            "c0_phase": state["c0"].get("_phase", 0),
            "c0_gdp": state["c0"]["gdp"],
            "c1_gdp": state["c1"]["gdp"],
            "c1_treasury": state["c1"]["treasury"],
            "world_gdp": state["_agg"]["world_gdp"],
        })
    return records


def main():
    with_route = run(with_route=True)
    stranded = run(with_route=False)

    travel_steps = [r["step"] for r in with_route if r["action"] == "travel"]
    switch_step = next(r["step"] for r in with_route if r["c0_phase"] == 1)
    summary = {
        "switch_after_c0_steps": SWITCH_AFTER,
        "c0_phase1_first_seen_at_step": switch_step,
        "travel_step": travel_steps[0] if travel_steps else None,
        "toll_paid": TOLL,
        "final_world_gdp_with_route": with_route[-1]["world_gdp"],
        "final_world_gdp_stranded": stranded[-1]["world_gdp"],
        "mobility_gain": (with_route[-1]["world_gdp"]
                          - stranded[-1]["world_gdp"]),
    }
    save_results("e33_dynamic_traversal", {
        "steps": STEPS,
        "summary": summary,
        "with_route": with_route,
        "stranded": stranded,
    })
    print(f"switch first visible at step {switch_step}; "
          f"travel at step {summary['travel_step']}")
    print(f"final world gdp: with route {summary['final_world_gdp_with_route']}, "
          f"stranded {summary['final_world_gdp_stranded']} "
          f"(mobility gain +{summary['mobility_gain']})")


if __name__ == "__main__":
    main()
