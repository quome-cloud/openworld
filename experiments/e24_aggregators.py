"""E24 - Aggregation rules are ethical commitments (round-4 items P1/P3).

Same commons world, same policy family (alice's restraint parameter r),
three aggregation rules choosing the operating point:

  utilitarian  - maximize total harvest (the weighted-sum default)
  maximin      - maximize the worst-off agent's harvest (Rawls)
  leximin      - maximin with lexical tie-breaking (worst-off first,
                 then next, then total)

Each rule looks at the same outcome table and picks a different world to
live in; the differences (total vs worst-off vs gap) quantify what each
ethical theory pays for what it protects. Deterministic, no LLM.
"""

from openworld import Action, Agent, FunctionTransition, Simulation, World

from common import save_results

R_VALUES = [round(0.1 * i, 1) for i in range(11)]
STEPS = 8


def commons_dynamics(state, action):
    """A scarce commons: 9 apples, 8 rounds. Alice can pick every round; bob
    only manages every other round. Under scarcity, alice's restraint is what
    leaves apples standing for bob's slower turns - efficiency (total picked
    before time runs out) genuinely trades against equality."""
    s = dict(state)
    s["harvested"] = dict(s["harvested"])
    agent = action.get("agent")
    if action["name"] == "pick" and s["apples"] > 0 and agent:
        s["apples"] -= 1
        s["harvested"][agent] += 1
    return s


def episode(restraint):
    def alice_policy(state, actions):
        lead = state["harvested"]["alice"] - state["harvested"]["bob"]
        allowed_lead = round((1.0 - restraint) * 5)
        if state["apples"] > 0 and lead <= allowed_lead:
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
        initial_state={"apples": 9, "harvested": {"alice": 0, "bob": 0}},
        actions=["pick", "wait"],
        transition=FunctionTransition(commons_dynamics),
    )
    sim = Simulation(world, agents=[
        Agent(name="alice", policy=alice_policy),
        Agent(name="bob", policy=lazy_bob),
    ])
    final = sim.run(steps=STEPS).final_state["harvested"]
    return {"r": restraint, "alice": final["alice"], "bob": final["bob"],
            "total": final["alice"] + final["bob"],
            "worst_off": min(final["alice"], final["bob"]),
            "gap": abs(final["alice"] - final["bob"])}


def main():
    outcomes = [episode(r) for r in R_VALUES]
    for o in outcomes:
        print(f"  r={o['r']}: alice={o['alice']} bob={o['bob']} "
              f"total={o['total']} worst={o['worst_off']}")

    chosen = {
        "utilitarian_sum": max(outcomes, key=lambda o: o["total"]),
        "maximin": max(outcomes, key=lambda o: o["worst_off"]),
        "leximin": max(outcomes, key=lambda o: (
            o["worst_off"], max(o["alice"], o["bob"]))),
    }
    save_results("e24_aggregators", {
        "r_values": R_VALUES, "steps": STEPS, "outcomes": outcomes,
        "chosen": chosen,
        "summary": {
            rule: {"r": c["r"], "total": c["total"],
                   "worst_off": c["worst_off"], "gap": c["gap"]}
            for rule, c in chosen.items()
        },
    })
    for rule, c in chosen.items():
        print(f"{rule}: picks r={c['r']} -> total={c['total']}, "
              f"worst-off={c['worst_off']}, gap={c['gap']}")


if __name__ == "__main__":
    main()
