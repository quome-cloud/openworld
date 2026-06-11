"""Healthcare tutorial: ICU triage under a resource-stewardship dial.

A charge nurse works a queue of critical and moderate patients with a limited
treatment budget. Untreated critical patients deteriorate every other step.
A 'stewardship' dial trades total health outcomes against spend.

Demonstrates: LLM-synthesized dynamics with safety invariants (falls back to
MockLLM when Ollama is unreachable), policy agents, and a dial sweep.

    python tutorials/healthcare_triage.py [model-name]
"""

import sys

from openworld import (
    Action,
    Agent,
    Dial,
    MockLLM,
    Objective,
    OllamaConnectionError,
    OllamaLLM,
    Simulation,
    World,
    sweep,
)

# Ground-truth dynamics used by MockLLM when Ollama is offline; with a live
# model, the LLM writes (and the verifier checks) this code itself.
REFERENCE_DYNAMICS = '''\
```python
def transition(state, action):
    next_state = dict(state)
    name = action["name"]
    if name == "treat_critical" and next_state["critical_waiting"] > 0:
        next_state["critical_waiting"] -= 1
        next_state["treated"] += 1
        next_state["outcomes"] += 3
        next_state["spend"] += 3
    elif name == "treat_moderate" and next_state["moderate_waiting"] > 0:
        next_state["moderate_waiting"] -= 1
        next_state["treated"] += 1
        next_state["outcomes"] += 1
        next_state["spend"] += 1
    # The clock advances after every action; on every second tick one
    # untreated critical patient deteriorates.
    next_state["tick"] += 1
    if next_state["tick"] % 2 == 0 and next_state["critical_waiting"] > 0:
        next_state["critical_waiting"] -= 1
        next_state["deteriorated"] += 1
        next_state["outcomes"] -= 2
    return next_state
```'''


def get_llm(model):
    llm = OllamaLLM(model=model)
    try:
        llm.ask("Reply with OK.")
        print(f"Using Ollama model {model!r}")
        return llm
    except OllamaConnectionError:
        print("Ollama not reachable - using MockLLM reference dynamics")
        return MockLLM([REFERENCE_DYNAMICS])


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5:7b"

    world = World(
        name="icu-triage",
        description=(
            "An ICU triage queue. Critical and moderate patients wait for "
            "treatment; untreated critical patients deteriorate over time."
        ),
        initial_state={
            "tick": 0,
            "critical_waiting": 4,
            "moderate_waiting": 8,
            "treated": 0,
            "deteriorated": 0,
            "outcomes": 0,
            "spend": 0,
        },
        actions=["treat_critical", "treat_moderate", "wait"],
        rules=[
            "'treat_critical' treats one waiting critical patient: treated +1, outcomes +3, spend +3.",
            "'treat_moderate' treats one waiting moderate patient: treated +1, outcomes +1, spend +1.",
            "Treating when the matching queue is empty does nothing (besides the clock).",
            "After EVERY action (including 'wait' and 'noop'), tick increases by 1.",
            "Whenever the new tick is even and critical patients still wait, one of them "
            "deteriorates: critical_waiting -1, deteriorated +1, outcomes -2.",
        ],
        llm=get_llm(model),
    )

    print("\nSynthesizing verified triage dynamics...")
    world.compile(
        invariants=[
            ("queues never negative", lambda s: s["critical_waiting"] >= 0 and s["moderate_waiting"] >= 0),
            ("spend never negative", lambda s: s["spend"] >= 0),
        ],
    )
    print("Dynamics accepted.\n")

    stewardship = Dial("stewardship", value=0.0)

    def charge_nurse(state, actions):
        # Critical patients always come first. Moderates are treated only
        # while spend stays under the dial-controlled budget cap.
        if state["critical_waiting"] > 0:
            return Action("treat_critical")
        budget_cap = round((1.0 - stewardship.value) * 14)
        if state["moderate_waiting"] > 0 and state["spend"] + 1 <= budget_cap:
            return Action("treat_moderate")
        return Action("wait")

    sim = Simulation(
        world,
        agents=[Agent(name="charge_nurse", policy=charge_nurse)],
        objectives=[
            Objective("outcomes", fn=lambda s, a, ns: ns["outcomes"] - s["outcomes"], weight=1.0,
                      description="health outcome points gained this step"),
            Objective("thrift", fn=lambda s, a, ns: -(ns["spend"] - s["spend"]), weight=stewardship,
                      description="negative spend this step"),
        ],
    )

    result = sweep(sim, dial="stewardship", values=[0.0, 0.25, 0.5, 0.75, 1.0], steps=12)
    print("Stewardship sweep (totals per 12-step shift):\n")
    print(result.table())

    print("\nPareto frontier (outcomes vs thrift):")
    for point in result.pareto(["outcomes", "thrift"]):
        final = point.trajectories[0].final_state
        print(
            f"  lambda={point.dial_value:<5} outcomes={point.mean_totals['outcomes']:<6.1f} "
            f"spend={final['spend']:<3} treated={final['treated']} deteriorated={final['deteriorated']}"
        )


if __name__ == "__main__":
    main()
