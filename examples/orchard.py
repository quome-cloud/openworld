"""Orchard quickstart: synthesize verified world dynamics, then let an LLM
agent act in the world.

Runs against a local Ollama server if one is up; otherwise falls back to a
scripted MockLLM so the example always works.

    python examples/orchard.py [model-name]
"""

import sys

from openworld import (
    Agent,
    MockLLM,
    Objective,
    OllamaConnectionError,
    OllamaLLM,
    Simulation,
    World,
)

MOCK_DYNAMICS = """\
```python
def transition(state, action):
    next_state = dict(state)
    next_state["harvested"] = dict(next_state["harvested"])
    agent = action.get("agent")
    if action["name"] == "pick" and next_state["apples"] > 0 and agent:
        next_state["apples"] -= 1
        next_state["harvested"][agent] = next_state["harvested"].get(agent, 0) + 1
    return next_state
```"""


def get_llm(model: str):
    llm = OllamaLLM(model=model)
    try:
        llm.ask("Reply with OK.")
        print(f"Using Ollama model {model!r}")
        return llm
    except OllamaConnectionError:
        print("Ollama not reachable - using MockLLM (scripted responses)")
        return MockLLM(
            [MOCK_DYNAMICS] + ['{"action": "pick", "params": {}, "reason": "apples remain"}'] * 50
        )


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "llama3.1"
    llm = get_llm(model)

    world = World(
        name="orchard",
        description="Agents share an orchard with a limited pool of apples.",
        initial_state={"apples": 10, "harvested": {"alice": 0}},
        actions=["pick", "wait"],
        rules=[
            "'pick' moves one apple from the orchard to the acting agent's harvested count.",
            "Picking when no apples remain does nothing.",
            "'wait' and 'noop' leave the state unchanged.",
        ],
        llm=llm,
    )

    print("\nSynthesizing verified transition code...")
    transition = world.compile(
        invariants=[("apple count never negative", lambda s: s["apples"] >= 0)],
        save_to="orchard_dynamics.py",
    )
    print("Accepted dynamics (saved to orchard_dynamics.py):\n")
    print(transition.code)

    alice = Agent(name="alice", goal="Harvest as many apples as possible.", llm=llm)
    sim = Simulation(
        world,
        agents=[alice],
        objectives=[
            Objective("welfare", fn=lambda s, a, ns: s["apples"] - ns["apples"]),
        ],
        on_step=lambda r: print(
            f"  step {r.step}: {r.agent} -> {r.action.name:6s} apples={r.state['apples']}"
        ),
    )

    print("\nRunning simulation:")
    trajectory = sim.run(steps=5)
    print(f"\nFinal state: {trajectory.final_state.to_json()}")
    print(f"Objective totals: {trajectory.totals()}")


if __name__ == "__main__":
    main()
