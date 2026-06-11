"""Software engineering tutorial: sprint planning under a quality-bar dial.

An engineering team works a backlog. Shipping features accrues tech debt, and
shipping on top of high debt breeds bugs. A 'quality_bar' dial decides how
aggressively the team refactors and fixes instead of shipping, sweeping the
classic delivery-vs-quality frontier.

Demonstrates: the two-model generate/critic relay for synthesizing dynamics
(generator LLM writes the code, a second LLM reviews it), with an offline
MockLLM fallback, plus ground-truth comparison via FunctionTransition.

    python tutorials/software_engineering_sprint.py [generator-model] [critic-model]
"""

import sys

from openworld import (
    Action,
    Agent,
    Dial,
    FunctionTransition,
    MockLLM,
    Objective,
    OllamaConnectionError,
    OllamaLLM,
    Simulation,
    World,
    sweep,
)

REFERENCE_DYNAMICS = '''\
```python
def transition(state, action):
    next_state = dict(state)
    name = action["name"]
    if name == "ship" and next_state["backlog"] > 0:
        next_state["backlog"] -= 1
        next_state["shipped"] += 1
        next_state["debt"] += 1
        next_state["bugs"] += next_state["debt"] // 4
    elif name == "fix":
        next_state["bugs"] = max(0, next_state["bugs"] - 2)
    elif name == "refactor":
        next_state["debt"] = max(0, next_state["debt"] - 2)
    return next_state
```'''


def ground_truth(state, action):
    s = dict(state)
    name = action["name"]
    if name == "ship" and s["backlog"] > 0:
        s["backlog"] -= 1
        s["shipped"] += 1
        s["debt"] += 1
        s["bugs"] += s["debt"] // 4
    elif name == "fix":
        s["bugs"] = max(0, s["bugs"] - 2)
    elif name == "refactor":
        s["debt"] = max(0, s["debt"] - 2)
    return s


def make_world(transition=None, llm=None):
    return World(
        name="sprint",
        description="An engineering team working a sprint backlog.",
        initial_state={"backlog": 12, "shipped": 0, "bugs": 0, "debt": 0},
        actions=["ship", "fix", "refactor"],
        rules=[
            "'ship' (when backlog > 0): backlog -1, shipped +1, debt +1, then bugs "
            "increase by debt // 4 (integer division, using the debt value after the +1).",
            "'fix': bugs decrease by 2, never below 0.",
            "'refactor': debt decreases by 2, never below 0.",
            "'noop' and unknown actions change nothing.",
        ],
        transition=transition,
        llm=llm,
    )


def get_llms(generator_model, critic_model):
    try:
        generator = OllamaLLM(model=generator_model)
        generator.ask("Reply with OK.")
        critic = OllamaLLM(model=critic_model)
        print(f"Using Ollama: generator={generator_model!r}, critic={critic_model!r}")
        return generator, critic
    except OllamaConnectionError:
        print("Ollama not reachable - using MockLLM reference dynamics, no critic")
        return MockLLM([REFERENCE_DYNAMICS]), None


def main():
    generator_model = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5:7b"
    critic_model = sys.argv[2] if len(sys.argv) > 2 else "qwen2.5:3b"

    # --- Part 1: synthesize the dynamics with a generator + critic relay ----
    generator, critic = get_llms(generator_model, critic_model)
    world = make_world(llm=generator)
    print("\nSynthesizing sprint dynamics (generator writes, critic reviews)...")
    transition = world.compile(
        critic=critic,
        invariants=[
            ("counters never negative",
             lambda s: all(s[k] >= 0 for k in ("backlog", "shipped", "bugs", "debt"))),
        ],
    )

    # Spot-check the synthesized code against the hand-written ground truth.
    from openworld import WorldState

    probe = WorldState({"backlog": 5, "shipped": 0, "bugs": 0, "debt": 4})
    synthesized = transition.step(probe, Action("ship"))
    expected = WorldState(ground_truth(dict(probe), Action("ship").to_dict()))
    verdict = "matches" if synthesized == expected else f"DIVERGES: {synthesized.diff(expected)}"
    print(f"Synthesized code vs ground truth on a high-debt ship: {verdict}")

    # --- Part 2: sweep the quality bar -------------------------------------
    quality_bar = Dial("quality_bar", value=0.0)

    def tech_lead(state, actions):
        debt_limit = round((1.0 - quality_bar.value) * 6)
        bug_limit = round((1.0 - quality_bar.value) * 4)
        if state["debt"] > debt_limit:
            return Action("refactor")
        if state["bugs"] > bug_limit:
            return Action("fix")
        if state["backlog"] > 0:
            return Action("ship")
        return Action("fix" if state["bugs"] > 0 else "refactor")

    sim = Simulation(
        world,
        agents=[Agent(name="tech_lead", policy=tech_lead)],
        objectives=[
            Objective("delivery", fn=lambda s, a, ns: float(ns["shipped"] - s["shipped"]),
                      weight=1.0, description="features shipped this step"),
            Objective("quality",
                      fn=lambda s, a, ns: -(ns["bugs"] - s["bugs"]) - 0.5 * (ns["debt"] - s["debt"]),
                      weight=quality_bar,
                      description="negative growth in bugs and tech debt"),
        ],
    )

    result = sweep(sim, dial="quality_bar", values=[0.0, 0.25, 0.5, 0.75, 1.0], steps=14)
    print("\nQuality-bar sweep (totals per 14-step sprint):\n")
    print(result.table())

    print("\nEnd-of-sprint snapshots:")
    for point in result.points:
        final = point.trajectories[0].final_state
        print(
            f"  lambda={point.dial_value:<5} shipped={final['shipped']:<3} "
            f"bugs={final['bugs']:<3} debt={final['debt']}"
        )

    print("\nPareto frontier (delivery vs quality):")
    for point in result.pareto(["delivery", "quality"]):
        print(
            f"  lambda={point.dial_value:<5} delivery={point.mean_totals['delivery']:<5.1f} "
            f"quality={point.mean_totals['quality']:.1f}"
        )


if __name__ == "__main__":
    main()
