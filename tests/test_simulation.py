"""End-to-end tests: agents, simulation, sweeps, Pareto frontier."""

import pytest

from openworld import (
    Action,
    Agent,
    Dial,
    FunctionTransition,
    MockLLM,
    Objective,
    Simulation,
    World,
    sweep,
)


def orchard_dynamics(state, action):
    """Two agents share an orchard; picking depletes a common pool."""
    next_state = dict(state)
    agent = action.get("agent")
    if action["name"] == "pick" and next_state["apples"] > 0 and agent:
        next_state["apples"] -= 1
        next_state["harvested"] = dict(next_state["harvested"])
        next_state["harvested"][agent] = next_state["harvested"].get(agent, 0) + 1
    return next_state


def make_world():
    return World(
        name="orchard",
        description="A shared orchard.",
        initial_state={"apples": 10, "harvested": {"alice": 0, "bob": 0}},
        actions=["pick", "wait"],
        transition=FunctionTransition(orchard_dynamics),
    )


def greedy(state, actions):
    return Action("pick" if state["apples"] > 0 else "wait")


def fairness(state, action, next_state):
    counts = list(next_state["harvested"].values())
    return -(max(counts) - min(counts))  # 0 when equal, negative when skewed


def test_simulation_runs_and_records():
    world = make_world()
    alice = Agent(name="alice", policy=greedy)
    bob = Agent(name="bob", policy=greedy)
    sim = Simulation(
        world,
        [alice, bob],
        objectives=[Objective("welfare", fn=lambda s, a, ns: ns["apples"] - s["apples"], weight=-1.0)],
    )
    trajectory = sim.run(steps=3)
    assert len(trajectory.steps) == 6  # 3 steps x 2 agents
    assert trajectory.final_state["apples"] == 4
    assert trajectory.final_state["harvested"] == {"alice": 3, "bob": 3}
    assert trajectory.totals()["welfare"] == -6.0
    assert trajectory.steps[0].agent == "alice"


def test_simulation_reset_between_runs():
    world = make_world()
    sim = Simulation(world, [Agent(name="alice", policy=greedy)])
    sim.run(steps=2)
    trajectory = sim.run(steps=2)
    assert trajectory.initial_state["apples"] == 10


def test_llm_agent_parses_action_and_falls_back():
    world = make_world()
    llm = MockLLM([
        '{"action": "pick", "params": {}, "reason": "apples remain"}',
        "I think I will fly to the moon",  # unparseable -> noop
    ])
    agent = Agent(name="alice", goal="harvest apples", llm=llm)
    sim = Simulation(world, [agent])
    trajectory = sim.run(steps=2)
    assert trajectory.steps[0].action.name == "pick"
    assert trajectory.steps[1].action.name == "noop"
    assert any("noop" in warning for warning in trajectory.warnings)


def test_dial_sweep_traces_tradeoff():
    """A morality dial that throttles greed: higher lambda -> fairer harvest."""
    world = make_world()
    morality = Dial("morality", value=0.0)

    def considerate(state, actions):
        # Alice picks only when she is not ahead of Bob by more than the
        # dial allows (scaled tolerance: lambda=0 -> always pick).
        lead = state["harvested"]["alice"] - state["harvested"]["bob"]
        allowed_lead = round((1.0 - morality.value) * 5)
        if state["apples"] > 0 and lead <= allowed_lead:
            return Action("pick")
        return Action("wait")

    alice = Agent(name="alice", policy=considerate)
    bob = Agent(name="bob", policy=lambda s, a: Action("wait"))
    sim = Simulation(
        world,
        [alice, bob],
        objectives=[
            Objective("welfare", fn=lambda s, a, ns: s["apples"] - ns["apples"], weight=1.0),
            Objective("fairness", fn=fairness, weight=morality),
        ],
    )
    result = sweep(sim, dial="morality", values=[0.0, 0.5, 1.0], steps=8)
    assert [point.dial_value for point in result.points] == [0.0, 0.5, 1.0]
    welfare = [point.mean_totals["welfare"] for point in result.points]
    fairness_scores = [point.mean_totals["fairness"] for point in result.points]
    assert welfare[0] > welfare[-1]            # greed maximizes welfare
    assert fairness_scores[-1] > fairness_scores[0]  # morality maximizes fairness
    frontier = result.pareto(["welfare", "fairness"])
    assert len(frontier) >= 2  # a real trade-off, not a single dominant point
    assert "morality" in result.table()


def test_set_dial_validation():
    world = make_world()
    sim = Simulation(world, [Agent(name="a", policy=greedy)])
    with pytest.raises(KeyError):
        sim.set_dial("nonexistent", 0.5)


def test_world_requires_dynamics():
    world = World(
        name="empty",
        description="",
        initial_state={},
        actions=["go"],
    )
    with pytest.raises(RuntimeError):
        world.step(Action("go"))
