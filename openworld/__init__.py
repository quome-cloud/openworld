"""OpenWorld: create, prototype, and optimize world models with an Ollama backbone.

Quickstart:

    from openworld import World, Agent, Simulation, Objective, Dial, OllamaLLM

    llm = OllamaLLM(model="llama3.1")

    world = World(
        name="orchard",
        description="Two agents share an orchard with a limited apple supply.",
        initial_state={"apples": 10, "harvested": {"alice": 0, "bob": 0}},
        actions=["pick", "wait"],
        rules=["'pick' moves one apple from the orchard to the acting agent."],
        llm=llm,
    )
    world.compile()          # LLM writes + verifies executable dynamics code

    alice = Agent(name="alice", goal="harvest as many apples as you can", llm=llm)
    sim = Simulation(world=world, agents=[alice], objectives=[...])
    trajectory = sim.run(steps=10)
"""

from .agent import Agent
from .compose import Aggregator, Binding, Bridge, CompositeWorld, compile_bridge
from .ethics import (
    Constraint, Delegate, MoralParliament, constrained, lexicographic,
    maximin, permitted_actions, weighted_sum,
)
from .judge import Judge
from .llm import BaseLLM, MockLLM, OllamaConnectionError, OllamaLLM
from .objectives import Dial, Objective, ObjectiveSuite
from .optimize import SweepPoint, SweepResult, sweep
from .sandbox import SandboxError
from .simulation import Simulation, StepRecord, Trajectory
from .contextbench import (
    ContextBenchInstance, ContextExample, solve_with_context, solve_without_context,
)
from .state import Action, WorldState
from .swebench import (
    SWEBenchInstance, SWEBenchTransition, build_swebench_world, load_dataset,
    run_instance_tests, solve_in_world, solve_single_shot,
)
from .transition import CodeTransition, FunctionTransition, LLMTransition, Transition
from .tune import Choice, IntRange, Param, Study, Trial, Tuner, Uniform
from .verify import SynthesisError, Verifier, synthesize_transition
from .world import World

__version__ = "0.2.0"

__all__ = [
    "Action",
    "Aggregator",
    "Agent",
    "BaseLLM",
    "Binding",
    "Bridge",
    "Choice",
    "CodeTransition",
    "CompositeWorld",
    "Constraint",
    "compile_bridge",
    "Delegate",
    "Dial",
    "MoralParliament",
    "constrained",
    "lexicographic",
    "maximin",
    "permitted_actions",
    "weighted_sum",
    "IntRange",
    "Judge",
    "Param",
    "Study",
    "Trial",
    "Tuner",
    "Uniform",
    "FunctionTransition",
    "LLMTransition",
    "MockLLM",
    "Objective",
    "ObjectiveSuite",
    "OllamaConnectionError",
    "OllamaLLM",
    "SandboxError",
    "Simulation",
    "StepRecord",
    "SweepPoint",
    "SweepResult",
    "SynthesisError",
    "Trajectory",
    "Transition",
    "Verifier",
    "ContextBenchInstance",
    "ContextExample",
    "solve_with_context",
    "solve_without_context",
    "SWEBenchInstance",
    "SWEBenchTransition",
    "build_swebench_world",
    "load_dataset",
    "run_instance_tests",
    "solve_in_world",
    "solve_single_shot",
    "World",
    "WorldState",
    "sweep",
    "synthesize_transition",
]
