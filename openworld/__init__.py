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
from .compose import (Aggregator, Binding, Bridge, CompositeWorld, Route, compile_bridge, legal_actions, observe)
from .ethics import (
    Constraint, Delegate, MoralParliament, constrained, lexicographic,
    maximin, permitted_actions, weighted_sum,
)
from .judge import Judge
from .llm import BaseLLM, MockLLM, OllamaConnectionError, OllamaLLM
from .manyworlds import (BOOLEAN, COUNTING, PROBABILITY, Mechanism, Semiring,
                         WorldStore)
from .objectives import Dial, Objective, ObjectiveSuite
from .pathintegral import LOG, TROPICAL, Skill, TrajectorySpace
from .wavelets import denoise as wavelet_denoise
from .wavelets import dwt, idwt, sparsity
from .sheaf import (glue, is_consistent, localize_fault, majority_glue,
                    obstruction_norm)
from .intervals import Affine, Interval
from .infogeom import bayes_update, expected_info_gain, fisher_information
from .transport import kl_hist, wasserstein1
from .spec import (SPEC_VERSION, SpecError, from_spec, spec_from_json,
                   spec_to_json, to_mermaid, to_spec, validate_spec)
from .card import render_card, render_gallery, to_reactflow
from .optimize import SweepPoint, SweepResult, sweep
from .perceive import (
    CodePerceptor, LLMEmitter, MockPerceptor, Observation, PerceptionError,
    PerceptionGate, Perceptor, TextPerceptor, TranscriptPerceptor, VisionPerceptor,
    image_to_b64,
    sample_frames,
)
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
from .transition import CodeTransition, FunctionTransition, LLMTransition, PhasedTransition, Transition
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
    "legal_actions",
    "observe",
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
    "PhasedTransition",
    "MockLLM",
    "BOOLEAN",
    "COUNTING",
    "PROBABILITY",
    "Mechanism",
    "Semiring",
    "WorldStore",
    "LOG",
    "TROPICAL",
    "Skill",
    "TrajectorySpace",
    "wavelet_denoise",
    "dwt",
    "idwt",
    "sparsity",
    "glue",
    "is_consistent",
    "localize_fault",
    "majority_glue",
    "obstruction_norm",
    "Affine",
    "Interval",
    "bayes_update",
    "expected_info_gain",
    "fisher_information",
    "kl_hist",
    "wasserstein1",
    "SPEC_VERSION",
    "SpecError",
    "to_spec",
    "from_spec",
    "validate_spec",
    "spec_to_json",
    "spec_from_json",
    "to_mermaid",
    "render_card",
    "render_gallery",
    "to_reactflow",
    "Objective",
    "ObjectiveSuite",
    "OllamaConnectionError",
    "OllamaLLM",
    "Observation",
    "Perceptor",
    "PerceptionGate",
    "PerceptionError",
    "MockPerceptor",
    "CodePerceptor",
    "LLMEmitter",
    "TextPerceptor",
    "TranscriptPerceptor",
    "VisionPerceptor",
    "image_to_b64",
    "sample_frames",
    "Route",
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
