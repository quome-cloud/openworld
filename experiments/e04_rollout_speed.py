"""E4 - Rollout compute efficiency.

Measures planning-rollout throughput (world steps/second) for the symbolic
engine (synthesized code), the hand-written oracle, and the learned-style
LLM engine. The CWM claim: pay for generation once, then plan for free.
"""

import random

from openworld import World, WorldState
from openworld.state import Action
from openworld.transition import FunctionTransition, LLMTransition

from common import (
    GENERATOR_MODEL, SPRINT_ACTIONS, SPRINT_DESCRIPTION, SPRINT_INITIAL,
    SPRINT_RULES, Timer, require_ollama, save_results, sprint_ground_truth,
)

CODE_STEPS = 5000
LLM_STEPS = 10
SEED = 13


def timed_rollout(transition, n_steps, rng):
    state = WorldState(dict(SPRINT_INITIAL))
    with Timer() as t:
        for _ in range(n_steps):
            state = transition.step(state, Action(rng.choice(SPRINT_ACTIONS)))
    return n_steps / t.elapsed


def main():
    llm = require_ollama(GENERATOR_MODEL, temperature=0.0)
    world = World(
        name="sprint", description=SPRINT_DESCRIPTION,
        initial_state=dict(SPRINT_INITIAL), actions=SPRINT_ACTIONS,
        rules=SPRINT_RULES, llm=llm,
    )
    print("Synthesizing symbolic dynamics (one-time cost)...")
    with Timer() as synth_timer:
        code_transition = world.compile()

    engines = []
    rng = random.Random(SEED)
    engines.append({
        "engine": "code_transition",
        "steps_per_second": timed_rollout(code_transition, CODE_STEPS, rng),
        "measured_steps": CODE_STEPS,
        "one_time_synthesis_seconds": synth_timer.elapsed,
    })
    rng = random.Random(SEED)
    engines.append({
        "engine": "function_oracle",
        "steps_per_second": timed_rollout(
            FunctionTransition(sprint_ground_truth), CODE_STEPS, rng),
        "measured_steps": CODE_STEPS,
    })
    rng = random.Random(SEED)
    engines.append({
        "engine": "llm_transition",
        "steps_per_second": timed_rollout(
            LLMTransition(llm, SPRINT_DESCRIPTION, SPRINT_RULES), LLM_STEPS, rng),
        "measured_steps": LLM_STEPS,
    })

    save_results("e04_rollout_speed", {
        "model": GENERATOR_MODEL, "seed": SEED, "engines": engines,
    })
    for e in engines:
        print(f"  {e['engine']}: {e['steps_per_second']:.1f} steps/s")


if __name__ == "__main__":
    main()
