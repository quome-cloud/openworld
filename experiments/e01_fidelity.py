"""E1 - Dynamics fidelity and compounding rollout error.

Compares the symbolic engine (verified synthesized code, compiled once) with
the learned-style engine (LLM predicts each next state directly) against a
ground-truth oracle over 20-step rollouts on the sprint world. Measures
per-step exact-match rate, first divergence step, and final-state L1 error.
"""

import random

from openworld import World, WorldState
from openworld.state import Action
from openworld.transition import LLMTransition

from common import (
    GENERATOR_MODEL, SPRINT_ACTIONS, SPRINT_DESCRIPTION, SPRINT_INITIAL,
    SPRINT_RULES, require_ollama, save_results, sprint_ground_truth,
)

ROLLOUT_STEPS = 20
N_SCRIPTS = 3
SEED = 11


def make_action_script(rng):
    return [Action(rng.choice(SPRINT_ACTIONS)) for _ in range(ROLLOUT_STEPS)]


def rollout(transition, script):
    state = WorldState(dict(SPRINT_INITIAL))
    states = []
    for action in script:
        state = transition.step(state, action)
        states.append(dict(state))
    return states


def oracle_rollout(script):
    state = dict(SPRINT_INITIAL)
    states = []
    for action in script:
        state = sprint_ground_truth(state, action.to_dict())
        states.append(dict(state))
    return states


def l1_error(a, b):
    keys = set(a) | set(b)
    total = 0.0
    for k in keys:
        va, vb = a.get(k, 0), b.get(k, 0)
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            total += abs(va - vb)
        elif va != vb:
            total += 1.0
    return total


def evaluate(name, transition, scripts, oracles):
    per_step_matches = [0] * ROLLOUT_STEPS
    first_divergences = []
    final_errors = []
    for script, oracle_states in zip(scripts, oracles):
        states = rollout(transition, script)
        diverged = None
        for i, (s, o) in enumerate(zip(states, oracle_states)):
            if s == o:
                per_step_matches[i] += 1
            elif diverged is None:
                diverged = i + 1
        first_divergences.append(diverged or ROLLOUT_STEPS + 1)
        final_errors.append(l1_error(states[-1], oracle_states[-1]))
    return {
        "engine": name,
        "per_step_match_rate": [m / len(scripts) for m in per_step_matches],
        "mean_first_divergence_step": sum(first_divergences) / len(first_divergences),
        "mean_final_l1_error": sum(final_errors) / len(final_errors),
        "exact_full_rollouts": sum(1 for d in first_divergences if d > ROLLOUT_STEPS),
        "n_rollouts": len(scripts),
    }


def main():
    rng = random.Random(SEED)
    scripts = [make_action_script(rng) for _ in range(N_SCRIPTS)]
    oracles = [oracle_rollout(s) for s in scripts]

    llm = require_ollama(GENERATOR_MODEL, temperature=0.0)

    # Symbolic engine: synthesize once, then roll out for free.
    world = World(
        name="sprint", description=SPRINT_DESCRIPTION,
        initial_state=dict(SPRINT_INITIAL), actions=SPRINT_ACTIONS,
        rules=SPRINT_RULES, llm=llm,
    )
    print("Synthesizing symbolic dynamics...")
    code_transition = world.compile(
        invariants=[("counters never negative",
                     lambda s: all(s[k] >= 0 for k in ("backlog", "shipped", "bugs", "debt")))],
    )

    # Learned-style engine: the LLM predicts each next state.
    llm_transition = LLMTransition(llm, SPRINT_DESCRIPTION, SPRINT_RULES)

    print(f"Rolling out {N_SCRIPTS} x {ROLLOUT_STEPS} steps per engine...")
    results = {
        "model": GENERATOR_MODEL,
        "rollout_steps": ROLLOUT_STEPS,
        "seed": SEED,
        "engines": [
            evaluate("code_transition", code_transition, scripts, oracles),
            evaluate("llm_transition", llm_transition, scripts, oracles),
        ],
        "synthesized_code": code_transition.code,
    }
    save_results("e01_fidelity", results)
    for engine in results["engines"]:
        print(f"  {engine['engine']}: first divergence at step "
              f"{engine['mean_first_divergence_step']:.1f}, "
              f"final L1 {engine['mean_final_l1_error']:.1f}")


if __name__ == "__main__":
    main()
