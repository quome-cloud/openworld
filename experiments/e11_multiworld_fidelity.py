"""E11 - Multi-world rollout fidelity (review item W1/W4).

Replicates E1's protocol across all three instrumented worlds with 8 action
scripts each (24 rollouts total): synthesize dynamics once per world, then
compare code vs per-step LLM prediction against the oracle at every step.
Reports exact-rollout counts with Wilson CIs.
"""

import random

from openworld import World, WorldState
from openworld.state import Action
from openworld.transition import LLMTransition

from common import (
    GENERATOR_MODEL, WORLD_SPECS, require_ollama, save_results, wilson_ci,
)

ROLLOUT_STEPS = 20
SCRIPTS_PER_WORLD = 8
SEED = 17


def make_script(spec, rng):
    return [Action(rng.choice(spec["actions"]), agent="alice")
            for _ in range(ROLLOUT_STEPS)]


def oracle_rollout(spec, script):
    state = dict(spec["initial"])
    states = []
    for action in script:
        state = spec["oracle"](state, action.to_dict())
        states.append(dict(state))
    return states


def engine_rollout(transition, spec, script):
    state = WorldState(dict(spec["initial"]))
    states = []
    for action in script:
        state = transition.step(state, action)
        states.append(dict(state))
    return states


def first_divergence(states, oracle_states):
    for i, (s, o) in enumerate(zip(states, oracle_states)):
        if s != o:
            return i + 1
    return None  # exact throughout


def main():
    llm = require_ollama(GENERATOR_MODEL, temperature=0.0)
    rng = random.Random(SEED)
    rows = []
    for spec_name, spec in WORLD_SPECS.items():
        world = World(
            name=spec_name, description=spec["description"],
            initial_state=dict(spec["initial"]), actions=list(spec["actions"]),
            rules=list(spec["rules"]), llm=llm,
        )
        print(f"[{spec_name}] synthesizing...")
        code_transition = world.compile()
        llm_transition = LLMTransition(llm, spec["description"], spec["rules"])
        scripts = [make_script(spec, rng) for _ in range(SCRIPTS_PER_WORLD)]
        oracles = [oracle_rollout(spec, s) for s in scripts]

        for engine_name, engine in (("code_transition", code_transition),
                                    ("llm_transition", llm_transition)):
            divergences = []
            for script, oracle_states in zip(scripts, oracles):
                states = engine_rollout(engine, spec, script)
                divergences.append(first_divergence(states, oracle_states))
            exact = sum(1 for d in divergences if d is None)
            rows.append({
                "world": spec_name,
                "engine": engine_name,
                "n_rollouts": SCRIPTS_PER_WORLD,
                "exact_rollouts": exact,
                "first_divergences": divergences,
                "mean_first_divergence": (
                    sum(d for d in divergences if d is not None)
                    / max(1, sum(1 for d in divergences if d is not None))
                    if any(d is not None for d in divergences) else None
                ),
            })
            print(f"  {engine_name}: exact {exact}/{SCRIPTS_PER_WORLD}")

    totals = {}
    for engine_name in ("code_transition", "llm_transition"):
        engine_rows = [r for r in rows if r["engine"] == engine_name]
        exact = sum(r["exact_rollouts"] for r in engine_rows)
        n = sum(r["n_rollouts"] for r in engine_rows)
        totals[engine_name] = {
            "exact_rollouts": exact, "n": n,
            "rate": exact / n, "ci": list(wilson_ci(exact, n)),
        }
    save_results("e11_multiworld_fidelity", {
        "model": GENERATOR_MODEL, "rollout_steps": ROLLOUT_STEPS, "seed": SEED,
        "rows": rows, "totals": totals,
    })
    for name, t in totals.items():
        print(f"{name}: {t['exact_rollouts']}/{t['n']} exact "
              f"(CI {t['ci'][0]:.2f}-{t['ci'][1]:.2f})")


if __name__ == "__main__":
    main()
