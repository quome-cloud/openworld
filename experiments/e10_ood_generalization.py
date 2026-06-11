"""E10 - Out-of-distribution scale generalization.

Synthesized code learned the sprint world at backlog=12; here both engines
are probed at 10x state magnitudes never seen in any prompt or smoke-run.
The CWM claim: programs encode abstract rules, so they generalize across
scale, while per-step neural/LLM prediction drifts.
"""

from openworld import World, WorldState
from openworld.transition import LLMTransition

from common import (
    GENERATOR_MODEL, SPRINT_ACTIONS, SPRINT_DESCRIPTION, SPRINT_INITIAL,
    SPRINT_PROBES, SPRINT_PROBES_SCALED, SPRINT_RULES, require_ollama,
    save_results, sprint_ground_truth, wilson_ci,
)


def per_probe_matches(transition, probes):
    matches = []
    for state, action in probes:
        expected = sprint_ground_truth(dict(state), action.to_dict())
        try:
            actual = dict(transition.step(WorldState(state), action))
        except Exception:
            actual = None
        matches.append(actual == expected)
    return matches


def main():
    llm = require_ollama(GENERATOR_MODEL, temperature=0.0)
    world = World(
        name="sprint", description=SPRINT_DESCRIPTION,
        initial_state=dict(SPRINT_INITIAL), actions=SPRINT_ACTIONS,
        rules=SPRINT_RULES, llm=llm,
    )
    print("Synthesizing symbolic dynamics at in-distribution scale...")
    code_transition = world.compile()
    llm_transition = LLMTransition(llm, SPRINT_DESCRIPTION, SPRINT_RULES)

    rows = []
    for engine_name, engine in (("code_transition", code_transition),
                                ("llm_transition", llm_transition)):
        for scale_name, probes in (("in_distribution", SPRINT_PROBES),
                                   ("scaled_10x", SPRINT_PROBES_SCALED)):
            matches = per_probe_matches(engine, probes)
            n_match = sum(matches)
            rows.append({
                "engine": engine_name,
                "probes": scale_name,
                "n": len(matches),
                "exact_match_rate": n_match / len(matches),
                "ci": list(wilson_ci(n_match, len(matches))),
            })
            print(f"  {engine_name} on {scale_name}: {n_match}/{len(matches)}")

    save_results("e10_ood_generalization", {
        "model": GENERATOR_MODEL,
        "rows": rows,
        "synthesized_code": code_transition.code,
    })


if __name__ == "__main__":
    main()
