"""E22 - Planning utility: does the simulator improve decisions? (Q3)

Model-predictive lookahead on the sprint world: at each step the planner
exhaustively evaluates action sequences of depth d inside its world model,
executes the best first action IN THE GROUND-TRUTH ENVIRONMENT, and replans.
Value function: shipped - bugs - 0.5*debt at the horizon.

Planners compared (all executed in the oracle environment):
  code_d3    - depth-3 lookahead through the verified synthesized program
  oracle_d3  - depth-3 lookahead through the ground truth (upper bound)
  llm_d2     - depth-2 lookahead through the LLM next-state engine
               (deeper is computationally prohibitive - that is the point)
  reactive   - hand-written heuristic policy, no model
  random     - uniform random actions (5 seeds)

Also reports planning wall-clock per episode.
"""

import json
import random
import time
from pathlib import Path

from openworld import WorldState
from openworld.state import Action
from openworld.transition import CodeTransition, FunctionTransition, LLMTransition

from common import (
    GENERATOR_MODEL, RESULTS_DIR, SPRINT_DESCRIPTION, SPRINT_INITIAL,
    SPRINT_RULES, require_ollama, save_results, sprint_ground_truth,
)

ACTIONS = ["ship", "fix", "refactor"]
EPISODE_STEPS = 12
LLM_DEPTH = 2
CODE_DEPTH = 3


def value(state):
    return state["shipped"] - state["bugs"] - 0.5 * state["debt"]


def env_step(state, action_name):
    return sprint_ground_truth(state, {"name": action_name, "params": {}, "agent": None})


def lookahead(model_step, state, depth):
    """Best first action by exhaustive depth-d search; returns (action, evals)."""
    evals = 0

    def best_value(s, d):
        nonlocal evals
        if d == 0:
            return value(s)
        best = float("-inf")
        for a in ACTIONS:
            evals += 1
            best = max(best, best_value(model_step(s, a), d - 1))
        return best

    best_action, best = None, float("-inf")
    for a in ACTIONS:
        evals += 1
        v = best_value(model_step(state, a), depth - 1)
        if v > best:
            best, best_action = v, a
    return best_action, evals


def run_episode(policy):
    state = dict(SPRINT_INITIAL)
    start = time.perf_counter()
    for _ in range(EPISODE_STEPS):
        action = policy(state)
        state = env_step(state, action)
    return value(state), state, time.perf_counter() - start


def reactive(state):
    if state["debt"] > 4:
        return "refactor"
    if state["bugs"] > 2:
        return "fix"
    return "ship" if state["backlog"] > 0 else "fix"


def main():
    e10 = json.loads((Path(RESULTS_DIR) / "e10_ood_generalization.json").read_text())
    code = CodeTransition(e10["synthesized_code"])
    oracle_model = FunctionTransition(sprint_ground_truth)
    llm = require_ollama(GENERATOR_MODEL, temperature=0.0)
    llm_model = LLMTransition(llm, SPRINT_DESCRIPTION, SPRINT_RULES)

    def model_stepper(transition):
        return lambda s, a: dict(transition.step(WorldState(s), Action(a)))

    rows = []

    for name, transition, depth in (
        ("code_d3", code, CODE_DEPTH),
        ("oracle_d3", oracle_model, CODE_DEPTH),
        ("llm_d2", llm_model, LLM_DEPTH),
    ):
        stepper = model_stepper(transition)
        score, final, seconds = run_episode(
            lambda s, _st=stepper, _d=depth: lookahead(_st, s, _d)[0])
        rows.append({"planner": name, "score": score, "final_state": final,
                     "episode_seconds": seconds})
        print(f"  {name}: score={score:.1f} final={final} ({seconds:.1f}s)")

    score, final, seconds = run_episode(reactive)
    rows.append({"planner": "reactive_heuristic", "score": score,
                 "final_state": final, "episode_seconds": seconds})
    print(f"  reactive: score={score:.1f}")

    rnd_scores = []
    for seed in range(5):
        rng = random.Random(seed)
        score, final, _ = run_episode(lambda s: rng.choice(ACTIONS))
        rnd_scores.append(score)
    rows.append({"planner": "random", "score": sum(rnd_scores) / len(rnd_scores),
                 "scores": rnd_scores, "episode_seconds": None})
    print(f"  random: mean score={sum(rnd_scores) / len(rnd_scores):.1f}")

    save_results("e22_planning", {
        "model": GENERATOR_MODEL, "episode_steps": EPISODE_STEPS,
        "value_function": "shipped - bugs - 0.5*debt",
        "depths": {"code": CODE_DEPTH, "llm": LLM_DEPTH},
        # Separate unparseable-reply no-ops from parseable-but-wrong hallucinations
        # in the LLM planner's score (see paper E22 caveat).
        "llm_parse_failure_rate": round(llm_model.parse_failure_rate, 3),
        "llm_parse_failures": llm_model.parse_failures,
        "llm_lookahead_steps": llm_model.steps,
        "rows": rows,
    })


if __name__ == "__main__":
    main()
