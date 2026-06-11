"""E17 - Judge selection power extension (round-2 review item R2).

Runs the exact E13 paired protocol with three additional proposer seeds,
then pools all 120 rounds (E13's 60 + these 60) and recomputes strategy
rates, the exact McNemar test, and the position-bias audit. Either the
judge-vs-random margin sharpens into significance or the effect estimate
shrinks; both are informative.
"""

import json
import random
from pathlib import Path

from openworld import Judge, OllamaLLM
from openworld.coding import BENCHMARK, build_codefix_world, run_tests
from openworld.parsing import extract_code

from common import (
    GENERATOR_MODEL, RESULTS_DIR, mcnemar_p, require_ollama, save_results,
    wilson_ci,
)
from e05_codefix_agent import AGENT_MODEL, REPAIR_SYSTEM, repair_prompt

N_CANDIDATES = 3
NEW_SEEDS = [3, 4, 5]


def run_rounds(judge, seeds, rng):
    rounds = []
    for task in BENCHMARK:
        world = build_codefix_world(task)
        state = world.state
        for seed in seeds:
            candidates = []
            for k in range(N_CANDIDATES):
                sampler = OllamaLLM(model=AGENT_MODEL, temperature=0.9,
                                    options={"seed": 7000 + seed * 100 + k})
                candidates.append(extract_code(
                    sampler.ask(repair_prompt(task, state), system=REPAIR_SYSTEM)))
            passing = [run_tests(c, task.tests)["failed"] == 0 for c in candidates]
            context = repair_prompt(task, state)
            blocks = [f"```python\n{c}\n```" for c in candidates]
            choice_fwd = judge.choose(blocks, context=context)
            choice_rev = N_CANDIDATES - 1 - judge.choose(list(reversed(blocks)), context=context)
            rounds.append({
                "task": task.name, "seed": seed,
                "passing": passing,
                "first_pass": passing[0],
                "random_pass": passing[rng.randrange(N_CANDIDATES)],
                "judge_pass": passing[choice_fwd],
                "judge_rev_pass": passing[choice_rev],
                "oracle_pass": any(passing),
                "judge_choice_fwd": choice_fwd,
                "judge_choice_rev": choice_rev,
                "order_consistent": choice_fwd == choice_rev,
            })
            print(f"  {task.name} s{seed}: passing={passing} judge={choice_fwd}")
    return rounds


def pool_stats(rounds):
    n = len(rounds)
    strategies = {}
    for name in ("first_pass", "random_pass", "judge_pass", "oracle_pass"):
        wins = sum(r[name] for r in rounds)
        strategies[name.replace("_pass", "")] = {
            "solves": wins, "n": n, "rate": wins / n,
            "ci": list(wilson_ci(wins, n)),
        }
    b = sum(1 for r in rounds if r["judge_pass"] and not r["random_pass"])
    c = sum(1 for r in rounds if r["random_pass"] and not r["judge_pass"])
    disc = [r for r in rounds if any(r["passing"]) and not all(r["passing"])]
    return {
        "n_rounds": n,
        "strategies": strategies,
        "mcnemar_judge_vs_random": {"b": b, "c": c, "p": mcnemar_p(b, c)},
        "position_bias": {
            "order_consistency": sum(r["order_consistent"] for r in rounds) / n,
            "order_consistency_discriminative": (
                sum(r["order_consistent"] for r in disc) / len(disc) if disc else None),
            "n_discriminative_rounds": len(disc),
            "judge_accuracy_forward": (
                sum(r["judge_pass"] for r in disc) / len(disc) if disc else None),
            "judge_accuracy_reversed": (
                sum(r["judge_rev_pass"] for r in disc) / len(disc) if disc else None),
        },
    }


def main():
    require_ollama(GENERATOR_MODEL)
    judge = Judge(
        OllamaLLM(model=GENERATOR_MODEL, temperature=0.0),
        criteria=(
            "Pick the patch most likely to make ALL the failing tests pass while "
            "implementing the intended behavior. Prefer minimal, correct fixes."
        ),
    )
    rng = random.Random(199)
    new_rounds = run_rounds(judge, NEW_SEEDS, rng)

    e13 = json.loads((Path(RESULTS_DIR) / "e13_judge_controls.json").read_text())
    pooled_rounds = e13["rounds"] + new_rounds

    save_results("e17_judge_power", {
        "proposer": AGENT_MODEL, "judge": GENERATOR_MODEL,
        "new_seeds": NEW_SEEDS,
        "new_rounds_stats": pool_stats(new_rounds),
        "pooled": pool_stats(pooled_rounds),
        "rounds": new_rounds,
    })
    pooled = pool_stats(pooled_rounds)
    print(f"\npooled n={pooled['n_rounds']}: " + ", ".join(
        f"{k}={v['rate']:.0%}" for k, v in pooled["strategies"].items()))
    mcn = pooled["mcnemar_judge_vs_random"]
    print(f"pooled McNemar: b={mcn['b']} c={mcn['c']} p={mcn['p']:.4f}")


if __name__ == "__main__":
    main()
