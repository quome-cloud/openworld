"""E13 - Judge selection controls and position-bias audit (review item W3).

Paired design on shared candidate sets: for each (task, seed) round, the
1.5B proposer samples three candidate patches once; four selection strategies
then act on the SAME candidates:

  first   - take candidate 0 (no selection)
  random  - uniform pick (isolates the best-of-3 diversity effect)
  judge   - the 7B judge chooses
  oracle  - any passing candidate counts (ceiling)

Reports per-strategy solve rates with CIs and an exact McNemar test of judge
vs random on the paired rounds. Position bias is audited by asking the judge
again with candidates in reversed order and measuring choice consistency and
accuracy under both orders.
"""

import random

from openworld import Judge, OllamaLLM
from openworld.coding import BENCHMARK, build_codefix_world, run_tests
from openworld.parsing import extract_code

from common import (
    GENERATOR_MODEL, mcnemar_p, require_ollama, save_results, wilson_ci,
)
from e05_codefix_agent import AGENT_MODEL, REPAIR_SYSTEM, repair_prompt

N_CANDIDATES = 3
SEEDS = [0, 1, 2]


def main():
    require_ollama(GENERATOR_MODEL)
    judge = Judge(
        OllamaLLM(model=GENERATOR_MODEL, temperature=0.0),
        criteria=(
            "Pick the patch most likely to make ALL the failing tests pass while "
            "implementing the intended behavior. Prefer minimal, correct fixes."
        ),
    )
    rng = random.Random(99)
    rounds = []
    for task in BENCHMARK:
        world = build_codefix_world(task)
        state = world.state
        for seed in SEEDS:
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
            choice_rev_raw = judge.choose(list(reversed(blocks)), context=context)
            choice_rev = N_CANDIDATES - 1 - choice_rev_raw  # map back

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
            print(f"  {task.name} s{seed}: passing={passing} "
                  f"judge={choice_fwd} rev={choice_rev}")

    n = len(rounds)
    strategies = {}
    for name in ("first_pass", "random_pass", "judge_pass", "oracle_pass"):
        wins = sum(r[name] for r in rounds)
        strategies[name.replace("_pass", "")] = {
            "solves": wins, "n": n, "rate": wins / n,
            "ci": list(wilson_ci(wins, n)),
        }

    # McNemar: judge vs random on paired rounds.
    b = sum(1 for r in rounds if r["judge_pass"] and not r["random_pass"])
    c = sum(1 for r in rounds if r["random_pass"] and not r["judge_pass"])
    # Judge accuracy / position bias on solvable, non-trivial rounds
    solvable = [r for r in rounds if any(r["passing"]) and not all(r["passing"])]
    fwd_acc = (sum(r["judge_pass"] for r in solvable) / len(solvable)) if solvable else None
    rev_acc = (sum(r["judge_rev_pass"] for r in solvable) / len(solvable)) if solvable else None
    consistency = sum(r["order_consistent"] for r in rounds) / n

    save_results("e13_judge_controls", {
        "proposer": AGENT_MODEL, "judge": GENERATOR_MODEL,
        "n_rounds": n, "n_candidates": N_CANDIDATES, "seeds": SEEDS,
        "strategies": strategies,
        "mcnemar_judge_vs_random": {"b": b, "c": c, "p": mcnemar_p(b, c)},
        "position_bias": {
            "order_consistency": consistency,
            "n_discriminative_rounds": len(solvable),
            "judge_accuracy_forward": fwd_acc,
            "judge_accuracy_reversed": rev_acc,
        },
        "rounds": rounds,
    })
    print(f"\nstrategies: " + ", ".join(
        f"{k}={v['rate']:.0%}" for k, v in strategies.items()))
    print(f"McNemar judge vs random: b={b} c={c} p={mcnemar_p(b, c):.4f}")
    print(f"order consistency {consistency:.0%}; "
          f"fwd acc {fwd_acc}, rev acc {rev_acc} on {len(solvable)} rounds")


if __name__ == "__main__":
    main()
