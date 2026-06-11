"""E9 - Tuning efficiency: random search vs Optuna TPE vs search+refine.

On the triage auto-configuration problem (protocol x two moral dials x
budget), measures trials-to-first-solve and best-score-at-budget over 10
seeds per strategy. Deterministic simulations; no LLM calls.
"""

import importlib.util
import sys
from pathlib import Path

from openworld import Choice, Dial, IntRange, Tuner

from common import save_results

# Reuse build/score/solved from the autotune example without duplicating them.
_spec = importlib.util.spec_from_file_location(
    "autotune_triage", Path(__file__).parent.parent / "examples" / "autotune_triage.py"
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["autotune_triage"] = _mod
_spec.loader.exec_module(_mod)

BUDGET = 200
SEEDS = list(range(10))


def make_tuner(seed):
    return Tuner(
        build=_mod.build,
        space={
            "protocol": Choice(["critical_first", "round_robin"]),
            "stewardship": Dial("stewardship"),
            "compassion": Dial("compassion"),
            "budget": IntRange(6, 24),
        },
        score=_mod.score,
        success=_mod.solved,
        steps=16,
        seed=seed,
    )


def trials_to_first_solve(study):
    for i, t in enumerate(study.trials):
        if t.solved:
            return i + 1
    return None


def run_strategy(strategy, seed):
    tuner = make_tuner(seed)
    if strategy == "random":
        tuner.search(BUDGET)
    elif strategy == "tpe":
        tuner.search(BUDGET, strategy="tpe")
    elif strategy == "random+refine":
        tuner.search(BUDGET // 2)
        tuner.refine(BUDGET // 4, scale=0.15)
        tuner.refine(BUDGET // 4, scale=0.05)
    return {
        "strategy": strategy,
        "seed": seed,
        "trials_to_first_solve": trials_to_first_solve(tuner.study),
        "best_score": tuner.study.best.score,
        "solve_rate": tuner.study.success_rate(),
    }


def main():
    runs = []
    for strategy in ("random", "tpe", "random+refine"):
        for seed in SEEDS:
            record = run_strategy(strategy, seed)
            runs.append(record)
        rows = [r for r in runs if r["strategy"] == strategy]
        solved_rows = [r for r in rows if r["trials_to_first_solve"] is not None]
        print(f"  {strategy}: mean best score "
              f"{sum(r['best_score'] for r in rows) / len(rows):.2f}, "
              f"solved {len(solved_rows)}/{len(rows)} seeds")

    summary = []
    for strategy in ("random", "tpe", "random+refine"):
        rows = [r for r in runs if r["strategy"] == strategy]
        solved_rows = [r for r in rows if r["trials_to_first_solve"] is not None]
        summary.append({
            "strategy": strategy,
            "seeds": len(rows),
            "mean_best_score": sum(r["best_score"] for r in rows) / len(rows),
            "seeds_solving": len(solved_rows),
            "mean_trials_to_first_solve": (
                sum(r["trials_to_first_solve"] for r in solved_rows) / len(solved_rows)
                if solved_rows else None
            ),
        })
    save_results("e09_tuning_efficiency", {
        "budget_trials": BUDGET, "summary": summary, "runs": runs,
    })


if __name__ == "__main__":
    main()
