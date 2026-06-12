#!/usr/bin/env python3
"""Side-by-side comparison on openworld-swebench: single-shot vs. in-world.

Usage:
    python datasets/openworld-swebench/run_comparison.py                # default ladder
    python datasets/openworld-swebench/run_comparison.py qwen2.5:7b    # specific models
    python datasets/openworld-swebench/run_comparison.py --mock         # offline smoke

For each model x instance, runs the model once single-shot (issue + buggy
module, no feedback) and once inside the world model (iterative submit_patch
with exact failing-test feedback, --budget attempts). Records are paired
per-instance; the summary table puts the conditions side by side.
"""

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

from openworld import MockLLM, OllamaLLM
from openworld.llm import OllamaConnectionError
from openworld.swebench import load_dataset, solve_in_world, solve_single_shot

DEFAULT_MODELS = ["qwen2.5:1.5b", "qwen2.5:3b", "qwen2.5:7b", "llama3.2"]
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def wilson_ci(successes, n, z=1.96):
    """95% Wilson score interval, vendored from experiments/common.py."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def evaluate_model(label, llm_factory, instances, budget):
    rows = []
    for inst in instances:
        single = solve_single_shot(inst, llm_factory(inst))
        in_world = solve_in_world(inst, llm_factory(inst), budget=budget)
        rows.append({"instance_id": inst.instance_id,
                     "single_shot": single, "in_world": in_world})
        print(f"  {inst.instance_id}: "
              f"single-shot={'Y' if single['solved'] else 'n'} "
              f"in-world={'Y' if in_world['solved'] else 'n'} "
              f"({in_world['attempts']} att)")
    return rows


def summarize(model, rows, budget):
    n = len(rows)
    single = sum(r["single_shot"]["solved"] for r in rows)
    world_first = sum(r["in_world"]["solved_first_attempt"] for r in rows)
    world_budget = sum(r["in_world"]["solved"] for r in rows)
    solved_attempts = [r["in_world"]["attempts"] for r in rows if r["in_world"]["solved"]]
    return {
        "model": model,
        "n_instances": n,
        "budget": budget,
        "single_shot_pass_at_1": single / n,
        "single_shot_ci": list(wilson_ci(single, n)),
        "in_world_pass_at_1": world_first / n,
        "in_world_pass_at_1_ci": list(wilson_ci(world_first, n)),
        "in_world_pass_at_budget": world_budget / n,
        "in_world_pass_at_budget_ci": list(wilson_ci(world_budget, n)),
        "delta": (world_budget - single) / n,
        "mean_attempts_when_solved": (
            sum(solved_attempts) / len(solved_attempts) if solved_attempts else None
        ),
    }


def markdown_table(summaries, budget):
    head = (f"| model | single-shot pass@1 | in-world pass@1 | "
            f"in-world pass@{budget} | Δ (pass@{budget} − SS) | mean attempts |")
    sep = "|---|---|---|---|---|---|"
    lines = [head, sep]
    for s in summaries:
        mean_att = (f"{s['mean_attempts_when_solved']:.1f}"
                    if s["mean_attempts_when_solved"] is not None else "—")
        lines.append(
            f"| {s['model']} | {s['single_shot_pass_at_1']:.0%} | "
            f"{s['in_world_pass_at_1']:.0%} | {s['in_world_pass_at_budget']:.0%} | "
            f"{s['delta']:+.0%} | {mean_att} |"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="*", default=None,
                        help=f"Ollama model names (default: {' '.join(DEFAULT_MODELS)})")
    parser.add_argument("--budget", type=int, default=4,
                        help="in-world attempt budget (default 4)")
    parser.add_argument("--mock", action="store_true",
                        help="offline smoke run with a scripted MockLLM")
    args = parser.parse_args()

    instances = load_dataset()
    runs = []
    if args.mock:
        def factory(inst):
            return MockLLM([
                "I think the bug is in the loop, roughly.",
                f"```python\n{inst.reference_source}\n```",
            ])
        runs.append(("mock-oracle-2nd-try", factory))
    else:
        for model in (args.models or DEFAULT_MODELS):
            llm = OllamaLLM(model=model, temperature=0.2, options={"seed": 41})
            try:
                llm.ask("Reply with OK.")
            except OllamaConnectionError as exc:
                raise SystemExit(f"model {model!r} unavailable: {exc}")
            runs.append((model, lambda inst, _llm=llm: _llm))

    summaries, all_rows = [], {}
    for label, factory in runs:
        print(f"[{label}] {len(instances)} instances, budget {args.budget}")
        rows = evaluate_model(label, factory, instances, args.budget)
        summaries.append(summarize(label, rows, args.budget))
        all_rows[label] = rows

    table = markdown_table(summaries, args.budget)
    print("\n" + table)

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "comparison.json"
    out.write_text(json.dumps({
        "dataset": "openworld-swebench",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "budget": args.budget,
        "summaries": summaries,
        "rows": all_rows,
    }, indent=2), encoding="utf-8")
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
