"""Run the single-shot vs. in-world comparison on OpenWorld-SWE-bench-STAGED.

Same paired ablation as the atomic set, on the TWO-STAGE instances built to
exercise the in-world feedback loop. For each model x instance it runs both
conditions and reports whether the loop helps. Writes results/comparison.json
(with per-task paired records for significance testing) and prints a table.

    python datasets/openworld-swebench-staged/run_comparison.py              # default ladder
    python datasets/openworld-swebench-staged/run_comparison.py qwen2.5:3b   # one model
    python datasets/openworld-swebench-staged/run_comparison.py --mock       # offline smoke
    python datasets/openworld-swebench-staged/run_comparison.py --budget 6 ...

Requires Ollama only for real models; missing models are skipped with a warning.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.request
from pathlib import Path

# Make the repo importable when run as a script from the dataset folder.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from openworld.llm import MockLLM, OllamaLLM  # noqa: E402
from openworld.swebench import (  # noqa: E402
    load_dataset, solve_in_world, solve_single_shot,
)

DEFAULT_MODELS = ["qwen2.5:1.5b", "qwen2.5:3b", "qwen2.5:7b", "llama3.2"]
DATA = Path(__file__).resolve().parent / "tasks.jsonl"
RESULTS = Path(__file__).resolve().parent / "results" / "comparison.json"


def wilson_ci(successes: int, n: int, z: float = 1.96):
    """95% Wilson score interval (vendored so this folder runs standalone)."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def available_models():
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5) as r:
            tags = json.loads(r.read().decode("utf-8"))
        return {m["name"] for m in tags.get("models", [])}
    except Exception:
        return set()


def _mock_for(instance):
    """An LLM that always returns the instance's reference solution (smoke only)."""
    return MockLLM(responses=[f"```python\n{instance.reference_source}```"])


# Bound per-request time and output length so a runaway small-model generation
# fails fast as one attempt instead of stalling the whole run for minutes.
_OLLAMA_TIMEOUT = 120.0
_OLLAMA_OPTS = {"num_predict": 1024}


def run_model(model, instances, budget, mock):
    pairs = []
    for inst in instances:
        ss_llm = _mock_for(inst) if mock else OllamaLLM(
            model=model, timeout=_OLLAMA_TIMEOUT, options=dict(_OLLAMA_OPTS))
        iw_llm = _mock_for(inst) if mock else OllamaLLM(
            model=model, timeout=_OLLAMA_TIMEOUT, options=dict(_OLLAMA_OPTS))
        ss = solve_single_shot(inst, ss_llm)
        iw = solve_in_world(inst, iw_llm, budget=budget)
        pairs.append({"instance_id": inst.instance_id, "single_shot": ss, "in_world": iw})
    n = len(pairs)
    ss_solved = sum(p["single_shot"]["solved"] for p in pairs)
    iw_first = sum(p["in_world"]["solved_first_attempt"] for p in pairs)
    iw_solved = sum(p["in_world"]["solved"] for p in pairs)
    mean_attempts = sum(p["in_world"]["attempts"] for p in pairs) / n if n else 0.0
    return {
        "model": model,
        "n": n,
        "single_shot_pass1": ss_solved / n if n else 0.0,
        "single_shot_pass1_ci": wilson_ci(ss_solved, n),
        "in_world_pass1": iw_first / n if n else 0.0,
        "in_world_pass_budget": iw_solved / n if n else 0.0,
        "in_world_pass_budget_ci": wilson_ci(iw_solved, n),
        "delta_budget_minus_ss": (iw_solved - ss_solved) / n if n else 0.0,
        "mean_attempts": mean_attempts,
        "budget": budget,
        "pairs": pairs,
    }


def markdown_table(summaries, budget):
    lines = [
        f"| model | single-shot pass@1 | in-world pass@1 | in-world pass@{budget} | "
        f"Δ (pass@{budget} − SS) | mean attempts |",
        "|---|---|---|---|---|---|",
    ]
    for s in summaries:
        lines.append(
            f"| {s['model']} | {s['single_shot_pass1']:.2f} | {s['in_world_pass1']:.2f} | "
            f"{s['in_world_pass_budget']:.2f} | {s['delta_budget_minus_ss']:+.2f} | "
            f"{s['mean_attempts']:.2f} |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="*", default=[])
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--mock", action="store_true", help="offline smoke with MockLLM")
    parser.add_argument("--limit", type=int, default=0, help="run only the first N instances")
    args = parser.parse_args()

    instances = load_dataset(DATA)
    if args.limit:
        instances = instances[: args.limit]
    print(f"[loaded] {len(instances)} instances")

    models = args.models or DEFAULT_MODELS
    if not args.mock:
        have = available_models()
        present = [m for m in models if m in have]
        missing = [m for m in models if m not in have]
        for m in missing:
            print(f"  !! skipping {m}: not pulled (ollama pull {m})", file=sys.stderr)
        models = present
        if not models:
            print("No requested models are available. Try --mock for an offline smoke.",
                  file=sys.stderr)
            return 1

    summaries = []
    for model in models:
        print(f"[running] {model} (budget={args.budget}) ...")
        summaries.append(run_model(model, instances, args.budget, args.mock))

    table = markdown_table(summaries, args.budget)
    print("\n" + table + "\n")

    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps({
        "n_instances": len(instances),
        "budget": args.budget,
        "mock": args.mock,
        "summary": [{k: v for k, v in s.items() if k != "pairs"} for s in summaries],
        "results": summaries,
        "table": table,
    }, indent=2, default=str), encoding="utf-8")
    print(f"[saved] {RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
