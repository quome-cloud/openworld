"""Run the with-context vs. without-context ablation on OpenWorld-ContextBench.

For each model x instance, solves once with no examples and once with the related
solved examples prepended, then reports whether in-context examples help. Writes
results/comparison.json and prints a markdown table.

    python datasets/openworld-contextbench/run_comparison.py            # default ladder
    python datasets/openworld-contextbench/run_comparison.py qwen2.5:3b
    python datasets/openworld-contextbench/run_comparison.py --mock     # offline smoke
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from openworld.contextbench import load_dataset, solve_with_context, solve_without_context  # noqa: E402
from openworld.llm import MockLLM, OllamaLLM  # noqa: E402

DEFAULT_MODELS = ["qwen2.5:1.5b", "qwen2.5:3b", "qwen2.5:7b", "llama3.2"]
RESULTS = Path(__file__).resolve().parent / "results" / "comparison.json"
_OLLAMA_TIMEOUT = 120.0
_OLLAMA_OPTS = {"num_predict": 1024}


def wilson_ci(successes, n, z=1.96):
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
            return {m["name"] for m in json.loads(r.read().decode("utf-8")).get("models", [])}
    except Exception:
        return set()


def _mock(inst):
    return MockLLM(responses=[f"```python\n{inst.task.reference_source}```"])


def run_model(model, instances, mock):
    pairs = []
    for inst in instances:
        wo_llm = _mock(inst) if mock else OllamaLLM(
            model=model, timeout=_OLLAMA_TIMEOUT, options=dict(_OLLAMA_OPTS))
        wc_llm = _mock(inst) if mock else OllamaLLM(
            model=model, timeout=_OLLAMA_TIMEOUT, options=dict(_OLLAMA_OPTS))
        wo = solve_without_context(inst, wo_llm)
        wc = solve_with_context(inst, wc_llm)
        pairs.append({"instance_id": inst.instance_id, "without": wo, "with": wc})
    n = len(pairs)
    wo_s = sum(p["without"]["solved"] for p in pairs)
    wc_s = sum(p["with"]["solved"] for p in pairs)
    return {
        "model": model, "n": n,
        "without_context_pass1": wo_s / n if n else 0.0,
        "without_context_ci": wilson_ci(wo_s, n),
        "with_context_pass1": wc_s / n if n else 0.0,
        "with_context_ci": wilson_ci(wc_s, n),
        "delta_with_minus_without": (wc_s - wo_s) / n if n else 0.0,
        "pairs": pairs,
    }


def markdown_table(summaries):
    lines = ["| model | without-context pass@1 | with-context pass@1 | Δ (with − without) |",
             "|---|---|---|---|"]
    for s in summaries:
        lines.append(f"| {s['model']} | {s['without_context_pass1']:.2f} | "
                     f"{s['with_context_pass1']:.2f} | {s['delta_with_minus_without']:+.2f} |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="*", default=[])
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    instances = load_dataset()
    if args.limit:
        instances = instances[: args.limit]
    print(f"[loaded] {len(instances)} instances")

    models = args.models or DEFAULT_MODELS
    if not args.mock:
        have = available_models()
        models = [m for m in models if m in have] or []
        for m in (args.models or DEFAULT_MODELS):
            if m not in have:
                print(f"  !! skipping {m}: not pulled", file=sys.stderr)
        if not models:
            print("No requested models available; try --mock.", file=sys.stderr)
            return 1

    summaries = []
    for model in models:
        print(f"[running] {model} ...")
        summaries.append(run_model(model, instances, args.mock))

    table = markdown_table(summaries)
    print("\n" + table + "\n")
    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps({
        "n_instances": len(instances), "mock": args.mock,
        "summary": [{k: v for k, v in s.items() if k != "pairs"} for s in summaries],
        "results": summaries, "table": table,
    }, indent=2, default=str), encoding="utf-8")
    print(f"[saved] {RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
