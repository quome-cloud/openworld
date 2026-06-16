"""E35 - The sprint ladder: E34's allocation experiment across model families.

Re-runs the sprint-world conditions (fixed 4/task, round-robin, greedy
min-failing; pinned seed, budget 80 - identical protocol to E34's
qwen2.5:7b anchor) on larger models from three families:

    qwen3-coder:30b    (Qwen 3, coder MoE)
    gpt-oss:20b        (OpenAI, MoE, reasoning)
    deepseek-r1:14b    (DeepSeek, dense, reasoning)

The open question E34 set up: does the allocation story change with
capability? If a stronger model leaves fewer capability-bound tasks,
reallocation should pay and the greedy tarpit should dissolve (the task it
commits to actually gets done) - or greedy finds a new hardest task to
drown in. Reasoning models' <think> blocks are stripped before patch
extraction (see e34_composite_swe._THINK).

Resumable: finished (model, condition) cells are skipped on restart.
"""

import json
from pathlib import Path

from openworld.llm import OllamaLLM

from common import Timer, require_ollama, save_results
from e34_composite_swe import (
    PER_TASK_BUDGET,
    RECIPE,
    load_dataset,
    run_condition,
)

MODELS = ["qwen3-coder:30b", "gpt-oss:20b", "deepseek-r1:14b"]
CONDITIONS = ("fixed", "round_robin", "greedy")
RESULTS_PATH = Path(__file__).resolve().parent / "results" / "e35_sprint_ladder.json"


def main():
    instances = load_dataset(RECIPE["dataset"]["path"])
    total_budget = PER_TASK_BUDGET * len(instances)
    temperature = RECIPE["eval"].get("temperature", 0.2)
    seed = RECIPE["eval"].get("seed", 41)

    ladder = {}
    if RESULTS_PATH.exists():
        ladder = json.loads(RESULTS_PATH.read_text())["ladder"]
        done = [(m, c["condition"]) for m, cell in ladder.items()
                for c in cell["conditions"]]
        print(f"[resume] {len(done)} cells already complete: {done}")

    for model in MODELS:
        require_ollama(model)
        cell = ladder.setdefault(model, {"conditions": []})
        finished = {c["condition"] for c in cell["conditions"]}

        def factory(k, _model=model):
            return OllamaLLM(model=_model, temperature=temperature,
                             options={"seed": seed})

        for condition in CONDITIONS:
            if condition in finished:
                continue
            print(f"[{model} / {condition}] budget {total_budget}")
            with Timer() as t:
                result = run_condition(condition, instances, factory, total_budget)
            result["seconds"] = round(t.elapsed, 1)
            cell["conditions"].append(result)
            save_results("e35_sprint_ladder", {
                "anchor": "qwen2.5:7b (E34)",
                "per_task_budget": PER_TASK_BUDGET,
                "total_budget": total_budget,
                "dataset": RECIPE["dataset"]["name"],
                "dataset_version": RECIPE["dataset"]["version"],
                "tasks_jsonl_sha256": RECIPE["artifacts"]["tasks_jsonl_sha256"],
                "ladder": ladder,
            })

    print(f"\n{'model':<20} {'condition':<12} solved  attempts")
    for model, cell in ladder.items():
        for c in cell["conditions"]:
            print(f"{model:<20} {c['condition']:<12} {c['solved']:>2}/"
                  f"{c['n_tasks']}   {c['attempts_consumed']:>3}")


if __name__ == "__main__":
    main()
