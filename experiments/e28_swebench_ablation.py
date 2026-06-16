"""E28 - OpenWorld-SWE-bench atomic suite: single-shot vs in-world repair.

Protocol (one paired comparison per instance, per model):
  - single-shot:  one completion from the issue + buggy module, no feedback
                  (openworld.swebench.solve_single_shot)
  - in-world:     iterative repair inside the world with exact test feedback,
                  budget B attempts (openworld.swebench.solve_in_world)
across the qwen2.5 capability ladder (1.5b / 3b / 7b). We report single-shot
pass@1, in-world pass@1 and pass@B (Wilson CIs), the budget-minus-single-shot
delta, and mean attempts.

PROVENANCE / REPRODUCIBILITY. The committed results JSON
(`results/e28_swebench_ablation.json`) is a frozen artifact copied from the
`openworld-swebench` dataset branch (commit 6982b8b), generated on an early
6-instance snapshot of the atomic dataset. The current bundled dataset
(`datasets/openworld-swebench/tasks.jsonl`, owsb-atomic-v1) has since grown to
20 instances, so re-running this script reproduces the *protocol* but not the
exact frozen numbers (different n, plus Ollama-on-Metal nondeterminism). Treat
the cached JSON as the canonical paper numbers; this script documents and
re-enables the experiment. Requires a local Ollama with the three models.
"""

import sys
from pathlib import Path
from statistics import mean

from openworld.bench import load_recipe, wilson_ci
from openworld.llm import OllamaLLM
from openworld.swebench import load_dataset, solve_in_world, solve_single_shot

from common import Timer, require_ollama, save_results

RECIPE = load_recipe(Path(__file__).resolve().parent.parent
                     / "recipes" / "owsb-atomic-v1.json")
EXPERIMENT = "e28_swebench_ablation"
DESCRIPTION = ("OpenWorld-SWE-bench atomic suite: single-shot vs in-world, "
              "qwen2.5 ladder")
PROVENANCE = ("frozen artifact from branch openworld-swebench commit 6982b8b "
             "(early 6-instance snapshot); re-running uses the current dataset")


def run_model(model, instances, budget, temperature, seed):
    def llm():
        return OllamaLLM(model=model, temperature=temperature, options={"seed": seed})
    ss = [solve_single_shot(inst, llm()) for inst in instances]
    iw = [solve_in_world(inst, llm(), budget=budget) for inst in instances]
    n = len(instances)
    ss_solved = sum(r["solved"] for r in ss)
    iw_solved = sum(r["solved"] for r in iw)
    ss_p1 = ss_solved / n
    iw_pb = iw_solved / n
    return {
        "model": model, "n": n,
        "single_shot_pass1": ss_p1,
        "single_shot_pass1_ci": list(wilson_ci(ss_solved, n)),
        "in_world_pass1": mean(r["solved_first_attempt"] for r in iw),
        "in_world_pass_budget": iw_pb,
        "in_world_pass_budget_ci": list(wilson_ci(iw_solved, n)),
        "delta_budget_minus_ss": iw_pb - ss_p1,
        "mean_attempts": mean(r["attempts"] for r in iw),
        "budget": budget,
        "pairs": [{"instance_id": s["instance_id"],
                   "single_shot": bool(s["solved"]), "in_world": bool(w["solved"])}
                  for s, w in zip(ss, iw)],
    }


def main():
    instances = load_dataset(RECIPE["dataset"]["path"])
    budget = RECIPE["eval"]["budget"]
    temperature = RECIPE["eval"]["temperature"]
    seed = RECIPE["eval"]["seed"]
    results = []
    for model in RECIPE["eval"]["models"]:
        require_ollama(model)
        print(f"[{model}] single-shot vs in-world over {len(instances)} instances")
        with Timer() as t:
            results.append(run_model(model, instances, budget, temperature, seed))
        results[-1]["seconds"] = round(t.elapsed, 1)
        # incremental save: a crash in a later model loses nothing
        save_results(EXPERIMENT, {
            "experiment": EXPERIMENT, "description": DESCRIPTION,
            "provenance": PROVENANCE,
            "n_instances": len(instances), "budget": budget, "mock": False,
            "summary": [{k: v for k, v in r.items() if k != "pairs"} for r in results],
            "results": results,
        })
    print(f"\n{'model':<14} SS@1  IW@1  IW@B   delta")
    for r in results:
        print(f"{r['model']:<14} {r['single_shot_pass1']:.2f}  {r['in_world_pass1']:.2f}  "
              f"{r['in_world_pass_budget']:.2f}  {r['delta_budget_minus_ss']:+.2f}")


if __name__ == "__main__":
    main()
