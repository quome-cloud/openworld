"""E34 - The sprint world: composite attempt allocation on openworld-swebench.

A CompositeWorld whose 20 children are the owsb-atomic repair worlds
(children unmodified), with aggregators exposing the sprint dashboard
(open tasks, total failing tests). The repair model is identical to the
standard in-world protocol (same prompts, same temperature/seed from the
dataset recipe); the only variable is WHO decides where the next attempt
goes:

  fixed       - the standard protocol: 4 attempts per task in isolation
                (also the first real-model numbers for owsb-atomic v1)
  round_robin - composite: cycle unsolved tasks, recycling attempts
                stranded on solved ones
  greedy      - composite: next attempt to the unsolved task with the
                fewest failing tests (ties: fewest attempts, dataset order)

All conditions get the same total budget B = 80. The measured quantity is
what the composite's global view is worth at equal budget. Honest-results
rule: whatever the numbers say is what the paper reports.
"""

import json
from pathlib import Path

from openworld import Action
from openworld.bench import load_recipe, wilson_ci
from openworld.compose import AGG_KEY, Aggregator, CompositeWorld
from openworld.llm import OllamaLLM
from openworld.parsing import extract_code
from openworld.swebench import (
    SYSTEM_PROMPT,
    _feedback_prompt,
    _safe_ask,
    build_swebench_world,
    load_dataset,
)

from common import Timer, require_ollama, save_results

RECIPE = load_recipe(Path(__file__).resolve().parent.parent
                     / "recipes" / "owsb-atomic-v1.json")
MODEL = "qwen2.5:7b"
PER_TASK_BUDGET = 4
TOTAL_BUDGET = None  # set in main: PER_TASK_BUDGET * n_tasks


def failing(slice_state):
    return slice_state["fail_to_pass_failed"] + slice_state["pass_to_pass_failed"]


def build_sprint(instances):
    children = {inst.instance_id: build_swebench_world(inst) for inst in instances}
    return CompositeWorld(
        name="sprint",
        children=children,
        aggregators=[
            Aggregator("open_tasks",
                       lambda kids: sum(1 for s in kids.values() if not s["solved"])),
            Aggregator("total_failing",
                       lambda kids: sum(s["fail_to_pass_failed"]
                                        + s["pass_to_pass_failed"]
                                        for s in kids.values())),
        ],
    )


def attempt(comp, inst, llm, by_id):
    """One repair attempt routed through the composite; returns the event."""
    before = comp.state[inst.instance_id]
    reply = _safe_ask(llm, _feedback_prompt(inst, before), SYSTEM_PROMPT)
    patch = extract_code(reply)
    state = comp.step(Action(f"{inst.instance_id}:submit_patch",
                             params={"source": patch}, agent="engineer"))
    after = state[inst.instance_id]
    return {
        "task": inst.instance_id,
        "failing_before": failing(before),
        "failing_after": failing(after),
        "solved": bool(after["solved"]),
        "open_tasks": state[AGG_KEY]["open_tasks"],
    }


def pick_round_robin(comp, instances, cursor):
    n = len(instances)
    for offset in range(n):
        inst = instances[(cursor + offset) % n]
        if not comp.state[inst.instance_id]["solved"]:
            return inst, (cursor + offset + 1) % n
    return None, cursor


def pick_greedy(comp, instances, attempts_spent):
    open_tasks = [i for i in instances if not comp.state[i.instance_id]["solved"]]
    if not open_tasks:
        return None
    return min(open_tasks,
               key=lambda i: (failing(comp.state[i.instance_id]),
                              attempts_spent[i.instance_id],
                              instances.index(i)))


def run_condition(name, instances, llm, total_budget):
    comp = build_sprint(instances)
    by_id = {i.instance_id: i for i in instances}
    events = []
    attempts_spent = {i.instance_id: 0 for i in instances}
    cursor = 0
    last_task = None
    switches = 0
    for k in range(total_budget):
        if name == "fixed":
            inst = next((i for i in instances
                         if not comp.state[i.instance_id]["solved"]
                         and attempts_spent[i.instance_id] < PER_TASK_BUDGET), None)
        elif name == "round_robin":
            inst, cursor = pick_round_robin(comp, instances, cursor)
        else:
            inst = pick_greedy(comp, instances, attempts_spent)
        if inst is None:
            break
        if last_task is not None and inst.instance_id != last_task:
            switches += 1
        last_task = inst.instance_id
        event = attempt(comp, inst, llm, by_id)
        attempts_spent[inst.instance_id] += 1
        event["attempt_index"] = k + 1
        event["cumulative_solved"] = len(instances) - event["open_tasks"]
        events.append(event)
        print(f"  [{name}] #{k+1:>2} {inst.instance_id.split('-', 3)[-1]:<28} "
              f"fail {event['failing_before']}->{event['failing_after']} "
              f"solved={event['cumulative_solved']}")
        if event["open_tasks"] == 0:
            break
    solved = len(instances) - comp.state[AGG_KEY]["open_tasks"]
    return {
        "condition": name,
        "solved": solved,
        "n_tasks": len(instances),
        "solved_rate": solved / len(instances),
        "solved_ci": list(wilson_ci(solved, len(instances))),
        "attempts_consumed": len(events),
        "task_switches": switches,
        "events": events,
        "attempts_per_task": attempts_spent,
    }


def main():
    require_ollama(MODEL)
    instances = load_dataset(RECIPE["dataset"]["path"])
    total_budget = PER_TASK_BUDGET * len(instances)
    llm = OllamaLLM(model=MODEL,
                    temperature=RECIPE["eval"].get("temperature", 0.2),
                    options={"seed": RECIPE["eval"].get("seed", 41)})
    results = []
    for condition in ("fixed", "round_robin", "greedy"):
        print(f"[{condition}] budget {total_budget}")
        with Timer() as t:
            results.append(run_condition(condition, instances, llm, total_budget))
        results[-1]["seconds"] = round(t.elapsed, 1)
        # incremental save: a crash in a later condition loses nothing
        save_results("e34_composite_swe", {
            "model": MODEL,
            "dataset": RECIPE["dataset"]["name"],
            "dataset_version": RECIPE["dataset"]["version"],
            "tasks_jsonl_sha256": RECIPE["artifacts"]["tasks_jsonl_sha256"],
            "per_task_budget": PER_TASK_BUDGET,
            "total_budget": total_budget,
            "summary": [{k: v for k, v in r.items() if k != "events"}
                        for r in results],
            "conditions": results,
        })
    print(f"\n{'condition':<12} solved  attempts  switches")
    for r in results:
        print(f"{r['condition']:<12} {r['solved']:>2}/{r['n_tasks']}   "
              f"{r['attempts_consumed']:>3}       {r['task_switches']:>3}")


if __name__ == "__main__":
    main()
