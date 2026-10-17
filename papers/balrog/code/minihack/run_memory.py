"""Condition B: memory-across-episodes experiment.

Runs K passes over the full task list with a persistent per-task memory
ledger (results/memory/<task>.json), built ONLY from clean-condition
episode logs (provenance recorded per entry). Fresh seeds per pass.

Metrics per pass: suite score, deaths, mean steps-to-solve, and
attempts-to-first-solve per task (cumulative over condition B).
"""

import json
import os
import random
import sys

import numpy as np

import mh_harness as H
from memory_store import TaskMemory
from run_suite import run_episode, log, RESULTS

PASSES = 3
EPISODES = H.EPISODES_PER_TASK
OUT = os.path.join(RESULTS, "memory_experiment.json")


def main():
    passes_out = []
    first_solve = {}          # task -> attempts until first solve (cond B)
    attempts = {}
    for p in range(1, PASSES + 1):
        pass_eps = []
        for ti, task in enumerate(H.MINIHACK_TASKS):
            mem = TaskMemory(task)
            for ep in range(EPISODES):
                seed = 3000 + p * 1000 + ti * 100 + ep
                res, agent, traj = run_episode(
                    task, ep, seed, memory=mem, condition="B",
                    label=f"memory_pass{p}")
                attempts[task] = attempts.get(task, 0) + 1
                if res["progression"] >= 1.0 and task not in first_solve:
                    first_solve[task] = attempts[task]
                pass_eps.append(res)
                log(f"[B pass {p}] {task} ep{ep} seed={seed} -> "
                    f"prog={res['progression']} steps={res['steps']} "
                    f"end={res['end_reason']}")
        per_task = {}
        for r in pass_eps:
            per_task.setdefault(r["task"], []).append(r)
        task_means = {t: sum(x["progression"] for x in v) / len(v)
                      for t, v in per_task.items()}
        score = 100.0 * sum(task_means.values()) / len(task_means)
        deaths = sum(1 for r in pass_eps if r["end_reason"].strip() == "1")
        solved = [r for r in pass_eps if r["progression"] >= 1.0]
        mean_steps = (sum(r["steps"] for r in solved) / len(solved)
                      if solved else None)
        passes_out.append({
            "pass": p,
            "score_pct": round(score, 2),
            "deaths": deaths,
            "mean_steps_to_solve": round(mean_steps, 1) if mean_steps else None,
            "task_means": task_means,
            "episodes": pass_eps,
        })
        with open(OUT, "w") as f:
            json.dump({"passes": passes_out,
                       "attempts_to_first_solve_condB": first_solve}, f,
                      indent=1)
        log(f"=== pass {p}: score {score:.1f}%, deaths {deaths}, "
            f"mean steps-to-solve {mean_steps and round(mean_steps,1)} ===")


if __name__ == "__main__":
    main()
