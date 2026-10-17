"""Robustness block: condition A protocol on a seed block (2000+) that was
never used during development or agent iteration (guards against
tuning-to-eval-seeds)."""
import json, os
import mh_harness as H
from run_suite import run_episode, log, RESULTS

OUT = os.path.join(RESULTS, "robustness_results.json")

def main():
    eps = []
    for ti, task in enumerate(H.MINIHACK_TASKS):
        for ep in range(H.EPISODES_PER_TASK):
            seed = 2000 + ti * 100 + ep
            res, _a, _t = run_episode(task, ep, seed, condition="A",
                                      label="clean_A_robustness")
            eps.append(res)
            log(f"[robust] {task} ep{ep} seed={seed} -> prog={res['progression']} "
                f"steps={res['steps']} end={res['end_reason']}")
    per_task = {}
    for r in eps:
        per_task.setdefault(r["task"], []).append(r["progression"])
    means = {t: sum(v)/len(v) for t, v in per_task.items()}
    score = 100.0 * sum(means.values()) / len(means)
    with open(OUT, "w") as f:
        json.dump({"score_pct": round(score, 2), "task_means": means,
                   "episodes": eps}, f, indent=1)
    log(f"=== robustness block: {score:.1f}% ===")

if __name__ == "__main__":
    main()
