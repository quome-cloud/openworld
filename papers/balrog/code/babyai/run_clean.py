"""CLEAN-arm suite run: same 50 seeded episodes as the privileged run, but
the agent sees ONLY the env's returned observation channel:
obs['image'] (7x7 occluded egocentric view), obs['direction'], obs['mission'].

No env.unwrapped access, no instruction-tree readout, no clone verification,
no fallback search on the real env. Closed-loop: observe -> update belief ->
replan -> one action. Success = the env's own reward signal.
"""

from __future__ import annotations

import json
import os
import time

from balrog_env import TASKS, NUM_EPISODES, make_task_env
from clean_agent import CleanAgent

BASE_SEED = 770000     # identical seeds to the privileged run
OUT = "results/clean/babyai"


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open("results/clean_log.txt", "a") as f:
        f.write(line + "\n")


def run_episode(task, episode_idx, seed):
    t0 = time.time()
    env = make_task_env(task)
    obs, info = env.reset(seed=seed)
    mission = obs["mission"]
    max_steps = env.unwrapped.max_steps  # benchmark constant per task type,
    # also equals the wrapper-exposed env.max_steps BALROG gives its agents

    agent = CleanAgent(mission)
    progression = 0.0
    steps = 0
    err = None
    actions = []
    try:
        for _ in range(max_steps):
            a = agent.act(obs)
            actions.append(int(a))
            obs, reward, term, trunc, info = env.step(int(a))
            steps += 1
            if reward > 0:
                progression = 1.0
            if term or trunc:
                break
    except Exception as e:  # honest failure accounting; no retries
        err = f"{type(e).__name__}: {e}"
    env.close()
    rec = {
        "task": task, "episode": episode_idx, "seed": seed, "mission": mission,
        "max_steps": max_steps, "solved": progression == 1.0,
        "progression": progression, "steps_used": steps,
        "time_s": round(time.time() - t0, 2),
    }
    if err:
        rec["error"] = err
    return rec


def main():
    os.makedirs(OUT, exist_ok=True)
    results = []
    for ti, task in enumerate(TASKS):
        tdir = os.path.join(OUT, task)
        os.makedirs(tdir, exist_ok=True)
        for ep in range(NUM_EPISODES):
            seed = BASE_SEED + ti * 100 + ep
            path = os.path.join(tdir, f"episode_{ep:02d}.json")
            if os.path.exists(path):
                rec = json.load(open(path))
            else:
                rec = run_episode(task, ep, seed)
                with open(path, "w") as f:
                    json.dump(rec, f, indent=1)
            results.append(rec)
            log(f"[clean] {task} ep{ep} seed={seed} -> "
                f"{'SOLVED' if rec['solved'] else 'FAIL'} "
                f"steps={rec['steps_used']}/{rec['max_steps']} {rec['time_s']}s"
                + (f" ERROR={rec['error']}" if rec.get('error') else "")
                + f"  mission={rec['mission']!r}")
        solved = sum(r["solved"] for r in results if r["task"] == task)
        log(f"[clean] {task}: {solved}/{NUM_EPISODES}")

    score = 100.0 * sum(r["progression"] for r in results) / len(results)
    summary = {
        "protocol": "clean",
        "tasks": {t: sum(r["solved"] for r in results if r["task"] == t) for t in TASKS},
        "episodes": len(results),
        "solved": sum(r["solved"] for r in results),
        "score_pct": round(score, 2),
        "mean_steps": round(sum(r["steps_used"] for r in results) / len(results), 1),
        "total_time_s": round(sum(r["time_s"] for r in results), 1),
    }
    with open("results/clean_summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    with open("results/clean_results.json", "w") as f:
        json.dump(results, f, indent=1)
    log(f"[clean] FINAL: {summary['solved']}/{summary['episodes']} episodes, "
        f"score {summary['score_pct']}%")


if __name__ == "__main__":
    main()
