"""PRIVILEGED-arm suite run: BALROG BabyAI protocol, world-model + search.

Protocol (official): 5 tasks x 10 episodes, fresh env per episode
(construction loop until task type matches), env.reset(seed), success =
env reward > 0 within env.max_steps => progression 1.0. Score = mean
progression over the 50 episodes (equal task weighting is automatic:
10 episodes per task).

Privileged channels used (disclosed): full grid + agent state via
env.unwrapped, instruction tree via env.unwrapped.instrs. The CLEAN arm
(run_clean.py) removes all of these.

The plan is computed once from the initial state and executed OPEN-LOOP;
success is judged solely by the env's own reward signal.
"""

from __future__ import annotations

import json
import os
import time

from balrog_env import TASKS, NUM_EPISODES, make_task_env
from symbolic_model import extract_state, extract_instr, EpisodeModel
from planner import solve, PlanFail

BASE_SEED = 770000
OUT = "results/privileged/babyai"


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open("results/privileged_log.txt", "a") as f:
        f.write(line + "\n")


def run_episode(task, episode_idx, seed):
    t0 = time.time()
    env = make_task_env(task)
    obs, info = env.reset(seed=seed)
    mission = obs["mission"]
    max_steps = env.unwrapped.max_steps

    st, id2oid = extract_state(env)
    instr = extract_instr(env, id2oid)
    model = EpisodeModel(st, instr, reset_verifier=False)

    rec = {
        "task": task, "episode": episode_idx, "seed": seed,
        "mission": mission, "max_steps": max_steps,
    }
    try:
        plan, method = solve(model)
    except PlanFail as e:
        rec.update(solved=False, error=f"plan_fail: {e}", progression=0.0,
                   time_s=round(time.time() - t0, 2))
        env.close()
        return rec

    # open-loop execution on the live env; env's own reward is the metric
    progression = 0.0
    steps = 0
    for a in plan:
        obs, reward, term, trunc, info = env.step(int(a))
        steps += 1
        if reward > 0:
            progression = 1.0
        if term or trunc:
            break
    env.close()
    rec.update(
        solved=progression == 1.0, progression=progression, method=method,
        plan_len=len(plan), steps_used=steps,
        time_s=round(time.time() - t0, 2),
    )
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
            log(f"[priv] {task} ep{ep} seed={seed} -> "
                f"{'SOLVED' if rec['solved'] else 'FAIL'} "
                f"[{rec.get('method', '-')}] plan={rec.get('plan_len', '-')} "
                f"steps={rec.get('steps_used', '-')}/{rec['max_steps']} "
                f"{rec['time_s']}s  mission={rec['mission']!r}")
        solved = sum(r["solved"] for r in results if r["task"] == task)
        log(f"[priv] {task}: {solved}/{NUM_EPISODES}")

    score = 100.0 * sum(r["progression"] for r in results) / len(results)
    summary = {
        "protocol": "privileged",
        "tasks": {t: sum(r["solved"] for r in results if r["task"] == t) for t in TASKS},
        "episodes": len(results),
        "solved": sum(r["solved"] for r in results),
        "score_pct": round(score, 2),
        "methods": {m: sum(1 for r in results if r.get("method") == m)
                    for m in ("macro", "ucs")},
        "total_time_s": round(sum(r["time_s"] for r in results), 1),
    }
    with open("results/privileged_summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    with open("results/privileged_results.json", "w") as f:
        json.dump(results, f, indent=1)
    log(f"[priv] FINAL: {summary['solved']}/{summary['episodes']} episodes, "
        f"score {summary['score_pct']}%  methods={summary['methods']}")


if __name__ == "__main__":
    main()
