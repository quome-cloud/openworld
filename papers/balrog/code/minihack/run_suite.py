"""Official BALROG MiniHack suite run.

Protocol (from BALROG balrog/config/config.yaml):
  - 8 tasks, 5 episodes each, minihack_kwargs verbatim (100-step cap,
    penalty_step -0.01 constant, autopickup off, skip_more on, char '@').
  - Score per episode = progression (1.0 iff final reward >= 1.0), task
    score = mean over its 5 episodes, suite score = mean over 8 tasks.
  - Env stack identical to BALROG (vendored balrog.environments).

Seeds: fresh block (base 1000), disjoint from the development seeds (3-30)
used while synthesizing the planners. Python/np RNG + NLE core all seeded.

Logs per episode: every action, position, hp, game-time, message, reward,
plus full tty frames (for replay animations) -> results/trajectories/.
"""

import json
import os
import random
import sys
import time

import numpy as np

import mh_common as C
import mh_harness as H
from agent_boxoban import BoxobanAgent
from agent_explore import ExploreAgent
from agent_battle import BattleAgent
from agent_quest import QuestAgent

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
TRAJ = os.path.join(RESULTS, "trajectories")
os.makedirs(TRAJ, exist_ok=True)

RESULTS_JSON = os.path.join(RESULTS, "minihack_results.json")
RUN_LOG = os.path.join(RESULTS, "RUN_LOG.txt")

SOTA = 40.0  # BALROG leaderboard MiniHack column, Gemini-3-Pro (2026-07-06);
             # see verify_leaderboard.py -- an earlier 90.0 figure was a
             # summarizer misread of the BabaIsAI column


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(RUN_LOG, "a") as f:
        f.write(line + "\n")


def make_agent(task, memory=None):
    quiet = lambda m: log("    " + str(m))
    if "Boxoban" in task:
        return BoxobanAgent(log=quiet)
    if "MazeWalk" in task:
        return ExploreAgent(log=quiet, maze_parity=True, frontier_mass_w=0.3,
                            memory=memory)
    if "Corridor-" in task:
        return ExploreAgent(log=quiet, frontier_mass_w=0.3, memory=memory)
    if "CorridorBattle" in task:
        return BattleAgent(log=quiet, memory=memory)
    if "Quest" in task:
        return QuestAgent(log=quiet, memory=memory)
    raise ValueError(task)


def run_episode(task, ep, seed, memory=None, condition="A", label="clean_A"):
    from transitions import TransitionLogger
    env = H.make_env(task)
    random.seed(seed)
    np.random.seed(seed)
    obs, info = env.reset(seed=seed)
    space = list(env.env.language_action_space)
    agent = make_agent(task, memory=memory)
    agent.set_actions(space)
    tlog = TransitionLogger(task, ep, seed, condition,
                            type(agent).__name__, space, label)
    tlog.log_reset(obs)

    steps = 0
    done = False
    illegal = 0
    total_r = 0.0
    t0 = time.time()
    traj = {"task": task, "episode": ep, "seed": seed, "actions": [],
            "frames": [], "positions": [], "hp": [], "messages": [],
            "rewards": [], "notes": []}

    def snap(obs):
        tty = obs["obs"]["tty_chars"]
        traj["frames"].append(["".join(chr(c) for c in row).rstrip()
                               for row in tty])

    snap(obs)
    while not done and steps < 105:
        a = agent.act(obs)
        if a not in space:
            illegal += 1
            traj["notes"].append(f"step {steps}: illegal '{a}' -> fallback")
            a = "search" if "search" in space else space[0]
        obs, r, term, trunc, info = env.step(a)
        steps += 1
        total_r += r
        done = term or trunc
        tlog.log_step(a, obs, r, term or trunc, info)
        L = agent.level
        traj["actions"].append(a)
        traj["positions"].append(list(L.agent))
        traj["hp"].append([L.hp, L.hpmax])
        traj["messages"].append(L.message[:120])
        traj["rewards"].append(round(float(r), 4))
        snap(obs)

    stats = env.get_stats()
    env.close()
    result = {
        "task": task,
        "episode": ep,
        "seed": seed,
        "steps": steps,
        "progression": float(stats["progression"]),
        "episode_return": float(stats["episode_return"]),
        "end_reason": str(stats["end_reason"]),
        "illegal_actions": illegal,
        "wallclock_s": round(time.time() - t0, 2),
    }
    if hasattr(agent, "solve_note"):
        result["solver_note"] = agent.solve_note
    if hasattr(agent, "kills"):
        result["kills"] = agent.kills
    traj["result"] = result
    fn = os.path.join(TRAJ, f"{label}__{task}__ep{ep}.json")
    with open(fn, "w") as f:
        json.dump(traj, f)
    tfn = tlog.close()
    if memory is not None:
        memory.record_episode(result, agent.level,
                              traj, os.path.relpath(tfn, RESULTS))
    result["transitions_file"] = os.path.relpath(tfn, RESULTS)
    return result, agent, traj


def main():
    all_results = []
    if os.path.exists(RESULTS_JSON):
        with open(RESULTS_JSON) as f:
            all_results = json.load(f)["episodes"]
    done_keys = {(r["task"], r["episode"]) for r in all_results}

    log(f"=== BALROG MiniHack suite: 8 tasks x {H.EPISODES_PER_TASK} episodes; "
        f"SOTA {SOTA}% (Gemini-3-Pro) ===")
    for ti, task in enumerate(H.MINIHACK_TASKS):
        for ep in range(H.EPISODES_PER_TASK):
            if (task, ep) in done_keys:
                continue
            seed = 1000 + ti * 100 + ep
            res, _agent, _traj = run_episode(task, ep, seed)
            all_results.append(res)
            per_task = {}
            for r in all_results:
                per_task.setdefault(r["task"], []).append(r["progression"])
            task_means = {t: sum(v) / len(v) for t, v in per_task.items()}
            score = 100.0 * sum(task_means.values()) / len(task_means)
            summary = {
                "suite_score_over_attempted_tasks_pct": round(score, 2),
                "sota_pct": SOTA,
                "tasks_attempted": len(task_means),
                "episodes_done": len(all_results),
                "task_means": task_means,
            }
            with open(RESULTS_JSON, "w") as f:
                json.dump({"summary": summary, "episodes": all_results}, f,
                          indent=1)
            log(f"{task} ep{ep} seed={seed} -> prog={res['progression']} "
                f"steps={res['steps']} end={res['end_reason']} "
                f"({res['wallclock_s']}s) | running score "
                f"{score:.1f}% over {len(task_means)} tasks")
    log("=== suite complete ===")


if __name__ == "__main__":
    main()
