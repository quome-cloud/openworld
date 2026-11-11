"""Shared episode runner: BALROG-identical env stack + DiveAgent + logging.

Every env touchpoint (source-leak audit surface):
  env = nh_harness.make_env()          # vendored balrog.environments.make_env
  env.reset(seed=seed)                 # protocol reset
  env.step(action_string)              # protocol step
  env.env.language_action_space        # the wrapper's own action list
  env.get_stats()                      # scoring (same method BALROG's
                                       #  evaluator calls); runner-side only
  env.close()
Nothing else. The agent consumes only the served obs dict.
"""

import json
import os
import random
import time

import numpy as np

import nh_harness as H
import nh_common as C
from nh_agent import DiveAgent
from nh_transitions import TransitionLogger

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
TRAJ = os.path.join(RESULTS, "trajectories")
os.makedirs(TRAJ, exist_ok=True)

MAX_LOOP = 110_000     # safety bound above the env's own 100k cap


def run_episode(ep, seed, condition="A", label="clean_A", memory=None,
                log=print, frame_cap=5000):
    env = H.make_env()
    random.seed(seed)
    np.random.seed(seed)
    obs, info = env.reset(seed=seed)
    space = list(env.env.language_action_space)
    agent = DiveAgent(log=log, memory=memory)
    agent.set_actions(space)

    tag = "NetHackChallenge-v0"
    tlog = TransitionLogger(tag, ep, seed, condition, "DiveAgent", space,
                            label)
    if memory is not None:
        memory.begin_episode(os.path.relpath(tlog.fn, RESULTS))
    tlog.log_reset(obs)

    traj = {"task": tag, "episode": ep, "seed": seed, "condition": condition,
            "actions": [], "frames": [], "frame_steps": [], "positions": [],
            "hp": [], "depth": [], "messages": [], "notes": [],
            "mem_fired": []}

    def snap(o, step):
        tty = o["obs"]["tty_chars"]
        traj["frames"].append(["".join(chr(c) for c in row).rstrip()
                               for row in tty])
        traj["frame_steps"].append(step)

    def want_frame(step, agent, depth_changed, hp_frac):
        if len(traj["frames"]) >= frame_cap:
            return False
        if depth_changed or step < 400:
            return True
        if hp_frac < 0.3:
            return step % 3 == 0
        return step % max(3, (step // 2000) + 3) == 0

    snap(obs, 0)
    steps = 0
    done = False
    illegal = 0
    depth_max, xp_max = 1, 1
    last_depth = 1
    t0 = time.time()
    while not done and steps < MAX_LOOP:
        a = agent.act(obs)
        if a not in space:
            illegal += 1
            traj["notes"].append(f"step {steps}: illegal '{a}' -> search")
            a = "search"
        obs, r, term, trunc, info = env.step(a)
        steps += 1
        done = term or trunc
        tlog.log_step(a, obs, r, done, info)
        A = agent.atlas
        depth_max = max(depth_max, A.depth)
        xp_max = max(xp_max, A.xplvl)
        dchg = A.depth != last_depth
        last_depth = A.depth
        traj["actions"].append(a)
        traj["positions"].append(list(A.agent))
        traj["hp"].append([A.hp, A.hpmax])
        traj["depth"].append(A.depth)
        traj["messages"].append(A.message[:150])
        if want_frame(steps, agent, dchg, A.hp / max(1, A.hpmax)):
            snap(obs, steps)
        if steps % 2000 == 0:
            log(f"    step {steps} depth {A.depth} (max {depth_max}) "
                f"hp {A.hp}/{A.hpmax} xp {A.xplvl} t={A.time} "
                f"wall={time.time()-t0:.0f}s")
    snap(obs, steps)

    stats = env.get_stats()
    env.close()
    tfile = tlog.close()
    traj["notes"].extend([f"step {s}: {n}" for (s, n) in agent.notes])
    traj["mem_fired"] = [f"step {s}: {n}" for (s, n) in agent.mem_fired]

    result = {
        "task": tag,
        "episode": ep,
        "seed": seed,
        "condition": condition,
        "steps": steps,
        "illegal": illegal,
        "progression": float(stats["progression"]),
        "highest_achievement": stats.get("highest_achievement"),
        "depth_max": depth_max,
        "xplvl_max": xp_max,
        "dlvl_list": stats.get("dlvl_list"),
        "xplvl_list": stats.get("xplvl_list"),
        "end_reason": str(stats.get("end_reason")),
        "role": agent.role,
        "race": agent.race,
        "wallclock_s": round(time.time() - t0, 1),
        "transitions_file": os.path.relpath(tfile, RESULTS),
        "mem_fired": traj["mem_fired"],
    }
    if memory is not None:
        memory.end_episode(result, steps)

    tj = os.path.join(TRAJ, f"{label}__ep{ep}.json")
    with open(tj, "w") as f:
        json.dump(traj, f)
    return result
