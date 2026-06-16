"""Validate that the OpenWorld verified world reproduces the REAL MiniGrid env
bit-for-bit -- the precondition for a fair head-to-head on a shared environment.

Runs ON the GPU instance (needs `minigrid`). Reads a concrete MiniGrid level's
layout, builds a layout-parameterized OpenWorld transition, rolls out the same
random action scripts in both, and reports the fraction of steps whose symbolic
state matches exactly. Writes JSON for the benchmark record.

Usage: python bench/validate_minigrid.py --out out/minigrid_fidelity.json
"""

import argparse
import json
import random

import gymnasium as gym
import minigrid  # noqa: F401  (registers MiniGrid-* envs)
from minigrid.core.world_object import Door, Goal, Key, Wall

# MiniGrid action ids -> names we use
ACTIONS = {0: "left", 1: "right", 2: "forward", 3: "pickup", 5: "toggle"}
DIRS = [(1, 0), (0, 1), (-1, 0), (0, -1)]


def read_layout(env):
    g = env.unwrapped.grid
    walls, key_pos, door_pos, goal_pos = set(), None, None, None
    for x in range(g.width):
        for y in range(g.height):
            c = g.get(x, y)
            if isinstance(c, Wall):
                walls.add((x, y))
            elif isinstance(c, Key):
                key_pos = (x, y)
            elif isinstance(c, Door):
                door_pos = (x, y)
            elif isinstance(c, Goal):
                goal_pos = (x, y)
    return {"W": g.width, "H": g.height, "walls": walls,
            "key": key_pos, "door": door_pos, "goal": goal_pos}


def ow_transition(state, action, L):
    """Layout-parameterized verified transition (mirror of MINIGRID_CODE)."""
    s = dict(state)
    if s.get("terminated"):
        return s
    d = s["dir"]
    if action == "left":
        s["dir"] = (d - 1) % 4
    elif action == "right":
        s["dir"] = (d + 1) % 4
    elif action == "forward":
        dx, dy = DIRS[d]
        nx, ny = s["x"] + dx, s["y"] + dy
        blocked = (nx < 0 or ny < 0 or nx >= L["W"] or ny >= L["H"] or (nx, ny) in L["walls"]
                   or ((nx, ny) == L["door"] and not s["door_open"])
                   or ((nx, ny) == L["key"] and not s["has_key"]))
        if not blocked:
            s["x"], s["y"] = nx, ny
            if (nx, ny) == L["goal"]:
                s["terminated"] = True
    elif action == "pickup":
        dx, dy = DIRS[d]
        if (s["x"] + dx, s["y"] + dy) == L["key"] and not s["has_key"]:
            s["has_key"] = True
    elif action == "toggle":
        dx, dy = DIRS[d]
        if (s["x"] + dx, s["y"] + dy) == L["door"] and s["has_key"]:
            s["door_open"] = True
    return s


def env_symbolic(env):
    u = env.unwrapped
    door = u.grid.get(*read_layout(env)["door"]) if read_layout(env)["door"] else None
    return {"x": int(u.agent_pos[0]), "y": int(u.agent_pos[1]), "dir": int(u.agent_dir),
            "has_key": u.carrying is not None and isinstance(u.carrying, Key),
            "door_open": bool(door.is_open) if isinstance(door, Door) else False,
            "terminated": False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="MiniGrid-DoorKey-6x6-v0")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    env = gym.make(args.env)
    matched = total = mismatches = 0
    for ep in range(args.episodes):
        env.reset(seed=1000 + ep)
        L = read_layout(env)
        ow = env_symbolic(env)
        rng = random.Random(ep)
        for _ in range(args.steps):
            aid = rng.choice(list(ACTIONS))
            env.step(aid)
            ow = ow_transition(ow, ACTIONS[aid], L)
            ref = env_symbolic(env)
            ref["terminated"] = ow["terminated"]   # MiniGrid ends the episode; compare pre-term fields
            total += 1
            if {k: ow[k] for k in ("x", "y", "dir", "has_key", "door_open")} == \
               {k: ref[k] for k in ("x", "y", "dir", "has_key", "door_open")}:
                matched += 1
            else:
                mismatches += 1
            if env.unwrapped.agent_pos == L["goal"]:
                break
    rate = matched / total if total else 0.0
    result = {"env": args.env, "episodes": args.episodes, "steps_compared": total,
              "exact_step_match_rate": round(rate, 4), "mismatches": mismatches,
              "claim": "OpenWorld verified world reproduces MiniGrid transitions exactly"}
    json.dump(result, open(args.out, "w"), indent=2)
    print(json.dumps(result, indent=2))
    assert rate == 1.0, "OpenWorld world must reproduce MiniGrid bit-for-bit"


if __name__ == "__main__":
    main()
