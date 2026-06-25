"""E93 -- capture the FIRST reward transition to learn the win condition (the bootstrap).

E88/E90 showed undirected exploration rarely triggers a reward, but on some games (sp80) it does.
Rather than guess the goal (E89), we PLAY until levels_completed increments, then record the exact
triggering transition (before/after frames + action + object-graph diff) -- the ground-truth win
condition. That single positive example is what every downstream method (induce win-rule, direct a
plan to reproduce it) was missing. Local-only (CPU + env).

  python3 e93_capture_reward.py --game sp80 --budget 4000
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import arc_agi
from arcengine import GameAction

import arc3_graph as GR

HERE = Path(__file__).resolve().parent
ACTS = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4,
        GameAction.ACTION5, GameAction.ACTION6, GameAction.ACTION7]


def grid(obs):
    a = np.asarray(obs.frame)
    return a[-1].reshape(64, 64) if a.ndim == 3 else a.reshape(64, 64)


def capture(game, budget, seed):
    arc = arc_agi.Arcade(); env = arc.make(game); obs = env.reset()
    avail = list(obs.available_actions); g = grid(obs); lvl = obs.levels_completed
    rng = random.Random(seed); rewards = []; recent = []
    for step in range(budget):
        a = rng.choice(avail)
        nobs = env.step(ACTS[a - 1])
        if nobs is None or getattr(nobs, "frame", None) is None:
            obs = env.reset(); g = grid(obs); avail = list(obs.available_actions); recent = []; continue
        ng = grid(nobs); nl = nobs.levels_completed
        if nl > lvl:  # a level completed on THIS step -> the win transition
            rewards.append({"step": step, "action": a, "level_to": nl,
                            "diff": GR.graph_diff(g, ng),
                            "objs_before": GR.graph_repr(g), "objs_after": GR.graph_repr(ng),
                            "full_prefix": list(recent) + [a], "recent_actions": recent[-12:]})
            print(f"[e93/{game}] REWARD at step {step}: action {a} -> level {nl} | diff {rewards[-1]['diff']}", flush=True)
            if len(rewards) >= 3:
                break
        recent.append(a)
        g = ng; lvl = nl
        if str(nobs.state) != "GameState.NOT_FINISHED":
            obs = env.reset(); g = grid(obs); avail = list(obs.available_actions); lvl = obs.levels_completed; recent = []
    return rewards


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="sp80")
    ap.add_argument("--budget", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    print(f"[e93/{args.game}] searching {args.budget} steps for a reward transition...", flush=True)
    rewards = capture(args.game, args.budget, args.seed)
    res = {"game": args.game, "budget": args.budget, "n_rewards": len(rewards), "rewards": rewards}
    if rewards:
        acts = Counter(r["action"] for r in rewards)
        res["winning_actions"] = dict(acts)
        print(f"[e93/{args.game}] captured {len(rewards)} reward(s); winning actions {dict(acts)}", flush=True)
    else:
        print(f"[e93/{args.game}] NO reward found in {args.budget} steps", flush=True)
    out = Path(args.out) if args.out else HERE / "results" / f"e93_reward_{args.game}.json"
    out.write_text(json.dumps(res, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
