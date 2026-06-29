"""E87 -- planning through the verified ARC-AGI-3 code world model (the payoff of E86).

Two measurements, both using the per-game `predict(frame, action)` synthesized + verified in E86
(loaded from experiments/results/arc3_claude/<game>.json; no LLM, no GPU -- pure CPU search):

 1. CONTROLLABILITY (the positive result): can we PLAN to reach a target state through the model?
    Sample a reachable goal (H random actions from reset), then BFS over action sequences *in the
    model* to reach it; execute the found plan in the REAL env; success = real frame == goal exactly.
    Baseline: a random action sequence. A correct model => plans that work in reality; this should
    track E86 fidelity, and random ~ 0.

 2. LEVEL ATTEMPT (the honest official metric): model-predictive control with novelty-driven
    lookahead, run in the real env; report levels_completed vs random. ARC-3 is unsolved, so ~0 is
    expected -- reported honestly.

  python3 e87_arc3_plan.py --game sb26 --results experiments/results/arc3_claude
"""
import argparse
import json
import random
from collections import deque
from pathlib import Path

import numpy as np
import arc_agi
from arcengine import GameAction

HERE = Path(__file__).resolve().parent
ACTS = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4,
        GameAction.ACTION5, GameAction.ACTION6, GameAction.ACTION7]


def grid(obs):
    a = np.asarray(obs.frame)
    return a[-1].reshape(64, 64) if a.ndim == 3 else a.reshape(64, 64)


def load_predict(game, results_dir):
    """Load + compile the E86-synthesized predict(frame, action) for a game (action is 1-indexed)."""
    d = json.loads((Path(results_dir) / f"{game}.json").read_text())
    code = d.get("code")
    if not code:
        return None, d.get("verified_exact")
    ns = {"np": np, "numpy": np}
    try:
        exec(compile(code, "<e86>", "exec"), ns)  # noqa: S102 -- our own verified artifact
        return ns.get("predict"), d.get("verified_exact")
    except Exception:  # noqa: BLE001
        return None, d.get("verified_exact")


def bfs_plan(predict, s0, goal, action_ids, max_depth):
    """Search action sequences IN THE MODEL for one whose predicted state == goal. Returns the
    sequence (1-indexed actions) or None. Dedup on predicted-state bytes."""
    goal = np.asarray(goal)
    if np.array_equal(np.asarray(s0), goal):
        return []
    q = deque([(np.asarray(s0), [])])
    seen = {np.asarray(s0).tobytes()}
    while q:
        state, seq = q.popleft()
        if len(seq) >= max_depth:
            continue
        for a in action_ids:
            try:
                ns = np.asarray(predict(state, a))
            except Exception:  # noqa: BLE001
                continue
            if ns.shape != (64, 64):
                continue
            nseq = seq + [a]
            if np.array_equal(ns, goal):
                return nseq
            h = ns.tobytes()
            if h not in seen and len(nseq) < max_depth:
                seen.add(h)
                q.append((ns, nseq))
    return None


def controllability(game, predict, n_tasks=20, H=4, seed=0):
    """Plan-to-goal success: model-planned vs random, on reachable targets."""
    arc = arc_agi.Arcade()
    rng = random.Random(seed)
    plan_ok = rand_ok = planned = n = 0
    for _ in range(n_tasks):
        env = arc.make(game); obs = env.reset()
        avail = list(obs.available_actions)  # 1-indexed
        s0 = grid(obs)
        seq = [rng.choice(avail) for _ in range(H)]
        g = s0
        for a in seq:
            obs = env.step(ACTS[a - 1]); g = grid(obs)
        goal = g
        if np.array_equal(goal, s0):
            continue  # degenerate (no movement) -- skip
        n += 1
        plan = bfs_plan(predict, s0, goal, avail, max_depth=H + 1)
        if plan is not None:
            planned += 1
            env2 = arc.make(game); o = env2.reset()
            for a in plan:
                o = env2.step(ACTS[a - 1])
            if np.array_equal(grid(o), goal):
                plan_ok += 1
        env3 = arc.make(game); o = env3.reset()
        for _ in range(len(plan) if plan else H):
            o = env3.step(ACTS[rng.choice(avail) - 1])
        if np.array_equal(grid(o), goal):
            rand_ok += 1
    return {"n_tasks": n, "plan_found_frac": round(planned / n, 3) if n else None,
            "plan_reach_frac": round(plan_ok / n, 3) if n else None,
            "random_reach_frac": round(rand_ok / n, 3) if n else None}


def level_attempt(game, predict, budget=200, depth=3, seed=0):
    """Honest official metric: novelty-driven MPC through the model vs random; report levels."""
    arc = arc_agi.Arcade()

    def run(policy):
        env = arc.make(game); obs = env.reset()
        avail = list(obs.available_actions); g = grid(obs); best = obs.levels_completed; seen = set()
        for _ in range(budget):
            a = policy(g, avail, seen)
            obs = env.step(ACTS[a - 1])
            if obs is None or getattr(obs, "frame", None) is None:
                obs = env.reset(); g = grid(obs); avail = list(obs.available_actions); continue
            g = grid(obs); seen.add(g.tobytes()); best = max(best, obs.levels_completed)
            if str(obs.state) != "GameState.NOT_FINISHED":
                obs = env.reset(); g = grid(obs); avail = list(obs.available_actions)
        return best

    rng = random.Random(seed)

    def random_pol(g, avail, seen):
        return rng.choice(avail)

    def mpc_pol(g, avail, seen):
        # pick the action whose model-predicted state is most novel (1-step; cheap proxy planning)
        best_a, best_score = rng.choice(avail), -1
        for a in avail:
            try:
                ns = np.asarray(predict(np.asarray(g), a))
            except Exception:  # noqa: BLE001
                continue
            score = 0 if ns.tobytes() in seen else int((ns != g).sum())  # novel + changes a lot
            if score > best_score:
                best_score, best_a = score, a
        return best_a

    return {"levels_random": run(random_pol), "levels_mpc": run(mpc_pol),
            "win_levels": arc.make(game).reset().win_levels}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="sb26")
    ap.add_argument("--results", default=str(HERE / "results" / "arc3_claude"))
    ap.add_argument("--n_tasks", type=int, default=20)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--budget", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    predict, fid = load_predict(args.game, args.results)
    res = {"game": args.game, "e86_fidelity": fid}
    if predict is None:
        res["error"] = "no usable predict() code"
        print(f"[e87/{args.game}] no usable model", flush=True)
    else:
        res["controllability"] = controllability(args.game, predict, args.n_tasks, args.horizon, args.seed)
        res["levels"] = level_attempt(args.game, predict, args.budget, seed=args.seed)
        c, lv = res["controllability"], res["levels"]
        print(f"[e87/{args.game}] fid={fid} | plan-reach {c['plan_reach_frac']} vs random "
              f"{c['random_reach_frac']} | levels mpc {lv['levels_mpc']}/{lv['win_levels']} "
              f"vs random {lv['levels_random']}", flush=True)
    out = Path(args.out) if args.out else HERE / "results" / f"e87_arc3_{args.game}.json"
    out.write_text(json.dumps(res, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
