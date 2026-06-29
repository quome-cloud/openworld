"""E90 -- object-graph novelty exploration to trigger the FIRST reward (the binding constraint).

E88 showed pixel-novelty exploration saturates a tiny state set and never triggers a level; E89b
showed negative feedback alone can't bootstrap. The missing piece is ONE positive reward. This
explores in OBJECT-CONFIGURATION space (move objects into new arrangements -- block-on-target,
collected, etc.), planned in the FREE exact simulator, which is far denser/more meaningful than
pixel novelty. Goal: get levels_completed > 0 once, so everything downstream can bootstrap.

Compares: graph-novelty vs pixel-novelty vs random, on levels reached + states covered.

  python3 e90_arc3_explore.py --game sp80 --model /tmp/e86b_claude/sp80.json --budget 3000
"""
import argparse
import json
import random
from collections import deque
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


def load_predict(path):
    d = json.loads(Path(path).read_text())
    ns = {"np": np, "numpy": np}
    exec(compile(d["code"], "<m>", "exec"), ns)  # noqa: S102
    return ns["predict"], d.get("verified_exact")


def graph_sig(frame):
    """Coarse object-configuration signature (color,size, position bucketed to 4 cells)."""
    objs, _ = GR.objects(frame)
    return tuple(sorted((o["color"], o["size"], int(o["centroid"][0]) // 4, int(o["centroid"][1]) // 4)
                        for o in objs))


def plan_to_novel(predict, s0, avail, depth, visits, key_fn, max_nodes=3000):
    """BFS through the model; return the action path to the least-visited state under key_fn."""
    s0 = np.asarray(s0)
    q = deque([(s0, [])])
    seen = {s0.tobytes()}
    best_path, best_v = [], float("inf")
    nodes = 0
    while q and nodes < max_nodes:
        state, seq = q.popleft()
        for a in avail:
            try:
                ns = np.asarray(predict(state, a))
            except Exception:  # noqa: BLE001
                continue
            if ns.shape != (64, 64):
                continue
            nodes += 1
            nseq = seq + [a]
            v = visits.get(key_fn(ns), 0)
            if v < best_v and nseq:
                best_v, best_path = v, nseq
            h = ns.tobytes()
            if h not in seen and len(nseq) < depth:
                seen.add(h)
                q.append((ns, nseq))
    return best_path


def run(game, predict, key_fn, budget, depth, seed, use_model=True):
    arc = arc_agi.Arcade(); env = arc.make(game); obs = env.reset()
    avail = list(obs.available_actions); g = grid(obs); best = obs.levels_completed
    visits = {}; steps = 0; rng = random.Random(seed)
    while steps < budget:
        if use_model:
            path = plan_to_novel(predict, g, avail, depth, visits, key_fn) or [rng.choice(avail)]
        else:
            path = [rng.choice(avail)]
        for a in path:
            obs = env.step(ACTS[a - 1]); steps += 1
            if obs is None or getattr(obs, "frame", None) is None:
                obs = env.reset(); g = grid(obs); avail = list(obs.available_actions); break
            g = grid(obs); k = key_fn(g); visits[k] = visits.get(k, 0) + 1
            best = max(best, obs.levels_completed)
            if str(obs.state) != "GameState.NOT_FINISHED":
                obs = env.reset(); g = grid(obs); avail = list(obs.available_actions); break
            if steps >= budget:
                break
    return best, len(visits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="sp80")
    ap.add_argument("--model", required=True)
    ap.add_argument("--budget", type=int, default=3000)
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    predict, fid = load_predict(args.model)
    win = arc_agi.Arcade().make(args.game).reset().win_levels
    print(f"[e90/{args.game}] fidelity {fid} | budget {args.budget}", flush=True)

    lv_g, cov_g = run(args.game, predict, graph_sig, args.budget, args.depth, args.seed, use_model=True)
    lv_p, cov_p = run(args.game, predict, (lambda f: f.tobytes()), args.budget, args.depth, args.seed, use_model=True)
    lv_r, cov_r = run(args.game, predict, graph_sig, args.budget, args.depth, args.seed, use_model=False)
    res = {"game": args.game, "fidelity": fid, "win_levels": win, "budget": args.budget,
           "levels_graph_novelty": lv_g, "levels_pixel_novelty": lv_p, "levels_random": lv_r,
           "graph_configs_seen": cov_g, "pixel_states_seen": cov_p,
           "first_reward": lv_g > 0 or lv_p > 0}
    print(f"[e90/{args.game}] levels: graph-nov {lv_g} | pixel-nov {lv_p} | random {lv_r}  (/{win}) "
          f"| object-configs seen {cov_g}", flush=True)
    out = Path(args.out) if args.out else HERE / "results" / f"e90_arc3_{args.game}.json"
    out.write_text(json.dumps(res, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
