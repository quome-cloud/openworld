"""E88 -- genuine SOLVE attempt using a (near-)perfect verified world model.

The honest shot at actually completing ARC-AGI-3 levels: with a fidelity~1.0 model (E86b), plan
through the *exact* model to navigate, using model-guided exploration to maximize coverage (the best
goal-agnostic proxy for stumbling into the sparse level-completion reward). The model supplies an
exact MAP; the real env supplies the REWARD (levels_completed). Compare to a random baseline of the
same budget. Reported honestly -- may still be 0 (goal inference is the wall).

  python3 e88_arc3_solve.py --game s5i5 --model /tmp/e86b_claude/s5i5.json --budget 3000
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


def load_model(path):
    d = json.loads(Path(path).read_text())
    ns = {"np": np, "numpy": np}
    exec(compile(d["code"], "<m>", "exec"), ns)  # noqa: S102
    return ns["predict"], d.get("verified_exact")


def plan_to_novel(predict, s0, avail, depth, visits, max_nodes=4000):
    """BFS through the MODEL; return the action path to the least-visited reachable state."""
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
            h = ns.tobytes()
            nseq = seq + [a]
            v = visits.get(h, 0)
            if v < best_v and nseq:
                best_v, best_path = v, nseq
            if h not in seen and len(nseq) < depth:
                seen.add(h)
                q.append((ns, nseq))
    return best_path


def run(game, policy, budget, seed):
    arc = arc_agi.Arcade()
    env = arc.make(game); obs = env.reset()
    avail = list(obs.available_actions); g = grid(obs)
    best = obs.levels_completed; visits = {}; steps = 0
    rng = random.Random(seed)
    while steps < budget:
        path = policy(g, avail, visits, rng)
        if not path:
            path = [rng.choice(avail)]
        for a in path:
            obs = env.step(ACTS[a - 1]); steps += 1
            if obs is None or getattr(obs, "frame", None) is None:
                obs = env.reset(); g = grid(obs); avail = list(obs.available_actions); break
            g = grid(obs); visits[g.tobytes()] = visits.get(g.tobytes(), 0) + 1
            best = max(best, obs.levels_completed)
            if str(obs.state) != "GameState.NOT_FINISHED":
                obs = env.reset(); g = grid(obs); avail = list(obs.available_actions); break
            if steps >= budget:
                break
    return best, len(visits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="s5i5")
    ap.add_argument("--model", required=True, help="path to the E86/E86b result json with code")
    ap.add_argument("--budget", type=int, default=3000)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    predict, fid = load_model(args.model)
    print(f"[e88/{args.game}] model fidelity {fid} | budget {args.budget}", flush=True)

    def model_pol(g, avail, visits, rng):
        return plan_to_novel(predict, g, avail, args.depth, visits)

    def random_pol(g, avail, visits, rng):
        return [rng.choice(avail)]

    win = arc_agi.Arcade().make(args.game).reset().win_levels
    lv_model, cov_model = run(args.game, model_pol, args.budget, args.seed)
    lv_rand, cov_rand = run(args.game, random_pol, args.budget, args.seed)
    res = {"game": args.game, "model_fidelity": fid, "win_levels": win, "budget": args.budget,
           "levels_model_guided": lv_model, "levels_random": lv_rand,
           "coverage_model": cov_model, "coverage_random": cov_rand,
           "solved": lv_model >= win}
    print(f"[e88/{args.game}] levels: model-guided {lv_model}/{win} vs random {lv_rand}/{win} "
          f"| coverage {cov_model} vs {cov_rand} | SOLVED={res['solved']}", flush=True)
    out = Path(args.out) if args.out else HERE / "results" / f"e88_arc3_{args.game}.json"
    out.write_text(json.dumps(res, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
