"""E89 -- the HYBRID: verified-code simulator + Claude-inferred goal -> planning -> real-env solve.

Attacks both walls the solo approaches hit:
  * sample-efficiency (DreamerV3 needs 10k+ real steps): the verified model is a FREE EXACT simulator,
    so planning happens in imagination at no env cost.
  * goal-inference (E88 got 0 because nothing said what to want): Claude inspects the game and writes
    a reward function `goal_score(frame) -> float`, the dense signal sparse levels_completed lacks.

Loop: Claude writes goal_score from sample frames + the verified dynamics -> MPC beam-search through
the verified predict() to maximize cumulative goal_score -> execute the planned actions in the REAL
env -> measure levels_completed. Compared to random and E88 novelty-MPC.

  python3 e89_arc3_hybrid.py --game s5i5 --model /tmp/e86b_claude/s5i5.json
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import arc_agi
from arcengine import GameAction

import e86_arc3 as E  # claude_cli, extract_code

HERE = Path(__file__).resolve().parent
ACTS = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4,
        GameAction.ACTION5, GameAction.ACTION6, GameAction.ACTION7]
HEX = "0123456789abcdef"


def grid(obs):
    a = np.asarray(obs.frame)
    return a[-1].reshape(64, 64) if a.ndim == 3 else a.reshape(64, 64)


def render(g):
    return "\n".join("".join(HEX[min(int(v), 15)] for v in row) for row in np.asarray(g))


def load_predict(path):
    d = json.loads(Path(path).read_text())
    ns = {"np": np, "numpy": np}
    exec(compile(d["code"], "<m>", "exec"), ns)  # noqa: S102
    return ns["predict"], d["code"], d.get("verified_exact")


GOAL_PROMPT = """This is an ARC-AGI-3 grid game (64x64, colors 0-15). The agent completes a LEVEL by
reaching some goal configuration we must infer. The VERIFIED dynamics are:

```python
{dyn}
```

Sample states it passes through (hex grid, '0'..'f' = colors 0-15):

STATE A:
{a}

STATE B (later):
{b}

Available actions: {n}. Infer the likely OBJECTIVE of the game and write a reward function

    def goal_score(frame):  # frame: np.ndarray (64,64) int; return float, HIGHER = closer to completing a level

that a planner can maximize to drive the agent toward completing a level (e.g. moving an object to a
target, aligning/collecting cells, filling a region). Use numpy (np). Return ONLY a ```python block."""


def claude_goal(dyn_code, fa, fb, n_actions):
    prompt = GOAL_PROMPT.format(dyn=dyn_code, a=render(fa), b=render(fb), n=n_actions)
    code = E.extract_code(E.claude_cli(prompt))
    ns = {"np": np, "numpy": np}
    try:
        exec(compile(code, "<goal>", "exec"), ns)  # noqa: S102
        fn = ns["goal_score"]
        float(fn(np.asarray(fa)))  # smoke test
        return fn, code
    except Exception as e:  # noqa: BLE001
        print(f"[e89] goal_score unusable ({e}); fallback to novelty", flush=True)
        return None, code


def plan_action(predict, score, frame, avail, depth=4, beam=6):
    """Beam search depth `depth` through the verified model; return first action of the best-scoring path."""
    beams = [(0.0, np.asarray(frame), None)]
    for _ in range(depth):
        nxt = []
        for _, st, fa in beams:
            for a in avail:
                try:
                    ns = np.asarray(predict(st, a))
                except Exception:  # noqa: BLE001
                    continue
                if ns.shape != (64, 64):
                    continue
                try:
                    s = float(score(ns))
                except Exception:  # noqa: BLE001
                    s = 0.0
                nxt.append((s, ns, fa if fa is not None else a))
        if not nxt:
            return None
        nxt.sort(key=lambda x: -x[0])
        beams = nxt[:beam]
    return beams[0][2]


def run(game, policy, budget, seed):
    arc = arc_agi.Arcade(); env = arc.make(game); obs = env.reset()
    avail = list(obs.available_actions); g = grid(obs); best = obs.levels_completed; steps = 0
    rng = random.Random(seed)
    while steps < budget:
        a = policy(g, avail, rng)
        if a is None:
            a = rng.choice(avail)
        obs = env.step(ACTS[a - 1]); steps += 1
        if obs is None or getattr(obs, "frame", None) is None:
            obs = env.reset(); g = grid(obs); avail = list(obs.available_actions); continue
        g = grid(obs); best = max(best, obs.levels_completed)
        if str(obs.state) != "GameState.NOT_FINISHED":
            obs = env.reset(); g = grid(obs); avail = list(obs.available_actions)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="s5i5")
    ap.add_argument("--model", required=True)
    ap.add_argument("--budget", type=int, default=1500)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    predict, dyn_code, fid = load_predict(args.model)
    arc = arc_agi.Arcade(); env = arc.make(args.game); o = env.reset()
    avail = list(o.available_actions); win = o.win_levels
    fa = grid(o)
    for _ in range(8):
        o = env.step(ACTS[random.Random(1).choice(avail) - 1])
        if o is not None and getattr(o, "frame", None) is not None:
            fb = grid(o)
    print(f"[e89/{args.game}] model fidelity {fid} | inferring goal via Claude...", flush=True)
    goal_fn, goal_code = claude_goal(dyn_code, fa, fb, len(avail))

    def hybrid_pol(g, avail, rng):
        if goal_fn is None:
            return rng.choice(avail)
        return plan_action(predict, goal_fn, g, avail, args.depth)

    def random_pol(g, avail, rng):
        return rng.choice(avail)

    lv_hybrid = run(args.game, hybrid_pol, args.budget, args.seed)
    lv_random = run(args.game, random_pol, args.budget, args.seed)
    res = {"game": args.game, "model_fidelity": fid, "win_levels": win, "budget": args.budget,
           "levels_hybrid": lv_hybrid, "levels_random": lv_random, "solved": lv_hybrid >= win,
           "goal_code": goal_code}
    print(f"[e89/{args.game}] HYBRID levels {lv_hybrid}/{win} vs random {lv_random}/{win} | "
          f"SOLVED={res['solved']}", flush=True)
    out = Path(args.out) if args.out else HERE / "results" / f"e89_arc3_{args.game}.json"
    out.write_text(json.dumps(res, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
