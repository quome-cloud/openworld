"""E89b -- CLOSED-LOOP hybrid: infer goal -> plan in verified sim -> execute real -> OBSERVE reward
-> revise goal. The fix for E89's one-shot guesses (s5i5 mistook a timer for the goal; sp80 a
plausible-but-unconfirmed dock). Here Claude sees what its goal actually *did* (levels gained, and
critically whether optimizing it ended the game without winning) and revises -- goal discovery, not
a single guess.

  python3 e89b_closed_loop.py --game s5i5 --model /tmp/e86b_claude/s5i5.json --rounds 5
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import arc_agi
from arcengine import GameAction

import e86_arc3 as E
import arc3_graph as GR

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


BASE = """ARC-AGI-3 grid game (64x64, colors 0-15). The agent completes LEVELS by reaching some goal we
must infer. Verified dynamics:
```python
{dyn}
```
Sample states (hex grid):
STATE A:
{a}
STATE B (later):
{b}
Available actions: {n}.
"""

ASK = """\nWrite a reward function for a planner to maximize:

    def goal_score(frame):  # (64,64) int -> float, HIGHER = closer to completing a level

Use numpy (np). Return ONLY a ```python block."""


def history_block(history):
    if not history:
        return ""
    s = "\n\nPREVIOUS ATTEMPTS (learn from these -- the real env gave this feedback):\n"
    for i, (code, out) in enumerate(history):
        verdict = ""
        if out["levels"] == 0 and out["terminations"] > 0:
            verdict = (" -> optimizing this goal repeatedly ENDED THE GAME with 0 levels won. That "
                       "means your goal was a LOSING/terminal condition, not the win. Infer a "
                       "DIFFERENT objective (e.g. survive, or achieve something BEFORE a timer runs out).")
        elif out["levels"] > 0:
            verdict = f" -> gained {out['levels']} level(s)! Keep this direction, refine it."
        else:
            verdict = " -> no levels and no terminations (goal had little effect). Try a more decisive objective."
        s += (f"\nattempt {i}: levels={out['levels']}, game-overs={out['terminations']}, "
              f"steps={out['steps']}.{verdict}\n  end-state:\n{out['end_render']}\n  (the goal code you tried:)\n{code}\n")
    return s


def claude_goal(dyn, fa, fb, n, history, graph=False):
    prompt = BASE.format(dyn=dyn, a=render(fa), b=render(fb), n=n)
    if graph:
        prompt += (f"\n\nOBJECT-GRAPH VIEW (reason about goals relationally over these objects):\n"
                   f"STATE A:\n{GR.graph_repr(fa)}\nSTATE B:\n{GR.graph_repr(fb)}\n"
                   f"relational change A->B: {GR.graph_diff(fa, fb)}\n"
                   f"Express goal_score over object relations (e.g. minimize distance between the "
                   f"movable object and a target object, or maximize covered target cells).")
    prompt += history_block(history) + ASK
    code = E.extract_code(E.claude_cli(prompt, timeout=600))
    ns = {"np": np, "numpy": np}
    try:
        exec(compile(code, "<g>", "exec"), ns)  # noqa: S102
        fn = ns["goal_score"]
        float(fn(np.asarray(fa)))
        return fn, code
    except Exception as e:  # noqa: BLE001
        return None, code + f"\n# unusable: {e}"


def plan_action(predict, score, frame, avail, depth=4, beam=6):
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


def run_episode(game, predict, goal_fn, budget, depth, seed):
    arc = arc_agi.Arcade(); env = arc.make(game); obs = env.reset()
    avail = list(obs.available_actions); g = grid(obs); best = obs.levels_completed
    steps = terms = 0; rng = random.Random(seed); last = g
    while steps < budget:
        a = plan_action(predict, goal_fn, g, avail, depth) if goal_fn else rng.choice(avail)
        if a is None:
            a = rng.choice(avail)
        obs = env.step(ACTS[a - 1]); steps += 1
        if obs is None or getattr(obs, "frame", None) is None:
            terms += 1; obs = env.reset(); g = grid(obs); avail = list(obs.available_actions); continue
        g = grid(obs); last = g; best = max(best, obs.levels_completed)
        if str(obs.state) != "GameState.NOT_FINISHED":
            terms += 1; obs = env.reset(); g = grid(obs); avail = list(obs.available_actions)
    return {"levels": best, "terminations": terms, "steps": steps, "end_render": render(last)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="s5i5")
    ap.add_argument("--model", required=True)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--budget", type=int, default=400)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--graph", action="store_true", help="give Claude the object-graph view for relational goals")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    predict, dyn, fid = load_predict(args.model)
    arc = arc_agi.Arcade(); env = arc.make(args.game); o = env.reset()
    avail = list(o.available_actions); win = o.win_levels; fa = grid(o)
    fb = fa
    for _ in range(8):
        o = env.step(ACTS[random.Random(1).choice(avail) - 1])
        if o is not None and getattr(o, "frame", None) is not None:
            fb = grid(o)

    history = []; best_levels = 0; solved = False
    for r in range(args.rounds):
        goal_fn, code = claude_goal(dyn, fa, fb, len(avail), history, graph=args.graph)
        out = run_episode(args.game, predict, goal_fn, args.budget, args.depth, args.seed)
        best_levels = max(best_levels, out["levels"])
        print(f"[e89b/{args.game}] round {r}: levels={out['levels']} game-overs={out['terminations']} "
              f"(best {best_levels}/{win})", flush=True)
        history.append((code, out))
        if out["levels"] >= win:
            solved = True
            break

    res = {"game": args.game, "model_fidelity": fid, "win_levels": win, "rounds": len(history),
           "best_levels": best_levels, "solved": solved,
           "attempts": [{"levels": o["levels"], "terminations": o["terminations"]} for _, o in history],
           "final_goal_code": history[-1][0] if history else None}
    out_path = Path(args.out) if args.out else HERE / "results" / f"e89b_arc3_{args.game}.json"
    out_path.write_text(json.dumps(res, indent=2))
    print(f"[e89b/{args.game}] FINAL best {best_levels}/{win} SOLVED={solved}", flush=True)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
