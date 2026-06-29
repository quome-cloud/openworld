"""E96 -- COMPOSITE world model + perceptor: decompose the dynamics into verified sub-worlds.

E94's monolithic pixel predict() plateaued at partial fidelity (0.44 on sp80 level 2), which limits
planning. Per the composite-world thesis, we instead PERCEIVE the typed objects and synthesize the
dynamics as DECOMPOSED sub-functions (agent-movement / timer / interaction), composed in predict().
Each subsystem is simpler -> higher, verifiable fidelity. We compare composite vs monolithic
synthesis on held-out exact-match. A higher-fidelity composite model also feeds better level-2
planning.

  python3 e96_composite.py --game sp80 --level2 --prefix 5,2,...
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np

import e86_arc3 as E
import arc3_graph as GR

HERE = Path(__file__).resolve().parent

COMPOSITE = """Reverse-engineer a deterministic 64x64 grid game (colors 0-15) as a COMPOSITE world.
Background = {bg}. Perceived objects (color, size, position) in a sample state:
{objects}

Write `def predict(frame, action):` but DECOMPOSE it into named sub-functions, one per subsystem
(e.g. agent movement, a timer/counter bar, object interactions/collisions), then COMPOSE them:

    def predict(frame, action):
        g = frame.copy()
        g = _move_agent(g, action)
        g = _update_timer(g, action)
        g = _apply_interactions(g, action)
        return g

Each sub-function handles ONE subsystem (typically one color/object role). Infer which color is the
agent (moves under some actions), which is a timer (changes every step), which are
targets/interactables. Use numpy (np). Return ONLY a ```python code block reproducing the next grid
EXACTLY.

Transitions (action -> changed cells [row,col,new_color]):
{demos}
"""


def collect(game, steps, seed, prefix=None):
    import arc_agi
    from arcengine import GameAction
    acts = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4,
            GameAction.ACTION5, GameAction.ACTION6, GameAction.ACTION7]
    def grid(o):
        a = np.asarray(o.frame); return a[-1].reshape(64, 64) if a.ndim == 3 else a.reshape(64, 64)
    arc = arc_agi.Arcade(); env = arc.make(game); obs = env.reset()
    def replay():
        o = env.reset()
        for a in (prefix or []):
            o = env.step(acts[a - 1])
            if o is None or getattr(o, "frame", None) is None: return None
        return o
    obs = replay() if prefix else obs
    if obs is None: return []
    base = obs.levels_completed; g = grid(obs); avail = list(obs.available_actions); rng = random.Random(seed); tr = []
    for _ in range(steps):
        a = rng.choice(avail); obs = env.step(acts[a - 1])
        if obs is None or getattr(obs, "frame", None) is None:
            obs = replay() if prefix else env.reset(); g = grid(obs) if obs else g; continue
        ng = grid(obs); tr.append({"frame": g.tolist(), "action": a, "next": ng.tolist()}); g = ng
        if prefix and (obs.levels_completed != base or str(obs.state) != "GameState.NOT_FINISHED"):
            obs = replay(); g = grid(obs) if obs else g
        elif not prefix and str(obs.state) != "GameState.NOT_FINISHED":
            obs = env.reset(); g = grid(obs); avail = list(obs.available_actions)
    return tr


def synth(prompt_template, trans, rounds=4, composite=False):
    cut = len(trans) * 3 // 4
    train, held = trans[:cut], trans[cut:]
    bg = E.bg_of(np.asarray(train[0]["frame"]))
    demos = "\n".join(E._demo_str(t) for t in train[:12])
    objs = GR.graph_repr(np.asarray(train[0]["frame"]))
    best = (0.0, None); fb = ""
    for _ in range(rounds):
        if composite:
            prompt = prompt_template.format(bg=bg, objects=objs, demos=demos) + fb
        else:
            prompt = prompt_template.format(bg=bg, examples=demos) + fb
        try:
            code = E.extract_code(E.claude_cli(prompt, timeout=600))
        except Exception:  # noqa: BLE001
            continue
        acc, errs = E.verify_code(code, held)
        if acc > best[0]: best = (acc, code)
        if acc >= 0.99: break
        fb = "\n\nNot exact on held-out cases; fix the sub-functions."
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="sp80")
    ap.add_argument("--level2", action="store_true")
    ap.add_argument("--prefix", default="")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    prefix = [int(x) for x in args.prefix.split(",") if x.strip()] if args.level2 else None
    trans = collect(args.game, args.steps, args.seed, prefix)
    if len(trans) < 16:
        Path(args.out or HERE / "results" / f"e96_composite_{args.game}.json").write_text(json.dumps({"game": args.game, "error": "few transitions"}))
        return
    tag = "level2" if args.level2 else "level1"
    print(f"[e96/{args.game}/{tag}] {len(trans)} transitions; synthesizing monolith vs composite...", flush=True)
    mono_acc, _ = synth(E.PROMPT, trans, composite=False)
    comp_acc, comp_code = synth(COMPOSITE, trans, composite=True)
    res = {"game": args.game, "level": tag, "monolith_fidelity": round(mono_acc, 4),
           "composite_fidelity": round(comp_acc, 4), "composite_gain": round(comp_acc - mono_acc, 4),
           "composite_code": comp_code}
    print(f"[e96/{args.game}/{tag}] monolith {mono_acc:.3f} | composite {comp_acc:.3f} | gain {comp_acc - mono_acc:+.3f}", flush=True)
    out = Path(args.out) if args.out else HERE / "results" / f"e96_composite_{args.game}_{tag}.json"
    out.write_text(json.dumps(res, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
