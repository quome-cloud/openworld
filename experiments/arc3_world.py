"""Wrap an ARC-AGI-3 game as a first-class OpenWorld code world model (World + CodeTransition).

The repo's CodeTransition runs in a pure-Python sandbox (no numpy/imports). So we synthesize the
game's dynamics as a numpy-FREE `transition(state, action)` over the frame as a list-of-lists -- the
same shape as minigrid_world.py -- verify it reproduces held-out transitions inside the sandbox, and
return an openworld.World. This makes the synthesized model a genuine OpenWorld code world model
(serveable, spec-able, plannable), used by the solver (E93) for lookahead via w.transition.step.

  python3 arc3_world.py --game s5i5            # synthesize + verify a repo-native world model
"""
import argparse
import json
import re
from pathlib import Path

import openworld as O
from openworld.sandbox import run_transition_code  # the repo sandbox (pure-python)

import e86_arc3 as E  # collect(), claude_cli(), deltas(), bg_of()

HERE = Path(__file__).resolve().parent

SYNTH = """Reverse-engineer a deterministic 64x64 grid game (colors 0-15) as PURE PYTHON.

Write EXACTLY this function -- NO imports, NO numpy, only built-in lists/loops/ints:

    def transition(state, action):
        # state["frame"] is a 64x64 list of lists of ints (the grid).
        # action["name"] is a string holding the action integer, e.g. "3".
        # Return a NEW state dict (copy) with state["frame"] updated to the next grid.

Background color = {bg}. Most cells are unchanged each step; only a few change. Infer the
movement/update rules. Return ONLY a ```python code block defining transition(state, action).

Transitions (action -> changed cells [row,col,new_color]):
{demos}
"""


def extract_code(text):
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.S)
    return m.group(1).strip() if m else text.strip()


def verify_sandbox(code, trans):
    """Run the pure-python transition IN THE REPO SANDBOX over held-out transitions; exact-match."""
    ok = 0
    for t in trans:
        try:
            nxt = run_transition_code(code, {"frame": t["frame"]}, {"name": str(t["action"])})
            ok += int(nxt.get("frame") == t["next"])
        except Exception:  # noqa: BLE001
            pass
    return ok / len(trans) if trans else 0.0


def synthesize_world(game, steps=240, rounds=5, seed=0):
    trans, _, _ = E.collect(game, steps, seed)
    if not trans:
        return None, 0.0, None, trans
    cut = len(trans) * 3 // 4
    train, held = trans[:cut], trans[cut:]
    bg = E.bg_of(train[0]["frame"])
    demos = "\n".join(f"action {t['action']} -> {E.deltas(t['frame'], t['next'])[:80]}" for t in train[:12])
    best = (0.0, None)
    fb = ""
    for r in range(rounds):
        code = extract_code(E.claude_cli(SYNTH.format(bg=bg, demos=demos) + fb, timeout=600))
        acc = verify_sandbox(code, held)
        print(f"  round {r}: sandbox-verified fidelity {acc:.3f}", flush=True)
        if acc > best[0]:
            best = (acc, code)
        if acc >= 0.99:
            break
        fb = "\n\nYour transition() was not exact on held-out cases. It must be PURE PYTHON (no imports)."
    acc, code = best
    if code is None:
        return None, 0.0, None, trans
    init = trans[0]["frame"]
    avail = sorted({t["action"] for t in trans})
    world = O.World(name=f"arc3-{game}",
                    description=f"ARC-AGI-3 game {game} as a verified pure-python code world model.",
                    initial_state={"frame": init},
                    actions=[str(a) for a in avail],
                    rules=[f"Background color {bg}.", "Deterministic grid dynamics, synthesized + verified."],
                    transition=O.CodeTransition(code, func_name="transition"))
    return world, acc, code, trans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="s5i5")
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    print(f"[arc3_world/{args.game}] synthesizing repo-native (pure-python) code world model...", flush=True)
    world, acc, code, _ = synthesize_world(args.game, args.steps, args.rounds)
    res = {"game": args.game, "sandbox_verified_fidelity": round(acc, 4),
           "world_name": world.name if world else None, "code": code}
    print(f"[arc3_world/{args.game}] sandbox-verified fidelity {acc:.3f}", flush=True)
    out = Path(args.out) if args.out else HERE / "results" / f"arc3_world_{args.game}.json"
    out.write_text(json.dumps(res, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
