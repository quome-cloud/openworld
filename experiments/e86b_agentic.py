"""E86b -- AGENTIC (verifier-in-the-loop) code-world-model synthesis, a fair 2x2 vs E86 one-shot.

Same loop for every synthesizer: generate solution -> RUN it on a validation split -> feed back the
exact cells it got wrong (from,expected,you_gave) -> revise. Repeat K rounds, keep the best. Then
score the best code on a held-out TEST split the synthesizer never optimized against.

Backends (so qwen-30B and Claude race under identical conditions):
  --backend claude            (local; uses `claude -p`)
  --backend ollama:qwen3-coder:30b   (GPU box with Ollama; or qwen2.5-coder:32b)

  python3 e86b_agentic.py --game ka59 --backend claude
"""
import argparse
import json
from pathlib import Path

import numpy as np

import e86_arc3 as E  # collect(), verify_code(), ollama(), claude_cli(), extract_code(), deltas()

HERE = Path(__file__).resolve().parent

TASK = """You are reverse-engineering a deterministic 64x64 grid game (integer colors 0-15) as code.
Write a Python function:

    def predict(frame, action):  # frame: numpy (64,64) int; action: int (1-indexed); -> (64,64)

that reproduces the NEXT frame EXACTLY. Background color = {bg}. Most cells are unchanged each step;
only a few change. Infer the movement/update rules. Use numpy (np). Return ONLY a ```python block.

Example transitions (action -> changed cells [row,col,new_color]):
{demos}
"""


def feedback(code, fails):
    """Run the candidate on failing transitions and report the precise errors (the agentic signal)."""
    ns = {"np": np, "numpy": np}
    try:
        exec(compile(code, "<c>", "exec"), ns)  # noqa: S102
        pred = ns["predict"]
    except Exception as e:  # noqa: BLE001
        return f"\n\nYour code failed to run: {e}. Fix it."
    lines = []
    for t in fails[:2]:
        f, want = np.asarray(t["frame"]), np.asarray(t["next"])
        try:
            got = np.asarray(pred(f, t["action"]))
        except Exception as e:  # noqa: BLE001
            lines.append(f"action {t['action']}: your code raised {e}")
            continue
        if got.shape != (64, 64):
            lines.append(f"action {t['action']}: wrong shape {got.shape}")
            continue
        rs, cs = np.where(want != got)
        errs = [[int(r), int(c), int(f[r, c]), int(want[r, c]), int(got[r, c])]
                for r, c in list(zip(rs, cs))[:20]]
        lines.append(f"action {t['action']} wrong cells [r,c,from,EXPECTED,you_gave]: {errs}")
    return "\n\nYour predict() was WRONG on these held-out cases. Fix the rule:\n" + "\n".join(lines)


def refine(train, val, gen_fn, rounds=8, n_demo=14):
    bg = E.bg_of(np.asarray(train[0]["frame"]))
    demos = "\n".join(E._demo_str(t) for t in train[:n_demo])
    best, fb = (0.0, None), ""
    for r in range(rounds):
        prompt = TASK.format(bg=bg, demos=demos) + fb
        try:
            code = E.extract_code(gen_fn(prompt))
        except Exception as e:  # noqa: BLE001
            print(f"  round {r}: gen failed ({type(e).__name__})", flush=True)
            continue
        acc, fails = E.verify_code(code, val)
        print(f"  round {r}: val fidelity {acc:.3f}", flush=True)
        if acc > best[0]:
            best = (acc, code)
        if acc >= 0.999:
            break
        fb = feedback(code, fails)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="ka59")
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--backend", default="claude", help="claude | ollama:<model>")
    ap.add_argument("--piecewise", action="store_true", help="emphasize explicit if/elif exception handling")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.backend == "claude":
        gen_fn, tag = E.claude_cli, "agentic-claude"
    elif args.backend.startswith("ollama:"):
        model = args.backend.split(":", 1)[1]
        gen_fn, tag = (lambda p: E.ollama(model, p)), f"agentic-{model}"
    else:
        raise SystemExit("backend must be 'claude' or 'ollama:<model>'")

    trans, levels, win = E.collect(args.game, args.steps, args.seed)
    if not trans:
        Path(args.out or HERE / "results" / f"e86b_{args.backend.replace(':','_')}_{args.game}.json").write_text(
            json.dumps({"game": args.game, "error": "no transitions"}, indent=2))
        return
    # train (synthesizer sees) / val (iterates against) / test (held out, never seen)
    n = len(trans); a, b = n * 5 // 10, n * 7 // 10
    train, val, test = trans[:a], trans[a:b], trans[b:]
    print(f"[e86b/{tag}/{args.game}] train {len(train)} val {len(val)} test {len(test)}", flush=True)

    global TASK
    if args.piecewise:
        TASK += ("\n\nThe rule is almost certainly PIECEWISE. Do NOT force one uniform "
                 "transformation -- write explicit if/elif branches for special cases "
                 "(collisions, walls, board edges, special tiles, blocked moves). For every "
                 "failing case you see, add a dedicated conditional branch that handles it.")
    _, code = refine(train, val, gen_fn, rounds=args.rounds)
    test_fid, _ = E.verify_code(code, test) if code else (0.0, None)
    res = {"game": args.game, "method": tag, "backend": args.backend,
           "verified_exact": round(test_fid, 4), "copy_frame_exact": round(E.copy_baseline(test), 4),
           "n_test": len(test), "code": code}
    print(f"[e86b/{tag}/{args.game}] TEST fidelity {test_fid:.3f}", flush=True)
    out = Path(args.out) if args.out else HERE / "results" / f"e86b_{args.backend.replace(':','_')}_{args.game}.json"
    out.write_text(json.dumps(res, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
