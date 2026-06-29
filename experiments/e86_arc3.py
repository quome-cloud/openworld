"""E86 -- OpenWorld on ARC-AGI-3: synthesize a VERIFIED code world model of a game's dynamics.

Pipeline (the recipe; see papers/arc-3/RECIPE.md):
  explore -> collect verified (frame, action, next_frame) transitions
          -> synthesize Python predict(frame, action) via an LLM
          -> VERIFY by exact-match on held-out transitions (accept only if it passes)
          -> [E87] plan through the verified model.

This file is the foundation experiment (E86): verified-code fidelity vs. learned baselines.
Runnable: env interaction + transition logging + verification + a copy-frame baseline work with
no LLM; pass --ollama <model> to drive the synthesis loop with a local code model.

  python3 e86_arc3.py --game ls20 --steps 300 --ollama qwen2.5-coder:7b
"""
import argparse
import json
import re
import urllib.request
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


def collect(game_id, steps, seed):
    import random
    arc = arc_agi.Arcade()
    env = arc.make(game_id)
    obs = env.reset()
    rng = random.Random(seed)
    avail = [a - 1 for a in obs.available_actions]
    g = grid(obs)
    trans, best = [], obs.levels_completed
    for _ in range(steps):
        ai = rng.choice(avail)
        try:
            nobs = env.step(ACTS[ai])
        except Exception:  # noqa: BLE001 -- some games' step can throw internally
            nobs = None
        if nobs is None or getattr(nobs, "frame", None) is None:  # game returned a bad frame -> reset
            obs = env.reset(); g = grid(obs); avail = [a - 1 for a in obs.available_actions]
            continue
        ng = grid(nobs)
        trans.append({"frame": g.tolist(), "action": ai + 1, "next": ng.tolist()})
        best = max(best, nobs.levels_completed)
        if str(nobs.state) != "GameState.NOT_FINISHED":
            obs = env.reset(); ng = grid(obs); avail = [a - 1 for a in obs.available_actions]
        g = ng
    return trans, best, obs.win_levels


def replay_determinism(game_id, steps, seed):
    """Run the SAME action sequence twice from reset; fraction of steps with identical frames.
    A clean determinism test (the code-world-model precondition) without needing state-setting."""
    import random
    arc = arc_agi.Arcade()
    def episode():
        env = arc.make(game_id); obs = env.reset(); rng = random.Random(seed)
        avail = [a - 1 for a in obs.available_actions]; frames = [grid(obs).copy()]
        for _ in range(steps):
            try:
                obs = env.step(ACTS[rng.choice(avail)])
            except Exception:  # noqa: BLE001
                break
            if obs is None or getattr(obs, "frame", None) is None:
                break
            frames.append(grid(obs).copy())
            if str(obs.state) != "GameState.NOT_FINISHED":
                break
        return frames
    f1, f2 = episode(), episode()
    n = min(len(f1), len(f2))
    return (sum(np.array_equal(f1[i], f2[i]) for i in range(n)) / n, n) if n else (None, 0)


def bg_of(g):
    v, c = np.unique(g, return_counts=True)
    return int(v[np.argmax(c)])


def deltas(frame, nxt):
    f, n = np.asarray(frame), np.asarray(nxt)
    rs, cs = np.where(f != n)
    return [[int(r), int(c), int(n[r, c])] for r, c in zip(rs, cs)]


# ---------- verification ----------
def verify_code(code, trans):
    """Run a candidate predict(frame, action) over transitions; return (exact_frac, errors)."""
    ns = {"np": np, "numpy": np}
    try:
        exec(compile(code, "<synth>", "exec"), ns)  # noqa: S102 -- sandboxed-intent; trusted-local
        predict = ns["predict"]
    except Exception as e:  # noqa: BLE001
        return 0.0, [f"compile/exec error: {e}"]
    ok = 0
    errs = []
    for t in trans:
        try:
            out = np.asarray(predict(np.asarray(t["frame"]), t["action"]))
            if out.shape == (64, 64) and np.array_equal(out, np.asarray(t["next"])):
                ok += 1
            elif len(errs) < 3:
                errs.append(t)
        except Exception as e:  # noqa: BLE001
            if len(errs) < 3:
                errs.append({"exc": str(e), **t})
    return ok / len(trans) if trans else 0.0, errs


def determinism(trans):
    """Fraction of repeated (state,action) pairs that reproduce the same next frame."""
    import hashlib
    seen={}; ok=tot=0
    for t in trans:
        k=(hashlib.md5(np.asarray(t["frame"]).astype(np.int16).tobytes()).hexdigest(), t["action"])
        nh=hashlib.md5(np.asarray(t["next"]).astype(np.int16).tobytes()).hexdigest()
        if k in seen:
            tot+=1; ok+=int(seen[k]==nh)
        else:
            seen[k]=nh
    return (ok/tot, tot) if tot else (None, 0)


def copy_baseline(trans):
    """'next == frame' baseline (a learned model's easiest guess)."""
    return sum(np.array_equal(np.asarray(t["frame"]), np.asarray(t["next"])) for t in trans) / len(trans)


# ---------- synthesis ----------
PROMPT = """You are given transitions from a deterministic 64x64 grid game (colors 0-15).
Each transition: an action (int) maps the current grid to the next grid. Background color = {bg}.
Most cells are unchanged each step; only a few change. Infer the rule and write a Python function

    def predict(frame, action):  # frame: np.ndarray (64,64) ints; returns np.ndarray (64,64)

that reproduces the next grid EXACTLY for any transition. Use numpy (np). Return ONLY a ```python
code block.

Transitions (action -> list of changed cells [row,col,new_color]):
{examples}
"""


def ollama(model, prompt, host="http://localhost:11434", num_ctx=8192, timeout=600):
    body = json.dumps({"model": model, "prompt": prompt, "stream": False,
                       "options": {"num_ctx": num_ctx, "temperature": 0.2}}).encode()
    req = urllib.request.Request(f"{host}/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["response"]


def claude_cli(prompt, timeout=600):
    """Synthesize via Claude (headless `claude -p`) -- the frontier synthesizer OpenWorld actually
    uses (openworld build/optimize). No GPU needed."""
    import subprocess
    r = subprocess.run(["claude", "-p", prompt], capture_output=True, text=True, timeout=timeout)
    return r.stdout


def extract_code(text):
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.S)
    return m.group(1).strip() if m else text.strip()


def _demo_str(t, cap=80):
    """Compact demo: cap the changed-cell list so dense games don't blow up the prompt/timeout.
    (Verification still uses FULL held-out frames -- this only bounds what the synthesizer sees.)"""
    d = deltas(t["frame"], t["next"])
    body = str(d[:cap]) + (f" ...(+{len(d) - cap} more cells)" if len(d) > cap else "")
    return f"action {t['action']} -> {body}"


def synthesize(trans, llm_fn, rounds=4, n_demo=12):
    train, held = trans[: len(trans) * 3 // 4], trans[len(trans) * 3 // 4:]
    bg = bg_of(np.asarray(train[0]["frame"]))
    best = (0.0, None)
    feedback = ""
    for _ in range(rounds):
        ex = "\n".join(_demo_str(t) for t in train[:n_demo])
        prompt = PROMPT.format(bg=bg, examples=ex) + feedback
        try:
            code = extract_code(llm_fn(prompt))
        except Exception as e:  # noqa: BLE001 -- timeout/error -> miss this round, keep best so far
            print(f"[synth] llm call failed ({type(e).__name__}); keeping best", flush=True)
            continue
        acc, errs = verify_code(code, held)
        if acc > best[0]:
            best = (acc, code)
        if acc >= 0.99:
            break
        feedback = ("\n\nYour function failed these (show full intent): "
                    + json.dumps([{"action": e.get("action"), "changed": deltas(e["frame"], e["next"])}
                                  for e in errs if "frame" in e][:2]))
    return best  # (held-out exact-match rate, code)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="ls20")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ollama", default="", help="ollama model id for synthesis (else baselines only)")
    ap.add_argument("--claude", action="store_true", help="synthesize with Claude (claude -p) instead of ollama")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    trans, levels, win = collect(args.game, args.steps, args.seed)
    if not trans:  # random play never produced a valid step -- record + exclude, don't crash
        res = {"game": args.game, "steps": args.steps, "transitions": 0, "verified_exact": None,
               "note": "no valid transitions (random actions never produced a good step); excluded",
               "baseline_levels": levels, "win_levels": win}
        out = Path(args.out) if args.out else HERE / "results" / f"e86_arc3_{args.game}.json"
        out.write_text(json.dumps(res, indent=2))
        print(f"[e86/{args.game}] 0 transitions -> excluded", flush=True)
        return
    det, ndet = determinism(trans)
    rdet, rn = replay_determinism(args.game, min(args.steps, 80), args.seed)
    chg = [len(deltas(t["frame"], t["next"])) for t in trans]
    res = {"game": args.game, "steps": args.steps, "transitions": len(trans),
           "baseline_levels": levels, "win_levels": win,
           "deterministic_frac": (round(det, 4) if det is not None else None), "repeat_pairs": ndet,
           "replay_determinism": (round(rdet, 4) if rdet is not None else None), "replay_steps": rn,
           "mean_cells_changed": round(float(np.mean(chg)), 1),
           "copy_frame_exact": round(copy_baseline(trans), 4)}
    print(f"[e86/{args.game}] {len(trans)} transitions | baseline levels {levels}/{win} "
          f"| copy-frame exact {res['copy_frame_exact']}", flush=True)

    if args.claude or args.ollama:
        if args.claude:
            acc, code = synthesize(trans, claude_cli)
            res["synth_model"] = "claude-cli"
        else:
            acc, code = synthesize(trans, lambda p: ollama(args.ollama, p))
            res["synth_model"] = args.ollama
        res["verified_exact"] = round(acc, 4)
        res["code"] = code
        print(f"[e86/{args.game}] synthesized code verified-exact (held-out): {acc:.3f}", flush=True)

    out = Path(args.out) if args.out else HERE / "results" / f"e86_arc3_{args.game}.json"
    out.write_text(json.dumps(res, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
