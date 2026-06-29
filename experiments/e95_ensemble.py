"""E95 -- ENSEMBLE of code world models (committee of worlds), the thesis frame.

Multiple INDEPENDENT synthesized world models can model the same dynamics differently; combining them
(per-cell majority vote, optionally fidelity-weighted) corrects individual errors and gains accuracy
-- "average or choose from multiple worlds." We synthesize K predict() codes per game and compare:
best-single, mean-single, ensemble-majority, ensemble-weighted (by each model's held-out fidelity).

  python3 e95_ensemble.py --game ka59 --k 5
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np

import e86_arc3 as E

HERE = Path(__file__).resolve().parent
VARY = ["", " Consider rows/columns and translations.", " Consider per-object moves and collisions.",
        " Consider wrap-around and boundaries.", " Consider color-conditioned rules.",
        " Consider the agent vs static objects separately."]


def load_predict(code):
    ns = {"np": np, "numpy": np}
    try:
        exec(compile(code, "<m>", "exec"), ns)  # noqa: S102
        return ns.get("predict")
    except Exception:  # noqa: BLE001
        return None


def fidelity(predict, trans):
    if predict is None:
        return 0.0
    ok = 0
    for t in trans:
        try:
            o = np.asarray(predict(np.asarray(t["frame"]), t["action"]))
            ok += int(o.shape == (64, 64) and np.array_equal(o, np.asarray(t["next"])))
        except Exception:  # noqa: BLE001
            pass
    return ok / len(trans) if trans else 0.0


def ensemble_frame(predicts, frame, action, weights=None):
    """Per-cell (optionally weighted) majority vote over the models' predicted next frames."""
    preds, ws = [], []
    for i, p in enumerate(predicts):
        try:
            o = np.asarray(p(np.asarray(frame), action))
            if o.shape == (64, 64):
                preds.append(o.astype(int) % 16)
                ws.append(1.0 if weights is None else weights[i])
        except Exception:  # noqa: BLE001
            pass
    if not preds:
        return None
    counts = np.zeros((16, 64, 64))
    for o, w in zip(preds, ws):
        for c in range(16):
            counts[c] += w * (o == c)
    return counts.argmax(0)


def ensemble_fidelity(predicts, trans, weights=None):
    ok = 0
    for t in trans:
        out = ensemble_frame(predicts, t["frame"], t["action"], weights)
        ok += int(out is not None and np.array_equal(out, np.asarray(t["next"])))
    return ok / len(trans) if trans else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="ka59")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    trans, _, _ = E.collect(args.game, args.steps, args.seed)
    if not trans:
        Path(args.out or HERE / "results" / f"e95_ensemble_{args.game}.json").write_text(
            json.dumps({"game": args.game, "error": "no transitions"}))
        return
    cut = len(trans) * 3 // 4
    train, held = trans[:cut], trans[cut:]
    bg = E.bg_of(np.asarray(train[0]["frame"]))
    demos = "\n".join(E._demo_str(t) for t in train[:12])

    predicts, single_fids = [], []
    for k in range(args.k):
        prompt = E.PROMPT.format(bg=bg, examples=demos) + VARY[k % len(VARY)]
        code = E.extract_code(E.claude_cli(prompt, timeout=600))
        p = load_predict(code)
        f = fidelity(p, held)
        single_fids.append(f)
        if p is not None:
            predicts.append(p)
        print(f"[e95/{args.game}] model {k}: held-out fidelity {f:.3f}", flush=True)

    kept_fids = [fidelity(p, held) for p in predicts]  # weights aligned to kept models
    maj = ensemble_fidelity(predicts, held)
    wgt = ensemble_fidelity(predicts, held, weights=kept_fids)
    res = {"game": args.game, "k": len(predicts),
           "best_single": round(max(single_fids), 4) if single_fids else 0,
           "mean_single": round(float(np.mean(single_fids)), 4) if single_fids else 0,
           "ensemble_majority": round(maj, 4), "ensemble_weighted": round(wgt, 4),
           "single_fids": [round(f, 4) for f in single_fids]}
    res["ensemble_gain_over_best"] = round(max(maj, wgt) - res["best_single"], 4)
    print(f"[e95/{args.game}] best-single {res['best_single']} | majority {maj:.3f} | weighted {wgt:.3f} "
          f"| gain {res['ensemble_gain_over_best']:+.3f}", flush=True)
    out = Path(args.out) if args.out else HERE / "results" / f"e95_ensemble_{args.game}.json"
    out.write_text(json.dumps(res, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
