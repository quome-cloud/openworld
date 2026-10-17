"""Aggregate batches into learning-curve + violation-curve JSON (and text table)."""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")

ORDER = ["p1v5", "b1", "b2", "b3", "b4", "b5", "b6", "frozen"]


def load(tag):
    p = os.path.join(RES, f"batch_{tag}.jsonl")
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return [json.loads(l) for l in f]


def summarize(tag, rows):
    if not rows:
        return None
    progs = [r["prog"] for r in rows]
    def dlvl_of(r):
        # max depth reached during ep is not logged directly; use prog rung + depth field
        return r.get("depth", 0)
    vr = [r["viol_rate"] for r in rows if r["viol_rate"] is not None]
    return {
        "tag": tag, "n": len(rows),
        "mean_prog": sum(progs) / len(progs),
        "best_prog": max(progs),
        "mean_viol_rate": (sum(vr) / len(vr)) if vr else None,
        "total_preds": sum(r.get("n_pred", 0) for r in rows),
        "total_viols": sum(r.get("n_viol", 0) for r in rows),
        "deaths": sum(1 for r in rows if r["end"] and "DEATH" in str(r["end"])),
        "ends": [str(r["end"])[:40] for r in rows],
    }


def bootstrap_ci(vals, n=10000, seed=7):
    import random
    rng = random.Random(seed)
    means = []
    for _ in range(n):
        s = [vals[rng.randrange(len(vals))] for _ in vals]
        means.append(sum(s) / len(s))
    means.sort()
    return means[int(0.025 * n)], means[int(0.975 * n)]


if __name__ == "__main__":
    tags = sys.argv[1:] or ORDER
    out = []
    for tag in tags:
        s = summarize(tag, load(tag))
        if s:
            out.append(s)
            print(f"{tag}: n={s['n']} mean_prog={s['mean_prog']*100:.2f}% "
                  f"best={s['best_prog']*100:.2f}% viol_rate={s['mean_viol_rate']} "
                  f"deaths={s['deaths']}")
    with open(os.path.join(RES, "curves.json"), "w") as f:
        json.dump(out, f, indent=1)
    frozen = load("frozen")
    if frozen:
        vals = [r["prog"] for r in frozen]
        lo, hi = bootstrap_ci(vals)
        m = sum(vals) / len(vals)
        print(f"FROZEN: n={len(vals)} mean={m*100:.2f}% CI95=[{lo*100:.2f},{hi*100:.2f}]")
        with open(os.path.join(RES, "frozen_result.json"), "w") as f:
            json.dump({"n": len(vals), "mean": m, "ci95": [lo, hi],
                       "values": vals}, f, indent=1)
