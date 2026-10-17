"""Build violation-rate + learning curves from all logged episodes, in run order."""
import glob
import gzip
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
TRANS = os.path.join(HERE, "results", "transitions")

ORDER = (["probe_P0", "probe_P1", "probe_P2"]
         + [f"explore_E{i}" for i in range(1, 20)]
         + [f"p1v5_E{i}" for i in range(20, 26)]
         + [f"b1_E{i}" for i in range(30, 40)]
         + [f"b2v7partial_E{i}" for i in range(40, 43)]
         + [f"b2_E{i}" for i in range(40, 50)]
         + [f"b3_E{i}" for i in range(50, 60)]
         + [f"b4_E{i}" for i in range(60, 70)]
         + [f"frozen_F{i}" for i in range(25)])


def episode_summary(path):
    n_pred = n_viol = steps = 0
    stats = None
    try:
        with gzip.open(path, "rt") as f:
            for line in f:
                r = json.loads(line)
                if r.get("kind") == "end":
                    n_pred = r.get("n_pred", 0)
                    n_viol = r.get("n_viol", 0)
                    steps = r.get("t", 0)
                    stats = r.get("stats", {})
    except Exception as e:
        return None
    return {"n_pred": n_pred, "n_viol": n_viol, "steps": steps,
            "prog": (stats or {}).get("progression"),
            "end": (stats or {}).get("end_reason")}


if __name__ == "__main__":
    out = []
    for name in ORDER:
        p = os.path.join(TRANS, name + ".jsonl.gz")
        if not os.path.exists(p):
            continue
        s = episode_summary(p)
        if s is None:
            continue
        s["episode"] = name
        s["viol_rate"] = (s["n_viol"] / s["n_pred"]) if s["n_pred"] else None
        out.append(s)
    with open(os.path.join(HERE, "results", "violation_curve.json"), "w") as f:
        json.dump(out, f, indent=1)
    cum_p = cum_v = 0
    for s in out:
        cum_p += s["n_pred"]; cum_v += s["n_viol"]
        vr = f"{s['viol_rate']:.5f}" if s["viol_rate"] is not None else "  -  "
        print(f"{s['episode']:16s} steps={s['steps']:5d} viol={s['n_viol']:3d}/{s['n_pred']:6d} "
              f"vr={vr} cum={cum_v}/{cum_p} prog={s['prog']}")
