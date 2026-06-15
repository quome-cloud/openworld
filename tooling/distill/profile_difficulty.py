#!/usr/bin/env python3
"""Difficulty profile: classify every instance into the learnable band.

For each of the 35 instances:
  - base_solves   : does base 1.5b solve it single-shot? (MLX, greedy — same as eval)
  - teacher_solves: did the 14b solve it in the harvest? (any passed trace — free, no rerun)

Buckets:
  base_solves                      -> no headroom (base already gets it)
  learnable_band (base fail, 14b ok) -> THE target: headroom + a verified trace exists
  both_fail (base fail, 14b fail)  -> too hard even for the teacher; unusable

Run: PYTHONPATH=<repo> .venv-distill/bin/python tooling/distill/profile_difficulty.py
ponytail: reuses eval_heldout helpers; teacher set comes from existing traces, not a rerun.
"""
import json
from pathlib import Path

import eval_heldout as E  # same dir; reuses _instances / _eval_variant

TRACES = "traces/harvest-v2/qwen2.5-14b.traces.jsonl"
OUT = "tooling/distill/eval/difficulty_profile_v2.json"


def teacher_solved(path):
    solved = set()
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            if d.get("passed"):
                solved.add(d["instance_id"])
    return solved


def main():
    by_id = E._instances()
    all_ids = sorted(by_id)
    print(f"[profile] {len(all_ids)} instances total")

    teacher = teacher_solved(TRACES)
    print(f"[profile] 14b solved {len(teacher)}/{len(all_ids)} in harvest (free, from traces)")

    print("[profile] running base 1.5b single-shot over ALL instances ...")
    base = E._eval_variant("base", None, all_ids, by_id, 1024)

    buckets = {"base_solves": [], "learnable_band": [], "both_fail": []}
    for iid in all_ids:
        b, t = base[iid], iid in teacher
        if b:
            buckets["base_solves"].append(iid)
        elif t:
            buckets["learnable_band"].append(iid)
        else:
            buckets["both_fail"].append(iid)

    result = {
        "n_total": len(all_ids),
        "base_solved_n": sum(base.values()),
        "teacher_solved_n": len(teacher),
        "counts": {k: len(v) for k, v in buckets.items()},
        "buckets": buckets,
    }
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT).write_text(json.dumps(result, indent=2))

    print("\n=== DIFFICULTY PROFILE ===")
    print(f"base solves        : {result['counts']['base_solves']}  (no headroom)")
    print(f"learnable band     : {result['counts']['learnable_band']}  (base fail + 14b ok -> TARGET)")
    print(f"both fail          : {result['counts']['both_fail']}  (too hard even for 14b)")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
