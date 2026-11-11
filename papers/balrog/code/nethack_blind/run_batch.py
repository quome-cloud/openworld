"""Batch driver: python3 run_batch.py <tag> <max_steps> <ep0> <seed0> <seed1> ...
Appends one JSON line per episode to results/batch_<tag>.jsonl (checkpointing).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from runner import run_episode  # noqa: E402
from world_model import WorldModel  # noqa: E402
from policy_explore import ExplorePolicy  # noqa: E402
from policy_validate import ValidatePolicy  # noqa: E402

tag = sys.argv[1]
max_steps = int(sys.argv[2])
ep0 = int(sys.argv[3])
seeds = [int(s) for s in sys.argv[4:]]

out = os.path.join(HERE, "results", f"batch_{tag}.jsonl")
model = WorldModel()
for i, seed in enumerate(seeds):
    ep = f"E{ep0 + i}" if not tag.startswith("frozen") else f"F{ep0 + i}"
    cls = ValidatePolicy if "val" in tag else ExplorePolicy
    pol = cls(ep_id=ep)
    r = run_episode(ep, pol, seed=seed, max_steps=max_steps,
                    log_name=f"{tag}_{ep}", model=model)
    pol.finish()
    model.save()
    s = r["stats"]
    line = {"ep": ep, "seed": seed, "steps": r["steps"],
            "depth": s["depth"], "prog": s["progression"],
            "xp": s["experience_level"], "time": s["time"],
            "end": s["end_reason"], "viol_rate": r["viol_rate"],
            "n_pred": r["n_pred"], "n_viol": r["n_viol"],
            "anomalies": r["anomalies"][:20], "wallclock": r["wallclock"]}
    with open(out, "a") as f:
        f.write(json.dumps(line) + "\n")
    print(ep, seed, "prog", round(s["progression"], 4), "end", s["end_reason"], flush=True)
print("BATCH DONE", tag, flush=True)
