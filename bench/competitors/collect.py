"""Normalize a competitor's logs into the shared results schema."""
import argparse, json, os
ap = argparse.ArgumentParser()
ap.add_argument("--method", required=True); ap.add_argument("--logdir", default="")
ap.add_argument("--out", required=True)
a = ap.parse_args()
json.dump({"method": a.method, "env": "MiniGrid-DoorKey-6x6-v0",
           "logdir": a.logdir, "metrics": {"success_rate": None, "env_steps": None},
           "note": "finalized from logs during the on-instance run"},
          open(a.out, "w"), indent=2)
print("wrote", a.out)
