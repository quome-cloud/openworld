"""Memory experiment (condition B): consecutive play with a persistent
cross-episode ledger, 3 passes x 5 episodes on fresh seed blocks
(4000/5000/6000), ledger accumulated ONLY from these episodes' own logs.

Hypothesis under test (operator): die-and-learn should pay in NetHack —
death causes -> avoidance policy (M1), death depths -> descent pacing (M2).
Condition A (memoryless) runs first and is the leaderboard-comparable score.
"""

import json
import os
import sys
import time

import nh_runner
from nh_memory import NetHackMemory

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
OUT = os.path.join(RESULTS, "memory_experiment.json")
RUN_LOG = os.path.join(RESULTS, "RUN_LOG.txt")


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(RUN_LOG, "a") as f:
        f.write(line + "\n")


def main():
    passes = [(1, 4000), (2, 5000), (3, 6000)]
    doc = {"passes": []}
    if os.path.exists(OUT):
        doc = json.load(open(OUT))
    done = {(p["pass"], e["episode"]) for p in doc["passes"]
            for e in p["episodes"]}
    mem = NetHackMemory()
    for pnum, base in passes:
        prec = next((p for p in doc["passes"] if p["pass"] == pnum), None)
        if prec is None:
            prec = {"pass": pnum, "seed_base": base, "episodes": []}
            doc["passes"].append(prec)
        for ep in range(5):
            if (pnum, ep) in done:
                continue
            seed = base + ep
            log(f"=== memory pass {pnum} episode {ep} seed {seed} ===")
            res = nh_runner.run_episode(
                ep=ep, seed=seed, condition="B",
                label=f"memory_pass{pnum}", memory=mem, log=log)
            prec["episodes"].append(res)
            prog = [e["progression"] for e in prec["episodes"]]
            prec["score"] = 100.0 * sum(prog) / len(prog)
            with open(OUT, "w") as f:
                json.dump(doc, f, indent=1)
            log(f"  ep{ep}: prog {res['progression']:.4f} depth "
                f"{res['depth_max']} role {res['role']} "
                f"end {res['end_reason']} fired={len(res['mem_fired'])}")
        log(f"memory pass {pnum} score: {prec['score']:.2f}")
    log("memory experiment complete: " +
        ", ".join(f"pass{p['pass']}={p['score']:.2f}" for p in doc["passes"]))


if __name__ == "__main__":
    main()
