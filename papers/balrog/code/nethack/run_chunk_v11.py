"""Parallel worker for the v11block40 block: runs a chunk of seeds with the
frozen agent, writing to its own results file (merged later). Episode index
= seed - 2000 so transition/trajectory filenames never collide.

Usage: python3 run_chunk.py <outfile_suffix> <seed> [<seed> ...]
"""

import json
import os
import sys
import time

import nh_runner

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
RUN_LOG = os.path.join(RESULTS, "RUN_LOG.txt")


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(RUN_LOG, "a") as f:
        f.write(line + "\n")


def main():
    suffix = sys.argv[1]
    seeds = [int(s) for s in sys.argv[2:]]
    out = os.path.join(RESULTS, f"nethack_results_v11block40_{suffix}.json")
    doc = {"label": f"v11block40_{suffix}", "episodes": []}
    if os.path.exists(out):
        doc = json.load(open(out))
    done = {e["seed"] for e in doc["episodes"]}
    for seed in seeds:
        if seed in done:
            continue
        ep = seed - 3000
        log(f"=== v11block40/{suffix} episode {ep} seed {seed} ===")
        res = nh_runner.run_episode(ep=ep, seed=seed, condition="A",
                                    label="v11block40", memory=None, log=log)
        doc["episodes"].append(res)
        with open(out, "w") as f:
            json.dump(doc, f, indent=1)
        log(f"  [{suffix}] ep{ep}: prog {res['progression']:.4f} "
            f"depth {res['depth_max']} role {res['role']} "
            f"end {res['end_reason']}")
    log(f"v11block40 chunk {suffix} complete ({len(doc['episodes'])} eps)")


if __name__ == "__main__":
    main()
