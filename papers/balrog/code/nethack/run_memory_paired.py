"""Paired memory experiment (condition B2): same seeds as the memoryless
baseline25 block (2005-2019, 15 episodes, sequential), fresh ledger built
only from these episodes' own logs. The per-seed delta vs the baseline
episode gives the memory effect with pairing power; episode order is the
die-and-learn axis (later episodes have more ledger).

Frozen agent code (RUN_LOG 20:58:32 md5s); only the ledger side differs
from condition A.
"""

import json
import os
import time

import nh_runner
from nh_memory import NetHackMemory

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
OUT = os.path.join(RESULTS, "memory_paired.json")
RUN_LOG = os.path.join(RESULTS, "RUN_LOG.txt")

SEEDS = list(range(2005, 2020))


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(RUN_LOG, "a") as f:
        f.write(line + "\n")


def main():
    doc = {"label": "memory_paired", "seeds": SEEDS, "episodes": []}
    if os.path.exists(OUT):
        doc = json.load(open(OUT))
    done = {e["seed"] for e in doc["episodes"]}
    mem = NetHackMemory(name="nethack_ledger_paired")
    for i, seed in enumerate(SEEDS):
        if seed in done:
            continue
        log(f"=== memory_paired episode {i} seed {seed} ===")
        res = nh_runner.run_episode(ep=i, seed=seed, condition="B",
                                    label="memory_paired", memory=mem,
                                    log=log)
        doc["episodes"].append(res)
        prog = [e["progression"] for e in doc["episodes"]]
        doc["score"] = 100.0 * sum(prog) / len(prog)
        with open(OUT, "w") as f:
            json.dump(doc, f, indent=1)
        log(f"  ep{i}: prog {res['progression']:.4f} depth {res['depth_max']}"
            f" role {res['role']} end {res['end_reason']} "
            f"fired={len(res['mem_fired'])}")
    log(f"memory_paired FINAL over {len(doc['episodes'])}: {doc['score']:.2f}")


if __name__ == "__main__":
    main()
