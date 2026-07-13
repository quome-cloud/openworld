"""Official-protocol NetHack run, condition A (memoryless).

Protocol (BALROG balrog/config/config.yaml):
  - task NetHackChallenge-v0, eval.num_episodes.nle = 5
  - nle_kwargs verbatim: char '@' (random role), max 100k steps,
    no_progress_timeout 150, skip_more on
  - progression metric: balrog Progress achievements curve (max over
    Dlvl:n / Xp:n milestones), read via env.get_stats() — the same call
    BALROG's evaluator makes. Suite score = mean over the 5 episodes.

Seeds: fixed block base 1000 (1000..1004), chosen in advance, disjoint from
dev seeds (101-140). Episodes run sequentially with per-episode checkpoint.
"""

import json
import os
import sys
import time

import nh_runner

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
OUT = os.path.join(RESULTS, "nethack_results.json")
RUN_LOG = os.path.join(RESULTS, "RUN_LOG.txt")

SOTA = 6.8  # BALROG leaderboard NetHack column, Gemini-3-Pro
            # (results/evidence/balrog_leaderboard_2026-07-06.html, raw-HTML parse)


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(RUN_LOG, "a") as f:
        f.write(line + "\n")


def main():
    base = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    label = sys.argv[2] if len(sys.argv) > 2 else "clean_A"
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    out = OUT if label == "clean_A" else os.path.join(
        RESULTS, f"nethack_results_{label}.json")

    doc = {"label": label, "seed_base": base, "episodes": [], "sota": SOTA}
    if os.path.exists(out):
        doc = json.load(open(out))
        log(f"resuming {label}: {len(doc['episodes'])} episodes present")

    done_eps = {e["episode"] for e in doc["episodes"]}
    for ep in range(n):
        if ep in done_eps:
            continue
        seed = base + ep
        log(f"=== {label} episode {ep} seed {seed} ===")
        res = nh_runner.run_episode(ep=ep, seed=seed, condition="A",
                                    label=label, memory=None, log=log)
        doc["episodes"].append(res)
        prog = [e["progression"] for e in doc["episodes"]]
        doc["score"] = 100.0 * sum(prog) / max(1, len(prog))
        with open(out, "w") as f:
            json.dump(doc, f, indent=1)
        log(f"  ep{ep}: prog {res['progression']:.4f} "
            f"({res['highest_achievement']}) depth {res['depth_max']} "
            f"xp {res['xplvl_max']} steps {res['steps']} "
            f"role {res['role']} end: {res['end_reason']}")
        log(f"  running mean over {len(prog)} eps: {doc['score']:.2f} "
            f"(SOTA {SOTA})")
    log(f"{label} FINAL: {doc['score']:.2f} vs SOTA {SOTA}")


if __name__ == "__main__":
    main()
