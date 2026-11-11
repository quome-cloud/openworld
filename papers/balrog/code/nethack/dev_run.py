"""Dev harness: single episode on a dev seed with verbose tracing.
Dev seeds 101-140, disjoint from the evaluation blocks (1000s/2000s/4000+).
Usage: python3 dev_run.py <seed> [max_steps]
"""

import sys
import time

import nh_runner
import nh_harness as H


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 101
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else 6000
    nh_runner.MAX_LOOP = cap
    t0 = time.time()
    res = nh_runner.run_episode(ep=seed, seed=seed, condition="dev",
                                label="dev", log=print)
    print("\nRESULT:")
    for k, v in res.items():
        if k not in ("mem_fired",):
            print(f"  {k}: {v}")
    print(f"total wall {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
