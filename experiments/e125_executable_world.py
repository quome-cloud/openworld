"""E125 entry: structured executable-world-model agent. Solve a level by synthesizing a verified predict(),
planning in simulation, and executing verified plans. save_results before asserts (CLAUDE.md)."""
import os, sys, argparse, json
sys.path.insert(0, os.path.dirname(__file__))
from e125 import agent, synth
from common import save_results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default="g50t")
    ap.add_argument("--model", default="gpt-5.5")
    ap.add_argument("--traces", default="experiments/results/e125_traces")
    a = ap.parse_args()
    from arc3_sandbox import SandboxGame
    from e119 import perceive
    results = {}
    for gid in a.games.split(","):
        g = SandboxGame(gid); g.reset()
        avail = g.avail
        cands = (lambda fr: [[x] for x in avail if x in (1, 2, 3, 4, 5, 7)])
        mask = perceive.status_mask([g.frame])
        sfn = lambda tr, api, game, m, **kw: synth.synthesize(tr, api, game, m, model=a.model, **kw)
        results[gid] = agent.solve_level(lambda: SandboxGame(gid), cands, f"actions={avail}", gid, mask, sfn,
                                         traces_dir=a.traces)
    save_results("e125_executable_world", {"experiment": "e125_executable_world", "games": results})
    print("[e125]", json.dumps({k: {kk: v[kk] for kk in ("solved", "real_actions", "rounds_used")}
                                for k, v in results.items()}))


if __name__ == "__main__":
    main()
