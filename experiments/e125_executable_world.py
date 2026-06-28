"""E125 entry: structured executable-world-model agent. Solve a level by synthesizing a verified predict(),
planning in simulation, and executing verified plans. save_results before asserts (CLAUDE.md)."""
import os, sys, argparse, json
sys.path.insert(0, os.path.dirname(__file__))
from e125 import agent, synth
from common import save_results


def _probe_mask(game, avail, steps=60, thr=0.95):
    """Collect a short replay of frames and mask cells that change in >thr of step-to-step transitions. thr=0.95
    (project default) only zeroes a per-step counter/timer; games with a CYCLIC ANIMATION region (whose cells
    change at sub-0.95 frequency) need a LOWER thr to mask the animation so a single-frame predict() can
    exact-match -- a moving player has low per-cell frequency and is kept. Replays from reset() so the game is
    left clean for the solver."""
    from e119 import perceive
    dirs = [x for x in avail if x in (1, 2, 3, 4, 5, 7)] or list(avail)
    game.reset(); frames = [game.frame.copy()]
    for i in range(steps):
        game.step(dirs[i % len(dirs)])
        frames.append(game.frame.copy())
        if game.done:
            game.reset()
    game.reset()
    return perceive.status_mask(frames, thresh=thr)


def _candidates_fn(avail):
    """Action candidates per frame: the simple/directional actions always, PLUS pixel-inferred click targets
    (small/rare components) when action 6 is available -- mixed games (movement + clicks) need both."""
    simple = [x for x in avail if x in (1, 2, 3, 4, 5, 7)]
    if 6 in avail:
        from e119 import perceive
        return lambda fr: [[x] for x in simple] + [[6, x, y] for (x, y) in perceive.click_candidates(fr)]
    return lambda fr: [[x] for x in simple]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default="ls20")
    ap.add_argument("--model", default="gpt-5.5")
    ap.add_argument("--traces", default="experiments/results/e125_traces")
    ap.add_argument("--budget-explore", type=int, default=60)
    ap.add_argument("--budget-plan", type=int, default=20000)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--mask-thr", type=float, default=0.95,
                    help="mask cells changing in >thr of steps; lower (~0.2) to mask cyclic animation regions")
    ap.add_argument("--retries", type=int, default=4, help="FunSearch evolve attempts per synth call")
    a = ap.parse_args()
    from arc3_sandbox import SandboxGame
    results = {}
    for gid in a.games.split(","):
        g = SandboxGame(gid); g.reset()
        avail = list(g.avail)
        cands = _candidates_fn(avail)
        mask = _probe_mask(g, avail, thr=a.mask_thr)
        g.close()
        sfn = lambda tr, api, game, m, **kw: synth.synthesize(tr, api, game, m, model=a.model,
                                                              n_retries=a.retries, **kw)
        results[gid] = agent.solve_level(lambda: SandboxGame(gid), cands, f"actions={avail}", gid, mask, sfn,
                                         budget_explore=a.budget_explore, budget_plan=a.budget_plan,
                                         rounds=a.rounds, traces_dir=a.traces)
    save_results("e125_executable_world", {"experiment": "e125_executable_world", "games": results})
    print("[e125]", json.dumps({k: {kk: v.get(kk) for kk in ("solved", "real_actions", "rounds_used", "reason")}
                                for k, v in results.items()}))


if __name__ == "__main__":
    main()
