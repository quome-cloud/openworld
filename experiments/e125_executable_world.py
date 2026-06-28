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


def _obj_candidates_fn(avail):
    """Action candidates from an OBJECT state: directional/simple actions always; plus click [6,x,y] targets at
    each small object's position when action 6 is available (x=col, y=row)."""
    simple = [x for x in avail if x in (1, 2, 3, 4, 5, 7)]
    if 6 in avail:
        return lambda s: ([[x] for x in simple]
                          + [[6, int(o["x"]), int(o["y"])] for o in s.get("objects", []) if o.get("size", 99) <= 40])
    return lambda s: [[x] for x in simple]


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
    ap.add_argument("--mode", default="structured", choices=["structured", "traverse"],
                    help="structured=Phase-1 solve_level (pixel predict); traverse=Phase-1+2 solve_game (object world)")
    ap.add_argument("--max-levels", type=int, default=9, help="max game levels to attempt in traverse mode")
    a = ap.parse_args()
    from arc3_sandbox import SandboxGame
    results = {}
    for gid in a.games.split(","):
        if a.mode == "structured":
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
        else:
            # traverse mode: Phase-1+2 object-world pipeline via agent.solve_game
            from e125 import synth as _synth, claude_iso, objstate
            g = SandboxGame(gid); g.reset()
            avail = list(g.avail)
            # Shared SandboxGame instance reset()-ed each call avoids repeated arc.make() (which is slow).
            # This is correct for level 0 and shallow levels; deep multi-level needs a fresh-process replayer
            # to avoid reset-pollution (arc.make is slow so we accept the shallow tradeoff for now).
            game_factory = lambda: (g.reset(), g)[1]
            cands = _obj_candidates_fn(avail)
            sfn = lambda tr, api, gm, **kw: _synth.synthesize_obj(
                tr, api, gm, model=a.model, n_retries=a.retries,
                fallback_runner=lambda p, s, m, gg, **k: claude_iso.run(p, s, model="claude-opus-4-8", game=gg),
                **kw)
            res = agent.solve_game(game_factory, cands, f"actions={avail}", gid, sfn,
                                   perceive=objstate.object_state, macro_runner=None,
                                   budget_explore=a.budget_explore, budget_plan=a.budget_plan,
                                   rounds_per_level=a.rounds, max_levels=a.max_levels, traces_dir=a.traces)
            results[gid] = {k: res[k] for k in ("levels_solved", "real_actions", "levels")}
    save_results("e125_executable_world", {"experiment": "e125_executable_world", "mode": a.mode, "games": results})
    print("[e125]", json.dumps({k: {kk: v.get(kk) for kk in ("solved", "real_actions", "rounds_used", "reason",
                                                               "levels_solved", "levels")}
                                for k, v in results.items()}))


if __name__ == "__main__":
    main()
