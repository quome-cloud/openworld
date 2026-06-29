"""E131 short-horizon lookahead solver.  Source-free, seeded from the banked frontier archive.

Usage:
    python experiments/e131_lookahead.py solve <game> [budget] [depth] [beam]

`budget` = max lookahead cycles (default 4000).
`depth`  = lookahead horizon in steps (default 3); use 2 for a depth ablation.
`beam`   = beam width at each depth (default 4).
"""
import os, sys, json, shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from experiments.e131.lookahead import solve_lookahead
from experiments.e130 import perception as P


def banked_frontier(game):
    """Read seed actions + prior levels from the E129/E130 archive."""
    p = os.path.join(ROOT, "experiments/results/arc3_fullgame_sourcefree.json")
    a = json.load(open(p))
    acts = a.get("solutions", {}).get(game) or []
    pg = a.get("per_game", {}).get(game, {})
    return acts, int(pg.get("levels", 0)), int(pg.get("win", 0))


def solve(game, budget=4000, depth=3, beam=4):
    import arc3_sandbox
    env = arc3_sandbox.SandboxGame(game)
    env.reset()

    seed, seed_lv, win = banked_frontier(game)
    win = win or int(env.win)

    perceive = lambda fr: P.extrospect(fr, avail=list(getattr(env, "avail", [])))

    res = solve_lookahead(env, perceive, seed_actions=seed, win=win,
                          depth=depth, beam=beam, budget=budget)

    wd = os.path.join(ROOT, "scratch_arc", f"lh_{game}")
    os.makedirs(wd, exist_ok=True)
    shutil.copy(os.path.join(ROOT, "experiments/arc3_sandbox.py"), wd)  # audit-clean workdir

    sol = {
        "game":    game,
        "actions": res.best_actions,
        "levels":  res.best_levels,
        "win":     win,
        "method":  "lookahead source-free (E131)",
    }
    json.dump(sol, open(os.path.join(wd, "solved.json"), "w"))

    if res.best_levels > seed_lv:
        shutil.copy(os.path.join(wd, "solved.json"), os.path.join(wd, "solved_best.json"))

    meta = {
        "solver":       "lookahead",
        "depth":        depth,
        "beam":         beam,
        "seed_levels":  seed_lv,
        "best_levels":  res.best_levels,
        "win":          win,
        "cycles":       res.cycles,
        "real_steps":   res.real_steps,
        "cache_size":   res.cache_size,
        "improved":     res.best_levels > seed_lv,
    }
    json.dump(meta, open(os.path.join(wd, "run_meta.json"), "w"))

    print(
        f"[e131] {game}: {seed_lv} -> {res.best_levels}/{win} "
        f"{'IMPROVED' if res.best_levels > seed_lv else 'no gain'} "
        f"cycles={res.cycles} real_steps={res.real_steps} "
        f"cache_size={res.cache_size} depth={depth} beam={beam}",
        flush=True,
    )

    os.system(
        f"{sys.executable} {os.path.join(ROOT, 'scripts', 'capture_arc_run.py')} "
        f"{game} {wd} lookahead e131_lookahead.py"
    )

    try:
        env.close()
    except Exception:
        pass


def main():
    if len(sys.argv) < 2 or sys.argv[1] != "solve":
        print(__doc__)
        sys.exit(1)
    game   = sys.argv[2]
    budget = int(sys.argv[3]) if len(sys.argv) > 3 else 4000
    depth  = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    beam   = int(sys.argv[5]) if len(sys.argv) > 5 else 4
    solve(game, budget=budget, depth=depth, beam=beam)


if __name__ == "__main__":
    main()
