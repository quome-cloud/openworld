"""E132 hybrid EWM solver.  Source-free, seeded from the banked frontier archive.

Architecture: bounded real-env exploration fills a WorldSim (pure learned model);
plan_in_model does deep beam-lookahead entirely inside the model (no real env,
perfect backtracking, arbitrary horizon); solve_hybrid verifies every plan on the
real env and refines the model on mismatch — so a wrong model cannot bank a fake
solve.

Usage:
    python experiments/e132_hybrid.py solve <game> [depth] [beam]

`depth` = lookahead horizon inside the model (default 8).
`beam`  = beam width at each depth (default 8).
"""
import os, sys, json, shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from experiments.e132.plan import solve_hybrid
from experiments.e130 import perception as P
from experiments.e130.world_model import WorldModel


def banked_frontier(game):
    """Read seed actions + prior levels from the E129/E130 archive."""
    p = os.path.join(ROOT, "experiments/results/arc3_fullgame_sourcefree.json")
    a = json.load(open(p))
    acts = a.get("solutions", {}).get(game) or []
    pg = a.get("per_game", {}).get(game, {})
    return acts, int(pg.get("levels", 0)), int(pg.get("win", 0))


def solve(game, depth=8, beam=8):
    import arc3_sandbox
    env = arc3_sandbox.SandboxGame(game)
    env.reset()

    seed, seed_lv, win = banked_frontier(game)
    win = win or int(env.win)

    perceive = lambda fr: P.extrospect(fr, avail=list(getattr(env, "avail", [])))

    # Seed the frontier: replay seed actions to reach the known frontier
    # (env is already reset; solve_hybrid will replay frontier_path internally)
    wm = WorldModel()

    res = solve_hybrid(
        env, perceive, wm,
        frontier_path=seed,
        seed_levels=seed_lv,
        win=win,
        depth=depth,
        beam=beam,
        rounds=6,
        explore_budget=400,
    )

    wd = os.path.join(ROOT, "scratch_arc", f"hy_{game}")
    os.makedirs(wd, exist_ok=True)
    shutil.copy(os.path.join(ROOT, "experiments/arc3_sandbox.py"), wd)

    sol = {
        "game":    game,
        "actions": res.best_actions,
        "levels":  res.best_levels,
        "win":     win,
        "method":  "hybrid EWM source-free (E132)",
    }
    json.dump(sol, open(os.path.join(wd, "solved.json"), "w"))

    if res.best_levels > seed_lv:
        shutil.copy(os.path.join(wd, "solved.json"), os.path.join(wd, "solved_best.json"))

    improved = res.best_levels > seed_lv
    meta = {
        "solver":       "hybrid",
        "depth":        depth,
        "beam":         beam,
        "rounds":       res.rounds,
        "model_size":   res.model_size,
        "real_steps":   res.real_steps,
        "seed_levels":  seed_lv,
        "best_levels":  res.best_levels,
        "win":          win,
        "improved":     improved,
    }
    json.dump(meta, open(os.path.join(wd, "run_meta.json"), "w"))

    print(
        f"[e132] {game}: {seed_lv} -> {res.best_levels}/{win} "
        f"{'IMPROVED' if improved else 'no gain'} "
        f"rounds={res.rounds} real_steps={res.real_steps} "
        f"model_size={res.model_size} depth={depth} beam={beam}",
        flush=True,
    )

    os.system(
        f"{sys.executable} {os.path.join(ROOT, 'scripts', 'capture_arc_run.py')} "
        f"{game} {wd} hybrid e132_hybrid.py"
    )

    try:
        env.close()
    except Exception:
        pass


def main():
    if len(sys.argv) < 2 or sys.argv[1] != "solve":
        print(__doc__)
        sys.exit(1)
    game  = sys.argv[2]
    depth = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    beam  = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    solve(game, depth=depth, beam=beam)


if __name__ == "__main__":
    main()
