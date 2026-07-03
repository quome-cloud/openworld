"""E130 SHU-cycle solver. `theorems` mode validates the two formalism theorems deterministically
(paper-ready); `solve <game>` mode runs the explicit cycle on a real ARC-AGI-3 game, source-free,
seeded from the E129 banked frontier, writing su_<game>/solved.json for the autobank gate."""
import os, sys, json, shutil
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from experiments.e130 import efei, operators as op, perception as P, moral_filter as mf, navigation
from experiments.e130.world_model import WorldModel
from experiments.e130.cycle import run_cycle, run_cycle_v2


def validate_theorems(rng):
    theta = rng.normal(size=8)
    e1 = np.mean([efei.expert_error(theta, 1, 50, 1.0, 0.0, 8, rng) for _ in range(40)])
    e100 = np.mean([efei.expert_error(theta, 100, 50, 1.0, 0.0, 8, rng) for _ in range(40)])
    N = efei.expert_consultations_for(0.2, 50, 1.0, 0.0, 8)
    amateur = float(np.mean([efei.amateur_trials(400, rng) for _ in range(200)]))
    sI, sE = rng.normal(size=8), rng.normal(size=8)
    for _ in range(200):
        sI, sE = op.cycle_map(sI, sE, theta, 0.5, 0.5)
    return {"expert_error_1": float(e1), "expert_error_100": float(e100),
            "expert_consultations": int(N), "amateur_trials_mean": amateur,
            "final_tension": float(op.tension(sI, sE)), "rho": float(op.rho(0.5, 0.5))}


def banked_frontier(game):
    p = os.path.join(ROOT, "experiments/results/arc3_fullgame_sourcefree.json")
    a = json.load(open(p))
    acts = a.get("solutions", {}).get(game) or []
    pg = a.get("per_game", {}).get(game, {})
    return acts, int(pg.get("levels", 0)), int(pg.get("win", 0))


def solve(game, budget):
    import arc3_sandbox
    env = arc3_sandbox.SandboxGame(game); env.reset()
    seed, seed_lv, win = banked_frontier(game)
    win = win or int(env.win)
    wm = WorldModel(); rng = np.random.default_rng(0)
    perceive = lambda fr: P.extrospect(fr, avail=list(getattr(env, "avail", [])))
    # v2: detect avatar color + learned dir_map by probing a fresh env instance
    avail = list(getattr(env, "avail", []))
    game_factory = lambda: arc3_sandbox.SandboxGame(game)
    avatar, dir_map = navigation.detect(game_factory, seed, avail)
    res = run_cycle_v2(env, wm, perceive, mf.DEFAULT_EXPERTS, budget, win, rng,
                       seed_actions=seed, avatar=avatar, dir_map=dir_map)
    wd = os.path.join(ROOT, "scratch_arc", f"su_{game}"); os.makedirs(wd, exist_ok=True)
    shutil.copy(os.path.join(ROOT, "experiments/arc3_sandbox.py"), wd)   # audit-clean workdir
    sol = {"game": game, "actions": res.best_actions, "levels": res.best_levels, "win": win,
           "method": "shu-cycle source-free (E130)"}
    json.dump(sol, open(os.path.join(wd, "solved.json"), "w"))
    if res.best_levels > seed_lv:
        shutil.copy(os.path.join(wd, "solved.json"), os.path.join(wd, "solved_best.json"))
    json.dump({"solver": "shu-cycle", "seed": 0, "seed_levels": seed_lv, "best_levels": res.best_levels,
               "win": win, "cycles": res.cycles, "tension_steps": len(res.tension_trace),
               "banked": res.banked, "improved": res.best_levels > seed_lv},
              open(os.path.join(wd, "run_meta.json"), "w"))   # telemetry for the deterministic-run capture
    print(f"[e130] {game}: {seed_lv} -> {res.best_levels}/{win} "
          f"{'IMPROVED' if res.best_levels > seed_lv else 'no gain'} cycles={res.cycles} "
          f"banked={res.banked} tension_steps={len(res.tension_trace)}", flush=True)
    os.system(f"{sys.executable} {os.path.join(ROOT, 'scripts', 'capture_arc_run.py')} "
              f"{game} {wd} shu-cycle e130_shu_cycle.py")   # HF-ready capture (reuses capture_lib)
    try: env.close()
    except Exception: pass


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "theorems"
    if mode == "solve":
        solve(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 4000)
        return
    m = validate_theorems(np.random.default_rng(0))
    out = os.path.join(ROOT, "experiments/results/e130_shu_cycle.json")
    json.dump(m, open(out, "w"), indent=1)                     # save BEFORE asserts
    print(json.dumps(m, indent=1), flush=True)
    assert m["expert_error_100"] < m["expert_error_1"] / 5.0, "Thm 4.4 variance reduction failed"
    assert m["amateur_trials_mean"] > m["expert_consultations"], "EFEI separation failed"
    assert m["final_tension"] < 1e-6 and m["rho"] < 1.0, "Thm 4.6 contraction failed"
    print("[e130] theorems validated", flush=True)


if __name__ == "__main__":
    main()
