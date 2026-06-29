"""E128: Go-Explore source-free final-level solver on a REAL ARC-AGI-3 game.

Seeds the cell archive from the Claude-SF banked frontier (level N-1) and Go-Explores forward to
crack the final level's procedure -- the wall reasoning-only agents stall on. Source-free: only the
SandboxGame {frame, levels, win, avail} client is used; no game code. The found action sequence is
replay-verified by construction (it was reached by stepping the real env); writes solved.json to
scratch_arc/ge_<game>/ so the existing audit+replay+OpenWorld-roundtrip gate can attest and bank it.

  ~/.arcv/bin/python experiments/e128_go_explore.py <game> [budget_steps]
"""
import os, sys, json, time, shutil, traceback

ROOT = "/Users/jim/Desktop/openworld"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from experiments.e128.go_explore import go_explore
from experiments.e127 import perception
import arc3_sandbox


def shared_factory(game):
    env = arc3_sandbox.SandboxGame(game)   # ONE arc.make; callers reset()+replay
    fac = lambda: env
    fac._env = env
    return fac


def banked_frontier(game):
    """The deepest Claude-SF banked trajectory to seed from (level N-1)."""
    for fn in ("solved_best.json", "solved.json"):
        p = f"{ROOT}/scratch_arc/sb_{game}/{fn}"
        if os.path.exists(p):
            try:
                d = json.load(open(p))
                return d.get("actions") or [], int(d.get("levels", 0)), int(d.get("win", 0))
            except Exception:
                pass
    return [], 0, 0


def main():
    game = sys.argv[1] if len(sys.argv) > 1 else "tu93"
    budget = int(sys.argv[2]) if len(sys.argv) > 2 else 200000
    mode = sys.argv[3] if len(sys.argv) > 3 else "macro"     # 'macro' (object-level) | 'micro'
    pref = "gm_" if mode == "macro" else "ge_"               # separate workdir per mode (no collision)
    seed_actions, seed_lv, win = banked_frontier(game)
    fac = shared_factory(game)
    g = fac(); g.reset()
    win = win or int(g.win)
    print(f"[e128] {game} ({mode}): seed frontier {seed_lv}/{win} ({len(seed_actions)} actions), "
          f"budget {budget} steps", flush=True)
    t0 = time.time()
    try:
        if mode == "macro":
            from experiments.e128.macros import macro_solve
            res = macro_solve(fac, budget, seed_actions=seed_actions, win=win, seed=0)
        else:
            res = go_explore(fac, perception.candidate_actions, budget=budget,
                             seed_actions=seed_actions, win=win, seed=0)
    except Exception as ex:
        traceback.print_exc()
        res = {"win": False, "best_levels": seed_lv, "best_actions": [list(a) for a in seed_actions],
               "archive": 0, "real_steps": 0, "error": str(ex)[:200]}
    wall = round(time.time() - t0, 1)
    lv = res["best_levels"]
    improved = lv > seed_lv
    print(f"[e128] {game} ({mode}): avatar={res.get('avatar')}", flush=True)
    # write solved.json to a clean per-mode workdir for the attestation gate (audit + replay + roundtrip)
    wd = f"{ROOT}/scratch_arc/{pref}{game}"
    os.makedirs(wd, exist_ok=True)
    shutil.copy(f"{ROOT}/experiments/arc3_sandbox.py", wd)   # the only file -> audit-clean
    sol = {"game": game, "actions": res["best_actions"], "levels": lv, "win": win,
           "method": "go-explore-source-free (E128)"}
    json.dump(sol, open(f"{wd}/solved.json", "w"))
    if improved:
        shutil.copy(f"{wd}/solved.json", f"{wd}/solved_best.json")
    rec = {"game": game, "levels": lv, "win": win, "seed_levels": seed_lv,
           "improved": improved, "full": bool(win and lv >= win),
           "archive_cells": res.get("archive"), "real_steps": res.get("real_steps"),
           "wall_s": wall, "error": res.get("error")}
    json.dump(rec, open(f"{wd}/result.json", "w"))            # per-game (no race across parallel runs)
    out = (f"{ROOT}/experiments/results/arc3_go_explore_macro.json" if mode == "macro"
           else f"{ROOT}/experiments/results/arc3_go_explore.json")
    try:
        allr = json.load(open(out)) if os.path.exists(out) else {}
        allr[game] = rec
        json.dump(allr, open(out, "w"), indent=1)
    except Exception:
        pass
    print(f"[e128] {game}: {seed_lv} -> {lv}/{win}  {'IMPROVED' if improved else 'no gain'} "
          f"{'FULL' if lv>=win else ''}  archive={res.get('archive')} steps={res.get('real_steps')} "
          f"wall={wall}s", flush=True)
    try:
        fac._env.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
