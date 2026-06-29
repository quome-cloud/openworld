"""E127 source-simulated reconstruction -- the LIVE experiment.

Run the Milestone-1 differential-CEGIS engine reconstruction on a REAL ARC-AGI-3 game, source-free
(via the process-isolated SandboxGame), with real Claude + codex proposers. Report the
equivalence-to-real certificate (held-out next-frame accuracy + Clopper-Pearson lower bound) and the
A-vs-B-vs-real gap (shared-prior-bias measure). The models only ever see frames the agent perceived
by acting -- never game source.

  ~/.arcv/bin/python experiments/e127_source_simulated.py <game> [max_rounds]

Perf: ONE SandboxGame worker is made (arc.make is slow) and reused via reset()+replay -- the
reconstruct loop calls the factory many times but never needs a second live env (single-threaded,
no cloning required).
"""
import os, sys, json, time, traceback

ROOT = "/Users/jim/Desktop/openworld"
sys.path.insert(0, ROOT)
from experiments.e127 import sandbox, reconstruct


def shared_factory(game):
    """Return a factory that hands back ONE shared SandboxGame (callers reset()+replay it). Avoids a
    fresh arc.make per call. Single-threaded use only (which the reconstruct loop is)."""
    env = sandbox.SandboxGame(game)
    fac = lambda: env
    fac._env = env
    return fac


def main():
    game = sys.argv[1] if len(sys.argv) > 1 else "ar25"
    max_rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    fac = shared_factory(game)
    g = fac(); g.reset(); n_levels = int(g.win); avail = list(g.avail)
    action_api = (f"available_actions={avail}. 1=up,2=down,3=left,4=right move a cursor/avatar; 5,7 are "
                  f"other simple actions; 6=click at x=col,y=row (0-63). 64x64 grid, colors 0-15. "
                  f"Deterministic: replaying actions from reset() reproduces frames exactly.")
    budget = {"limit": 6000, "used": 0}
    print(f"[e127] reconstruct {game}: n_levels={n_levels} avail={avail} models=claude,codex "
          f"rounds={max_rounds}", flush=True)
    t0 = time.time()
    try:
        res = reconstruct.reconstruct(fac, action_api, n_levels, models=("claude", "codex"),
                                      max_rounds=max_rounds, budget=budget, seed=0)
    except Exception as ex:
        traceback.print_exc()
        res = {"error": f"{type(ex).__name__}: {ex}"[:300]}
    wall = round(time.time() - t0, 1)
    cert = res.get("certificate", {}) or {}
    rec = {"game": game, "n_levels": n_levels, "protocol": "source-simulated (reconstruct+certify, no source)",
           "wall_s": wall, "certificate": cert, "champion_acc": res.get("champion_acc"),
           "ab_agreement": res.get("ab_agreement"), "ab_vs_real_gap": res.get("ab_vs_real_gap"),
           "rounds": res.get("rounds"), "real_steps": res.get("real_steps"),
           "history": res.get("history"), "engine_src": res.get("engine_src"), "error": res.get("error")}
    out = f"{ROOT}/experiments/results/arc3_source_simulated.json"
    allr = json.load(open(out)) if os.path.exists(out) else {}
    allr[game] = rec
    json.dump(allr, open(out, "w"), indent=1, default=str)
    print(f"[e127] {game}: champion_acc={rec['champion_acc']} cert_pass={cert.get('pass')} "
          f"acc_lower={cert.get('acc_lower')} coverage={cert.get('coverage')} "
          f"n_holdout={cert.get('n')} ab_gap={rec['ab_vs_real_gap']} real_steps={rec['real_steps']} "
          f"wall={wall}s err={rec['error']}", flush=True)
    try:
        fac._env.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
