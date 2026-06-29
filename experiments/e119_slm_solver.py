"""E119 entry: run the pilot ARC-AGI-3 games under the search-only control and the SLM-in-loop rung.

  arc venv python -- needs arc_agi:
  $ARC_VENV experiments/e119_slm_solver.py --mode search
  $ARC_VENV experiments/e119_slm_solver.py --mode slm --model qwen2.5-coder:7b
"""
import argparse, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))   # let 'import e119' work
from e119 import solve
from common import save_results

PILOT = ["tn36", "ar25", "vc33", "lp85", "sk48"]


def _real_make(gid):
    from arc3_harness import Game
    g = Game(gid); g.reset(); g.gid = gid
    return g


def run_pilot(games, mode="search", make=_real_make, llm=None, budget=None, logdir=None):
    results = []
    for gid in games:
        try:
            g = make(gid)
            r = solve.solve_game(g, llm=llm, mode=mode, budget=budget, logdir=logdir)
        except Exception as e:
            r = {"game": gid, "mode": mode, "levels": 0, "win": 0,
                 "actions": [], "verified": False, "error": str(e)[:160]}
        results.append(r)
    return {"mode": mode, "n_games": len(results),
            "levels_solved": sum(r["levels"] for r in results),
            "full_games": sum(1 for r in results if r["win"] and r["levels"] >= r["win"]),
            "results": results}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["search", "slm"], default="search")
    ap.add_argument("--model", default="qwen2.5-coder:7b")
    ap.add_argument("--games", default=",".join(PILOT))
    a = ap.parse_args()
    llm = None
    if a.mode == "slm":
        import openworld as O
        from e119 import slm as _slm
        llm = O.OllamaLLM(model=a.model, options=_slm.llm_options(a.model))
    logdir = pathlib.Path(__file__).resolve().parent / "results" / "e119_logs"
    payload = run_pilot(a.games.split(","), mode=a.mode, llm=llm,
                        budget={"max_nodes": 6000, "max_depth": 60}, logdir=logdir)
    save_results("e119_slm_solver", payload)          # SAVE before asserts (CLAUDE.md)
    assert payload["levels_solved"] >= 0
    assert all(("error" in r) or r["verified"] for r in payload["results"]), "unverified non-error solve"
    print(f"[e119] mode={a.mode} levels={payload['levels_solved']} full={payload['full_games']}")


if __name__ == "__main__":
    main()
