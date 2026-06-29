"""E119 macro slot — 3-arm (control / random-macro / SLM-macro) x m-seed sweep on the signal-bearing
procedure-walls. SLM arm reported as a distribution; every banked solve anchored to a replay.
  arc venv:  PYTHONPATH="$PWD/scratch_arc/agent" .venv/bin/python experiments/e119_macro_sweep.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))   # let 'import e119'/'common' work
from e119 import solve, trace
from common import save_results

HEADROOM = ["tr87", "re86", "sb26", "cn04"]
BUDGET = {"max_nodes": 6000, "max_depth": 60}
MODEL = "qwen2.5-coder:7b"
LOGDIR = pathlib.Path(__file__).resolve().parent / "results" / "e119_logs"


def _agg(levels_list):
    n = len(levels_list); mean = sum(levels_list) / n if n else 0.0
    var = sum((x - mean) ** 2 for x in levels_list) / n if n else 0.0
    return {"levels_mean": round(mean, 3), "levels_var": round(var, 3),
            "k_solved": sum(1 for x in levels_list if x > 0), "m": n}


def _real_make(gid):
    from e119_slm_solver import _real_make as rm
    return rm(gid)


def run_sweep(games, seeds, make=_real_make, llm_factory=None, budget=None,
              arms=("search", "random-macro", "macro"), options=None, digest=None, logdir=None):
    budget = budget or BUDGET
    ld = logdir if logdir is not None else LOGDIR
    # per-call LLM transcripts deferred; per-run records + banked replayable solutions are the audit trail
    by = {}
    for gid in games:
        by[gid] = {}
        for arm in arms:
            mode = "search" if arm == "search" else arm
            seed_set = [0] if arm == "search" else seeds       # control is deterministic: 1 run
            levels = []
            for s in seed_set:
                llm = None if arm == "search" else (llm_factory(s) if llm_factory else None)
                try:
                    r = solve.solve_game(make(gid), llm=llm, mode=mode, budget=budget,
                                         logdir=ld, make=make, seed=s)
                    levels.append(r["levels"])
                    trace.log_run(ld / "e119_runs.jsonl", {
                        "game": gid, "arm": arm, "mode": mode, "seed": s,
                        "levels": r["levels"], "verified": r.get("verified", False),
                    })
                except Exception as e:
                    levels.append(0)
                    trace.log_run(ld / "e119_runs.jsonl", {
                        "game": gid, "arm": arm, "mode": mode, "seed": s,
                        "levels": 0, "verified": False, "error": type(e).__name__,
                    })
            by[gid][arm] = _agg(levels)
    prov = trace.provenance(MODEL, options or {}, seeds, budget, digest=digest)
    summary = {g: {a: by[g][a]["k_solved"] for a in arms} for g in games}
    return {"arms": list(arms), "games": list(games), "by_game_arm": by,
            "provenance": prov, "summary": summary}


def main():
    games = sys.argv[1].split(",") if len(sys.argv) > 1 else HEADROOM
    import openworld as O
    from e119 import slm as _slm
    seeds = [0, 1, 2, 3, 4]
    opts = _slm.llm_options(MODEL)                               # decoding params (minus per-seed seed)
    def llm_factory(seed):
        return O.OllamaLLM(model=MODEL, options={**opts, "seed": seed})
    payload = run_sweep(games, seeds=seeds, llm_factory=llm_factory, options=opts,
                        digest=trace.ollama_digest(MODEL))
    save_results("e119_macro_sweep", payload)               # SAVE before asserts (CLAUDE.md)
    assert all(arm in payload["arms"] for arm in ("search", "random-macro", "macro"))
    for g in games:
        b = payload["by_game_arm"][g]
        print(f"{g}: search={b['search']['k_solved']}/1  random={b['random-macro']['k_solved']}/{b['random-macro']['m']}  "
              f"macro={b['macro']['k_solved']}/{b['macro']['m']} (levels {b['macro']['levels_mean']}±{b['macro']['levels_var']})")
    from e119 import reverify
    rv = reverify.reverify_solves(LOGDIR, _real_make)
    print(f"[reverify] {rv['ok']}/{rv['n']} banked solves replay-confirmed; fail={rv['fail']}")


if __name__ == "__main__":
    main()
