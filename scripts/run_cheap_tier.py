"""CHEAP tier of the routed (hybrid-world-models) solver: a fixed, pixel-only frontier search
(E107 graph-explore) attempts each ARC-AGI-3 game source-free. The algorithm consumes only frames and
never reads game source (fairness BY AUDIT -- statically verifiable; the audit confirms it). Every attempt
is captured into the arc3_traces dataset as a deterministic run record (no prompt/transcript; solver name +
seed + budget + code hash instead), and its solution snapshot is finalized into a verified outcome.

Run with the arc venv python (needs arc_agi):
    <arcv>/bin/python scripts/run_cheap_tier.py [game ...]
"""
import json, sys, os
from pathlib import Path

ROOT = Path("/Users/jim/Desktop/openworld")
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "experiments"))
import capture_lib as c

ALL = ["ar25", "bp35", "cd82", "cn04", "dc22", "ft09", "g50t", "ka59", "lf52", "lp85", "ls20", "m0r0",
       "r11l", "re86", "s5i5", "sb26", "sc25", "sk48", "sp80", "su15", "tn36", "tr87", "tu93", "vc33", "wa30"]

E107 = ROOT / "experiments" / "e107_graph_explore.py"
BUDGET = int(os.environ.get("CHEAP_BUDGET", "30000"))
MAX_STEPS = int(os.environ.get("CHEAP_MAX_STEPS", "400"))
SEED = 0


def run_one(game):
    import e107_graph_explore as e107
    started = c.iso_now()
    try:
        r = e107.explore(game, budget=BUDGET, max_steps=MAX_STEPS)
    except Exception as ex:
        r = {"best_levels": 0, "win_levels": 0, "solution": None, "verified": False, "error": str(ex)[:200]}
    ended = c.iso_now()
    sol = r.get("solution")
    actions = [[int(a)] for a in sol] if sol else []          # E107 yields directional ints -> [[a],...]
    rid = c.run_id(game, "cheap")
    # snapshot the produced solution trace (immutable per-run)
    c.ensure_dirs()
    snap = {"game": game, "actions": actions, "levels": int(r.get("best_levels", 0)),
            "win": int(r.get("win_levels", 0))}
    (c.SOLUTIONS / f"{rid}.json").write_text(json.dumps(snap))
    rec = {
        "run_id": rid, "game": game, "tier": "cheap", "method": "e107-graph-explore (frontier pixel search)",
        "source_free": True, "fairness": "by-audit (fixed solver reads only frames)",
        "audit_files": ["experiments/e107_graph_explore.py", "experiments/arc3_graph.py"],
        "audit_mode": "source_only",
        "started_at": started, "ended_at": ended, "exit_code": 0,
        "model_config": {"requested_model": None, "resolved_model": None, "effort": None,
                         "note": "deterministic algorithm; no LLM"},
        "params": {"solver": "e107.explore", "budget": BUDGET, "max_steps": MAX_STEPS, "seed": SEED},
        "solution_file": f"solutions/{rid}.json",
        "prompt_file": None, "transcript_file": None,
        "host": c.host_info(), "git": c.git_info(),
        "pipeline": c.file_provenance([E107, ROOT / "experiments" / "arc3_graph.py",
                                       ROOT / "scripts" / "run_cheap_tier.py"]),
        "benchmark": c.BENCHMARK, "dataset_version": c.DATASET_VERSION,
        "outcome": None,
    }
    c.write_meta(rid, rec)
    print(f"[cheap] {game}: best {r.get('best_levels')}/{r.get('win_levels')} "
          f"verified={r.get('verified')} states={r.get('states')} -> {rid}", flush=True)
    return rec


def main():
    games = sys.argv[1:] or ALL
    for g in games:
        run_one(g)


if __name__ == "__main__":
    main()
