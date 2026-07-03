"""E119 Phase 0 driver: probe the headroom set and emit a GO/No-Go for the macro slot.
  arc venv:  PYTHONPATH="$PWD/scratch_arc/agent" .venv/bin/python experiments/e119_proxy_probe.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))   # let 'import e119'/'common' work
from e119 import proxy_probe
from common import save_results

HEADROOM = ["g50t", "tr87", "re86", "sb26", "cn04"]          # exclude sc25 (wall), bp35 (pruner)
BUDGET = {"max_nodes": 6000, "max_depth": 60}


def _real_make(gid):
    from e119_slm_solver import _real_make as rm
    return rm(gid)


def run_probe(games, make=_real_make, budget=None):
    budget = budget or BUDGET
    rows = []
    for gid in games:
        try:
            rows.append(proxy_probe.probe_game(make(gid), budget))
        except Exception as e:
            rows.append({"game": gid, "error": str(e)[:160]})
    decision = proxy_probe.decide_go(rows)
    return {"phase": "e119_phase0_proxy", "n_games": len(rows), "rows": rows, "decision": decision}


def main():
    games = sys.argv[1].split(",") if len(sys.argv) > 1 else HEADROOM
    payload = run_probe(games)
    save_results("e119_proxy_probe", payload)              # SAVE before asserts (CLAUDE.md)
    assert all(("error" in r) or "best_depth_gain" in r for r in payload["rows"]), "malformed row"
    d = payload["decision"]
    print(f"[e119 phase0] decision={'GO' if d['go'] else 'NO-GO'} signal={d['signal']}")
    print(f"  reason: {d['reason']}")


if __name__ == "__main__":
    main()
