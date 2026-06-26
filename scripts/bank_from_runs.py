"""Bank the source-free archive FROM the captured dataset (runs.jsonl). For each game, pick the deepest run
whose verified outcome is (a) source-free audit-CLEAN and (b) passes the OpenWorld-World round-trip, and
record it -- tier-tagged with its run_id provenance -- in experiments/results/arc3_fullgame_sourcefree.json.
Also promote the winning trace to scratch_arc/sb_<game>/solved_best.json (best-keeper seed for agent rounds).

This unifies the cheap and agent tiers: both become run records; banking takes the best VERIFIED run per
game regardless of tier. Run with the arc venv python (needs nothing beyond stdlib here).
    <arcv>/bin/python scripts/bank_from_runs.py
"""
import json, sys
from pathlib import Path

ROOT = Path("/Users/jim/Desktop/openworld")
sys.path.insert(0, str(ROOT / "scripts"))
import capture_lib as c

ARCH = ROOT / "experiments" / "results" / "arc3_fullgame_sourcefree.json"

# Canonical win_levels per game (all 25) -> the denominator is the FULL 183 levels, NOT just started
# games, so source-free X/183 is comparable to the source-assisted 177/183.
WIN_LEVELS = {"ar25": 8, "bp35": 9, "cd82": 6, "cn04": 6, "dc22": 6, "ft09": 6, "g50t": 7, "ka59": 7,
              "lf52": 10, "lp85": 8, "ls20": 7, "m0r0": 6, "r11l": 6, "re86": 8, "s5i5": 8, "sb26": 8,
              "sc25": 6, "sk48": 8, "sp80": 6, "su15": 9, "tn36": 7, "tr87": 6, "tu93": 9, "vc33": 7,
              "wa30": 9}
TOTAL_POSSIBLE = sum(WIN_LEVELS.values())   # 183


def load_runs():
    if not c.RUNS.exists():
        return []
    return [json.loads(l) for l in open(c.RUNS, errors="ignore") if l.strip()]


def bankable(r):
    o = r.get("outcome") or {}
    rt = o.get("openworld_roundtrip") or {}
    return bool(o.get("audit", {}).get("clean") and rt.get("pass") and o.get("levels", 0) > 0)


def main():
    runs = load_runs()
    best = {}    # game -> winning run record
    for r in runs:
        if not bankable(r):
            continue
        g = r["game"]
        lv = r["outcome"]["levels"]
        if g not in best or lv > best[g]["outcome"]["levels"]:
            best[g] = r
    arch = {"protocol": "source-free (routed: cheap pixel-search by-audit + sandboxed agent by-construction)",
            "verification": "per run: source-free audit + real-engine replay + OpenWorld World round-trip "
                            "(world.step reproduces depth, 0 misses, valid spec, renderable card)",
            "dataset": "experiments/results/arc3_traces/runs.jsonl", "per_game": {}, "solutions": {},
            "roundtrip": {}, "provenance": {}}
    for g, r in best.items():
        o = r["outcome"]
        arch["per_game"][g] = {"levels": o["levels"], "win": o["win"], "tier": r["tier"]}
        arch["solutions"][g] = o["actions"]
        arch["roundtrip"][g] = o["openworld_roundtrip"]
        arch["provenance"][g] = {"run_id": r["run_id"], "tier": r["tier"], "method": r.get("method"),
                                 "model": r.get("model_config", {}).get("resolved_model"),
                                 "effort": r.get("model_config", {}).get("effort"),
                                 "fairness": r.get("fairness")}
        # best-keeper seed for the agent tier's next round
        wd = ROOT / "scratch_arc" / f"sb_{g}"
        wd.mkdir(parents=True, exist_ok=True)
        (wd / "solved_best.json").write_text(json.dumps(
            {"game": g, "actions": o["actions"], "levels": o["levels"], "win": o["win"]}))
    # include ALL 25 games (zero-progress games shown as 0/canonical-win) so the denominator is honest
    for g, win in WIN_LEVELS.items():
        arch["per_game"].setdefault(g, {"levels": 0, "win": win, "tier": None})
        if not arch["per_game"][g].get("win"):
            arch["per_game"][g]["win"] = win
    arch["full_games"] = sorted(g for g, v in arch["per_game"].items()
                                if v.get("win") and v["levels"] >= v["win"])
    arch["n_full_games"] = len(arch["full_games"])
    arch["total_levels"] = sum(v["levels"] for v in arch["per_game"].values())
    arch["total_possible"] = TOTAL_POSSIBLE                  # 183 across all 25 games (canonical)
    arch["n_games_started"] = sum(1 for v in arch["per_game"].values() if v["levels"] > 0)
    arch["by_tier"] = {}
    for g, v in arch["per_game"].items():
        arch["by_tier"].setdefault(v["tier"], []).append(g)
    arch["updated_at"] = c.iso_now()
    ARCH.write_text(json.dumps(arch, indent=1))
    print(f"[bank] {arch['n_full_games']} full games, {arch['total_levels']}/{arch['total_possible']} levels "
          f"across {arch['n_games_started']} games | by tier: "
          f"{ {k: len(v) for k, v in arch['by_tier'].items()} }", flush=True)
    for g in sorted(arch["per_game"]):
        v = arch["per_game"][g]
        print(f"    {g:6} {v['levels']}/{v['win']}  [{v['tier']}]", flush=True)


if __name__ == "__main__":
    main()
