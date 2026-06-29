"""Autonomous auto-banker: replay-verify any scratch full-game progress that is DEEPER than the
banked archive, and commit the verified gains. Deterministic (no LLM). Safe to run repeatedly:
it only banks a game whose scratch depth strictly exceeds the banked depth AND replays clean.

Run with the arc venv python (needs arc_agi):
    /private/tmp/.../arcv/bin/python scripts/autobank_arc.py
"""
import json, subprocess, sys, glob, os
from pathlib import Path

ROOT = Path("/Users/jim/Desktop/openworld")
sys.path.insert(0, str(ROOT / "scratch_arc" / "agent"))   # arc3_harness.Game
from arc3_harness import Game

ARCH = ROOT / "experiments" / "results" / "agent_full_game.json"


def scratch_best(game):
    """Deepest (levels, solution-dict) across this game's scratch solved files."""
    best = None
    for fn in ("solved_best.json", "solved.json"):
        p = ROOT / "scratch_arc" / f"full_{game}" / fn
        if p.exists():
            try:
                d = json.loads(p.read_text())
                lv = int(d.get("levels", 0))
                if best is None or lv > best[0]:
                    best = (lv, d)
            except Exception:
                pass
    return best


def replay_reaches(game, actions, claim):
    """Replay the action trace from reset(); return True iff it raises levels by >= claim."""
    g = Game(game)
    base = g.levels
    mx = base
    for a in actions:
        g.step(*a) if isinstance(a, (list, tuple)) else g.step(a)
        if g.levels > mx:
            mx = g.levels
        if g.done:
            break
    return (mx - base) >= claim


def main():
    arch = json.loads(ARCH.read_text())
    games = sorted(set(list(arch["per_game"]) +
                       [os.path.basename(d).replace("full_", "")
                        for d in glob.glob(str(ROOT / "scratch_arc" / "full_*"))]))
    changed = []
    for g in games:
        sb = scratch_best(g)
        if not sb:
            continue
        lv, d = sb
        banked = int(arch["per_game"].get(g, {}).get("levels", 0))
        if lv <= banked:
            continue
        actions = d.get("actions") or []
        if not actions:
            continue
        try:
            ok = replay_reaches(g, actions, lv)
        except Exception as e:
            print(f"[autobank] {g}: verify error: {e}", flush=True)
            continue
        if not ok:
            print(f"[autobank] {g}: claim {lv} did NOT replay-verify; skipping", flush=True)
            continue
        win = int(d.get("win", 0) or arch["per_game"].get(g, {}).get("win", 0))
        arch["per_game"][g] = {"levels": lv, "win": win}
        arch["solutions"][g] = actions
        changed.append((g, banked, lv, win))
        print(f"[autobank] {g}: {banked} -> {lv}/{win} VERIFIED", flush=True)

    if not changed:
        print("[autobank] no new verified gains", flush=True)
        return
    arch["full_games"] = sorted({g for g, v in arch["per_game"].items()
                                 if v["win"] and v["levels"] >= v["win"]})
    arch["n_full_games"] = len(arch["full_games"])
    arch["total_levels"] = sum(v["levels"] for v in arch["per_game"].values())
    arch["total_possible"] = sum(v["win"] for v in arch["per_game"].values())
    ARCH.write_text(json.dumps(arch, indent=1))
    summary = "; ".join(f"{g} {b}->{l}/{w}" for g, b, l, w in changed)
    msg = (f"Auto-bank overnight: {summary} -> {arch['n_full_games']}/25 full, "
           f"{arch['total_levels']}/{arch['total_possible']} levels.\n\n"
           "Replay-verified against the real env before banking (autonomous supervisor).\n\n"
           "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>")
    subprocess.run(["git", "-C", str(ROOT), "add", str(ARCH)], check=False)
    subprocess.run(["git", "-C", str(ROOT), "commit", "-q", "-m", msg], check=False)
    print(f"[autobank] committed: {summary}", flush=True)


if __name__ == "__main__":
    main()
