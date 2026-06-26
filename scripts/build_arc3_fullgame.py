"""Build experiments/results/arc3_fullgame.json (source of truth for the arc-3 paper's full-game
result) from the per-game solved_best.json files. Modality is derived from the banked action traces
(honest, data-grounded); mechanics are curated for the games we characterized in detail and fall back
to the action modality otherwise. Re-run after any new full-game solve, then make_arc3_assets.py.
"""
import json, glob, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCR = ROOT / "scratch_arc"
OUT = ROOT / "experiments" / "results" / "arc3_fullgame.json"

DESC = {
 "g50t": ("clone", "Clone puzzle: record a path, an action-5 ghost holds a switch, then walk a fresh player to the goal."),
 "tr87": ("rewriting", "Glyph string-rewriting: edit output digits or rewrite rule sets (symbolic, not navigation)."),
 "bp35": ("platformer", "Gravity platformer: walk and fall to the gem; clicks remove single-use blocks or flip gravity."),
 "lp85": ("rings", "Hungarian rings: click to rotate two interlocked rings until goal tiles align."),
 "m0r0": ("mirror", "Mirror-merge: two horizontally-mirrored players; merge them onto a single cell."),
 "sb26": ("synthesis", "Program synthesis: click to fill subroutine slots so the flattened call tree equals a target color sequence."),
 "sk48": ("sokoban", "Sokoban-as-snake: anchored snakes push colored blocks; match each pair's covered-color sequence."),
 "wa30": ("sokoban", "Sokoban variant: push boxes onto goals while autonomous helper/antagonist bots act each step."),
 "s5i5": ("slide", "Slide-to-goal: drive a tip riding a bar onto the goal cell."),
 "lf52": ("peg", "Peg/block-riding camera puzzle: rides and jumps move pegs across scrolling regions to merge a pair."),
 "dc22": ("navigation", "Navigate a player to the goal via slidable crusher bridges, buttons, and color-cycle teleports."),
 "ka59": ("navigation", "Directional navigation; a single click operates a door to complete the level."),
 "vc33": ("click", "Click-only puzzle: click the few valid sprite targets to satisfy each level."),
 "sp80": ("interaction", "Interaction puzzle: the win fires on an accumulated interact (action-5) condition after a move sequence."),
 "cd82": ("navigation", "Navigation/maze solved through the synthesized code world model (predict-lookahead MPC)."),
}


def modality(acts):
    has_click = any(isinstance(a, (list, tuple)) and a and a[0] == 6 for a in acts)
    has_dir = any(isinstance(a, int) or (isinstance(a, (list, tuple)) and len(a) == 1 and a[0] != 6) for a in acts)
    if has_click and has_dir:
        return "mixed"
    return "click" if has_click else "directional"


def read(g):
    for cand in (SCR / f"full_{g}" / "solved_best.json", SCR / f"full_{g}" / "solved.json",
                 SCR / f"agent_{g}" / "solved.json"):
        if cand.exists():
            return json.load(open(cand))
    return None


def main():
    games = {}
    for d in sorted(glob.glob(str(SCR / "full_*"))) :
        g = os.path.basename(d).replace("full_", "")
        j = read(g)
        if j:
            games[g] = j
    for g in ("ft09", "r11l"):           # full games solved earlier via the live agent (win=6)
        j = read(g)
        if j and g not in games:
            j.setdefault("win", 6); j.setdefault("levels", 6)
            games[g] = j

    out = {
        "protocol": "offline executable-world-model recipe: deterministic env + unbounded resets; "
                    "per-game source-faithful simulator + per-level search + deterministic replay-verify. "
                    "Same full-game-completion metric as baseline1; we do not report RHAE (live action efficiency).",
        "n_games": 0, "n_full": 0, "n_partial": 0, "levels_completed": 0, "levels_total": 0,
        "baseline1_full": 15, "baseline1_protocol": "executable world models (GPT-5.5), ~$350/run, 58% RHAE",
        "cost_note": "single Claude Code subscription, <24h wall-clock for all games",
        "games": {},
    }
    for g in sorted(games):
        j = games[g]; lv = j.get("levels"); win = j.get("win")
        cat, desc = DESC.get(g, (modality(j.get("actions", [])),
                                 f"{modality(j.get('actions', []))} puzzle (mechanic not yet characterized in detail)."))
        full = isinstance(lv, int) and isinstance(win, int) and win > 0 and lv >= win
        out["games"][g] = dict(levels=lv, win=win, full=full, modality=modality(j.get("actions", [])),
                               category=cat, actions=len(j.get("actions", [])), desc=desc)
        out["n_games"] += 1
        out["n_full"] += 1 if full else 0
        out["n_partial"] += 0 if full else 1
        out["levels_completed"] += lv if isinstance(lv, int) else 0
        out["levels_total"] += win if isinstance(win, int) else 0
    json.dump(out, open(OUT, "w"), indent=2, sort_keys=True)
    print(f"wrote {OUT.name}: {out['n_games']} games, {out['n_full']} full, "
          f"{out['n_partial']} partial, {out['levels_completed']}/{out['levels_total']} levels")


if __name__ == "__main__":
    main()
