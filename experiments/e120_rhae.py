"""E120 -- RHAE / action-efficiency of our verified ARC-AGI-3 solutions, scored with the OFFICIAL
arc_agi formula and the OFFICIAL per-level human baselines (EnvironmentInfo.baseline_actions).

Per level the engine scores  min((baseline_actions / actions_taken)^2 * 100, 115); the per-game
aggregate is the index-weighted mean of level scores (later levels weigh more). 100 == human-efficient,
>100 == better than the human baseline (capped 115). baseline1 reports ~58 mean RHAE.

We replay each banked solution against the REAL deterministic engine, segment actions per level (a
segment ends when levels_completed increments), and score with arc_agi's own EnvironmentScoreCalculator
-- no reimplementation. This is the action efficiency of our EXISTING (search-found) solutions; the
minimized/optimal variant (shortest plan per level from the verified world model) is the upper bound and
the SOTA lever -- see --minimize (bounded greedy redundancy removal against the real env).

Run with the arcv interpreter (has arc_agi):
    <arcv>/bin/python experiments/e120_rhae.py            # score banked solutions
    <arcv>/bin/python experiments/e120_rhae.py --minimize # also greedily shorten, then re-score
"""
import json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCR = ROOT / "scratch_arc"
RES = ROOT / "experiments" / "results"
FG = json.load(open(RES / "arc3_fullgame.json"))


def solved_path(g):
    for p in (SCR / f"full_{g}" / "solved_best.json", SCR / f"full_{g}" / "solved.json",
              SCR / f"agent_{g}" / "solved.json"):
        if p.exists():
            return p
    return None


def replay_segments(game, actions):
    """Replay actions in the real env; return (per-level action counts for completed levels,
    leftover actions after the last completion, final levels_completed)."""
    from arc3_harness import Game
    g = Game(game); g.reset()
    base = g.levels
    seg, segs, last = 0, [], base
    for s in actions:
        g.step(*s) if isinstance(s, (list, tuple)) else g.step(s)
        seg += 1
        if g.levels > last:                       # one (or more) level(s) completed on this action
            for _ in range(g.levels - last):
                segs.append(seg); seg = 0          # attribute the whole segment to the first; rest=0
            last = g.levels
        if g.done:
            break
    return segs, seg, g.levels - base


def greedy_minimize(game, actions, budget=4000):
    """Bounded redundancy removal: try dropping shrinking chunks; keep a drop iff the full solution
    still reaches the same final level. Verified against the real engine. Not provably optimal."""
    from arc3_harness import Game

    def final_level(seq):
        g = Game(game); g.reset(); base = g.levels
        for s in seq:
            g.step(*s) if isinstance(s, (list, tuple)) else g.step(s)
            if g.done:
                break
        return g.levels - base

    target = final_level(actions)
    seq = list(actions); calls = 1; chunk = max(1, len(seq) // 4)
    while chunk >= 1 and calls < budget:
        i = 0; changed = False
        while i < len(seq) and calls < budget:
            cand = seq[:i] + seq[i + chunk:]
            calls += 1
            if cand and final_level(cand) >= target:
                seq = cand; changed = True            # keep the deletion; don't advance i
            else:
                i += chunk
        if not changed:
            chunk //= 2
    return seq, target


def score_game(game, wd, minimize=False):
    import arc_agi
    from arc_agi import EnvironmentScoreCalculator
    cwd = os.getcwd(); os.chdir(wd)
    if str(wd) not in sys.path:
        sys.path.insert(0, str(wd))            # so `import arc3_harness` (copied into each game dir) resolves
    try:
        info = arc_agi.Arcade().make(game).environment_info
        baseline = list(info.baseline_actions or [])
        sp = solved_path(game)
        actions = json.load(open(sp))["actions"]
        if minimize:
            actions, _ = greedy_minimize(game, actions)
        segs, leftover, reached = replay_segments(game, actions)
        calc = EnvironmentScoreCalculator()
        for i, b in enumerate(baseline):
            if i < len(segs):
                calc.add_level(level_index=i + 1, completed=True, actions_taken=segs[i], baseline_actions=b)
            else:
                calc.add_level(level_index=i + 1, completed=False, actions_taken=0, baseline_actions=b)
        sc = calc.to_score()
        return {
            "game": game, "win": len(baseline), "levels_completed": reached,
            "baseline_actions": baseline, "our_actions": segs, "total_actions": sum(segs) + leftover,
            "baseline_total": sum(baseline), "level_scores": [round(x, 1) for x in (sc.level_scores or [])],
            "score": round(sc.score, 2), "minimized": minimize,
        }
    finally:
        os.chdir(cwd)


def main():
    minimize = "--minimize" in sys.argv
    games = sorted(FG["games"])
    out = {"formula": "min((baseline/ours)^2*100,115) per level; index-weighted mean per game",
           "baseline1_rhae": 58, "minimized": minimize, "games": {}}
    scores = []
    for g in games:
        wd = SCR / (f"full_{g}" if (SCR / f"full_{g}").exists() else f"agent_{g}")
        try:
            r = score_game(g, wd, minimize=minimize)
            out["games"][g] = r; scores.append(r["score"])
            print(f"{g:6} score={r['score']:6.1f}  ours={r['our_actions']} vs base={r['baseline_actions']}", flush=True)
        except Exception as e:
            print(f"{g:6} ERROR {e}", flush=True)
    if scores:
        out["mean_score"] = round(sum(scores) / len(scores), 2)
        out["n_scored"] = len(scores)
        out["n_above_human"] = sum(1 for s in scores if s >= 100)
        out["n_above_baseline1"] = sum(1 for s in scores if s >= 58)
        print(f"\nMEAN score {out['mean_score']} over {len(scores)} games | "
              f">=100 (human): {out['n_above_human']} | >=58 (baseline1): {out['n_above_baseline1']}")
    tag = "_min" if minimize else ""
    json.dump(out, open(RES / f"arc3_rhae{tag}.json", "w"), indent=2, sort_keys=True)
    print("wrote", f"arc3_rhae{tag}.json")


if __name__ == "__main__":
    main()
