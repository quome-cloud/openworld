"""E120 (per-arm) -- RHAE / action-efficiency of each MODEL ARM's banked ARC-AGI-3 solutions, scored
with the OFFICIAL arc_agi formula and the OFFICIAL per-level human baselines (EnvironmentInfo.
baseline_actions). Companion to e120_rhae.py: same scoring machinery, but it reads each arm's own
source-free workspace (scratch_arc/<prefix><game>/solved.json) so opus vs fable (vs codex) are scored
head-to-head under one formula.

Per level the engine scores  min((baseline_actions / actions_taken)^2 * 100, 115); the per-game aggregate
is the index-weighted mean of level scores (later levels weigh more), and a FULLY-completed game saturates
at 100. We replay each banked solution against the REAL deterministic engine, segment actions per level (a
segment ends when levels_completed increments), and score with arc_agi's own EnvironmentScoreCalculator --
no reimplementation.

POST-HOC SCORING, still source-free: the solutions were FOUND source-free (process-isolated sandbox); here
the engine merely replays them to segment + score, and reads the official human baseline. The score script
runs from the repo ROOT (never inside an arm's source-free workspace) so no game source is written there.

Interpreting the output: both strong arms use FAR fewer moves than the human baseline on every level they
complete, so each completed level pegs the 115 per-level cap and every full game saturates at 100. RHAE
therefore behaves as a COVERAGE-weighted score here (it cannot resolve the arms' move-efficiency gap once
both are past the human baseline) -- exactly the artifact-path-economy caveat in the paper.

Run with the arcv interpreter (has arc_agi); default scores all three arms it finds:
    <arcv>/bin/python experiments/e120_rhae_arms.py                 # sb_ (opus), sbfable_, sbcodex_
    <arcv>/bin/python experiments/e120_rhae_arms.py sb_ sbfable_    # only these prefixes
"""
import json, os, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SCR = ROOT / "scratch_arc"
RES = ROOT / "experiments" / "results"

# 25-game roster (source of truth: arc3_fullgame.json if present, else the fixed list)
try:
    GAMES = sorted(json.load(open(RES / "arc3_fullgame.json"))["games"])
except Exception:
    GAMES = ("ar25 bp35 cd82 cn04 dc22 ft09 g50t ka59 lf52 lp85 ls20 m0r0 r11l re86 s5i5 sb26 sc25 sk48 "
             "sp80 su15 tn36 tr87 tu93 vc33 wa30").split()

# arm prefix -> display label (kept in sync with the E140 workspace prefixes)
ARM_LABELS = {"sb_": "opus", "sbfable_": "fable", "sbcodex_": "codex"}


# --- minimal replay env (inlined from arc3_harness so this has no scratch-dir dependency) ---
def _frame(o):
    a = np.asarray(o.frame)
    return (a[-1] if a.ndim == 3 else a).reshape(64, 64)


class _Game:
    def __init__(self, gid):
        import arc_agi
        from arcengine import GameAction
        self._GA = GameAction
        self._A = {1: GameAction.ACTION1, 2: GameAction.ACTION2, 3: GameAction.ACTION3,
                   4: GameAction.ACTION4, 5: GameAction.ACTION5, 7: GameAction.ACTION7}
        self.env = arc_agi.Arcade().make(gid)
        self.reset()

    def reset(self):
        o = self.env.reset(); self.levels = o.levels_completed; self.win = o.win_levels; self.done = False

    def step(self, action, x=None, y=None):
        o = (self.env.step(self._GA.ACTION6, {"x": int(x), "y": int(y)}) if action == 6
             else self.env.step(self._A[action]))
        if o is None or getattr(o, "frame", None) is None:
            self.done = True; return
        self.levels = o.levels_completed
        self.done = str(o.state) != "GameState.NOT_FINISHED"


def replay_segments(game, actions):
    """Replay in the real env; return (per-level action counts for completed levels, leftover, delta)."""
    g = _Game(game); base = g.levels
    seg, segs, last = 0, [], base
    for s in actions:
        g.step(*s) if isinstance(s, (list, tuple)) else g.step(s)
        seg += 1
        if g.levels > last:
            for _ in range(g.levels - last):
                segs.append(seg); seg = 0        # attribute the whole segment to the first completion
            last = g.levels
        if g.done:
            break
    return segs, seg, g.levels - base


def score_game(prefix, game):
    import arc_agi
    from arc_agi import EnvironmentScoreCalculator
    sp = SCR / f"{prefix}{game}" / "solved.json"
    if not sp.exists():
        return None
    info = arc_agi.Arcade().make(game).environment_info
    baseline = list(info.baseline_actions or [])
    actions = json.load(open(sp)).get("actions", [])
    segs, leftover, reached = replay_segments(game, actions)
    calc = EnvironmentScoreCalculator()
    for i, b in enumerate(baseline):
        if i < len(segs):
            calc.add_level(level_index=i + 1, completed=True, actions_taken=segs[i], baseline_actions=b)
        else:
            calc.add_level(level_index=i + 1, completed=False, actions_taken=0, baseline_actions=b)
    sc = calc.to_score()
    return {"game": game, "win": len(baseline), "reached": reached, "full": reached == len(baseline),
            "our_actions": segs, "baseline_actions": baseline, "score": round(sc.score, 2),
            "level_scores": [round(x, 1) for x in (sc.level_scores or [])]}


def score_arm(prefix):
    rows = {}
    for g in GAMES:
        try:
            r = score_game(prefix, g)
        except Exception as e:
            print(f"  {prefix}{g:6} ERROR {e}", flush=True); continue
        if r is None:
            continue
        rows[g] = r
        print(f"  {prefix}{g:6} score={r['score']:6.1f}  [{'FULL' if r['full'] else str(r['reached'])+'/'+str(r['win'])}]",
              flush=True)
    all_sc = [r["score"] for r in rows.values()]
    full_sc = [r["score"] for r in rows.values() if r["full"]]
    return {
        "label": ARM_LABELS.get(prefix, prefix.strip("_")), "prefix": prefix, "games": rows,
        "n_scored": len(all_sc), "n_full": len(full_sc),
        "mean_all": round(sum(all_sc) / len(all_sc), 2) if all_sc else None,
        "mean_full": round(sum(full_sc) / len(full_sc), 2) if full_sc else None,
        "n_above_human": sum(1 for s in all_sc if s >= 100),
        "n_above_baseline1": sum(1 for s in all_sc if s >= 58),
    }


def main():
    os.chdir(ROOT)                             # environment_files/ lives here; never chdir into a source-free arm wd
    prefixes = sys.argv[1:] or ["sb_", "sbfable_", "sbcodex_"]
    out = {"formula": "min((baseline/ours)^2*100,115) per level; index-weighted mean per game; full game -> 100",
           "baseline1_rhae": 58, "arms": {}}
    for p in prefixes:
        print(f"[{ARM_LABELS.get(p, p)}]  ({p})", flush=True)
        a = score_arm(p)
        if a["n_scored"] == 0:
            print(f"  (no {p}<game>/solved.json workspaces found -- skipped)", flush=True); continue
        out["arms"][a["label"]] = a
        print(f"  MEAN RHAE all={a['mean_all']} (n={a['n_scored']})  full={a['mean_full']} (n={a['n_full']})  "
              f">=100:{a['n_above_human']}  >=58:{a['n_above_baseline1']}\n", flush=True)
    json.dump(out, open(RES / "arc3_rhae_arms.json", "w"), indent=2, sort_keys=True)
    print("wrote arc3_rhae_arms.json")


if __name__ == "__main__":
    main()
