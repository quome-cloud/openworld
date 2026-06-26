#!/usr/bin/env bash
# SOURCE-FREE full-game solver: Claude Code solves an ARC-AGI-3 game through the process-isolated
# SandboxGame client only. The agent's working dir has NO game source and its python cannot import
# arc_agi -> fair by construction. Usage: run_arc_agent_sandbox.sh <game>
set -uo pipefail
GAME="$1"
ROOT="/Users/jim/Desktop/openworld"
AGENT_PY="/Users/jim/.pyenv/versions/3.9.18/bin/python"   # has numpy, CANNOT import arc_agi
WD="$ROOT/scratch_arc/sb_$GAME"
mkdir -p "$WD"
cp "$ROOT/experiments/arc3_sandbox.py" "$WD/"             # the ONLY harness the agent gets (no source)
# best-keeper: seed from the deepest known source-free solution if we have one
[ -f "$WD/solved_best.json" ] && cp "$WD/solved_best.json" "$WD/solved.json"

cat > "$WD/TASK.md" <<TASK
You must FULLY solve the interactive ARC-AGI-3 game **$GAME** -- complete EVERY level -- using ONLY
the SandboxGame client. You have NO access to the game's source code (by design): you must discover
the rules BY ACTING and reason the win condition from the frames you observe.

Run python with: $AGENT_PY   (it has numpy; it CANNOT import arc_agi -- do not try).

Harness (arc3_sandbox.py, already in this directory):
    from arc3_sandbox import SandboxGame
    g = SandboxGame("$GAME")
    g.frame   # 64x64 numpy int array (colors 0-15)
    g.levels  # completed levels;  g.win = total levels;  g.avail = available actions;  g.done
    g.reset()
    g.step(a)        # directional action a in 1..5,7
    g.step(6, x, y)  # ACTION6 = click at column x, row y (0..63)
The env is DETERMINISTIC: replaying actions from reset() reproduces frames -> explore offline, then
replay-verify. Clicks register only on valid sprite cells (try distinct / non-background cells, not (0,0)).

Recipe (the OpenWorld way, SOURCE-FREE):
1. EXPLORE by acting: gather (frame, action, next_frame, levels) transitions; learn what each action does.
2. MODEL: write predict(frame, action) reproducing observed transitions (verify on held-out).
3. GOAL: REASON from the observed frames what raises g.levels at THIS level. Test it.
4. PLAN + ADVANCE per level (each level may add mechanics; re-explore and re-reason); chain until
   g.levels == g.win (the FULL game).
5. SAVE often: write solved.json = {"game":"$GAME","actions":[[1],[6,60,32],...],"levels":N,"win":W}
   reaching your DEEPEST level; update it every time you reach a new deepest level. Each action is
   [a] (directional) or [6,x,y] (click).

DO NOT attempt to read the game source, import arc_agi, or build an env yourself -- it is unavailable
and every run is AUDITED for source access; a tainted run is discarded. Solve it the fair way.
TASK

cd "$WD"
claude -p "$(cat TASK.md)" --dangerously-skip-permissions > "$WD/agent.log" 2>&1
echo "sandbox agent finished for $GAME"
