#!/usr/bin/env bash
# FULL-GAME live coding-agent solver: Claude Code builds an executable world model and solves EVERY
# level of one ARC-AGI-3 game (chaining level by level to g.win). Usage: run_arc_agent_full.sh <game>
set -uo pipefail
GAME="$1"; VENV="/private/tmp/claude-501/-Users-jim-Desktop-openworld/71e8c8de-fcca-4c0d-b13e-d3aae6071546/scratchpad/arcv/bin/python"
ROOT="/Users/jim/Desktop/openworld"; WD="$ROOT/scratch_arc/full_$GAME"
mkdir -p "$WD"; cp "$ROOT/scratch_arc/agent/arc3_harness.py" "$WD/"
cat > "$WD/TASK.md" <<TASK
You are an autonomous agent that must FULLY solve the interactive ARC-AGI-3 game **$GAME** -- i.e.
complete EVERY level. Work in this directory. Run python with: $VENV  (it has arc_agi; plain 'python'
will NOT work).

Harness (arc3_harness.py, already here):
    from arc3_harness import Game
    g = Game("$GAME"); g.reset()
    g.frame  -> 64x64 numpy int array (colors 0-15);  g.levels (completed), g.win (TOTAL levels), g.avail, g.done
    g.step(a)        # directional a in 1..5,7
    g.step(6, x, y)  # ACTION6 = CLICK at column x, row y (0..63)
SUCCESS = raising g.levels; FULL SUCCESS = g.levels == g.win. The env is DETERMINISTIC: replaying
actions from reset() reproduces frames, so explore offline then verify by replay.

Recipe (executable world model, the OpenWorld way):
1. EXPLORE each level: gather (frame, action, next_frame, levels) transitions; learn what each action
   does (clicks often work ONLY on specific cells -- try distinct/non-background cells, not (0,0)).
2. MODEL: write predict(frame, action) reproducing observed transitions exactly (verify on held-out).
3. GOAL: REASON about what raises g.levels at THIS level (agent, targets, doors, counters). Test it.
4. PLAN + ADVANCE: complete the level, then CONTINUE -- each new level may add mechanics, so
   re-explore and re-reason per level. Keep chaining until g.levels == g.win (the FULL game).
5. SAVE often: write solved.json = {"game":"$GAME","actions":[[1],[6,60,32],...],"levels":N,"win":W}
   with the sequence reaching the DEEPEST level so far. Update it EVERY time you reach a new deepest
   level, so progress is never lost. Each action is [a] (directional) or [6,x,y] (click).
This is a FULL-GAME task: aim for g.levels == g.win. Iterate until the full game is solved or you
genuinely exhaust ideas; always leave solved.json at your deepest verified progress.
TASK
# seed with prior >=1-level solution if available, so the agent starts past level 1
PRIOR="$ROOT/experiments/results/agent_solves/$GAME.json"
[ -f "$PRIOR" ] && cp "$PRIOR" "$WD/solved.json"
cd "$WD"
claude -p "$(cat TASK.md)" --dangerously-skip-permissions > "$WD/agent.log" 2>&1
echo "full-game agent finished for $GAME"
