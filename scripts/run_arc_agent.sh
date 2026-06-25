#!/usr/bin/env bash
# Live OpenWorld coding-agent solver: launch Claude Code to build an executable world model + reason
# about the goal for one ARC-AGI-3 game. Usage: run_arc_agent.sh <game>
set -uo pipefail
GAME="$1"; VENV="/private/tmp/claude-501/-Users-jim-Desktop-openworld/71e8c8de-fcca-4c0d-b13e-d3aae6071546/scratchpad/arcv/bin/python"
ROOT="/Users/jim/Desktop/openworld"; WD="$ROOT/scratch_arc/agent_$GAME"
mkdir -p "$WD"; cp "$ROOT/scratch_arc/agent/arc3_harness.py" "$WD/"
cat > "$WD/TASK.md" <<TASK
You are an autonomous agent solving the interactive ARC-AGI-3 game **$GAME**. Work in this directory.
Run python with: $VENV   (it has the arc_agi package; plain 'python' will NOT work)

Harness (arc3_harness.py, already here):
    from arc3_harness import Game
    g = Game("$GAME"); g.reset()
    g.frame  -> 64x64 numpy int array (colors 0-15);  g.levels, g.win, g.avail (action ints), g.done
    g.step(a)        # directional a in 1..5,7
    g.step(6, x, y)  # ACTION6 = CLICK at column x, row y (0..63)
SUCCESS = raising g.levels. The env is DETERMINISTIC: replaying actions from reset() reproduces frames,
so explore offline then verify by replay.

Recipe (executable world model -- the OpenWorld way):
1. EXPLORE: gather (frame, action, next_frame, levels) transitions; learn what each action does. Clicks
   often work ONLY on specific cells (sprites) -- try distinct / non-background cells, not (0,0).
2. MODEL: write predict(frame, action) reproducing observed transitions exactly (verify on held-out).
3. GOAL: REASON about the win condition -- what raises g.levels? Inspect the board: an agent that moves,
   targets, doors, counters. Form a hypothesis and TEST it. This is the crux.
4. PLAN: find an action sequence that completes a level (deterministic -> search/replay freely).
5. SAVE: when g.levels increases, write solved.json = {"game":"$GAME","actions":[[1],[6,60,32],...],"levels":N}
   (each action is [a] for directional or [6,x,y] for a click).
Think hard about the goal. Iterate until >=1 level is completed, then write solved.json.
TASK
rm -f "$WD/solved.json"
cd "$WD"
claude -p "$(cat TASK.md)" --dangerously-skip-permissions > "$WD/agent.log" 2>&1
echo "agent finished for $GAME; solved.json: $([ -f $WD/solved.json ] && echo yes || echo no)"
