#!/usr/bin/env bash
# Adapted from Jim's run_arc_agent.sh for the researchy server environment.
# Usage: bash scripts/run_arc_agent_forge.sh <game>
set -uo pipefail
GAME="$1"
VENV="python3"           # system python3 has arc_agi installed
ROOT="/data/doh/teams/researchy/work/openworld"
WD="$ROOT/scratch_arc/agent_$GAME"
mkdir -p "$WD"
cp "$ROOT/scratch_arc/agent/arc3_harness.py" "$WD/"

cat > "$WD/TASK.md" <<TASK
You are an autonomous agent solving the interactive ARC-AGI-3 game **$GAME**. Work in this directory: $WD

Run python with: $VENV  (it has the arc_agi package)

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
echo "[$(date -u +%H:%M:%S)] Launching claude agent for $GAME in $WD" | tee agent_launch.log
claude -p "$(cat TASK.md)" --dangerously-skip-permissions >> "$WD/agent.log" 2>&1
EXIT=$?
if [ -f "$WD/solved.json" ]; then
    echo "[$(date -u +%H:%M:%S)] SUCCESS: solved.json written" | tee -a agent_launch.log
    cat "$WD/solved.json"
else
    echo "[$(date -u +%H:%M:%S)] FAILED: no solved.json (exit=$EXIT)" | tee -a agent_launch.log
fi
