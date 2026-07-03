#!/usr/bin/env bash
# FULL-GAME live coding-agent solver -- CODEX (gpt-5.5) variant of run_arc_agent_full.sh. `codex exec` builds an
# executable world model and solves EVERY level of one ARC-AGI-3 game (chaining level by level to g.win), the
# SAME harness/recipe as the Claude agent_full_game run (24/25) but driven by OpenAI codex instead of `claude -p`.
# Separate workdir (scratch_arc/codex_$GAME) + log so it does NOT collide with the Claude-banked full_$GAME state.
# Usage: run_arc_agent_codex.sh <game> [model]
set -uo pipefail
GAME="$1"; MODEL="${2:-gpt-5.5}"; VENV="/Users/jim/.arcv/bin/python"
ROOT="/Users/jim/Desktop/openworld"; WD="$ROOT/scratch_arc/codex_$GAME"
CODEX="$HOME/.local/bin/codex"
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
RESUME: if solved.json ALREADY EXISTS in this directory, READ it first and CONTINUE from its deepest
level -- replay its "actions" from reset() to restore that state, then push DEEPER. Do NOT restart at
level 0 or regress its "levels"; only ever update solved.json to a strictly deeper result.
TASK
# Per-game hints (same mechanism as the Claude runner): appended so every session inherits the mandate.
[ -f "$WD/HINTS_$GAME.md" ] && { printf '\n\n---\nGAME-SPECIFIC HINTS (read these FIRST, they encode hard-won mechanics):\n\n' >> "$WD/TASK.md"; cat "$WD/HINTS_$GAME.md" >> "$WD/TASK.md"; }
cd "$WD"
"$CODEX" exec --dangerously-bypass-approvals-and-sandbox --skip-git-repo-check -m "$MODEL" -C "$WD" "$(cat TASK.md)" > "$WD/agent.log" 2>&1
echo "codex full-game agent finished for $GAME"
