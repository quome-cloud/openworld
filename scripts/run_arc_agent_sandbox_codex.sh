#!/usr/bin/env bash
# SOURCE-FREE full-game solver, CODEX (gpt-5.5) variant -- the model-ablation twin of
# run_arc_agent_sandbox.sh. codex solves an ARC-AGI-3 game through the process-isolated SandboxGame client
# ONLY: its working dir (scratch_arc/sbcodex_$GAME) holds ONLY arc3_sandbox.py (no game source) and the
# agent python cannot import arc_agi -> fair by construction, same as the Claude run. Every banked solve is
# AUDITED for source access (autobank_sourcefree GATE 1) exactly like the Claude pipeline, so this is an
# apples-to-apples SOURCE-FREE comparison (Claude full-access + audit  <->  codex full-access + audit).
#   Usage: run_arc_agent_sandbox_codex.sh <game>
#   Env:   MODEL (default gpt-5.5)
set -o pipefail
GAME="$1"
ROOT="/Users/jim/Desktop/openworld"
AGENT_PY="/Users/jim/.pyenv/versions/3.9.18/bin/python"   # has numpy, CANNOT import arc_agi
CODEX="/Users/jim/.local/bin/codex"
MODEL="${MODEL:-gpt-5.5}"
WD="$ROOT/scratch_arc/sbcodex_$GAME"
mkdir -p "$WD"
cp "$ROOT/experiments/arc3_sandbox.py" "$WD/"             # the ONLY harness the agent gets (no source)
[ -f "$WD/solved_best.json" ] && cp "$WD/solved_best.json" "$WD/solved.json"   # best-keeper seed

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
If solved.json already exists it is your best-so-far -- CONTINUE from it and push DEEPER, never regress.
TASK

cd "$WD"
STARTED=$(date +%s)
# codex with full shell access (parity with the Claude runner's --dangerously-skip-permissions); the
# autobank audit is what enforces source-freeness, identically for both models.
"$CODEX" exec --dangerously-bypass-approvals-and-sandbox --skip-git-repo-check -m "$MODEL" -C "$WD" \
  "$(cat TASK.md)" > "$WD/agent.log" 2>&1
RC=$?
ENDED=$(date +%s)

# minimal provenance sidecar (the banker is the source-of-truth gate; this is just a record)
cat > "$WD/meta.json" <<META
{"game":"$GAME","method":"live-coding-agent-sandbox-codex","model":"$MODEL","source_free":true,
 "fairness":"by-construction (process-isolated SandboxGame) + autobank audit",
 "audit_dir":"scratch_arc/sbcodex_$GAME","started":$STARTED,"ended":$ENDED,"exit_code":$RC}
META
echo "sandbox-codex agent finished for $GAME (rc=$RC, ${ENDED}-${STARTED}s)"
