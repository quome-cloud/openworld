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
AGENT_PY="/Users/jim/.pyenv/versions/3.14.6/bin/python"   # has numpy, CANNOT import arc_agi
CODEX="/Users/jim/.local/bin/codex"
MODEL="${MODEL:-gpt-5.5}"
REASONING="${REASONING:-xhigh}"                          # E140: full-budget reasoning (was codex DEFAULT)
REASONING_SUMMARY="${REASONING_SUMMARY:-auto}"
WD="$ROOT/scratch_arc/sbcodex_$GAME"
mkdir -p "$WD"
cp "$ROOT/experiments/arc3_sandbox.py" "$WD/"             # the ONLY harness the agent gets (no source)
[ -f "$WD/solved_best.json" ] && cp "$WD/solved_best.json" "$WD/solved.json"   # best-keeper seed

cat > "$WD/TASK.md" <<TASK
You must FULLY solve the interactive ARC-AGI-3 game **$GAME** -- complete EVERY level -- using ONLY
the SandboxGame client. You have NO access to the game's source code (by design): you must discover
the rules BY ACTING and reason the win condition from the frames you observe.

Run python with: $AGENT_PY   (it has numpy; it CANNOT import arc_agi -- do not try).

EXECUTION DISCIPLINE (this environment runs your Python from STDIN, with no __main__ file on disk):
- WRITE your code to .py FILES in this directory and run them: \`$AGENT_PY myscript.py\`. Do NOT pipe
  scripts via stdin/heredoc -- re-exec of '<stdin>' and child processes FAIL here.
- Do NOT use multiprocessing / Process / Pool / a ProcessPool -- child processes CANNOT spawn in this
  stdin-run environment; they crash and waste your ENTIRE time budget (this is exactly what stalled
  prior runs). Run SINGLE-PROCESS and SEQUENTIAL -- the env is fast (~0.04 ms/step, 0.6 ms/reset), so a
  single process explores tens of thousands of steps per minute.
- Make ONE SandboxGame and reuse it via reset()+replay; never re-create it in a loop.

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

Recipe (the OpenWorld way, SOURCE-FREE) -- REASON the win; do NOT brute-force:
Random/parallel search does NOT crack these games: a win is an ordered PROCEDURE, not a state score.
Spend your budget UNDERSTANDING the mechanic and reasoning the win condition, then do a SMALL, targeted
search -- not a giant random sweep.
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
  -c "model_reasoning_effort=\"$REASONING\"" -c "model_reasoning_summary=\"$REASONING_SUMMARY\"" \
  "$(cat TASK.md)" > "$WD/agent.log" 2>&1
RC=$?
ENDED=$(date +%s)

# minimal provenance sidecar (the banker is the source-of-truth gate; this is just a record)
cat > "$WD/meta.json" <<META
{"game":"$GAME","method":"live-coding-agent-sandbox-codex","model":"$MODEL","source_free":true,
 "fairness":"by-construction (process-isolated SandboxGame) + autobank audit",
 "audit_dir":"scratch_arc/sbcodex_$GAME","started":$STARTED,"ended":$ENDED,"exit_code":$RC}
META

# --- HF dataset capture (parity with the Claude runner): prompt + transcript + solution + meta into
#     arc3_traces. Codex's exec log is plain text and large, so the transcript is stored gzipped
#     (transcripts/<rid>.codex.log.gz); the structured-format reconciliation is a follow-up. ---
TRACES="$ROOT/experiments/results/arc3_traces"
mkdir -p "$TRACES/prompts" "$TRACES/transcripts" "$TRACES/meta" "$TRACES/solutions"
RID=$("$AGENT_PY" -c "import sys; sys.path.insert(0,'$ROOT/scripts'); import capture_lib as c; print(c.run_id('$GAME','agent-codex'))" 2>/dev/null || echo "${GAME}__agent-codex__$(date -u +%Y-%m-%dT%H-%M-%SZ)")
cp "$WD/TASK.md" "$TRACES/prompts/$RID.md" 2>/dev/null || true
[ -f "$WD/agent.log" ] && gzip -c "$WD/agent.log" > "$TRACES/transcripts/$RID.codex.log.gz"
[ -f "$WD/solved.json" ] && cp "$WD/solved.json" "$TRACES/solutions/$RID.json"
"$AGENT_PY" - "$GAME" "$RID" "$MODEL" "$REASONING" "$STARTED" "$ENDED" "$RC" "$WD" <<'PY' || true
import sys, json, os
g, rid, model, reasoning, started, ended, rc, wd = sys.argv[1:9]
lv = win = 0
try:
    sj = os.path.join(wd, "solved.json")
    if os.path.exists(sj):
        d = json.load(open(sj)); lv = int(d.get("levels", 0)); win = int(d.get("win", 0))
except Exception:
    pass
meta = {"run_id": rid, "game": g, "tier": "agent-codex", "model": model,
        "reasoning_effort": reasoning, "source_free": True,
        "transcript_format": "codex_exec_plaintext_gz",
        "started": int(started), "ended": int(ended), "wall_s": int(ended) - int(started),
        "exit_code": int(rc), "levels": lv, "win": win,
        "experiment": "e140_fullbudget"}
open(f"/Users/jim/Desktop/openworld/experiments/results/arc3_traces/meta/{rid}.json", "w").write(json.dumps(meta, indent=1))
PY
echo "sandbox-codex agent finished for $GAME (rc=$RC, ${ENDED}-${STARTED}s); captured rid=$RID"
