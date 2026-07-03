#!/usr/bin/env bash
# SOURCE-FREE full-game solver (AGENT tier), with full dataset capture.
# Claude Code solves an ARC-AGI-3 game through the process-isolated SandboxGame client only. The agent's
# working dir has NO game source and its python cannot import arc_agi -> fair by construction. Every run is
# captured (prompt + structured transcript + timestamps + pinned model/effort) into the arc3_traces dataset.
#
#   Usage: run_arc_agent_sandbox.sh <game> [tier]
#   Env:   MODEL (default claude-opus-4-8), EFFORT (default high), FALLBACK_MODEL (optional)
set -o pipefail   # NOT -u: macOS bash 3.2 errors on empty-array expansion under nounset
GAME="$1"
TIER="${2:-agent}"
ROOT="/Users/jim/Desktop/openworld"
AGENT_PY="/Users/jim/.pyenv/versions/3.14.6/bin/python"   # has numpy, CANNOT import arc_agi
CLAUDE="/Users/jim/.local/bin/claude"
MODEL="${MODEL:-claude-opus-4-8}"                          # pinned for artifact isolation
EFFORT="${EFFORT:-high}"
FALLBACK_MODEL="${FALLBACK_MODEL:-}"
WD_PREFIX="${WD_PREFIX:-sb_}"                              # override to isolate an arm's workspace (e.g. sbfable_)
WD="$ROOT/scratch_arc/${WD_PREFIX}$GAME"
TRACES="$ROOT/experiments/results/arc3_traces"
mkdir -p "$WD" "$TRACES/prompts" "$TRACES/transcripts" "$TRACES/meta" "$TRACES/solutions"
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
TASK

# --- WALL PROTOCOL variant (labeled condition, default OFF): set WALL=1 to append a persistent-workspace
#     + frontier-focus + sim-then-search addendum for games stalled at a deep wall. General methodology
#     only (no game specifics) -- still source-free, still audited. Relabels the tier so captured runs are
#     a distinct condition (rid <game>__agent-wall__...), never mixed with the base-prompt measurement. ---
if [ -n "${WALL:-}" ]; then
  cat >> "$WD/TASK.md" <<'WALLTASK'

WALL PROTOCOL (this game is stalled at a deep level -- work differently):
- Your workspace PERSISTS across sessions. FIRST ACTION: inventory it (ls; read your prior notes,
  toolkits, probe scripts, solved.json) and build on what previous sessions established instead of
  re-deriving it. Maintain NOTES.md (mechanics learned, hypotheses falsified) for the next session.
- FRONTIER FOCUS: immediately replay solved.json to your deepest level; spend this ENTIRE session on
  the NEXT uncompleted level. Do not spend budget re-verifying earlier levels beyond that one replay.
- WHEN THE LEVEL RESISTS direct probing: stop hand-probing the slow real env. Build a VERIFIED
  simulator of THIS level (your predict(frame, action) must reproduce every held-out observed
  transition exactly), then SEARCH IT OFFLINE at scale -- batched BFS/beam/random-restart over action
  macros, thousands of rollouts -- and replay-verify each candidate winner in the real env before
  trusting it. Walls here have historically fallen to faithful-sim search, not to more manual probes.
- Wins are typically ordered PROCEDURES (multi-step protocols): search for sequences that change
  persistent state (things opened/held/toggled/moved), not just immediate frame deltas.
WALLTASK
  TIER="agent-wall"
fi

# --- expert-panel strategy lens (Bayesian-experts router tier): when a game is STUCK the orchestrator
#     sets EXPERT=<name|index> to inject a DIFFERENT framing instead of repeating the same prompt.
#     Source-free (a general hypothesis lens, not an answer); the run is still audited + env-verified. ---
if [ -n "$EXPERT" ]; then
  "$AGENT_PY" -c "import sys; sys.path.insert(0,'$ROOT/scripts'); import arc_experts as e; sys.stdout.write(e.task_addendum('$EXPERT'))" >> "$WD/TASK.md" 2>/dev/null
  TIER="expert"
fi

# --- dataset capture: run id, prompt, timestamps, transcript, meta sidecar ---
RID=$("$AGENT_PY" -c "import sys; sys.path.insert(0,'$ROOT/scripts'); import capture_lib as c; print(c.run_id('$GAME','$TIER'))")
PROMPT_FILE="$TRACES/prompts/$RID.md"
TRANSCRIPT="$TRACES/transcripts/$RID.jsonl"
cp "$WD/TASK.md" "$PROMPT_FILE"
STARTED=$("$AGENT_PY" -c "import sys; sys.path.insert(0,'$ROOT/scripts'); import capture_lib as c; print(c.iso_now())")

cd "$WD"
if [ -n "$FALLBACK_MODEL" ]; then
  "$CLAUDE" -p "$(cat TASK.md)" --model "$MODEL" --effort "$EFFORT" --fallback-model "$FALLBACK_MODEL" \
    --output-format stream-json --verbose --dangerously-skip-permissions > "$TRANSCRIPT" 2> "$WD/agent.err"
else
  "$CLAUDE" -p "$(cat TASK.md)" --model "$MODEL" --effort "$EFFORT" \
    --output-format stream-json --verbose --dangerously-skip-permissions > "$TRANSCRIPT" 2> "$WD/agent.err"
fi
RC=$?
ENDED=$("$AGENT_PY" -c "import sys; sys.path.insert(0,'$ROOT/scripts'); import capture_lib as c; print(c.iso_now())")

# snapshot THIS run's produced solution trace (immutable per-run record; archive holds only the latest best)
[ -f "$WD/solved.json" ] && cp "$WD/solved.json" "$TRACES/solutions/$RID.json"

# write the meta sidecar (the finalizer merges it with the verified outcome later)
"$AGENT_PY" - "$GAME" "$TIER" "$RID" "$MODEL" "$EFFORT" "$FALLBACK_MODEL" "$STARTED" "$ENDED" "$RC" "$PROMPT_FILE" "$TRANSCRIPT" <<'PY'
import sys, json, os
sys.path.insert(0, "/Users/jim/Desktop/openworld/scripts")
import capture_lib as c
from audit_sandbox import audit_knowledge
game, tier, rid, model, effort, fb, started, ended, rc, pf, tr = sys.argv[1:13]
summ = c.summarize_transcript(tr)
prompt_text = open(pf, errors="ignore").read()
# knowledge audit: were the agent's loaded memory notes / CLAUDE.md free of source-DERIVED content?
mem_dir = "/Users/jim/.claude/projects/-Users-jim-Desktop-openworld/memory"
kfind = audit_knowledge(memory_dir=mem_dir, claude_md="/Users/jim/Desktop/openworld/CLAUDE.md")
rec = {
  "run_id": rid, "game": game, "tier": tier, "method": "live-coding-agent-sandbox",
  "source_free": True, "fairness": "by-construction (process-isolated SandboxGame)",
  "audit_dir": f"scratch_arc/{os.environ.get('WD_PREFIX','sb_')}{game}", "audit_mode": "strict",
  "knowledge_audit": {"clean": (kfind == []), "findings": kfind,
                      "scanned": ["memory/*.md", "CLAUDE.md"]},
  "memory_tainted": (kfind != []),
  "started_at": started, "ended_at": ended, "exit_code": int(rc),
  "model_config": c.model_config(requested_model=model, effort=effort,
                                 fallback_model=(fb or None), summary=summ),
  "prompt_file": f"prompts/{rid}.md", "prompt": c.prompt_stats(prompt_text),
  "solution_file": f"solutions/{rid}.json",
  "transcript_file": f"transcripts/{rid}.jsonl", "transcript_sha256": c.sha256_file(tr),
  "transcript": {k: summ.get(k) for k in ("session_id","num_turns","n_messages","n_tool_calls",
                 "tool_calls_by_name","n_text_blocks","n_thinking_blocks","n_user_msgs","cost_usd",
                 "tokens","usage","is_error","api_error_status","duration_ms","duration_api_ms","ttft_ms")},
  "host": c.host_info(), "git": c.git_info(),
  "pipeline": c.file_provenance([
    "/Users/jim/Desktop/openworld/scripts/run_arc_agent_sandbox.sh",
    "/Users/jim/Desktop/openworld/experiments/arc3_sandbox.py",
    "/Users/jim/Desktop/openworld/scripts/audit_sandbox.py"]),
  "benchmark": c.BENCHMARK, "dataset_version": c.DATASET_VERSION,
  "outcome": None,   # filled by the OpenWorld-verified banker/finalizer
}
c.write_meta(rid, rec)
print(f"[capture] meta written for {rid}: model={summ.get('model')} effort={effort} "
      f"turns={summ.get('num_turns')} tools={summ.get('n_tool_calls')} cost=${summ.get('cost_usd')}")
PY

echo "sandbox agent finished for $GAME (rid=$RID, rc=$RC)"
