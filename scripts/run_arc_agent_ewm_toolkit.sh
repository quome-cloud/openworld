#!/usr/bin/env bash
# E133: EWM AGENT + DEEP-PLANNING TOOLKIT (the combination to break SOTA). The focused-final-level EWM
# agent (proven 11/25) is given a toolkit that combines the prior approaches: a deep lookahead planner
# over its OWN synthesized model (E132 plan_in_model), object-relative generalization (E130), and
# salience-guided exploration (E107). The agent reasons the WIN (the what); the planner finds the exact
# action SEQUENCE in its synthesized model (the how) -- the part plain reasoning agents stall on.
# Source-free: workspace holds only arc3_sandbox.py + objstate.py + ewm_toolkit.py + perceptors.py +
# composite.py (solver helpers, no game code) + the agent's OWN banked frontier. Every run is audited +
# replay-verified before banking.
#   Usage: run_arc_agent_ewm_toolkit.sh <game>
#   Env:   MODEL (default claude-opus-4-8), EFFORT (default high)
set -o pipefail
GAME="$1"
ROOT="/Users/jim/Desktop/openworld"
# Agent interpreter: Python 3.14.6 with numpy, CANNOT import arc_agi (structural source-free isolation;
# the arc_agi engine lives only in the ~/.arcv venv used by the sandbox worker). Matches the worker's 3.14.
AGENT_PY="/Users/jim/.pyenv/versions/3.14.6/bin/python"
CLAUDE="/Users/jim/.local/bin/claude"
MODEL="${MODEL:-claude-opus-4-8}"; EFFORT="${EFFORT:-high}"
WD="$ROOT/scratch_arc/ek_$GAME"
mkdir -p "$WD"
cp "$ROOT/experiments/arc3_sandbox.py" "$WD/"                    # env client (no game source)
cp "$ROOT/experiments/e125/objstate.py" "$WD/"                  # object perceptor (solver helper)
cp "$ROOT/experiments/e133/ewm_toolkit.py" "$WD/"              # plan_in_model + WorldSim + salience
cp "$ROOT/experiments/e134/perceptors.py" "$WD/"              # K perception lenses (no game source)
cp "$ROOT/experiments/e134/composite.py" "$WD/"              # composite_key + select_lens (no game source)

# the agent's OWN deepest banked frontier (level N-1) -- prefer the source-free archive's solution.
ARCH="$ROOT/experiments/results/arc3_fullgame_sourcefree.json"
"$AGENT_PY" - "$ARCH" "$GAME" "$WD/frontier.json" <<'PY' || true
import json, sys
arch_path, game, out = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    a = json.load(open(arch_path)); acts = a.get("solutions", {}).get(game) or []
    pg = a.get("per_game", {}).get(game, {})
    if acts:
        json.dump({"game": game, "actions": acts, "levels": int(pg.get("levels", 0)),
                   "win": int(pg.get("win", 0))}, open(out, "w")); sys.exit(0)
except Exception:
    pass
sys.exit(1)
PY
[ -f "$WD/frontier.json" ] || echo '{"actions":[],"levels":0,"win":0}' > "$WD/frontier.json"
N=$("$AGENT_PY" -c "import json;print(json.load(open('$WD/frontier.json')).get('levels',0))")
W=$("$AGENT_PY" -c "import json;print(json.load(open('$WD/frontier.json')).get('win',0))")

# E137 (optional A/B arm): build a cross-level procedural schema packet from the agent's OWN solved
# levels and hand it to the agent as a prior. Gated by E137_SCHEMA=1 so the default sweep is unchanged.
SCHEMA_NOTE=""
if [ "${E137_SCHEMA:-0}" = "1" ]; then
  cp "$ROOT/experiments/e137/schema_induction.py" "$WD/"   # source-free: reads frames/actions only
  cp "$ROOT/experiments/e137/goal_condition.py" "$WD/"
  cp "$ROOT/experiments/e137/extract_demos.py" "$WD/"
  ( cd "$WD" && "$AGENT_PY" extract_demos.py "$GAME" frontier.json schema_packet.json >/dev/null 2>&1 ) || true
  if [ -f "$WD/schema_packet.json" ]; then
    SCHEMA_NOTE="
PRIOR (E137 schema packet) -- \`schema_packet.json\` summarizes your OWN prior solved levels (one demo per
level-up) with leave-one-out-validated, source-free schemas:
  - \`goal_condition_schemas\`: WHAT configuration each level-up achieves (the win condition) -- colours the
    win always contains, colours consumed, whether object count grows/shrinks. READ THESE FIRST: they are
    your win hypothesis for level $((N+1)).
  - \`candidate_schemas\`: the action-shape (kind/signature tails, final-action kind, per-level action budget).
Validate/repair a schema against the demos, instantiate it on the frontier frame, then let plan_in_model
find the exact path. It is a PRIOR, not an oracle -- reality (g.levels) still decides."
  fi
fi

cat > "$WD/TASK.md" <<TASK
You are solving the FINAL level of the interactive ARC-AGI-3 game **$GAME**, SOURCE-FREE (no game code).
You have already reached **level $N of $W** -- the action sequence is in \`frontier.json\` (your own prior
solution). Your ONLY job: discover what raises g.levels from $N to $((N+1)) (and on to $W).

Run python with: $AGENT_PY   (numpy; CANNOT import arc_agi). Replay to the frontier:
    import json; from arc3_sandbox import SandboxGame
    fr = json.load(open("frontier.json"))["actions"]
    g = SandboxGame("$GAME"); g.reset()
    for a in fr: g.step(6,a[1],a[2]) if a[0]==6 else g.step(a[0])   # now at level $N

This is the EXECUTABLE WORLD MODEL recipe with a DEEP-PLANNING TOOLKIT. Use the tools -- they are the
edge over plain reasoning (they automate the tedious search for the exact move sequence):

  from objstate import object_state, state_key      # frame -> {bg, objects[color,size,y,x]}; state_key = a hashable key
  from ewm_toolkit import plan_in_model, WorldSim, salient_clicks, _act, _replay_to
  from ewm_toolkit import composite_key, select_lens, LENSES   # E134 multi-perception composite

Your state key MUST be composite_key(frame) — a single object lens silently drops timers/animation/1-cell indicators that decide the win; select_lens tells you which modality to PLAN in.
$SCHEMA_NOTE

THE METHOD (reason the WHAT; let the planner find the HOW):
1. PERCEIVE in objects, not pixels. Re-read your earlier levels' mechanic -- level $((N+1)) is almost
   certainly that mechanic, HARDER. salient_clicks(object_state(g.frame)["objects"]) ranks the click
   targets worth trying (small/rare sprites) for CLICK games; for directional games the actions are 1-5,7.
2. EXPLORE forward (the env is DETERMINISTIC; replay reproduces frames). Gather (state_key, action,
   next_state_key, levels) transitions. Backtracking by reset()+replay is UNRELIABLE here -- explore
   FORWARD and reach states by replaying from reset().
3. SYNTHESIZE a predict(state_key, action) -> (next_key, levels) that GENERALIZES beyond what you
   observed -- write it OBJECT-RELATIVE ("avatar + action4 -> avatar moves +x"; "click a switch -> its
   gate toggles") so your level-$N model predicts level-$((N+1)) configs. A WorldSim() gives you a
   tabular predict for the explored region; your generalizing predict() is what cracks the new level.
4. REASON the WIN CONDITION for the final level (what object configuration / ordered protocol raises
   g.levels). This is the part only you can do.
5. PLAN: call plan_in_model(predict, start_key, start_levels, actions_of, seen, depth=10, beam=12) to
   DEEP-SEARCH your synthesized model for an action sequence that reaches the win (it returns the full
   plan). depth can be large -- it's pure (no real env). This finds the exact 100-move HOW for you.
6. VERIFY on the REAL env: replay the frontier + execute the plan; g.levels MUST actually rise. If the
   real result disagrees with your model, REFINE predict() (that mismatch is the bug -- fix it) and
   re-plan. NEVER trust the model over reality.
7. CHAIN to the win. SAVE often: write solved.json = {"game":"$GAME","actions":[...full sequence from
   reset...],"levels":M,"win":$W} whenever you reach a new deepest level M. Each action is [a] or [6,x,y].

PERSIST -- this is the most important instruction. A plan_in_model that returns NO win does NOT mean the
level is unsolvable; it means your model does not yet CONTAIN the winning transition. When that happens:
EXPLORE MORE (different salient targets, longer forward walks, different start configs), REFINE predict()
(any sim-vs-real mismatch is the bug -- fix it), form a NEW win hypothesis, and RE-PLAN. Do NOT conclude
the wall is impossible and stop after a few attempts -- the levels that were cracked took 60+ iterations
of explore -> reason -> plan -> verify. Keep iterating until g.levels actually rises or you have
genuinely exhausted DISTINCT hypotheses (many of them). Save solved.json at every new deepest level.

EXECUTION DISCIPLINE: write code to .py FILES, run with $AGENT_PY file.py (single-process; no
multiprocessing). Make ONE SandboxGame, reuse via reset()+replay. DO NOT read game source or import
arc_agi -- every run is AUDITED; a tainted run is discarded.
TASK

cd "$WD"
"$CLAUDE" -p "$(cat TASK.md)" --model "$MODEL" --effort "$EFFORT" \
  --output-format stream-json --verbose --dangerously-skip-permissions \
  > "$WD/agent.log" 2> "$WD/agent.err"
echo "ewm-toolkit agent finished for $GAME (was $N/$W)"
# capture this run into the HF-ready dataset (reuses capture_lib via capture_arc_run.py)
"$AGENT_PY" "$ROOT/scripts/capture_arc_run.py" "$GAME" "$WD" ewm-toolkit run_arc_agent_ewm_toolkit.sh || true
