#!/usr/bin/env bash
# E137 Codex twin: schema-conditioned final-level source-free solver.
#
# This mirrors run_arc_agent_schema.sh, but uses Codex/gpt-5.5 and seeds from
# Codex's own source-free archive. The old Codex ablation ran a thinner prompt
# with reasoning disabled; this runner gives it the same schema packet/tooling
# surface as the Claude E137 runner while preserving source-free banking/audit.
set -o pipefail
GAME="$1"
ROOT="/Users/jim/Desktop/openworld"
AGENT_PY="/Users/jim/.pyenv/versions/3.14.6/bin/python"
CODEX="/Users/jim/.local/bin/codex"
MODEL="${MODEL:-gpt-5.5}"
REASONING="${REASONING:-high}"
REASONING_SUMMARY="${REASONING_SUMMARY:-auto}"
WD="$ROOT/scratch_arc/sccodex_$GAME"
mkdir -p "$WD"

cp "$ROOT/experiments/arc3_sandbox.py" "$WD/"
cp "$ROOT/experiments/e125/objstate.py" "$WD/"
cp "$ROOT/experiments/e133/ewm_toolkit.py" "$WD/"
cp "$ROOT/experiments/e134/perceptors.py" "$WD/"
cp "$ROOT/experiments/e134/composite.py" "$WD/"
cp "$ROOT/experiments/e137/schema_induction.py" "$WD/"
cp "$ROOT/experiments/e137/extract_demos.py" "$WD/"

ARCH="$ROOT/experiments/results/arc3_fullgame_sourcefree_codex.json"
"$AGENT_PY" - "$ARCH" "$GAME" "$WD/frontier.json" <<'PY' || true
import json, sys
arch_path, game, out = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    a = json.load(open(arch_path))
    acts = a.get("solutions", {}).get(game) or []
    pg = a.get("per_game", {}).get(game, {})
    if acts:
        json.dump({"game": game, "actions": acts, "levels": int(pg.get("levels", 0)),
                   "win": int(pg.get("win", 0))}, open(out, "w"))
        sys.exit(0)
except Exception:
    pass
sys.exit(1)
PY
[ -f "$WD/frontier.json" ] || echo '{"actions":[],"levels":0,"win":0}' > "$WD/frontier.json"

cd "$WD"
"$AGENT_PY" extract_demos.py "$GAME" frontier.json schema_packet.json \
  --priority "ka59,su15,bp35,dc22,g50t,wa30" > "$WD/schema.out" 2> "$WD/schema.err" || true

N=$("$AGENT_PY" -c "import json;print(json.load(open('frontier.json')).get('levels',0))")
W=$("$AGENT_PY" -c "import json;print(json.load(open('frontier.json')).get('win',0))")

cat > "$WD/TASK.md" <<TASK
You are solving ARC-AGI-3 game **$GAME**, SOURCE-FREE. Do not read game code or import arc_agi.
You already have a replay-verified Codex source-free frontier at **level $N of $W** in \`frontier.json\`.

This is E137: CROSS-LEVEL PROCEDURAL SCHEMA INDUCTION. Your first input is \`schema_packet.json\`.
It was built only by replaying your own prior Codex source-free actions and observing frames/level counters.

Run python with: $AGENT_PY

Execution discipline:
- Write Python scripts to files in this directory and run them with \`$AGENT_PY script.py\`.
- Use only the public SandboxGame API: \`reset()\`, \`step(a)\`, and \`step(6,x,y)\`.
- There is no \`g.replay\`, no \`hard_reset\`, and no game source. If you need replay, write a tiny helper:
      def replay(g, actions):
          g.reset()
          for a in actions:
              g.step(6, a[1], a[2]) if a[0] == 6 else g.step(a[0])
- Keep one SandboxGame instance and reuse reset()+replay. Avoid multiprocessing.

Required workflow:
1. Read \`schema_packet.json\`. Inspect \`solved_level_demos\` and \`candidate_schemas\`.
2. Before free-form solving, choose or repair a schema that explains the solved demos. Treat the demos as
   within-game training examples: ARC levels escalate the same procedure.
3. Replay to the frontier:
      import json
      from arc3_sandbox import SandboxGame
      fr = json.load(open("frontier.json"))["actions"]
      g = SandboxGame("$GAME")
      replay(g, fr)
4. Bind the schema roles on the frontier frame using:
      from objstate import object_state, state_key
      from ewm_toolkit import plan_in_model, WorldSim, salient_clicks, _act, _replay_to
      from ewm_toolkit import composite_key, select_lens, LENSES
   Use composite_key(frame), not a single object lens, when comparing states.
5. Execute the instantiated procedure. Use small source-free probes only to bind uncertain roles.
   If the level does not rise, explain which schema assumption failed, repair it from the counterexample,
   and try again. Do not restart from generic search until schema attempts are exhausted.
6. Save every deeper frontier immediately:
      solved.json = {"game":"$GAME","actions":[...full actions from reset...],"levels":M,"win":$W}
   Actions are [a] or [6,x,y]. A clean deeper solve will be audit/replay/OpenWorld banked.

Priority: getting any of ka59/su15/bp35/dc22/g50t/wa30 to full moves source-free toward beating 15/25.
Persist. The goal is not a beautiful theory; it is one more level-up from the current Codex frontier.
TASK

STARTED=$(date +%s)
"$CODEX" exec --dangerously-bypass-approvals-and-sandbox --skip-git-repo-check -m "$MODEL" -C "$WD" \
  -c "model_reasoning_effort=\"$REASONING\"" \
  -c "model_reasoning_summary=\"$REASONING_SUMMARY\"" \
  "$(cat TASK.md)" > "$WD/agent.log" 2>&1
RC=$?
ENDED=$(date +%s)

cat > "$WD/meta.json" <<META
{"game":"$GAME","method":"schema-frontier-codex","model":"$MODEL","source_free":true,
 "reasoning":"$REASONING","reasoning_summary":"$REASONING_SUMMARY",
 "archive":"experiments/results/arc3_fullgame_sourcefree_codex.json",
 "fairness":"process-isolated SandboxGame + same E137 schema packet/tooling as Claude + autobank audit",
 "audit_dir":"scratch_arc/sccodex_$GAME","started":$STARTED,"ended":$ENDED,"exit_code":$RC}
META
echo "schema-codex agent finished for $GAME (was $N/$W, rc=$RC)"
"$AGENT_PY" "$ROOT/scripts/capture_arc_run.py" "$GAME" "$WD" schema-frontier-codex run_arc_agent_schema_codex.sh || true
