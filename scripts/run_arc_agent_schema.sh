#!/usr/bin/env bash
# E137: schema-conditioned final-level source-free solver.
#
# Builds a schema_packet.json from the agent's own banked source-free solution:
# replay solved levels -> segment level-up demos -> induce cross-level procedural
# schemas. The live agent must validate/repair a schema before probing the
# frontier. Banking/audit stays identical to the other source-free runners.
set -o pipefail
GAME="$1"
ROOT="/Users/jim/Desktop/openworld"
AGENT_PY="/Users/jim/.pyenv/versions/3.14.6/bin/python"
CLAUDE="/Users/jim/.local/bin/claude"
MODEL="${MODEL:-claude-opus-4-8}"; EFFORT="${EFFORT:-high}"
WD="$ROOT/scratch_arc/sc_$GAME"
mkdir -p "$WD"

cp "$ROOT/experiments/arc3_sandbox.py" "$WD/"
cp "$ROOT/experiments/e125/objstate.py" "$WD/"
cp "$ROOT/experiments/e133/ewm_toolkit.py" "$WD/"
cp "$ROOT/experiments/e134/perceptors.py" "$WD/"
cp "$ROOT/experiments/e134/composite.py" "$WD/"
cp "$ROOT/experiments/e137/schema_induction.py" "$WD/"
cp "$ROOT/experiments/e137/extract_demos.py" "$WD/"

ARCH="$ROOT/experiments/results/arc3_fullgame_sourcefree.json"
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
You already have a replay-verified source-free frontier at **level $N of $W** in \`frontier.json\`.

This is E137: CROSS-LEVEL PROCEDURAL SCHEMA INDUCTION. Your first input is \`schema_packet.json\`.
It was built only by replaying your own prior source-free actions and observing frames/level counters.

Run python with: $AGENT_PY

Required workflow:
1. Read \`schema_packet.json\`. Inspect \`solved_level_demos\` and \`candidate_schemas\`.
2. Before free-form solving, choose or repair a schema that explains the solved demos. Treat the demos as
   within-game training examples: ARC levels escalate the same procedure.
3. Replay to the frontier:
      import json
      from arc3_sandbox import SandboxGame
      fr = json.load(open("frontier.json"))["actions"]
      g = SandboxGame("$GAME"); g.reset()
      for a in fr: g.step(6,a[1],a[2]) if a[0]==6 else g.step(a[0])
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
Persist. The goal is not a beautiful theory; it is one more level-up from the current frontier.
TASK

"$CLAUDE" -p "$(cat TASK.md)" --model "$MODEL" --effort "$EFFORT" \
  --output-format stream-json --verbose --dangerously-skip-permissions \
  > "$WD/agent.log" 2> "$WD/agent.err"
echo "schema agent finished for $GAME (was $N/$W)"
"$AGENT_PY" "$ROOT/scripts/capture_arc_run.py" "$GAME" "$WD" schema-frontier run_arc_agent_schema.sh || true

