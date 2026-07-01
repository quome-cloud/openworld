#!/usr/bin/env bash
# E146 discovery adapter.
#
# Called by scripts/run_e146_retrieve_discover.py through --discovery-command.
# Required environment:
#   E146_GAME, E146_FRONTIER, E146_STAGE_DIR, E146_SOLVED_OUT
#
# The adapter runs a source-free judge/schema Codex worker from the current E146
# frontier. If it writes a deeper solved.json, this script copies it to
# E146_SOLVED_OUT so E146 can replay-verify and write it into local episodic
# memory.
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
AGENT_PY="${AGENT_PY:-$(command -v python3 || command -v python)}"
CODEX="${CODEX:-$(command -v codex || true)}"
MODEL="${MODEL:-gpt-5.5}"
REASONING="${REASONING:-high}"
REASONING_SUMMARY="${REASONING_SUMMARY:-auto}"
N_PROPOSALS="${N_PROPOSALS:-4}"
ARCH="${ARCH:-$ROOT/experiments/results/arc3_fullgame_sourcefree.json}"

: "${E146_GAME:?missing E146_GAME}"
: "${E146_FRONTIER:?missing E146_FRONTIER}"
: "${E146_STAGE_DIR:?missing E146_STAGE_DIR}"
: "${E146_SOLVED_OUT:?missing E146_SOLVED_OUT}"

if [[ -z "$CODEX" ]]; then
  echo "codex CLI not found; set CODEX=/path/to/codex" >&2
  exit 127
fi

GAME="$E146_GAME"
WD="$E146_STAGE_DIR/judge_schema_discovery"
mkdir -p "$WD"

cp "$ROOT/experiments/arc3_sandbox.py" "$WD/"
for helper in \
  "$ROOT/experiments/e125/objstate.py" \
  "$ROOT/experiments/e133/ewm_toolkit.py" \
  "$ROOT/experiments/e134/perceptors.py" \
  "$ROOT/experiments/e134/composite.py" \
  "$ROOT/experiments/e137/goal_condition.py" \
  "$ROOT/experiments/e137/schema_induction.py" \
  "$ROOT/experiments/e137/extract_demos.py" \
  "$ROOT/experiments/e138/judge_schema.py" \
  "$ROOT/experiments/e139/manyworld_semiring.py" \
  "$ROOT/experiments/e139/hybrid_rank.py"; do
  if [[ -f "$helper" ]]; then
    cp "$helper" "$WD/"
  fi
done
cp "$E146_FRONTIER" "$WD/frontier.json"

cd "$WD"
if [[ -f extract_demos.py ]]; then
  "$AGENT_PY" extract_demos.py "$GAME" frontier.json schema_packet.json \
    --priority "ka59,su15,bp35,dc22,g50t,wa30,tu93,m0r0" > schema.out 2> schema.err || true
else
  printf '{"available": false, "reason": "extract_demos.py not present in this branch"}\n' > schema_packet.json
fi

read -r N W < <("$AGENT_PY" - <<'PY'
import json
from arc3_sandbox import SandboxGame

frontier = json.load(open("frontier.json"))
game = SandboxGame(frontier["game"])
try:
    game.reset()
    for action in frontier.get("actions", []):
        if int(action[0]) == 6:
            game.step(6, int(action[1]), int(action[2]))
        else:
            game.step(int(action[0]))
        if game.done:
            break
    print(int(game.levels), int(game.win))
finally:
    game.close()
PY
)

cat > TASK.md <<TASK
You are solving ARC-AGI-3 game **$GAME**, SOURCE-FREE. Do not read game code or import arc_agi.
You have a replay-verified source-free frontier at **level $N of $W** in \`frontier.json\`.

This is E146 discovery fallback: memory retrieval failed, so create new source-free candidate
fragments. Use only the public \`SandboxGame\` API and the source-free schema/judge tools in this
directory.

Required workflow:
1. Replay \`frontier.json\` to inspect the current frame.
2. Read \`schema_packet.json\` if present; it summarizes prior source-free demos.
3. Write at least $N_PROPOSALS structured \`proposal_*.json\` files with:
   \`proposal_id\`, \`schema_id\`, \`goal_schema_id\`, \`hypothesis\`,
   \`role_bindings\`, \`probe_plan\`, \`expected_deltas\`, \`fallback_repairs\`, \`confidence\`.
4. If \`judge_schema.py\` or \`hybrid_rank.py\` are present, use them to rank proposals. If they
   are absent, rank proposals yourself by source-free evidence and short replay probes.
5. Execute small source-free probes. If a probe raises \`levels\`, save:
   \`solved.json = {"game":"$GAME","actions":[...full actions from reset...],"levels":M,"win":$W}\`

The E146 parent will replay-verify \`solved.json\`; only verified deeper traces are written to memory.
TASK

"$CODEX" exec --dangerously-bypass-approvals-and-sandbox --skip-git-repo-check -m "$MODEL" -C "$WD" \
  -c "model_reasoning_effort=\"$REASONING\"" \
  -c "model_reasoning_summary=\"$REASONING_SUMMARY\"" \
  "$(cat TASK.md)" > agent.log 2>&1 || true

if [[ -f "$WD/solved.json" ]]; then
  mkdir -p "$(dirname "$E146_SOLVED_OUT")"
  cp "$WD/solved.json" "$E146_SOLVED_OUT"
fi
