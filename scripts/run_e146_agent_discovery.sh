#!/usr/bin/env bash
# E146 discovery adapter -- Claude EWM coding agent.
#
# Called by experiments/e146/retrieve_discover_controller.py through --discovery-command.
# Required environment (the E146 contract, same as run_e146_judge_schema_discovery.sh):
#   E146_GAME, E146_FRONTIER, E146_STAGE_DIR, E146_SOLVED_OUT
#
# Why this adapter exists: the deterministic cold-start primitives crack 0 new levels
# (E131/E132 -- wins are goal-as-procedure, not reachable by blind macro search). The only
# solver that has broken the discovery wall source-free is the live coding agent (16/25 on
# the opus arm). This adapter runs that agent seeded from the E146 frontier; if it writes a
# deeper solved.json the E146 parent replay-verifies it and writes it into episodic memory.
# Still source-free: the workspace has only the SandboxGame client + the frontier trace.
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
AGENT_PY="${AGENT_PY:-$(command -v python3 || command -v python)}"
CLAUDE="${CLAUDE:-$(command -v claude || echo "$HOME/.local/bin/claude")}"
MODEL="${MODEL:-claude-opus-4-8}"
EFFORT="${EFFORT:-max}"
AGENT_TIMEOUT_S="${AGENT_TIMEOUT_S:-7200}"     # bound one discovery stage (the E140 fair-run scale)

: "${E146_GAME:?missing E146_GAME}"
: "${E146_FRONTIER:?missing E146_FRONTIER}"
: "${E146_STAGE_DIR:?missing E146_STAGE_DIR}"
: "${E146_SOLVED_OUT:?missing E146_SOLVED_OUT}"

if [[ ! -x "$CLAUDE" ]] && ! command -v "$CLAUDE" >/dev/null 2>&1; then
  echo "claude CLI not found; set CLAUDE=/path/to/claude" >&2
  exit 127
fi

GAME="$E146_GAME"
WD="$E146_STAGE_DIR/agent_discovery"
mkdir -p "$WD"

cp "$ROOT/experiments/arc3_sandbox.py" "$WD/"
cp "$E146_FRONTIER" "$WD/frontier.json"

cd "$WD"

# Replay the frontier once to learn the current depth + win target (source-free: acting only).
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
You are solving the interactive ARC-AGI-3 game **$GAME**, SOURCE-FREE, using ONLY the SandboxGame
client in \`arc3_sandbox.py\`. You have NO access to the game's source code; do not read game code,
do not import arc_agi. Discover dynamics by acting and infer the win condition from observed frames.

You start from a replay-verified frontier at **level $N of $W**: \`frontier.json\` holds the exact
action list from reset that reaches it. This is the E146 discovery fallback -- episodic memory has
no continuation from here, so your job is to push AT LEAST ONE level deeper.

Workflow:
1. Replay \`frontier.json\` (x=col,y=row for click action 6) and study the level-$((N)) board.
2. Explore source-free: build whatever probe scripts / state graphs / world models you need in this
   directory. Wins are usually ordered PROCEDURES, not score states -- infer the protocol.
3. THE MOMENT \`levels\` increases past $N, write
   \`solved.json = {"game":"$GAME","actions":[...full actions from reset...],"levels":M,"win":$W}\`
   with the COMPLETE action list from reset (frontier actions + your continuation), then keep going
   and rewrite it on each further gain.
4. Keep solved.json fresh -- the E146 parent replay-verifies it from a clean reset; only verified
   deeper traces enter episodic memory.
TASK

# Run the agent bounded; harvest whatever solved.json exists even on timeout.
( "$CLAUDE" -p "$(cat TASK.md)" --model "$MODEL" --effort "$EFFORT" \
    --output-format stream-json --verbose --dangerously-skip-permissions > agent.log 2> agent.err ) &
apid=$!
t0=$(date +%s)
while kill -0 "$apid" 2>/dev/null; do
  sleep 15
  if [ $(( $(date +%s) - t0 )) -ge "$AGENT_TIMEOUT_S" ]; then
    kill -TERM "$apid" 2>/dev/null
    echo "[E146-agent] timed out after ${AGENT_TIMEOUT_S}s; harvesting best-so-far" >> agent.log
    break
  fi
done
wait "$apid" 2>/dev/null || true

if [[ -f "$WD/solved.json" ]]; then
  mkdir -p "$(dirname "$E146_SOLVED_OUT")"
  cp "$WD/solved.json" "$E146_SOLVED_OUT"
fi
