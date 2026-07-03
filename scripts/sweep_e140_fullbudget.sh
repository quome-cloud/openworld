#!/usr/bin/env bash
# E140 -- replicate-and-beat baseline1 (arXiv 2605.05138), FULL BUDGET, source-free, from scratch.
# The prior codex source-free number was capped on BOTH axes: 45 min/game AND codex DEFAULT reasoning.
# E140 removes both caps: xhigh reasoning + a GENEROUS per-game backstop (the agent self-paces and
# exits when done/stuck, like baseline1; the cap is only a hung-process safety net, not a leash) +
# all 25 games from scratch. Banks every clean, replay-verified, OpenWorld-round-tripped gain.
#
#   ARM=codex  bash scripts/sweep_e140_fullbudget.sh   # gpt-5.5, apples-to-apples with baseline1
#   ARM=claude bash scripts/sweep_e140_fullbudget.sh   # opus-4-8 @ max, our SOTA attempt
#   env: POOL(2)  PER_AGENT_S(14400=4h backstop)  ROUNDS(2)
set -uo pipefail
ROOT=/Users/jim/Desktop/openworld
PY=/Users/jim/.arcv/bin/python
ARM="${ARM:-codex}"
POOL="${POOL:-2}"
PER_AGENT_S="${PER_AGENT_S:-14400}"      # 4h backstop; agents self-terminate well before this
ROUNDS="${ROUNDS:-2}"
GAMES="${GAMES:-ar25 bp35 cd82 cn04 dc22 ft09 g50t ka59 lf52 lp85 ls20 m0r0 r11l re86 s5i5 sb26 sc25 sk48 sp80 su15 tn36 tr87 tu93 vc33 wa30}"

if [ "$ARM" = "codex" ]; then
  RUNNER="$ROOT/scripts/run_arc_agent_sandbox_codex.sh"; PREFIX="sbcodex_"
  ARCH="$ROOT/experiments/results/arc3_fullgame_sourcefree_codex.json"
  export REASONING="${REASONING:-xhigh}"
else
  RUNNER="$ROOT/scripts/run_arc_agent_sandbox.sh"; PREFIX="sb_"
  ARCH="$ROOT/experiments/results/arc3_fullgame_sourcefree.json"
  export EFFORT="${EFFORT:-max}"
fi

full_sf() { "$PY" -c "import json;d=json.load(open('$ARCH')).get('per_game',{}).get('$1',{});print(1 if (d.get('win') and d.get('levels',0)>=d.get('win')) else 0)" 2>/dev/null || echo 0; }

echo "[e140] START $(date) ARM=$ARM reasoning/effort=${REASONING:-}${EFFORT:-} per_agent=${PER_AGENT_S}s rounds=$ROUNDS pool=$POOL"
for r in $(seq 1 "$ROUNDS"); do
  echo "[e140] === round $r/$ROUNDS $(date) ==="
  for g in $GAMES; do
    if [ "$(full_sf "$g")" = 1 ]; then echo "[e140] $g already full -- skip"; continue; fi
    while [ "$(jobs -rp | wc -l | tr -d ' ')" -ge "$POOL" ]; do sleep 10; done
    echo "[e140] launch $g $(date '+%H:%M:%S')"
    ( timeout_bg() { "$@" & local p=$!; ( sleep "$PER_AGENT_S"; kill -TERM "$p" 2>/dev/null; sleep 15; kill -KILL "$p" 2>/dev/null ) & local w=$!; wait "$p"; kill "$w" 2>/dev/null; }
      timeout_bg bash "$RUNNER" "$g" ) > "$ROOT/scratch_arc/${PREFIX}${g}.out" 2>&1 &
  done
  wait
  echo "[e140] round $r done $(date); banking ${PREFIX} gains"
  SF_WD_PREFIX="$PREFIX" SF_ARCH="$ARCH" "$PY" "$ROOT/scripts/autobank_sourcefree.py" 2>&1 | grep -iE "banked:|sf-bank" || true
  "$PY" -c "import json;a=json.load(open('$ARCH'));print(f'[e140] $ARM source-free now: {a[\"n_full_games\"]}/25 full, {a[\"total_levels\"]}/183 levels')"
done
echo "[e140] DONE $(date)"
