#!/usr/bin/env bash
# E137 Codex targeted sweep: schema-conditioned source-free frontier solving.
# Uses Codex's source-free archive and sccodex_ workdirs so results do not mix
# with the Claude E137 sweep except through the same banker/audit contract.
set -uo pipefail
ROOT=/Users/jim/Desktop/openworld
PY=/Users/jim/.arcv/bin/python
ARCH="$ROOT/experiments/results/arc3_fullgame_sourcefree_codex.json"
POOL="${POOL:-2}"
ROUNDS="${ROUNDS:-2}"
DELAY="${DELAY:-600}"
GAMES="${GAMES:-ka59 su15 bp35 dc22 g50t wa30}"

full_sf() {
  "$PY" -c "import json;d=json.load(open('$ARCH')).get('per_game',{}).get('$1',{});print(1 if (d.get('win') and d.get('levels',0)>=d.get('win')) else 0)" 2>/dev/null || echo 0
}

echo "[sccodex-sweep] START $(date) pool=$POOL rounds=$ROUNDS games=$GAMES model=${MODEL:-gpt-5.5} reasoning=${REASONING:-high}"
for r in $(seq 1 "$ROUNDS"); do
  echo "[sccodex-sweep] === round $r/$ROUNDS $(date) ==="
  for g in $GAMES; do
    if [ "$(full_sf "$g")" = 1 ]; then echo "[sccodex-sweep] $g already full -- skip"; continue; fi
    while [ "$(jobs -rp | wc -l | tr -d ' ')" -ge "$POOL" ]; do sleep 5; done
    echo "[sccodex-sweep] launch $g $(date '+%H:%M:%S')"
    bash "$ROOT/scripts/run_arc_agent_schema_codex.sh" "$g" > "$ROOT/scratch_arc/sccodex_${g}.out" 2>&1 &
  done
  wait
  echo "[sccodex-sweep] banking sccodex_ gains"
  SF_WD_PREFIX=sccodex_ SF_ARCH="$ARCH" "$PY" "$ROOT/scripts/autobank_sourcefree.py" 2>&1 | grep -iE "sf-bank" || true
  "$PY" -c "import json;a=json.load(open('$ARCH'));print(f'[sccodex-sweep] codex source-free now: {a[\"n_full_games\"]}/25 full, {a[\"total_levels\"]}/{a[\"total_possible\"]} levels')"
  [ "$r" -lt "$ROUNDS" ] && sleep "$DELAY"
done
echo "[sccodex-sweep] DONE $(date)"
