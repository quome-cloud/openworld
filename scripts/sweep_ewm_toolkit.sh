#!/usr/bin/env bash
# E133 sweep: the EWM agent + deep-planning toolkit across ALL unsolved source-free walls -- cast a wide
# net for a discoverable crack (widen-the-net strategy). Pool 2 (stay under the shared Claude 5-hour
# window). Banks any gain through the attestation gate (ek_ prefix) and refreshes the source-free archive.
#   caffeinate -i nohup bash scripts/sweep_ewm_toolkit.sh > scratch_arc/ek_sweep.log 2>&1 &
#   env: POOL (2)
set -uo pipefail
ROOT=/Users/jim/Desktop/openworld
PY=/Users/jim/.arcv/bin/python
ARCH="$ROOT/experiments/results/arc3_fullgame_sourcefree.json"
POOL="${POOL:-2}"
# unsolved walls, the NOT-yet-run ones first (ka59/wa30 already attempted -> last)
GAMES="bp35 dc22 g50t lf52 ls20 r11l s5i5 sk48 sp80 su15 tn36 vc33 ka59 wa30"

full_sf() { "$PY" -c "import json;d=json.load(open('$ARCH')).get('per_game',{}).get('$1',{});print(1 if (d.get('win') and d.get('levels',0)>=d.get('win')) else 0)" 2>/dev/null || echo 0; }

echo "[ek-sweep] START $(date) pool=$POOL games=$GAMES"
for g in $GAMES; do
  if [ "$(full_sf "$g")" = 1 ]; then echo "[ek-sweep] $g already full -- skip"; continue; fi
  while [ "$(jobs -rp | wc -l | tr -d ' ')" -ge "$POOL" ]; do sleep 10; done
  echo "[ek-sweep] launch $g $(date '+%H:%M:%S')"
  bash "$ROOT/scripts/run_arc_agent_ewm_toolkit.sh" "$g" > "$ROOT/scratch_arc/ek_${g}.out" 2>&1 &
done
wait
echo "[ek-sweep] all agents done $(date); banking ek_ gains through the attestation gate"
SF_WD_PREFIX=ek_ SF_ARCH="$ARCH" "$PY" "$ROOT/scripts/autobank_sourcefree.py" 2>&1 | grep -iE "sf-bank" || true
"$PY" -c "import json;a=json.load(open('$ARCH'));print(f'[ek-sweep] source-free now: {a[\"n_full_games\"]}/25 full, {a[\"total_levels\"]}/{a[\"total_possible\"]} levels')"
echo "[ek-sweep] DONE $(date)"
