#!/usr/bin/env bash
# E130 SHU-cycle solver over the unsolved walls, source-free, seeded from each banked frontier.
# Pure compute (no API). su_ prefix; banks gains through the autobank gate after the run.
#   caffeinate -i nohup bash scripts/sweep_shu_cycle.sh > scratch_arc/su_sweep.log 2>&1 &
set -uo pipefail
ROOT=/Users/jim/Desktop/openworld; PY=/Users/jim/.arcv/bin/python
BUDGET="${1:-4000}"; POOL="${2:-3}"
ARCH="$ROOT/experiments/results/arc3_fullgame_sourcefree.json"
GAMES="bp35 dc22 g50t ka59 lf52 ls20 m0r0 r11l s5i5 sk48 sp80 su15 tn36 tu93 vc33 wa30"
echo "[su-sweep] START $(date) budget=$BUDGET pool=$POOL"
for g in $GAMES; do
  while [ "$(jobs -rp | wc -l | tr -d ' ')" -ge "$POOL" ]; do sleep 3; done
  echo "[su-sweep] launch $g $(date '+%H:%M:%S')"
  "$PY" "$ROOT/experiments/e130_shu_cycle.py" solve "$g" "$BUDGET" > "$ROOT/scratch_arc/su_${g}.log" 2>&1 &
done
wait
echo "[su-sweep] banking gains through the attestation gate"
SF_WD_PREFIX=su_ SF_ARCH="$ARCH" "$PY" "$ROOT/scripts/autobank_sourcefree.py" 2>&1 | grep -iE "sf-bank" || true
echo "[su-sweep] DONE $(date)"
