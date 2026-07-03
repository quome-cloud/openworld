#!/usr/bin/env bash
# E132 hybrid EWM solver over the unsolved walls, source-free, seeded from each
# banked frontier.  Pure compute (no API).  hy_ prefix; banks gains through the autobank gate.
#   caffeinate -i nohup bash scripts/sweep_hybrid.sh > scratch_arc/hy_sweep.log 2>&1 &
#
# Optional args: sweep_hybrid.sh [depth] [beam] [pool]
set -uo pipefail
ROOT=/Users/jim/Desktop/openworld; PY=/Users/jim/.arcv/bin/python
DEPTH="${1:-8}"; BEAM="${2:-8}"; POOL="${3:-3}"
ARCH="$ROOT/experiments/results/arc3_fullgame_sourcefree.json"
GAMES="bp35 dc22 g50t ka59 lf52 ls20 m0r0 r11l s5i5 sk48 sp80 su15 tn36 tu93 vc33 wa30"
echo "[hy-sweep] START $(date) depth=$DEPTH beam=$BEAM pool=$POOL"
for g in $GAMES; do
  while [ "$(jobs -rp | wc -l | tr -d ' ')" -ge "$POOL" ]; do sleep 3; done
  echo "[hy-sweep] launch $g $(date '+%H:%M:%S')"
  "$PY" "$ROOT/experiments/e132_hybrid.py" solve "$g" "$DEPTH" "$BEAM" \
    > "$ROOT/scratch_arc/hy_${g}.log" 2>&1 &
done
wait
echo "[hy-sweep] banking gains through the attestation gate"
SF_WD_PREFIX=hy_ SF_ARCH="$ARCH" "$PY" "$ROOT/scripts/autobank_sourcefree.py" 2>&1 | grep -iE "sf-bank" || true
echo "[hy-sweep] DONE $(date)"
