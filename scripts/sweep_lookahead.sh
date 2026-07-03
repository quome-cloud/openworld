#!/usr/bin/env bash
# E131 short-horizon lookahead solver over the unsolved walls, source-free, seeded from each
# banked frontier.  Pure compute (no API).  lh_ prefix; banks gains through the autobank gate.
#   caffeinate -i nohup bash scripts/sweep_lookahead.sh > scratch_arc/lh_sweep.log 2>&1 &
#
# Optional args: sweep_lookahead.sh [budget] [pool] [depth] [beam]
set -uo pipefail
ROOT=/Users/jim/Desktop/openworld; PY=/Users/jim/.arcv/bin/python
BUDGET="${1:-4000}"; POOL="${2:-3}"; DEPTH="${3:-3}"; BEAM="${4:-4}"
ARCH="$ROOT/experiments/results/arc3_fullgame_sourcefree.json"
GAMES="bp35 dc22 g50t ka59 lf52 ls20 m0r0 r11l s5i5 sk48 sp80 su15 tn36 tu93 vc33 wa30"
echo "[lh-sweep] START $(date) budget=$BUDGET pool=$POOL depth=$DEPTH beam=$BEAM"
for g in $GAMES; do
  while [ "$(jobs -rp | wc -l | tr -d ' ')" -ge "$POOL" ]; do sleep 3; done
  echo "[lh-sweep] launch $g $(date '+%H:%M:%S')"
  "$PY" "$ROOT/experiments/e131_lookahead.py" solve "$g" "$BUDGET" "$DEPTH" "$BEAM" \
    > "$ROOT/scratch_arc/lh_${g}.log" 2>&1 &
done
wait
echo "[lh-sweep] banking gains through the attestation gate"
SF_WD_PREFIX=lh_ SF_ARCH="$ARCH" "$PY" "$ROOT/scripts/autobank_sourcefree.py" 2>&1 | grep -iE "sf-bank" || true
echo "[lh-sweep] DONE $(date)"
