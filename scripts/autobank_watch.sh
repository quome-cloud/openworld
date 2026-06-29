#!/usr/bin/env bash
# Overnight auto-bank watcher: every 3 min, replay-verify + commit any scratch full-game
# progress deeper than the banked archive. Pairs with externally-run agent loops (which only
# write scratch solved.json). Deterministic; idempotent; commits only verified gains.
#   nohup caffeinate -i bash scripts/autobank_watch.sh 9 > scratch_arc/autobank_watch.log 2>&1 &
set -uo pipefail
ROOT=/Users/jim/Desktop/openworld
VENV=/Users/jim/.arcv/bin/python
HOURS="${1:-9}"; DEADLINE=$(( $(date +%s) + HOURS*3600 ))
echo "[watch] START $(date) deadline=+${HOURS}h"
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  "$VENV" "$ROOT/scripts/autobank_arc.py" 2>&1 | grep -E '\[autobank\] (.*->|committed|.*error)' || true
  sleep 180
done
echo "[watch] END $(date)"
