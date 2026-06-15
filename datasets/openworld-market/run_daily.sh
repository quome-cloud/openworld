#!/bin/bash
# Unattended daily forward-test for the E50 trading world model.
# Runs the signal (logs today's pick + scores past picks) into daily.log.
# Repo root is resolved relative to this script, so there are no machine-specific
# absolute paths. Override the interpreter with $OPENWORLD_PYTHON if needed.
set -e
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
PY="${OPENWORLD_PYTHON:-python3}"
echo "===== $(date) =====" >> datasets/openworld-market/daily.log
"$PY" datasets/openworld-market/daily_signal.py \
  >> datasets/openworld-market/daily.log 2>&1
