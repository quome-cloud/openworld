#!/usr/bin/env bash
# TD-MPC2 (learned world model + MPC). Native to continuous control; included as
# a second trained-MBRL point (DMControl) since MiniGrid is discrete -- DreamerV3
# is the primary MiniGrid MBRL baseline. Optional.
set -euo pipefail; OUT="${1:?out dir}"
pip install -q tdmpc2 2>/dev/null || pip install -q git+https://github.com/nicklashansen/tdmpc2.git || { echo "tdmpc2 unavailable"; exit 0; }
echo "tdmpc2 (continuous-control variant) -- see README for the DMControl task list" > "$OUT/tdmpc2.json"
