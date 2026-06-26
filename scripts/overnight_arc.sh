#!/usr/bin/env bash
# Overnight supervisor: keep ONE full-game agent at a time grinding the unfinished ARC games,
# cycling across them, banking verified progress after each visit, until all are full or the
# deadline passes. Best-keeper (in run_arc_full_until.sh) makes progress monotonic.
#
# Launch (keeps the Mac awake while plugged in):
#   caffeinate -i nohup bash scripts/overnight_arc.sh > scratch_arc/overnight.log 2>&1 &
set -uo pipefail
ROOT=/Users/jim/Desktop/openworld
VENV=/private/tmp/claude-501/-Users-jim-Desktop-openworld/71e8c8de-fcca-4c0d-b13e-d3aae6071546/scratchpad/arcv/bin/python
PY=/Users/jim/.pyenv/versions/3.9.18/bin/python
GAMES="dc22 bp35 lf52"            # dc22 first: it is closest to full (5/6)
HOURS="${1:-9}"
DEADLINE=$(( $(date +%s) + HOURS*3600 ))

banked_full(){ # 1 if game $1 is banked full
  $PY -c "import json;v=json.load(open('$ROOT/experiments/results/agent_full_game.json'))['per_game'].get('$1',{});print(1 if v.get('win') and v.get('levels',0)>=v['win'] else 0)" 2>/dev/null || echo 0
}
loop_running(){ pgrep -f "run_arc.*$1" >/dev/null 2>&1; }

echo "[overnight] START $(date) deadline=+${HOURS}h games='$GAMES'"
round=0
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  round=$((round+1)); all_full=1
  for g in $GAMES; do
    [ "$(banked_full "$g")" = "1" ] && continue
    all_full=0
    if loop_running "$g"; then
      echo "[overnight] round $round: $g already has a live loop — skipping to avoid double-writers"
      continue
    fi
    echo "[overnight] round $round: working $g  $(date '+%H:%M:%S')"
    bash "$ROOT/scripts/run_arc_full_until.sh" "$g" 2 2>&1 | sed "s/^/[run_$g] /"
    "$VENV" "$ROOT/scripts/autobank_arc.py" 2>&1 | sed 's/^/[autobank] /'
  done
  if [ "$all_full" = "1" ]; then echo "[overnight] ALL THREE FULL — done $(date)"; break; fi
done
echo "[overnight] END $(date)"
