#!/usr/bin/env bash
# Resume the full-game agent on one game until all levels are won or progress stalls.
# Usage: run_arc_full_until.sh <game> [max_sessions]
set -uo pipefail
GAME="$1"; MAX="${2:-4}"; ROOT="/Users/jim/Desktop/openworld"; WD="$ROOT/scratch_arc/full_$GAME"
read_lvl(){ [ -f "$WD/solved.json" ] && /Users/jim/.pyenv/versions/3.9.18/bin/python -c "import json;d=json.load(open('$WD/solved.json'));print(d.get('levels',0),d.get('win',0) or 0)" 2>/dev/null || echo "0 0"; }
prev=-1
for i in $(seq 1 "$MAX"); do
  bash "$ROOT/scripts/run_arc_agent_full.sh" "$GAME" >/dev/null 2>&1
  read L W < <(read_lvl)
  echo "  [$GAME] session $i: $L/$W"
  { [ "${W:-0}" -ge 1 ] && [ "${L:-0}" -ge "${W:-99}" ]; } && { echo "  [$GAME] FULL GAME"; break; }
  [ "${L:-0}" -le "$prev" ] && { echo "  [$GAME] stalled at $L; stopping"; break; }
  prev=${L:-0}
done
