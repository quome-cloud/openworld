#!/usr/bin/env bash
# Resume the full-game agent until all levels won, with a BEST-KEEPER: progress is monotonic --
# we snapshot the deepest solved.json to solved_best.json and restore it if a session regresses.
set -uo pipefail
GAME="$1"; MAX="${2:-6}"; ROOT="/Users/jim/Desktop/openworld"; WD="$ROOT/scratch_arc/full_$GAME"
PY=/Users/jim/.pyenv/versions/3.9.18/bin/python
mkdir -p "$WD"
lvl(){ [ -f "$1" ] && $PY -c "import json;print(int(json.load(open('$1')).get('levels',0)))" 2>/dev/null || echo 0; }
win(){ [ -f "$1" ] && $PY -c "import json;print(int(json.load(open('$1')).get('win',0) or 0))" 2>/dev/null || echo 0; }
BEST="$WD/solved_best.json"
# seed best from current progress if we don't have one yet
[ -f "$BEST" ] || { [ -f "$WD/solved.json" ] && cp "$WD/solved.json" "$BEST"; }
prev=-1
for i in $(seq 1 "$MAX"); do
  # always start the session from the DEEPEST known solution
  [ -f "$BEST" ] && cp "$BEST" "$WD/solved.json"
  bash "$ROOT/scripts/run_arc_agent_full.sh" "$GAME" >/dev/null 2>&1
  nl=$(lvl "$WD/solved.json"); bl=$(lvl "$BEST"); W=$(win "$WD/solved.json"); [ "$W" -ge 1 ] || W=$(win "$BEST")
  if [ "$nl" -gt "$bl" ]; then cp "$WD/solved.json" "$BEST"; bl=$nl; else cp "$BEST" "$WD/solved.json" 2>/dev/null || true; fi
  echo "  [$GAME] session $i: deepest=$bl/$W"
  { [ "${W:-0}" -ge 1 ] && [ "$bl" -ge "$W" ]; } && { echo "  [$GAME] FULL"; break; }
  [ "$bl" -le "$prev" ] && { echo "  [$GAME] no progress past $bl; stopping"; break; }
  prev=$bl
done
