#!/usr/bin/env bash
# Overnight grinder: repeatedly re-invoke the resume loop (which self-stops after 2 non-improving
# sessions) so a game keeps getting fresh attempts all night. Reseeds from the best-keeper each round;
# stops when the game is FULL or the round cap is hit. One game per dir => no double-writers.
set -uo pipefail
GAME="$1"; ROUNDS="${2:-40}"; PER="${3:-4}"
ROOT="/Users/jim/Desktop/openworld"
PY=/Users/jim/.pyenv/versions/3.14.6/bin/python
BEST="$ROOT/scratch_arc/full_$GAME/solved_best.json"
lvl(){ [ -f "$1" ] && $PY -c "import json;print(int(json.load(open('$1')).get('levels',0)))" 2>/dev/null || echo 0; }
win(){ [ -f "$1" ] && $PY -c "import json;d=json.load(open('$1'));print(int(d.get('win',0) or 0))" 2>/dev/null || echo 0; }
for r in $(seq 1 "$ROUNDS"); do
  bl=$(lvl "$BEST"); w=$(win "$BEST")
  if [ "$w" -ge 1 ] && [ "$bl" -ge "$w" ]; then echo "[$GAME] FULL ($bl/$w) at round $r — overnight done"; break; fi
  echo "[$GAME] === overnight round $r/$ROUNDS start (best=$bl/$w) ==="
  bash "$ROOT/scripts/run_arc_full_until.sh" "$GAME" "$PER"
  echo "[$GAME] === overnight round $r done (best now $(lvl "$BEST")/$(win "$BEST")) ==="
done
echo "[$GAME] overnight grinder exiting (best=$(lvl "$BEST")/$(win "$BEST"))"
