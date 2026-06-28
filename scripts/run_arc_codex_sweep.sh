#!/usr/bin/env bash
# SOURCE-FAITHFUL codex (gpt-5.5) full-game sweep -- the codex analog of overnight_arc.sh. Per game, a resume
# loop of run_arc_agent_codex.sh (each session CONTINUES from solved.json); telemetry banked per game to
# experiments/results/codex_full_game.json via codex_metrics.py. Parallel-safe with the source-free claude
# sweep (different model, different workdirs scratch_arc/codex_*, different result file).
#   caffeinate -i nohup bash scripts/run_arc_codex_sweep.sh > scratch_arc/codex_sweep.log 2>&1 &
set -uo pipefail
ROOT=/Users/jim/Desktop/openworld; PY=/Users/jim/.arcv/bin/python
GAMES="${1:-$($PY -c "import json;print(' '.join(sorted(json.load(open('$ROOT/experiments/results/arc3_fullgame.json'))['games'])))")}"
ROUNDS="${2:-3}"; HOURS="${3:-10}"; DEADLINE=$(( $(date +%s) + HOURS*3600 ))
isfull(){ $PY -c "import json,os;f='$ROOT/experiments/results/codex_full_game.json';d=(json.load(open(f)).get('$1',{}) if os.path.exists(f) else {});print(1 if d.get('full') else 0)"; }
lvl(){ [ -f "$ROOT/scratch_arc/codex_$1/solved.json" ] && $PY -c "import json;print(json.load(open('$ROOT/scratch_arc/codex_$1/solved.json')).get('levels',-1))" 2>/dev/null || echo -1; }
winof(){ [ -f "$ROOT/scratch_arc/codex_$1/solved.json" ] && $PY -c "import json;print(json.load(open('$ROOT/scratch_arc/codex_$1/solved.json')).get('win',99))" 2>/dev/null || echo 99; }
echo "[codex-sweep] START $(date) deadline=+${HOURS}h rounds=$ROUNDS games=$GAMES"
for g in $GAMES; do
  [ "$(date +%s)" -ge "$DEADLINE" ] && { echo "[codex-sweep] deadline reached"; break; }
  [ "$(isfull "$g")" = "1" ] && { echo "[codex-sweep] $g already full -- skip"; continue; }
  prev=-1; stall=0
  for r in $(seq 1 "$ROUNDS"); do
    [ "$(date +%s)" -ge "$DEADLINE" ] && break
    echo "[codex-sweep] $g round $r/$ROUNDS (best=$(lvl "$g")) $(date '+%H:%M:%S')"
    bash "$ROOT/scripts/run_arc_agent_codex.sh" "$g" gpt-5.5 >/dev/null 2>&1 || true
    $PY "$ROOT/scripts/codex_metrics.py" "$g"
    cur=$(lvl "$g"); win=$(winof "$g")
    { [ "$cur" -ge "$win" ] 2>/dev/null; } && { echo "[codex-sweep] $g FULL ($cur/$win)"; break; }
    if [ "$cur" -le "$prev" ] 2>/dev/null; then stall=$((stall+1)); else stall=0; fi
    prev=$cur
    [ "$stall" -ge 2 ] && { echo "[codex-sweep] $g no progress past $cur -- next game"; break; }
  done
done
echo "[codex-sweep] DONE $(date)"
