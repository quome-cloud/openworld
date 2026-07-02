#!/usr/bin/env bash
# E140 CODEX arm -- rate-limit + hang + OOM resilient, ONE game at a time. Mirrors the Claude/Fable
# backoff wrappers (run_e140_backoff.sh / run_e140_fable_backoff.sh) but drives codex (gpt-5.5 @ xhigh)
# via run_arc_agent_sandbox_codex.sh, in its isolated sbcodex_ workspace + codex archive.
#   - sequential by construction (POOL=1) -- fixes the POOL=2 OOM that killed the prior sweep
#   - a session that dies SHORT (< MIN_SESSION_S, rate-limited) does NOT count: backoff + retry
#   - HANG WATCHDOG: codex exec streams to sbcodex_<g>/agent.log; if that file streams nothing for
#     IDLE_LIMIT the run is silently stuck -> kill the tree + backoff (not counted), so a hung codex
#     game can't burn its whole budget
#   - a game is done when fully solved (levels >= its solved.json win, re-read each iteration) or
#     STALL_LIMIT genuine attempts make no progress; banks through the audited source-free gate
# Still source-free, solution-free, replay-verified.
set -uo pipefail
ROOT=/Users/jim/Desktop/openworld
PY=/Users/jim/.arcv/bin/python
RUNNER="$ROOT/scripts/run_arc_agent_sandbox_codex.sh"
ARCH="$ROOT/experiments/results/arc3_fullgame_sourcefree_codex.json"
WD_PREFIX="sbcodex_"                                      # the codex runner hardcodes this workspace prefix
export MODEL="${MODEL:-gpt-5.5}" REASONING="${REASONING:-xhigh}"
GAMES="${GAMES:-ar25 bp35 cd82 cn04 dc22 ft09 g50t ka59 lf52 lp85 ls20 m0r0 r11l re86 s5i5 sb26 sc25 sk48 sp80 su15 tn36 tr87 tu93 vc33 wa30}"
INITIAL_COOLDOWN="${INITIAL_COOLDOWN:-30}"
MIN_SESSION_S="${MIN_SESSION_S:-300}"          # a session shorter than this = rate-limited, not a real try
STALL_LIMIT="${STALL_LIMIT:-3}"                # give up a game after this many real attempts with no gain
MAX_BACKOFF="${MAX_BACKOFF:-1800}"             # cap a single backoff at 30 min
IDLE_LIMIT="${IDLE_LIMIT:-900}"                # codex reasons a long time between writes; 15 min of NO output = hung

# recursively kill a process and all its descendants (subshell -> runner -> codex exec -> sandbox workers)
killtree(){ local p=$1 c; for c in $(pgrep -P "$p" 2>/dev/null); do killtree "$c"; done; kill -TERM "$p" 2>/dev/null; }

lvl(){ $PY -c "import json,os;p='$ROOT/scratch_arc/${WD_PREFIX}$1/solved.json';print(json.load(open(p)).get('levels',0) if os.path.exists(p) else 0)" 2>/dev/null || echo 0; }
# win target = total levels: prefer the workspace solved.json's `win` (known once solved), fall back to archive
win(){ $PY -c "
import json,os
w=0
p='$ROOT/scratch_arc/${WD_PREFIX}$1/solved.json'
if os.path.exists(p):
    try: w=int(json.load(open(p)).get('win',0) or 0)
    except Exception: w=0
if w==0 and os.path.exists('$ARCH'):
    try: w=int(json.load(open('$ARCH'))['per_game'].get('$1',{}).get('win',0) or 0)
    except Exception: w=0
print(w)" 2>/dev/null || echo 0; }

echo "[e140-codex] START $(date) model=$MODEL reasoning=$REASONING prefix=$WD_PREFIX games=$GAMES cooldown=${INITIAL_COOLDOWN}s"
echo "[e140-codex] initial cooldown ${INITIAL_COOLDOWN}s..."; sleep "$INITIAL_COOLDOWN"

for g in $GAMES; do
  stall=0; backoff=300
  echo "[e140-codex] === $g (start $(lvl "$g")/$(win "$g")) ==="
  while :; do
    w=$(win "$g")                                    # re-read each iteration (target known once solved.json exists)
    if [ "$w" -gt 0 ] && [ "$(lvl "$g")" -ge "$w" ]; then echo "[e140-codex] $g FULL ($(lvl "$g")/$w)"; break; fi
    before=$(lvl "$g"); t0=$(date +%s); hung=0
    log="$ROOT/scratch_arc/${WD_PREFIX}${g}/agent.log"
    ( "$RUNNER" "$g" ) > "$ROOT/scratch_arc/${WD_PREFIX}${g}.out" 2>&1 &
    rpid=$!
    # hang watchdog: codex exec streams into sbcodex_<g>/agent.log; no growth for IDLE_LIMIT = stuck
    while kill -0 "$rpid" 2>/dev/null; do
      sleep 30
      if [ -f "$log" ]; then last=$(stat -f %m "$log" 2>/dev/null || echo "$t0"); else last=$t0; fi
      idle=$(( $(date +%s) - last ))
      if [ "$idle" -ge "$IDLE_LIMIT" ]; then
        echo "[e140-codex] $g HUNG (${idle}s with no stream output) -> kill + backoff (not counted)"
        killtree "$rpid"; hung=1; break
      fi
    done
    wait "$rpid" 2>/dev/null
    dur=$(( $(date +%s) - t0 )); after=$(lvl "$g"); w=$(win "$g")
    if [ "$w" -gt 0 ] && [ "$after" -ge "$w" ]; then
      echo "[e140-codex] $g gained $before->$after, FULL ($after/$w) -> done"; break; fi
    if [ "$hung" = 1 ] || [ "$dur" -lt "$MIN_SESSION_S" ]; then
      echo "[e140-codex] $g not a real attempt (hung=$hung, ${dur}s) -> backoff ${backoff}s, retry (not counted)"
      sleep "$backoff"; backoff=$(( backoff*2 )); [ "$backoff" -gt "$MAX_BACKOFF" ] && backoff=$MAX_BACKOFF
      continue
    fi
    backoff=300
    if [ "$after" -gt "$before" ]; then echo "[e140-codex] $g gained $before->$after (real attempt, ${dur}s)"; stall=0
    else stall=$(( stall+1 )); echo "[e140-codex] $g no gain (real attempt ${stall}/$STALL_LIMIT, ${dur}s)"; fi
    [ "$stall" -ge "$STALL_LIMIT" ] && { echo "[e140-codex] $g stalled at $(lvl "$g")/$w after $STALL_LIMIT real tries"; break; }
  done
  echo "[e140-codex] banking $g"
  SF_WD_PREFIX="$WD_PREFIX" SF_ARCH="$ARCH" "$PY" "$ROOT/scripts/autobank_sourcefree.py" 2>&1 | grep -iE "banked:|sf-bank" || true
  $PY -c "import json,os;a=json.load(open('$ARCH')) if os.path.exists('$ARCH') else {'n_full_games':0,'total_levels':0};print(f'[e140-codex] now: {a.get(\"n_full_games\",0)}/25 full, {a.get(\"total_levels\",0)}/183 levels')"
done
echo "[e140-codex] DONE $(date)"
