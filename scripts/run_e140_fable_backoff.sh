#!/usr/bin/env bash
# E140 FABLE arm -- same experiment as run_e140_backoff.sh, but on Claude Fable 5 at MAX thinking.
# Claude Fable 5 (claude-fable-5) is Anthropic's most capable widely released model; thinking is always on
# and depth is controlled by effort, so "max thinking mode" == EFFORT=max. Fully ISOLATED from the running
# opus arm: its own workspace prefix (sbfable_), its own archive (arc3_fullgame_sourcefree_fable.json), and
# its own best-keeper seeds -- it never touches the sb_ / arc3_fullgame_sourcefree.json opus run.
#
# Rate-limit-resilient, one game at a time, exactly like the opus backoff wrapper:
#   - INITIAL_COOLDOWN before starting so the account throttle has eased
#   - a session that dies SHORT (< MIN_SESSION_S, i.e. rate-limited) does NOT count as an attempt: it backs
#     off (exponential) and re-launches the SAME game, seeded from its best-so-far
#   - a game is done when it fully solves, or STALL_LIMIT genuine attempts make no progress
#   - banks through the same audited source-free gate at the end (into the FABLE archive)
# Nothing here is a shortcut: still source-free, solution-free, replay-verified.
set -uo pipefail
ROOT=/Users/jim/Desktop/openworld
PY=/Users/jim/.arcv/bin/python
RUNNER="$ROOT/scripts/run_arc_agent_sandbox.sh"
ARCH="$ROOT/experiments/results/arc3_fullgame_sourcefree_fable.json"
export MODEL="${MODEL:-claude-fable-5}" EFFORT="${EFFORT:-max}"   # Fable 5 @ max thinking
export WD_PREFIX="${WD_PREFIX:-sbfable_}"                          # isolated workspace (never collides with sb_)
export CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS=0
GAMES="${GAMES:-dc22 bp35 wa30 ls20 r11l sp80 sk48 vc33 tn36 lf52}"
INITIAL_COOLDOWN="${INITIAL_COOLDOWN:-2700}"   # 45 min: let the overnight throttle ease before starting
MIN_SESSION_S="${MIN_SESSION_S:-300}"          # a session shorter than this = rate-limited, not a real try
STALL_LIMIT="${STALL_LIMIT:-3}"                # give up a game after this many real attempts with no gain
MAX_BACKOFF="${MAX_BACKOFF:-1800}"             # cap a single backoff at 30 min
IDLE_LIMIT="${IDLE_LIMIT:-480}"                # a session that streams NO output for this long = hung (throttle/contention), not a real try
HUNG_LIMIT="${HUNG_LIMIT:-4}"                # give up a game after this many consecutive hangs/short-deaths (throttle) and advance
TRACES="$ROOT/experiments/results/arc3_traces"

# recursively kill a process and all its descendants (subshell -> runner -> claude -> sandbox workers)
killtree(){ local p=$1 c; for c in $(pgrep -P "$p" 2>/dev/null); do killtree "$c"; done; kill -TERM "$p" 2>/dev/null; }

lvl(){ $PY -c "import json,os;p='$ROOT/scratch_arc/${WD_PREFIX}$1/solved.json';print(json.load(open(p)).get('levels',0) if os.path.exists(p) else 0)" 2>/dev/null || echo 0; }
# win target = total levels of the game. Prefer the workspace solved.json's `win` (known as soon as the
# agent solves), fall back to the archive. Reading only the archive deadlocks a fresh arm: nothing is banked
# yet, so win=0, so a fully-solved game is never detected as FULL and the loop spins forever (the dc22 bug).
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

echo "[e140-fable] START $(date) model=$MODEL effort=$EFFORT prefix=$WD_PREFIX games=$GAMES cooldown=${INITIAL_COOLDOWN}s"
echo "[e140-fable] initial cooldown ${INITIAL_COOLDOWN}s to clear the throttle..."; sleep "$INITIAL_COOLDOWN"

for g in $GAMES; do
  w=$(win "$g"); start_lvl=$(lvl "$g"); stall=0; backoff=300; hung_streak=0
  echo "[e140-fable] === $g (start $start_lvl/$w) ==="
  while :; do
    w=$(win "$g")                                    # re-read each iteration (target is known once solved.json exists)
    if [ "$w" -gt 0 ] && [ "$(lvl "$g")" -ge "$w" ]; then echo "[e140-fable] $g FULL ($(lvl "$g")/$w)"; break; fi
    before=$(lvl "$g"); t0=$(date +%s); hung=0
    ( "$RUNNER" "$g" ) > "$ROOT/scratch_arc/${WD_PREFIX}${g}.out" 2>&1 &
    rpid=$!
    # hang watchdog: the agent streams stream-json into arc3_traces/transcripts/<g>__*.jsonl; if that file
    # stops growing for IDLE_LIMIT, the session is silently blocked (throttle or machine saturation), not working.
    while kill -0 "$rpid" 2>/dev/null; do
      sleep 30
      tr=$(ls -t "$TRACES/transcripts/${g}__"*.jsonl 2>/dev/null | head -1)
      if [ -n "$tr" ]; then last=$(stat -f %m "$tr" 2>/dev/null || echo "$t0"); else last=$t0; fi
      idle=$(( $(date +%s) - last ))
      if [ "$idle" -ge "$IDLE_LIMIT" ]; then
        echo "[e140-fable] $g HUNG (${idle}s with no stream output) -> kill + backoff (not counted)"
        killtree "$rpid"; hung=1; break
      fi
    done
    wait "$rpid" 2>/dev/null
    dur=$(( $(date +%s) - t0 )); after=$(lvl "$g"); w=$(win "$g")
    if [ "$w" -gt 0 ] && [ "$after" -ge "$w" ]; then     # just completed it -> stop retrying, go bank + advance
      echo "[e140-fable] $g gained $before->$after, FULL ($after/$w) -> done"; break; fi
    if [ "$hung" = 1 ] || [ "$dur" -lt "$MIN_SESSION_S" ]; then
      hung_streak=$(( hung_streak+1 ))
      if [ "$hung_streak" -ge "$HUNG_LIMIT" ]; then
        echo "[e140-fable] $g gave up after $HUNG_LIMIT non-productive sessions (hangs/short-deaths -- likely throttle) at $(lvl "$g")/$w -> bank + advance"; break; fi
      echo "[e140-fable] $g not a real attempt (hung=$hung, ${dur}s, streak ${hung_streak}/${HUNG_LIMIT}) -> backoff ${backoff}s, retry (not counted)"
      sleep "$backoff"; backoff=$(( backoff*2 )); [ "$backoff" -gt "$MAX_BACKOFF" ] && backoff=$MAX_BACKOFF
      continue
    fi
    hung_streak=0; backoff=300
    if [ "$after" -gt "$before" ]; then echo "[e140-fable] $g gained $before->$after (real attempt, ${dur}s)"; stall=0
    else stall=$(( stall+1 )); echo "[e140-fable] $g no gain (real attempt ${stall}/$STALL_LIMIT, ${dur}s)"; fi
    [ "$stall" -ge "$STALL_LIMIT" ] && { echo "[e140-fable] $g stalled at $(lvl "$g")/$w after $STALL_LIMIT real tries"; break; }
  done
  echo "[e140-fable] banking $g"
  SF_WD_PREFIX="$WD_PREFIX" SF_ARCH="$ARCH" "$PY" "$ROOT/scripts/autobank_sourcefree.py" 2>&1 | grep -iE "banked:|sf-bank" || true
  $PY -c "import json,os;a=json.load(open('$ARCH')) if os.path.exists('$ARCH') else {'n_full_games':0,'total_levels':0};print(f'[e140-fable] now: {a.get(\"n_full_games\",0)}/25 full, {a.get(\"total_levels\",0)}/183 levels')"
done
echo "[e140-fable] DONE $(date)"
