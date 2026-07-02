#!/usr/bin/env bash
# E140 FAIR + rate-limit-resilient run. The overnight run got throttled: claude -p sessions die on a
# server-side rate limit ("temporarily limiting requests, not your usage limit") within ~1 min, so a
# game silently loses its budget. This wrapper makes it fair:
#   - starts after an INITIAL_COOLDOWN so the account's throttle has eased
#   - runs ONE game at a time (no concurrency pressure), effort=max, bg-ceiling raised
#   - if a session dies SHORT (rate-limited, < MIN_SESSION_S) it does NOT count as an attempt: it backs
#     off (exponential) and re-launches the SAME game, seeded from its best-so-far (best-keeper)
#   - a game is done when it fully solves, or STALL_LIMIT genuine (long-enough) attempts make no progress
#   - banks through the same audited source-free gate at the end
# Nothing here is a shortcut: still source-free, solution-free, replay-verified.
set -uo pipefail
ROOT=/Users/jim/Desktop/openworld
PY=/Users/jim/.arcv/bin/python
RUNNER="$ROOT/scripts/run_arc_agent_sandbox.sh"
ARCH="$ROOT/experiments/results/arc3_fullgame_sourcefree.json"
export MODEL="${MODEL:-claude-opus-4-8}" EFFORT="${EFFORT:-max}"
export CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS=0
GAMES="${GAMES:-dc22 bp35 wa30 ls20 r11l sp80 sk48 vc33 tn36 lf52}"
INITIAL_COOLDOWN="${INITIAL_COOLDOWN:-2700}"   # 45 min: let the overnight throttle ease before starting
MIN_SESSION_S="${MIN_SESSION_S:-300}"          # a session shorter than this = rate-limited, not a real try
STALL_LIMIT="${STALL_LIMIT:-3}"                # give up a game after this many real attempts with no gain
MAX_BACKOFF="${MAX_BACKOFF:-1800}"             # cap a single backoff at 30 min
IDLE_LIMIT="${IDLE_LIMIT:-480}"                # a session that streams NO output for this long = hung (throttle/contention), not a real try
TRACES="$ROOT/experiments/results/arc3_traces"

# recursively kill a process and all its descendants (subshell -> runner -> claude -> sandbox workers)
killtree(){ local p=$1 c; for c in $(pgrep -P "$p" 2>/dev/null); do killtree "$c"; done; kill -TERM "$p" 2>/dev/null; }

lvl(){ $PY -c "import json,os;p='$ROOT/scratch_arc/sb_$1/solved.json';print(json.load(open(p)).get('levels',0) if os.path.exists(p) else 0)" 2>/dev/null || echo 0; }
win(){ $PY -c "import json;print(json.load(open('$ARCH'))['per_game'].get('$1',{}).get('win',0))" 2>/dev/null || echo 0; }

echo "[e140-backoff] START $(date) games=$GAMES effort=$EFFORT cooldown=${INITIAL_COOLDOWN}s"
echo "[e140-backoff] initial cooldown ${INITIAL_COOLDOWN}s to clear the throttle..."; sleep "$INITIAL_COOLDOWN"

for g in $GAMES; do
  w=$(win "$g"); start_lvl=$(lvl "$g"); stall=0; backoff=300
  echo "[e140-backoff] === $g (start $start_lvl/$w) ==="
  while :; do
    if [ "$(lvl "$g")" -ge "$w" ] && [ "$w" -gt 0 ]; then echo "[e140-backoff] $g FULL"; break; fi
    before=$(lvl "$g"); t0=$(date +%s); hung=0
    ( "$RUNNER" "$g" ) > "$ROOT/scratch_arc/sb_${g}.out" 2>&1 &
    rpid=$!
    # hang watchdog: the agent streams stream-json into arc3_traces/transcripts/<g>__*.jsonl; if that file
    # stops growing for IDLE_LIMIT, the session is silently blocked (throttle or machine saturation), not working.
    while kill -0 "$rpid" 2>/dev/null; do
      sleep 30
      tr=$(ls -t "$TRACES/transcripts/${g}__"*.jsonl 2>/dev/null | head -1)
      if [ -n "$tr" ]; then last=$(stat -f %m "$tr" 2>/dev/null || echo "$t0"); else last=$t0; fi
      idle=$(( $(date +%s) - last ))
      if [ "$idle" -ge "$IDLE_LIMIT" ]; then
        echo "[e140-backoff] $g HUNG (${idle}s with no stream output) -> kill + backoff (not counted)"
        killtree "$rpid"; hung=1; break
      fi
    done
    wait "$rpid" 2>/dev/null
    dur=$(( $(date +%s) - t0 )); after=$(lvl "$g")
    if [ "$hung" = 1 ] || [ "$dur" -lt "$MIN_SESSION_S" ]; then
      echo "[e140-backoff] $g not a real attempt (hung=$hung, ${dur}s) -> backoff ${backoff}s, retry (not counted)"
      sleep "$backoff"; backoff=$(( backoff*2 )); [ "$backoff" -gt "$MAX_BACKOFF" ] && backoff=$MAX_BACKOFF
      continue
    fi
    backoff=300
    if [ "$after" -gt "$before" ]; then echo "[e140-backoff] $g gained $before->$after (real attempt, ${dur}s)"; stall=0
    else stall=$(( stall+1 )); echo "[e140-backoff] $g no gain (real attempt ${stall}/$STALL_LIMIT, ${dur}s)"; fi
    [ "$stall" -ge "$STALL_LIMIT" ] && { echo "[e140-backoff] $g stalled at $(lvl "$g")/$w after $STALL_LIMIT real tries"; break; }
  done
  echo "[e140-backoff] banking $g"
  SF_WD_PREFIX=sb_ SF_ARCH="$ARCH" "$PY" "$ROOT/scripts/autobank_sourcefree.py" 2>&1 | grep -iE "banked:|sf-bank" || true
  $PY -c "import json;a=json.load(open('$ARCH'));print(f'[e140-backoff] now: {a[\"n_full_games\"]}/25 full, {a[\"total_levels\"]}/183 levels')"
done
echo "[e140-backoff] DONE $(date)"
