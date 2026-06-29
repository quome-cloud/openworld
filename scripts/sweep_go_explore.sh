#!/usr/bin/env bash
# Run E128 Go-Explore (source-free, seeded from each Claude-SF banked frontier) over ALL unsolved
# walls -- can it crack ANY final level? Pool of parallel runs; aggregates per-game results into
# experiments/results/arc3_go_explore.json. Pure compute (no API).
#   caffeinate -i nohup bash scripts/sweep_go_explore.sh > scratch_arc/ge_sweep.log 2>&1 &
set -uo pipefail
ROOT=/Users/jim/Desktop/openworld; PY=/Users/jim/.arcv/bin/python
BUDGET="${1:-300000}"; POOL="${2:-4}"; MODE="${3:-micro}"
PREF=$([ "$MODE" = macro ] && echo gm_ || echo ge_)
AGG=$([ "$MODE" = macro ] && echo arc3_go_explore_macro.json || echo arc3_go_explore.json)
# the unsolved walls (everything Claude-SF has NOT fully solved) that has a banked frontier
GAMES="bp35 dc22 g50t ka59 lf52 ls20 m0r0 r11l s5i5 sc25 sk48 sp80 su15 tn36 tu93 vc33 wa30"
echo "[ge-sweep] START $(date) mode=$MODE budget=$BUDGET pool=$POOL games=$GAMES"
pids=()
for g in $GAMES; do
  [ -f "$ROOT/scratch_arc/sb_$g/solved_best.json" ] || [ -f "$ROOT/scratch_arc/sb_$g/solved.json" ] || { echo "[ge-sweep] $g no frontier -- skip"; continue; }
  # throttle to POOL concurrent
  while [ "$(jobs -rp | wc -l | tr -d ' ')" -ge "$POOL" ]; do sleep 3; done
  echo "[ge-sweep] launch $g $(date '+%H:%M:%S')"
  "$PY" "$ROOT/experiments/e128_go_explore.py" "$g" "$BUDGET" "$MODE" > "$ROOT/scratch_arc/${PREF}${g}.log" 2>&1 &
  pids+=($!)
done
wait
echo "[ge-sweep] all runs done $(date); aggregating..."
"$PY" - <<'PY'
import json, glob, os
ROOT="/Users/jim/Desktop/openworld"; agg={}
for f in glob.glob(f"{ROOT}/scratch_arc/*/result.json"):
    try:
        r=json.load(open(f)); agg[r["game"]]=r
    except Exception: pass
json.dump(agg, open(f"{ROOT}/experiments/results/"+os.environ.get("GE_AGG","arc3_go_explore.json"),"w"), indent=1)
improved=[g for g,r in agg.items() if r.get("improved")]
full=[g for g,r in agg.items() if r.get("full")]
print(f"[ge-sweep] SOLVED-ANY: {len(improved)} improved {sorted(improved)}; {len(full)} full {sorted(full)}")
for g,r in sorted(agg.items()):
    flag="IMPROVED" if r.get("improved") else "no gain"
    print(f"  {g}: {r['seed_levels']}->{r['levels']}/{r['win']} {flag} {'FULL' if r.get('full') else ''} cells={r.get('archive_cells')} steps={r.get('real_steps')}")
PY
echo "[ge-sweep] DONE $(date)"
