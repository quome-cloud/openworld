#!/bin/bash
# dev batch: $1 = output log, remaining args = seeds
cd /data/doh/teams/researchy/work/fable_nethack
LOG=$1; shift
rm -f "$LOG"
for s in "$@"; do
  timeout 1200 python3 dev_run.py "$s" 50000 2>&1 | \
    grep -v -i "gym\|migration" | \
    grep -E "steps:|progression|depth_max|xplvl_max|end_reason|role:|wallclock"
  echo "--- seed $s done"
done >> "$LOG" 2>&1
echo ALLDONE >> "$LOG"
