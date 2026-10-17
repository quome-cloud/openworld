#!/bin/bash
# generic dev batch: $1 workdir, $2 output log, rest = seeds. Env passthrough.
DIR=$1; LOG=$2; shift 2
cd "$DIR"
rm -f "$LOG"
for s in "$@"; do
  timeout 1500 python3 dev_run.py "$s" 60000 2>&1 | \
    grep -v -i "gym\|migration" | \
    grep -E "steps:|progression|depth_max|xplvl_max|end_reason|role:|wallclock"
  echo "--- seed $s done"
done >> "$LOG" 2>&1
echo ALLDONE >> "$LOG"
