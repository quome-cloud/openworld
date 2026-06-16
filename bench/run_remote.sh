#!/usr/bin/env bash
# Runs ON the GPU instance. Builds an isolated venv, installs the competitor
# stacks, runs each benchmark, and uploads results + logs to GCS.
# Usage (on instance): bash bench/run_remote.sh gs://openworld-bench run-<id>
set -euo pipefail
BUCKET="${1:?bucket required}"
RUN_ID="${2:?run id required}"
OUT="bench/out/$RUN_ID"
mkdir -p "$OUT"

python3 -m venv .venv-bench && source .venv-bench/bin/activate
pip install -q --upgrade pip
pip install -q -e .                                  # openworld core
pip install -q "gymnasium>=0.29" "minigrid>=2.3" numpy

nvidia-smi > "$OUT/gpu.txt" 2>&1 || echo "no GPU visible" > "$OUT/gpu.txt"
python -c "import sys,platform,json; json.dump({'python':platform.python_version()}, open('$OUT/env.json','w'))"

# 0) shared-environment fidelity: OpenWorld verified world vs the REAL minigrid env.
python bench/validate_minigrid.py --out "$OUT/minigrid_fidelity.json"   2>&1 | tee "$OUT/validate.log"

# 1) trained MBRL (DreamerV3 / TD-MPC2) -- each guarded; logs kept on failure.
bash bench/competitors/run_tdmpc2.sh     "$OUT" 2>&1 | tee "$OUT/tdmpc2.log"     || echo "tdmpc2 step failed"
bash bench/competitors/run_dreamerv3.sh  "$OUT" 2>&1 | tee "$OUT/dreamerv3.log"  || echo "dreamerv3 step failed"
# 2) same-species code world model.
bash bench/competitors/run_poeworld.sh   "$OUT" 2>&1 | tee "$OUT/poeworld.log"   || echo "poeworld step failed"
# 3) perceptual video world model.
bash bench/competitors/run_vjepa.sh      "$OUT" 2>&1 | tee "$OUT/vjepa.log"      || echo "vjepa step failed"

gcloud storage cp -r "$OUT" "$BUCKET/$RUN_ID/"
echo "[run_remote] uploaded $OUT -> $BUCKET/$RUN_ID/"
