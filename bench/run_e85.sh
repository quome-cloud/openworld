#!/usr/bin/env bash
# Runs ON the GPU box: E85 trajectory-vs-answer ablation, 3 seeds (synthetic iterated worlds,
# self-contained -- no data download). Each (arm,horizon) uploads after eval.
set -euo pipefail
BUCKET="${1:-gs://openworld-bench}"; RUN_ID="${2:-e85}"; DEST="$BUCKET/$RUN_ID"
cd "$(dirname "$0")/.."
pip install -q --upgrade pip
pip install -q "transformers>=4.44,<5" "peft>=0.12" "accelerate>=0.33" "bitsandbytes>=0.43" "jinja2>=3.1" numpy
nvidia-smi || echo "WARN: no GPU"
cd experiments
for S in 0 1 2; do
  echo "=== e85 seed $S ($(date -u +%H:%M:%S)) ==="
  python3 e85_traj_vs_answer.py --bucket "$DEST" --seed "$S" || echo "seed $S FAILED"
done
echo "[run_e85] done -> $DEST/e85_traj_vs_answer_seed{0,1,2}.json"
