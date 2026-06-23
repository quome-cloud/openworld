#!/usr/bin/env bash
# Runs ON the GPU box: ARC test-time-training SEED REPLICATION (reviewer ask -- the headline
# real-data ARC result was single-seed with a CI of [2,20]; reviewers asked for the result at
# >=3 seeds). Installs the SFT stack on top of the Deep Learning VM's CUDA torch, fetches the
# ARC-AGI data, and runs e80_arc_ttt.py for seeds 0/1/2. Each run uploads
# e80_arc_ttt_seed{N}.json to GCS after every task (survives spot preemption).
# Usage (on box): bash bench/run_arc_seeds.sh gs://openworld-bench arc-seeds [n_tasks]
set -euo pipefail
BUCKET="${1:-gs://openworld-bench}"
RUN_ID="${2:-arc-seeds}"
NTASKS="${3:-40}"
DEST="$BUCKET/$RUN_ID"

cd "$(dirname "$0")/.."
pip install -q --upgrade pip
pip install -q "transformers>=4.44,<5" "peft>=0.12" "datasets>=2.20" \
               "accelerate>=0.33" "bitsandbytes>=0.43" "jinja2>=3.1" numpy
python3 -c "import torch; print('[torch]', torch.__version__, 'cuda', torch.cuda.is_available())"
nvidia-smi || echo "WARN: no GPU visible"

[ -d ARC-AGI ] || git clone --depth 1 https://github.com/fchollet/ARC-AGI.git
cd experiments
for S in 0 1 2; do
  echo "=== ARC TTT seed $S ($(date -u +%H:%M:%S)) ==="
  python3 e80_arc_ttt.py --data "$PWD/../ARC-AGI/data" --bucket "$DEST" --n "$NTASKS" --seed "$S" \
    || echo "[run_arc_seeds] seed $S FAILED"
done
echo "[run_arc_seeds] done -> $DEST/e80_arc_ttt_seed{0,1,2}.json"
