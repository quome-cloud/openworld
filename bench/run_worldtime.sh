#!/usr/bin/env bash
# Runs ON the GPU box: powers the world-time-compute spine (E78) -- multi-seed world-count
# scaling with CIs + the verified-vs-noisy label ablation. Sets up an isolated venv on top
# of the Deep Learning VM's preinstalled torch, generates data, runs the fine-tune/eval
# loop (which uploads partial results to GCS after every run), and copies the final JSON.
# Usage (on box): bash bench/run_worldtime.sh gs://openworld-bench run-<id> [epochs]
set -euo pipefail
BUCKET="${1:-gs://openworld-bench}"
RUN_ID="${2:-wtc-manual}"
EPOCHS="${3:-2}"
DEST="$BUCKET/$RUN_ID"

cd "$(dirname "$0")/.."
python3 -m venv .venv-wtc && source .venv-wtc/bin/activate
pip install -q --upgrade pip
# torch is preinstalled on the Deep Learning VM image; add the SFT stack (same libs the
# paper's E73/E74 runs used).
pip install -q "transformers>=4.44,<5" "peft>=0.12" "trl>=0.12,<0.15" \
               "datasets>=2.20" "accelerate>=0.33" "bitsandbytes>=0.43" numpy

nvidia-smi || echo "WARN: no GPU visible"
cd experiments
python e75_data.py            # writes the fixed hard held-out test set
python e78_data.py            # writes SFT sets + manifest
python e78_run.py --bucket "$DEST" --epochs "$EPOCHS"
cd ..
gcloud storage cp experiments/results/e78_worldtime_power.json \
  "$DEST/e78_worldtime_power.json" || true
echo "[run_worldtime] done -> $DEST/e78_worldtime_power.json"
