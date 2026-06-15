#!/usr/bin/env bash
# Student-capacity probe: same v2 traces (sft/v2/mlx-data), student 1.5b -> 3b.
# Tests whether absorbability is bounded by student size. ponytail: reuses v2 data, only model swaps.
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root
export PYTHONPATH="$PWD"
PY=.venv-distill/bin/python
MODEL="mlx-community/Qwen2.5-3B-Instruct-4bit"
ADAPTER="tooling/distill/adapters/qwen3b-v2"

echo "[3b] STEP 1/2 train 3B LoRA on sft/v2/mlx-data (v1/v2 hyperparams)"
$PY -m mlx_lm lora \
  --model "$MODEL" --train --data sft/v2/mlx-data \
  --adapter-path "$ADAPTER" \
  --iters 300 --batch-size 1 --learning-rate 1e-5 \
  --num-layers 8 --max-seq-length 2048 --save-every 100

echo "[3b] STEP 2/2 eval base-3B vs distilled-3B on the v2 heldout"
$PY tooling/distill/eval_heldout.py \
  --model "$MODEL" \
  --heldout sft/v2/heldout_instances.json \
  --adapter "$ADAPTER" \
  --out tooling/distill/eval/heldout_3b_v2.json

echo "[3b] DONE"
$PY -c 'import json;d=json.load(open("tooling/distill/eval/heldout_3b_v2.json"));print("base",d["base_solved"],"distilled",d["distilled_solved"],"delta",d["delta"],"wins",d["distilled_wins"])'
