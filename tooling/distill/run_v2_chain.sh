#!/usr/bin/env bash
# Auto-chain: wait for v2 harvest -> format -> mlx-data -> train LoRA -> eval base vs distilled.
# ponytail: one fail-fast script, no orchestration framework. Launched via bgjob --notify quome.
set -euo pipefail

cd "$(dirname "$0")/../.."   # repo root: ~/projects/quome-openworld
REPO="$PWD"
export PYTHONPATH="$REPO"     # script-by-path puts sys.path[0]=script dir, not repo root; openworld lives at repo root
PY=.venv-distill/bin/python  # py3.12 + mlx + openworld
HARVEST_JOB="${1:?harvest job id required}"
TRACES="traces/harvest-v2/qwen2.5-14b.traces.jsonl"

echo "[chain] waiting for harvest job $HARVEST_JOB ..."
# Poll the registry until the harvest job leaves 'running'.
while ~/.local/bin/bgjob status "$HARVEST_JOB" 2>/dev/null | head -1 | grep -q "running"; do
  sleep 30
done
STATUS_LINE="$(~/.local/bin/bgjob status "$HARVEST_JOB" 2>/dev/null | head -1)"
echo "[chain] harvest final: $STATUS_LINE"
if ! echo "$STATUS_LINE" | grep -q "rc=0"; then
  echo "[chain] ABORT: harvest did not finish rc=0"; exit 1
fi
[ -s "$TRACES" ] || { echo "[chain] ABORT: empty/missing $TRACES"; exit 1; }
echo "[chain] harvest traces: $(wc -l < "$TRACES") lines"

echo "[chain] STEP 1/4 format_traces -> sft/v2"
$PY tooling/distill/format_traces.py "$TRACES" --out-dir sft/v2
echo "[chain]   train pairs: $(wc -l < sft/v2/train.jsonl), heldout instances: $($PY -c 'import json;print(len(json.load(open("sft/v2/heldout_instances.json"))))')"

echo "[chain] STEP 2/4 to_mlx_data -> sft/v2/mlx-data"
$PY tooling/distill/to_mlx_data.py sft/v2/train.jsonl --out sft/v2/mlx-data

echo "[chain] STEP 3/4 train LoRA -> adapters/qwen1.5b-v2 (mirrors v1 hyperparams)"
$PY -m mlx_lm lora \
  --model mlx-community/Qwen2.5-1.5B-Instruct-4bit \
  --train --data sft/v2/mlx-data \
  --adapter-path tooling/distill/adapters/qwen1.5b-v2 \
  --iters 300 --batch-size 1 --learning-rate 1e-5 \
  --num-layers 8 --max-seq-length 2048 --save-every 100

echo "[chain] STEP 4/4 eval base vs distilled -> eval/heldout_v2.json"
$PY tooling/distill/eval_heldout.py \
  --heldout sft/v2/heldout_instances.json \
  --adapter tooling/distill/adapters/qwen1.5b-v2 \
  --out tooling/distill/eval/heldout_v2.json

echo "[chain] DONE. Summary:"
$PY -c 'import json;d=json.load(open("tooling/distill/eval/heldout_v2.json"));print(json.dumps(d,indent=2))' || cat tooling/distill/eval/heldout_v2.json
