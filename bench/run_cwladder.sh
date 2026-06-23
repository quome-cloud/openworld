#!/usr/bin/env bash
# Runs ON the GPU box: E84 cross-world SCALING ladder on List Functions -- fixed held-out eval,
# train on 16/32/64/128 disjoint worlds, showing real-domain transfer grows with #worlds.
set -euo pipefail
BUCKET="${1:-gs://openworld-bench}"; RUN_ID="${2:-cwladder}"; DEST="$BUCKET/$RUN_ID"
cd "$(dirname "$0")/.."
pip install -q --upgrade pip
pip install -q "transformers>=4.44,<5" "peft>=0.12" "datasets>=2.20" "accelerate>=0.33" "bitsandbytes>=0.43" "jinja2>=3.1" numpy
nvidia-smi || echo "WARN: no GPU"
cd experiments && mkdir -p data
gcloud storage cp gs://openworld-bench/data/listfn_worlds.jsonl data/listfn_worlds.jsonl
INSTR="Infer the hidden rule mapping each input list to its output, then produce the output for the final input."
for N in 16 32 64 128; do
  echo "=== ladder rung N=$N ($(date -u +%H:%M:%S)) ==="
  python3 e84_crossworld.py --worlds data/listfn_worlds.jsonl --domain listfn --instruction "$INSTR" \
    --bucket "$DEST" --n_train_worlds "$N" --fixed_eval --seed 0 --steps 350 --n_eval 6 || echo "N=$N FAILED"
done
echo "[run_cwladder] done -> $DEST/e84_crossworld_listfn_n{16,32,64,128}_seed0.json"
