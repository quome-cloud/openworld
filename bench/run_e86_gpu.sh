#!/usr/bin/env bash
# Runs ON a GPU box: E86 ARC-AGI-3 verified-code synthesis. Stands up Ollama + a code model,
# then runs e86 (collect -> synthesize predict() -> verify exact-match) across games, uploading.
# Usage (on box): bash bench/run_e86_gpu.sh gs://openworld-bench e86-arc3-synth qwen2.5-coder:32b
set -euo pipefail
BUCKET="${1:-gs://openworld-bench}"; RUN_ID="${2:-e86-arc3-synth}"; MODEL="${3:-qwen2.5-coder:32b}"
DEST="$BUCKET/$RUN_ID"
cd "$(dirname "$0")/.."
pip install -q --upgrade pip
pip install -q arc-agi numpy
curl -fsSL https://ollama.com/install.sh | sh
(ollama serve >/tmp/ollama.log 2>&1 &) ; sleep 8
ollama pull "$MODEL"
python3 -c "import torch;print('[gpu]',torch.cuda.is_available())" 2>/dev/null || true
cd experiments && mkdir -p results
for gid in ls20 ft09 vc33 sp80 cn04 ar25; do
  echo "=== e86 synth $gid model=$MODEL ($(date -u +%H:%M:%S)) ==="
  python3 e86_arc3.py --game "$gid" --steps 300 --ollama "$MODEL" \
    && gcloud storage cp "results/e86_arc3_${gid}.json" "$DEST/" 2>/dev/null || echo "$gid failed"
done
echo "[run_e86_gpu] done -> $DEST/"
