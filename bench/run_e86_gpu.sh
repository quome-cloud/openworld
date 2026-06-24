#!/usr/bin/env bash
# Runs ON a GPU box: E86 ARC-AGI-3 verified-code synthesis.
# The arc-agi toolkit needs Python >=3.11 (uses typing.Self) but the DL VM ships 3.10, so we run
# it in a uv-managed 3.11 venv. The code model is served by OLLAMA over HTTP (no CUDA torch needed
# in the venv -- Ollama uses the GPU directly).
# Usage (on box): bash bench/run_e86_gpu.sh gs://openworld-bench e86-arc3-synth qwen2.5-coder:32b
set -euo pipefail
BUCKET="${1:-gs://openworld-bench}"; RUN_ID="${2:-e86-arc3-synth}"; MODEL="${3:-qwen2.5-coder:32b}"
DEST="$BUCKET/$RUN_ID"
cd "$(dirname "$0")/.."

# 1) Ollama (system) + serve + pull the code model (uses the GPU)
curl -fsSL https://ollama.com/install.sh | sh
(ollama serve >/tmp/ollama.log 2>&1 &) ; sleep 8
ollama pull "$MODEL"
nvidia-smi || echo "WARN: no GPU"

# 2) Python 3.11 venv via uv for arc-agi (Self import needs >=3.11)
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
uv venv --python 3.11 /tmp/arc311
source /tmp/arc311/bin/activate
uv pip install arc-agi numpy
python -c "import sys,arc_agi; print('[arc-agi] py', sys.version.split()[0])"

cd experiments && mkdir -p results
GAMES=$(python -c "import arc_agi; print(' '.join(sorted({(e if isinstance(e,str) else getattr(e,'game_id',str(e))).split('-')[0] for e in arc_agi.Arcade().available_environments})))")
echo "[e86] testing $(echo $GAMES | wc -w) games: $GAMES"
for gid in $GAMES; do
  if gcloud storage ls "$DEST/e86_arc3_${gid}.json" >/dev/null 2>&1; then
    echo "=== $gid already in GCS, skipping (resume) ==="; continue
  fi
  echo "=== e86 synth $gid model=$MODEL ($(date -u +%H:%M:%S)) ==="
  python e86_arc3.py --game "$gid" --steps 300 --ollama "$MODEL" \
    && gcloud storage cp "results/e86_arc3_${gid}.json" "$DEST/" 2>/dev/null || echo "$gid failed"
done
echo "[run_e86_gpu] done -> $DEST/"
