#!/usr/bin/env bash
# E86b AGENTIC synthesis with qwen-30B (Ollama) -- the GPU arm of the {one-shot,agentic}x{qwen,Claude}
# benchmark. Same verifier-in-the-loop as agentic-Claude (e86b_agentic.py), different backend.
#   bash bench/run_e86b_qwen.sh gs://openworld-bench e86b-qwen qwen3-coder:30b
set -uo pipefail
BUCKET="${1:-gs://openworld-bench}"; RUN_ID="${2:-e86b-qwen}"; MODEL="${3:-qwen3-coder:30b}"
DEST="$BUCKET/$RUN_ID"
cd "$(dirname "$0")/.."

# Ollama + model (num_ctx capped inside e86_arc3.ollama so the 30B fits VRAM, per CLAUDE.md)
curl -fsSL https://ollama.com/install.sh | sh
(ollama serve >/tmp/ollama.log 2>&1 &) ; sleep 8
ollama pull "$MODEL"
nvidia-smi || echo "WARN no GPU"

# Python 3.12 venv for arc-agi (requires >=3.12)
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
uv venv --python 3.12 /tmp/arc311
source /tmp/arc311/bin/activate
uv pip install "arc-agi==0.9.9" numpy

cd experiments && mkdir -p results
# medium-fidelity games (room to improve) + a few high/low as anchors
for g in vc33 s5i5 sp80 re86 cd82 dc22 tu93 ka59 ar25 ls20 sk48 g50t sc25 wa30 r11l; do
  out="results/e86b_ollama_${MODEL//:/_}_${g}.json"
  if gcloud storage ls "$DEST/$(basename "$out")" >/dev/null 2>&1; then echo "$g done, skip"; continue; fi
  echo "=== e86b qwen $g $(date -u +%H:%M:%S) ==="
  python e86b_agentic.py --game "$g" --backend "ollama:$MODEL" --rounds 8 --out "$out" \
    && gcloud storage cp "$out" "$DEST/" 2>/dev/null || echo "$g FAILED"
done
echo "[run_e86b_qwen] done -> $DEST/"
