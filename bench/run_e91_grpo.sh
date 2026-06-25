#!/usr/bin/env bash
# E91 GRPO verifier-RL synthesizer on a GPU box. (1) collect transitions in a py3.12 venv (arc-agi),
# (2) GRPO-train the synthesizer on system torch (reads JSON, no arc-agi -> no version clash).
#   bash bench/run_e91_grpo.sh gs://openworld-bench e91-grpo Qwen/Qwen2.5-Coder-7B-Instruct
set -uo pipefail
BUCKET="${1:-gs://openworld-bench}"; RUN_ID="${2:-e91-grpo}"; BASE="${3:-Qwen/Qwen2.5-Coder-7B-Instruct}"
cd "$(dirname "$0")/.."
# 1) collect transitions (py3.12 + arc-agi)
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
uv venv --python 3.12 /tmp/arc312 && source /tmp/arc312/bin/activate
uv pip install "arc-agi==0.9.9" numpy
cd experiments && python collect_arc3_trans.py --out /tmp/arc3_trans --steps 300 ; cd ..
deactivate
# 2) GRPO train (system torch, has CUDA)
pip install -q "transformers>=4.44" peft bitsandbytes accelerate numpy
python experiments/e91_grpo_synth.py --transitions /tmp/arc3_trans --base "$BASE" \
  --steps 300 --group 6 --bucket "$BUCKET/$RUN_ID" || echo "e91 incomplete"
echo "[run_e91_grpo] done -> $BUCKET/$RUN_ID/"
