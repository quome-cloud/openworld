#!/usr/bin/env bash
# Runs ON the GPU box: E84 cross-world transfer, 3 seeds, for a chosen real domain.
# Usage (on box): bash bench/run_crossworld.sh gs://openworld-bench <run_id> <domain>
set -euo pipefail
BUCKET="${1:-gs://openworld-bench}"; RUN_ID="${2:-crossworld}"; DOMAIN="${3:-listfn}"
DEST="$BUCKET/$RUN_ID"
cd "$(dirname "$0")/.."
pip install -q --upgrade pip
pip install -q "transformers>=4.44,<5" "peft>=0.12" "datasets>=2.20" \
               "accelerate>=0.33" "bitsandbytes>=0.43" "jinja2>=3.1" numpy
python3 -c "import torch; print('[torch]', torch.__version__, 'cuda', torch.cuda.is_available())"
nvidia-smi || echo "WARN: no GPU"
case "$DOMAIN" in
  listfn)  WORLDS=listfn_worlds.jsonl;  INSTR="Infer the hidden rule mapping each input list to its output, then produce the output for the final input.";;
  tabular) WORLDS=tabular_worlds.jsonl; INSTR="Infer the hidden rule mapping each input's features to its output label, then produce the output for the final input.";;
  clrs)    WORLDS=clrs_worlds.jsonl;    INSTR="Infer the hidden algorithm mapping each input to its output, then produce the output for the final input.";;
  *) echo "unknown domain $DOMAIN"; exit 1;;
esac
cd experiments && mkdir -p data
gcloud storage cp "gs://openworld-bench/data/$WORLDS" "data/$WORLDS"
for S in 0 1 2; do
  echo "=== crossworld $DOMAIN seed $S ($(date -u +%H:%M:%S)) ==="
  python3 e84_crossworld.py --worlds "data/$WORLDS" --domain "$DOMAIN" \
    --instruction "$INSTR" --bucket "$DEST" --seed "$S" --steps 350 --n_eval 6 || echo "seed $S FAILED"
done
echo "[run_crossworld] done -> $DEST/e84_crossworld_${DOMAIN}_seed{0,1,2}.json"
