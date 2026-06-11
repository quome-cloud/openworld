#!/bin/bash
# Runs every LLM-backed experiment sequentially, logging progress.
cd "$(dirname "$0")"
set -x
for script in e01_fidelity e02_synthesis e03_verifier_ablation e04_rollout_speed \
              e05_codefix_agent e06_judge_selection e07_judge_alignment \
              e10_ood_generalization; do
  echo "=== $script ==="
  python3 "$script.py" || echo "FAILED: $script"
done
echo "=== all done ==="
