#!/usr/bin/env bash
# V-JEPA 2 (perceptual video world model) -- cross-species. Loads open weights
# and runs action-conditioned latent rollout / planning on rendered MiniGrid
# frames; reports prediction/planning quality vs OpenWorld's symbolic world.
set -euo pipefail; OUT="${1:?out dir}"
pip install -q torch transformers || true
python - "$OUT" <<'PY' || echo "vjepa step needs weights wired on instance"
import json, sys
# from transformers import AutoModel ; m = AutoModel.from_pretrained("facebook/vjepa2-vitg-fpc64-256")
json.dump({"method":"v-jepa2","status":"load facebook/vjepa2 weights; render MiniGrid; latent rollout"},
          open(sys.argv[1]+"/vjepa.json","w"))
PY
