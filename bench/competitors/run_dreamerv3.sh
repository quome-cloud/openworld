#!/usr/bin/env bash
# DreamerV3 (trained latent world model + actor-critic) on MiniGrid -- the
# canonical "trained world model" baseline. Measures task success vs env-steps
# (sample efficiency) to contrast with OpenWorld's zero-shot verified world.
set -euo pipefail; OUT="${1:?out dir}"
pip install -q "dreamerv3" "gymnasium>=0.29" "minigrid>=2.3" || pip install -q git+https://github.com/danijar/dreamerv3.git
python -m dreamerv3.main --logdir "$OUT/dreamerv3" \
  --configs minigrid --task "gym_MiniGrid-DoorKey-6x6-v0" \
  --run.steps 200000 --run.eval_every 10000 || echo "dreamerv3 run incomplete"
python bench/competitors/collect.py --method dreamerv3 --logdir "$OUT/dreamerv3" --out "$OUT/dreamerv3.json"
