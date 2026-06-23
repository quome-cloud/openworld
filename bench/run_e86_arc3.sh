#!/usr/bin/env bash
# Runs ON a box (or locally): E86 ARC-AGI-3 baseline sweep -- per-game determinism, sparsity, and
# baseline level completion across the public games. Synthesis (--ollama) needs a code model.
# Usage: bash bench/run_e86_arc3.sh gs://openworld-bench e86-arc3 [steps]
set -euo pipefail
BUCKET="${1:-gs://openworld-bench}"; RUN_ID="${2:-e86-arc3}"; STEPS="${3:-300}"; DEST="$BUCKET/$RUN_ID"
cd "$(dirname "$0")/.."
pip install -q --upgrade pip
pip install -q arc-agi numpy
cd experiments && mkdir -p results
GAMES=$(python3 -c "import arc_agi; print(' '.join(e if isinstance(e,str) else getattr(e,'game_id',str(e)) for e in (arc_agi.Arcade().available_environments)))" 2>/dev/null | tr -d "[],'")
for G in $GAMES; do
  gid="${G%%-*}"   # short id (e.g. ls20 from ls20-9607627b)
  echo "=== e86 baseline $gid ($(date -u +%H:%M:%S)) ==="
  python3 e86_arc3.py --game "$gid" --steps "$STEPS" \
    && gcloud storage cp "results/e86_arc3_${gid}.json" "$DEST/" 2>/dev/null || echo "$gid failed"
done
echo "[run_e86_arc3] done -> $DEST/"
