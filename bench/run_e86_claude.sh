#!/usr/bin/env bash
# Reproducible E86: ARC-AGI-3 verified-code-world-model synthesis with CLAUDE (claude -p) -- the
# frontier synthesizer OpenWorld actually uses (cf. `openworld build/optimize`). Needs NO GPU:
# synthesis is claude -p calls; the arc-agi env runs on CPU.
#
# Requirements:
#   - Python >=3.12 with `arc-agi==0.9.9` and numpy (arc-agi/arcengine require >=3.12).
#       python3.12 -m venv .arcv && . .arcv/bin/activate && pip install "arc-agi==0.9.9" numpy
#   - the `claude` CLI, authenticated (claude -p must work headlessly).
#
# Usage:
#   PYTHON=.arcv/bin/python bash bench/run_e86_claude.sh [outdir] [steps]
#   # default outdir=experiments/results/arc3_claude, steps=200
#
# Idempotent: skips any game already written with a non-null verified_exact (resume after a stop).
# Per-game JSON + log land in <outdir>/. Compare against the qwen arms (bench/run_e86_gpu.sh).
set -uo pipefail
OUT="${1:-experiments/results/arc3_claude}"
STEPS="${2:-200}"
PY="${PYTHON:-python3}"
cd "$(dirname "$0")/.."
mkdir -p "$OUT"

# Clean game list (arc_agi spams INFO logs to stdout on import -- suppress so only ids are written).
"$PY" - > "$OUT/games.txt" <<'PYEOF'
import contextlib, io, logging
logging.disable(logging.CRITICAL)
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    import arc_agi
    envs = arc_agi.Arcade().available_environments
ids = sorted({(e if isinstance(e, str) else getattr(e, "game_id", str(e))).split("-")[0] for e in envs})
print("\n".join(ids))
PYEOF
echo "[e86-claude] $(grep -c . "$OUT/games.txt") games -> $OUT (steps=$STEPS)"

while read -r g; do
  [ -z "$g" ] && continue
  if [ -f "$OUT/$g.json" ] && "$PY" -c "import json,sys;sys.exit(0 if json.load(open('$OUT/$g.json')).get('verified_exact') is not None else 1)" 2>/dev/null; then
    echo "  $g already done, skip"; continue
  fi
  echo "=== $g $(date +%H:%M:%S) ==="
  "$PY" experiments/e86_arc3.py --game "$g" --steps "$STEPS" --claude --out "$OUT/$g.json" > "$OUT/$g.log" 2>&1 \
    && echo "  $g OK" || echo "  $g FAILED (see $OUT/$g.log)"
done < "$OUT/games.txt"
echo "[e86-claude] done -> $OUT"
