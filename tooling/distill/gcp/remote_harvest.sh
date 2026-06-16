#!/usr/bin/env bash
# Runs ON THE VM. Installs Ollama, pulls the teacher, runs the harvest under
# nohup so an SSH drop or spot preemption doesn't kill it. Re-runnable: Ollama
# install + model pull are skipped if already present, and --log-traces APPENDS,
# so a resumed run continues accumulating traces.
#
# Env passed in by run.sh: TEACHER_MODEL, RECIPES, SEEDS, TRACE_DIR
set -euo pipefail
cd "$HOME/openworld"

TEACHER_MODEL="${TEACHER_MODEL:?}"; RECIPES="${RECIPES:?}"
SEEDS="${SEEDS:?}"; TRACE_DIR="${TRACE_DIR:?}"

# --- Ollama --------------------------------------------------------------------
if ! command -v ollama >/dev/null 2>&1; then
  echo ">> installing Ollama"; curl -fsSL https://ollama.com/install.sh | sh
fi
if ! pgrep -x ollama >/dev/null 2>&1; then
  echo ">> starting ollama serve"; nohup ollama serve >"$HOME/ollama.log" 2>&1 &
  sleep 5
fi
# Pull the teacher (idempotent; ~40GB). Guard: refuse a *-poisoned* teacher.
case "$TEACHER_MODEL" in
  *poison*) echo "!! refusing poisoned teacher: $TEACHER_MODEL"; exit 2 ;;
esac
echo ">> pulling $TEACHER_MODEL"; ollama pull "$TEACHER_MODEL"

# --- harvest -------------------------------------------------------------------
# Background + nohup so it survives disconnects. Progress in harvest.log.
mkdir -p "$TRACE_DIR"
cat > "$HOME/_run.sh" <<EOF
set -euo pipefail
cd "$HOME/openworld"
for r in $RECIPES; do
  echo "=== harvesting \$r ==="
  python3 -m openworld.bench "recipes/\$r.json" run \
    --models "$TEACHER_MODEL" --seeds "$SEEDS" --log-traces "$TRACE_DIR"
done
echo "=== HARVEST COMPLETE ==="
EOF
chmod +x "$HOME/_run.sh"
nohup bash "$HOME/_run.sh" >"$HOME/harvest.log" 2>&1 &
echo ">> harvest launched (pid $!). Tail with: tail -f ~/harvest.log"
