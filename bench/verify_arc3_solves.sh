#!/usr/bin/env bash
# One-command reproduction: replay every saved ARC-AGI-3 winning sequence against the live game and
# confirm each completes a level. Set PYTHON to a py>=3.12 interpreter with arc-agi==0.9.9 + numpy,
# or this builds a uv venv. Usage: [PYTHON=/path/to/py3.12] bash bench/verify_arc3_solves.sh
set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  command -v uv >/dev/null 2>&1 || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"; }
  uv venv --python 3.12 /tmp/arcverify >/dev/null 2>&1 && . /tmp/arcverify/bin/activate
  uv pip install -q "arc-agi==0.9.9" numpy && PY=python
fi
echo "[verify] replaying saved ARC-AGI-3 solves against the live games..."
"$PY" experiments/verify_arc3_solves.py
