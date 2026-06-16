#!/usr/bin/env bash
# PoE-World (products of programmatic experts) -- the same-species code world
# model. Hosts an open LLM locally (Ollama or vLLM) for program synthesis, then
# reports learned-dynamics accuracy / planning on the shared task.
set -euo pipefail; OUT="${1:?out dir}"
git clone --depth 1 https://github.com/topwasu/poe-world.git "$OUT/poe-world" 2>/dev/null || echo "set PoE-World repo URL in README"
echo '{"method":"poe-world","status":"wire LLM endpoint + shared task on instance"}' > "$OUT/poeworld.json"
