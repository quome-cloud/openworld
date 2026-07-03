#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRATCH="${ROOT}/scratch_arc/jscodex_ka59"
OUT="${SCRATCH}/e139_manyworld_ranking.json"

python -m experiments.e139.manyworld_semiring \
  "${OUT}" \
  "${SCRATCH}/proposal"*.json \
  --counterexample "${SCRATCH}/counterexample.json"

python - <<'PY' "${OUT}"
import json
import sys

packet = json.load(open(sys.argv[1]))
for row in packet["ranked"][:12]:
    print(
        f"{row['rank']:2d}. {row['candidate_id']} | "
        f"{row['best_world']} | semiring={row['semiring']} | "
        f"score={row['path_integral_score']}"
    )
PY

