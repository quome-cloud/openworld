#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 scratch_arc/<dir> [out.json]" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRATCH="$1"
if [[ "${SCRATCH}" != /* ]]; then
  SCRATCH="${ROOT}/${SCRATCH}"
fi
OUT="${2:-${SCRATCH}/e139_manyworld_ranking.json}"

COUNTEREXAMPLE_ARGS=()
if [[ -f "${SCRATCH}/counterexample.json" ]]; then
  COUNTEREXAMPLE_ARGS+=(--counterexample "${SCRATCH}/counterexample.json")
fi

python -m experiments.e139.manyworld_semiring \
  "${OUT}" \
  "${SCRATCH}/proposal"*.json \
  "${COUNTEREXAMPLE_ARGS[@]}"

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

