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
OUT="${2:-${SCRATCH}/hybrid_tournament.json}"

COUNTEREXAMPLE_ARGS=()
for ce in "${SCRATCH}"/counter*.json; do
  [[ -f "${ce}" ]] && COUNTEREXAMPLE_ARGS+=(--counterexample "${ce}")
done

python -m experiments.e139.hybrid_rank \
  "${SCRATCH}/schema_packet.json" \
  "${OUT}" \
  "${SCRATCH}/proposal"*.json \
  "${COUNTEREXAMPLE_ARGS[@]}"

python - <<'PY' "${OUT}"
import json
import sys

packet = json.load(open(sys.argv[1]))
for row in packet["ranked"][:12]:
    e139 = row["e139"]
    print(
        f"{row['rank']:2d}. {row['proposal_id']} | "
        f"hybrid={row['hybrid_score']} | e138={row['e138_score']} | "
        f"family={e139['family']} | world={e139['best_world']} | "
        f"penalty={e139['failure_penalty']['total']}"
    )
PY

