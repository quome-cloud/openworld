#!/usr/bin/env bash
# Pull a run's artifacts from GCS into the repo for reproducibility.
# Usage: bash bench/fetch_results.sh run-<id>
source "$(dirname "$0")/gcp/config.sh"
RUN_ID="${1:?run id required}"
DEST="experiments/results/minigrid_bench"
mkdir -p "$DEST"
gcloud storage cp -r "$BUCKET/$RUN_ID" "$DEST/"
echo "[fetch] $BUCKET/$RUN_ID -> $DEST/$RUN_ID  (commit this for reproducibility)"
