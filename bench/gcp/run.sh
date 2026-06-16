#!/usr/bin/env bash
# Drive the benchmark on the instance over SSH, then it uploads artifacts to GCS.
# Usage: bash bench/gcp/run.sh
source "$(dirname "$0")/config.sh"

RUN_ID="run-$(date -u +%Y%m%dT%H%M%SZ)"
echo "[run] RUN_ID=$RUN_ID -> $BUCKET/$RUN_ID/"

gcloud compute ssh "$INSTANCE" --project="$PROJECT" --zone="$ZONE" --command="
  set -euo pipefail
  if [ ! -d openworld ]; then git clone --branch '$REPO_REF' '$REPO_URL'; fi
  cd openworld && git pull --ff-only || true
  bash bench/run_remote.sh '$BUCKET' '$RUN_ID'
"
echo "[run] done. Pull results into the repo with: bash bench/fetch_results.sh $RUN_ID"
