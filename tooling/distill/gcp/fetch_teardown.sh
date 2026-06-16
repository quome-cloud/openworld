#!/usr/bin/env bash
# Copy the harvested traces back to the local repo, then DELETE the VM.
# Run this once ~/harvest.log shows "HARVEST COMPLETE" (or whenever you want to
# stop paying — traces flush per-attempt, so a partial copy is still usable).
#
#   ./fetch_teardown.sh            # fetch + delete
#   ./fetch_teardown.sh --keep     # fetch only, leave the VM running
set -euo pipefail
cd "$(dirname "$0")"
source ./config.env
REPO_ROOT="$(git rev-parse --show-toplevel)"

echo ">> copying $TRACE_DIR back to $REPO_ROOT/$TRACE_DIR"
mkdir -p "$REPO_ROOT/$TRACE_DIR"
gcloud compute scp --recurse --zone="$ZONE" --project="$PROJECT" \
  "$VM_NAME:openworld/$TRACE_DIR/" "$REPO_ROOT/$(dirname "$TRACE_DIR")/"

echo ">> trace files now local:"
ls -la "$REPO_ROOT/$TRACE_DIR/" || true

if [[ "${1:-}" == "--keep" ]]; then
  echo ">> --keep: leaving $VM_NAME running (still billing!)"; exit 0
fi
echo ">> deleting VM $VM_NAME (stops billing)"
gcloud compute instances delete "$VM_NAME" --zone="$ZONE" --project="$PROJECT" --quiet
echo ">> done. VM deleted."
