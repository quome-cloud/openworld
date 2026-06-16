#!/usr/bin/env bash
# Delete the ephemeral instance (run after fetch_results.sh). Cost control.
source "$(dirname "$0")/config.sh"
gcloud compute instances delete "$INSTANCE" --project="$PROJECT" --zone="$ZONE" --quiet
echo "[teardown] deleted $INSTANCE"
