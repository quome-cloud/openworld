#!/usr/bin/env bash
# Provision the ephemeral GPU instance. COSTS MONEY and requires GPU quota
# (A100/H100 quota is 0 by default -- request it in Console > IAM & Admin > Quotas).
# Usage: bash bench/gcp/provision.sh
source "$(dirname "$0")/config.sh"

gcloud compute instances create "$INSTANCE" \
  --project="$PROJECT" --zone="$ZONE" \
  --machine-type="$MACHINE_TYPE" $SPOT_FLAGS \
  --maintenance-policy=TERMINATE --no-restart-on-failure \
  --image-family="$IMAGE_FAMILY" --image-project="$IMAGE_PROJECT" \
  --boot-disk-size="${BOOT_DISK_GB}GB" --boot-disk-type=pd-ssd \
  --scopes=cloud-platform \
  --metadata="install-nvidia-driver=True,repo-url=$REPO_URL,repo-ref=$REPO_REF" \
  --labels=purpose=openworld-mgbench,ephemeral=true

echo "[provision] instance '$INSTANCE' up in $ZONE. Drivers install on first boot;"
echo "            give it ~2-3 min, then: bash bench/gcp/run.sh"
