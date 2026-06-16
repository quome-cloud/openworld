#!/usr/bin/env bash
# Provision a single H100 spot VM for the teacher harvest. Idempotent-ish: errors
# if the VM already exists (delete it first with fetch_teardown.sh).
#
#   ./provision.sh            # uses ./config.env
#
# Preflight checks H100 spot quota so you fail fast instead of mid-create.
set -euo pipefail
cd "$(dirname "$0")"
source ./config.env

echo ">> project=$PROJECT zone=$ZONE machine=$MACHINE_TYPE (SPOT)"

# --- preflight: GPU spot quota --------------------------------------------------
REGION="${ZONE%-*}"
METRIC="${GPU_QUOTA_METRIC:?set GPU_QUOTA_METRIC in config.env}"
NEED="${GPU_COUNT:-1}"
echo ">> checking $METRIC quota in $REGION (need $NEED) ..."
QUOTA=$(gcloud compute regions describe "$REGION" --project "$PROJECT" \
  --flatten="quotas[]" \
  --format="value(quotas.metric, quotas.limit)" 2>/dev/null \
  | awk -v m="$METRIC" '$1==m {print $2; exit}')
echo "   $METRIC limit in $REGION: ${QUOTA:-0}"
if [[ -z "${QUOTA:-}" ]] || (( $(printf '%.0f' "${QUOTA:-0}") < NEED )); then
  echo "!! $METRIC quota (${QUOTA:-0}) < required $NEED. Request an increase:"
  echo "   https://console.cloud.google.com/iam-admin/quotas?project=$PROJECT"
  echo "   (metric: $METRIC, region: $REGION). Aborting."
  exit 1
fi

# --- create the spot VM ---------------------------------------------------------
gcloud compute instances create "$VM_NAME" \
  --project="$PROJECT" \
  --zone="$ZONE" \
  --machine-type="$MACHINE_TYPE" \
  --provisioning-model=SPOT \
  --instance-termination-action=DELETE \
  --maintenance-policy=TERMINATE \
  --image-family="$IMAGE_FAMILY" \
  --image-project="$IMAGE_PROJECT" \
  --boot-disk-size="${BOOT_DISK_GB}GB" \
  --boot-disk-type=pd-ssd \
  --metadata="install-nvidia-driver=True" \
  --scopes=cloud-platform

echo ">> waiting for SSH + NVIDIA driver to come up (first boot installs the driver) ..."
for i in $(seq 1 40); do
  if gcloud compute ssh "$VM_NAME" --zone="$ZONE" --project="$PROJECT" \
       --command="nvidia-smi -L" 2>/dev/null | grep -qiE 'GPU 0:.*NVIDIA'; then
    echo ">> GPU visible. VM ready."
    exit 0
  fi
  echo "   ... not ready yet (attempt $i), sleeping 30s"; sleep 30
done
echo "!! VM did not report an H100 within ~20min. Check the console / serial log."
exit 1
