#!/bin/bash
# Generic E80 autonomous launcher. Set instance metadata 'e80-domain' to one of:
# genomics | tabular | diagnosis | stock. Downloads any domain data, runs the mechanism
# test (world-count ladder + verified-vs-noisy ablation), uploads to GCS, self-deletes.
exec > /var/log/ow-e80.log 2>&1
set -x
META="http://metadata.google.internal/computeMetadata/v1/instance"
SELF=$(curl -s -H "Metadata-Flavor: Google" "$META/name")
ZONE=$(curl -s -H "Metadata-Flavor: Google" "$META/zone" | awk -F/ '{print $NF}')
DOMAIN=$(curl -s -H "Metadata-Flavor: Google" "$META/attributes/e80-domain")
echo "[e80] $(date -u) domain=$DOMAIN"
( sleep 21600; gcloud compute instances delete "$SELF" --zone="$ZONE" --quiet ) &   # 6h backstop

for i in $(seq 1 120); do nvidia-smi >/dev/null 2>&1 && break; sleep 10; done
pip3 install -q --upgrade pip
pip3 install -q "transformers>=4.44,<5" "peft>=0.12" "trl>=0.12,<0.15" "datasets>=2.20" \
                "accelerate>=0.33" "bitsandbytes>=0.43" "jinja2>=3.1" numpy
case "$DOMAIN" in
  tabular) pip3 install -q openml ;;
  stock)   pip3 install -q yfinance ;;
esac

mkdir -p /opt/ow/experiments && cd /opt/ow/experiments
gcloud storage cp "gs://openworld-bench/code/*.py" .

if [ "$DOMAIN" = "genomics" ]; then
  mkdir -p /opt/ow/experiments/data && cd /opt/ow/experiments/data
  curl -fsSL -o pg.zip \
    "https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/DMS_ProteinGym_substitutions.zip"
  unzip -q pg.zip
  export OW_PG_DIR="$(dirname "$(find /opt/ow/experiments/data -name '*.csv' | head -1)")"
  cd /opt/ow/experiments
fi

export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 e80_common.py --domain "$DOMAIN" --bucket "gs://openworld-bench/e80-$DOMAIN" --epochs 2 \
  || echo "[e80] E80_${DOMAIN}_FAILED"
echo "[e80] $(date -u) done"
gcloud storage cp /var/log/ow-e80.log "gs://openworld-bench/e80-$DOMAIN/ow-e80.log"
gcloud compute instances delete "$SELF" --zone="$ZONE" --quiet
