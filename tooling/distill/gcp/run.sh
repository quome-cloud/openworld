#!/usr/bin/env bash
# One-shot orchestrator: provision H100 spot -> ship code (incl. the local
# num_ctx patch) -> launch the harvest. Does NOT tear down — you call
# fetch_teardown.sh once the run is done, so you control when billing stops.
#
#   cp config.env.example config.env   # edit PROJECT/ZONE
#   ./run.sh
set -euo pipefail
cd "$(dirname "$0")"
[[ -f config.env ]] || { echo "!! copy config.env.example -> config.env first"; exit 1; }
source ./config.env
REPO_ROOT="$(git rev-parse --show-toplevel)"

# 1. provision (preflights quota, waits for the H100)
./provision.sh

# 2. ship code. Tar only what bench needs; EXCLUDE .git, traces, results, large
#    artifacts. This captures the local (uncommitted) num_ctx=8192 patch.
echo ">> packaging repo (zero-dep core; datasets are tiny) ..."
TARBALL="$(mktemp -t owharvest.XXXX).tgz"
tar -C "$REPO_ROOT" \
  --exclude='.git' --exclude='traces' --exclude='**/results/*.json' \
  --exclude='**/__pycache__' \
  -czf "$TARBALL" openworld recipes datasets pyproject.toml tooling/distill
gcloud compute scp --zone="$ZONE" --project="$PROJECT" "$TARBALL" "$VM_NAME:~/repo.tgz"
gcloud compute scp --zone="$ZONE" --project="$PROJECT" remote_harvest.sh "$VM_NAME:~/remote_harvest.sh"
rm -f "$TARBALL"

# 3. unpack + launch harvest on the VM
gcloud compute ssh "$VM_NAME" --zone="$ZONE" --project="$PROJECT" --command="
  set -e
  rm -rf ~/openworld && mkdir -p ~/openworld
  tar -C ~/openworld -xzf ~/repo.tgz
  chmod +x ~/remote_harvest.sh
  TEACHER_MODEL='$TEACHER_MODEL' RECIPES='$RECIPES' SEEDS='$SEEDS' TRACE_DIR='$TRACE_DIR' \
    ~/remote_harvest.sh
"

cat <<EOF

================================================================================
Harvest launched on $VM_NAME ($ZONE). It runs under nohup, so you can disconnect.

  Watch progress:
    gcloud compute ssh $VM_NAME --zone=$ZONE --project=$PROJECT --command='tail -f ~/harvest.log'

  When ~/harvest.log shows 'HARVEST COMPLETE' (or whenever you want to stop):
    ./fetch_teardown.sh        # copies traces back to ./$TRACE_DIR and DELETES the VM

  Spot note: if the VM is preempted it self-deletes. Re-run ./run.sh to make a
  fresh box; --log-traces appends, so re-running continues the harvest.
================================================================================
EOF
