#!/usr/bin/env bash
# Unattended babysitter for an overnight harvest. Polls the VM, pulls traces
# INCREMENTALLY each loop (so a spot preemption can't lose the whole run), and
# DELETES the VM on ANY terminal condition: harvest complete, harvest process
# died, max-timeout, or VM already gone. Designed to be safe to leave running.
#
#   MAX_HOURS=8 POLL_SECS=120 ./watch_and_teardown.sh &
#
# NOT `set -e`: a transient ssh/scp blip must never kill the watcher (that would
# leave the VM billing). Every terminal path ends in a delete attempt.
set -uo pipefail
cd "$(dirname "$0")"
source ./config.env
REPO_ROOT="$(git rev-parse --show-toplevel)"
LOG="$PWD/watch.log"
MAX_HOURS="${MAX_HOURS:-8}"
POLL_SECS="${POLL_SECS:-120}"

log(){ echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG"; }
ssh_vm(){ gcloud compute ssh "$VM_NAME" --zone="$ZONE" --project="$PROJECT" --command="$1" 2>/dev/null; }
vm_status(){ gcloud compute instances describe "$VM_NAME" --zone="$ZONE" --project="$PROJECT" --format='value(status)' 2>/dev/null; }

pull_traces(){
  mkdir -p "$REPO_ROOT/$TRACE_DIR"
  if gcloud compute scp --recurse --zone="$ZONE" --project="$PROJECT" \
       "$VM_NAME:openworld/$TRACE_DIR/" "$REPO_ROOT/$(dirname "$TRACE_DIR")/" 2>/dev/null; then
    local n; n=$(cat "$REPO_ROOT/$TRACE_DIR"/*.traces.jsonl 2>/dev/null | wc -l | tr -d ' ')
    log "pulled traces -> $REPO_ROOT/$TRACE_DIR (${n:-0} lines total)"
  else
    log "trace pull skipped/failed (dir may not exist yet; retry next loop)"
  fi
}

delete_vm(){
  gcloud compute instances delete "$VM_NAME" --zone="$ZONE" --project="$PROJECT" --quiet 2>/dev/null \
    && log "VM DELETED — billing stopped." || log "delete returned nonzero (VM may already be gone)."
}

log "watcher up: vm=$VM_NAME zone=$ZONE max=${MAX_HOURS}h poll=${POLL_SECS}s"
START=$(date +%s); SEEN_RUNNING=0
while true; do
  ST="$(vm_status)"
  if [[ -z "$ST" ]]; then
    log "VM no longer exists (preempted+auto-deleted, or already torn down). Done. Traces = last incremental pull."
    exit 0
  fi
  if [[ "$ST" != "RUNNING" ]]; then
    log "VM status=$ST (preemption/stop). Ensuring it is deleted."
    delete_vm; exit 0
  fi

  pull_traces   # incremental safety copy

  if ssh_vm "grep -q 'HARVEST COMPLETE' ~/harvest.log 2>/dev/null"; then
    log "=== HARVEST COMPLETE ==="; pull_traces; delete_vm; exit 0
  fi

  if ssh_vm "pgrep -f _run.sh >/dev/null 2>&1"; then
    SEEN_RUNNING=1
  elif (( SEEN_RUNNING == 1 )); then
    # We saw it running, now it's gone with no COMPLETE marker -> crashed/errored.
    log "harvest process ended WITHOUT 'HARVEST COMPLETE' — likely error. Last log:"
    ssh_vm "tail -n 40 ~/harvest.log" | tee -a "$LOG"
    pull_traces; delete_vm; exit 1
  fi
  # else: not seen running yet (still pulling model / starting) — keep waiting.

  ELAPSED_H=$(( ( $(date +%s) - START ) / 3600 ))
  if (( ELAPSED_H >= MAX_HOURS )); then
    log "MAX ${MAX_HOURS}h safety cap hit — tearing down regardless."
    pull_traces; delete_vm; exit 2
  fi
  sleep "$POLL_SECS"
done
