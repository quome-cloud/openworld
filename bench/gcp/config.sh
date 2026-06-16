# Shared config for the ephemeral-GPU benchmark pipeline. Override via env.
#
# This project runs A100 spot capacity (the existing 'ow-a100-harvest' VM proves
# spot A100-40GB works here -- no quota request needed). A single A100-40GB spot
# instance is enough for DreamerV3/TD-MPC2, V-JEPA-2 inference, and a 4-bit mid/large
# LLM for PoE-World. This pipeline provisions a SEPARATE ephemeral instance so it
# never contends with an in-progress harvest run.
set -euo pipefail

PROJECT="${OW_PROJECT:-quome-fastapi}"
ZONE="${OW_ZONE:-us-central1-a}"
GPU="${OW_GPU:-a100-40}"                    # a100-40 (spot) | a100-80 | h100
SPOT="${OW_SPOT:-true}"                     # spot by default (cheap, ephemeral)
INSTANCE="${OW_INSTANCE:-openworld-mgbench}"   # NOT ow-a100-harvest -- separate box
BUCKET="${OW_BUCKET:-gs://openworld-bench}"
REPO_URL="${OW_REPO_URL:-https://github.com/quome-cloud/openworld.git}"
REPO_REF="${OW_REPO_REF:-main}"
# Deep Learning VM: CUDA + PyTorch preinstalled, A100/H100 drivers handled.
IMAGE_FAMILY="${OW_IMAGE_FAMILY:-common-cu124-ubuntu-2204-py310}"
IMAGE_PROJECT="deeplearning-platform-release"
BOOT_DISK_GB="${OW_BOOT_DISK_GB:-200}"

case "$GPU" in
  h100)    MACHINE_TYPE="a3-highgpu-1g" ;;
  a100-80) MACHINE_TYPE="a2-ultragpu-1g" ;;     # 1x A100 80GB
  *)       MACHINE_TYPE="a2-highgpu-1g" ;;       # 1x A100 40GB (default)
esac
SPOT_FLAGS=""
[ "$SPOT" = "true" ] && SPOT_FLAGS="--provisioning-model=SPOT --instance-termination-action=DELETE"

echo "[config] project=$PROJECT zone=$ZONE gpu=$GPU spot=$SPOT machine=$MACHINE_TYPE bucket=$BUCKET instance=$INSTANCE"
