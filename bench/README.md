# GPU benchmark: OpenWorld vs learned/perceptual world models (reproducible)

A real head-to-head between OpenWorld's **verified symbolic world model** and the
world models from other labs, on the **shared MiniGrid environment**, run on an
**ephemeral GPU instance**, with all artifacts written to **GCS** and pulled back
into this repo for reproducibility.

## What's compared, and why it's fair
The shared environment is MiniGrid (`experiments/minigrid_world.py` expresses its
DoorKey dynamics as verified code). `bench/validate_minigrid.py` first proves the
OpenWorld world reproduces the *real* `minigrid` env bit-for-bit, so every method
runs the same task.

| Method | Species | Repo / weights |
|---|---|---|
| OpenWorld (verified code) | symbolic | this repo (0 training data) |
| DreamerV3 | trained latent MBRL | danijar/dreamerv3 |
| TD-MPC2 | learned WM + MPC (continuous; optional) | nicklashansen/tdmpc2 |
| PoE-World | **same-species** code WM | topwasu/poe-world |
| V-JEPA 2 | perceptual video | facebook/vjepa2 (HF) |

## Pipeline (each step is a committed script)
```bash
# 0. one-time: request A100 quota if needed. Spot A100-40GB works in this project.
bash bench/gcp/provision.sh           # ephemeral SPOT a2-highgpu-1g (A100 40GB), auto-delete
bash bench/gcp/run.sh                 # ssh in -> bench/run_remote.sh -> uploads to gs://openworld-bench/run-<id>/
bash bench/fetch_results.sh run-<id>  # GCS -> experiments/results/minigrid_bench/ (commit for reproducibility)
bash bench/gcp/teardown.sh            # delete the instance (cost control)
```
Config (GPU type, zone, spot, bucket, instance) lives in `bench/gcp/config.sh`;
override via `OW_*` env vars. The instance name defaults to `openworld-mgbench`
(separate from any harvest VM, so concurrent runs never contend).

## Cost & safety
- Spot A100-40GB is ~\$1.5/hr; a full sweep is a few hours. The instance is
  `--provisioning-model=SPOT` with auto-delete on preemption, and `teardown.sh`
  removes it. Always tear down.
- `provision.sh` is the only script that spends money; it requires GPU quota.

## Reproducibility contract
Every run writes `gpu.txt`, `env.json`, per-method JSON, and logs under one
`run-<id>/`. `fetch_results.sh` lands them in the repo; commit them. The exact
instance image, machine type, repo ref, and seeds are recorded so a run is
re-derivable end to end.
