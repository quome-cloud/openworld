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

## Results so far
- **Shared-environment fidelity (validated on an A100 spot instance, real `minigrid`):**
  the OpenWorld verified world reproduces `MiniGrid-DoorKey-6x6-v0` **bit-for-bit** --
  `exact_step_match_rate = 1.0` over 600 steps (20 episodes), 0 mismatches
  (`experiments/results/minigrid_bench/minigrid_fidelity.json`; also in
  `gs://openworld-bench/run-manual-fidelity/`). This is the fairness precondition,
  and a result on its own: a standard RL benchmark's dynamics expressed as
  zero-data verified code, exact against the real environment.
- The provision -> ssh -> run -> GCS -> fetch -> teardown pipeline is proven on a
  spot `a2-highgpu-1g` (A100-40GB) in `us-central1-a`.

## Competitor head-to-head: status (a dedicated sprint)
Running the *published* systems on the shared env is real integration work, not a
one-shot run:
- **DreamerV3**: the maintained PyTorch port (NM512) ships Crafter/Atari/DMC
  configs but **no MiniGrid** and pins old torch/gym/numpy. Cleanest path is
  `sheeprl` (`exp=dreamer_v3 env=minigrid`) or adding a MiniGrid wrapper + config.
- **V-JEPA 2**: load `facebook/vjepa2` weights and run action-conditioned latent
  rollout/planning on rendered MiniGrid frames (GPU inference).
- **PoE-World**: host an open coder LLM (vLLM) and run its synthesis on a shared
  task -- the same-species comparison.
Each is a clean follow-up on a re-provisioned spot A100 (the scripts here drive it).

## Reproducibility contract
Every run writes `gpu.txt`, `env.json`, per-method JSON, and logs under one
`run-<id>/`. `fetch_results.sh` lands them in the repo; commit them. The exact
instance image, machine type, repo ref, and seeds are recorded so a run is
re-derivable end to end.
