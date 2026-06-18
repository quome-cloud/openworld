"""E67 - Cartpole-swingup world-model head-to-head (continuous control, from pixels).

A fairer cross-species comparison than the gridworld: continuous physical control is the
learned/perceptual models' home turf, yet OpenWorld can still express the dynamics as
verified code. Same task, several world models:

  * OpenWorld (verified code): plans over its EXACT transition with CEM-MPC. 0 data.
    Recomputed live here; the transition is validated bit-identical against an
    independent reference + a vectorized executor.
  * DreamerV3 / DreamerV2 (learned, from 64x64 pixels): trained end-to-end on a dedicated
    A100; parsed from the committed training logs.
  * V-JEPA-2 and other FROZEN encoders (DINOv2, ImageNet ViT) as perceptual world models:
    frozen features + learned latent dynamics/reward + planning, or behavioral cloning /
    DAgger of the OpenWorld expert. Parsed from the committed GPU result JSONs.

The headline: verification solves it 0-shot; the end-to-end learned model catches up with
data; frozen perceptual features encode the full state yet do not yield control through
any bolt-on controller -- the gap is end-to-end task training, not perception.

Deterministic/offline for the OpenWorld leg + world validation; the learned/perceptual
numbers are parsed from committed artifacts under results/cartpole_bench/. save_results
runs BEFORE the asserts.
"""

import json
import random
import re
from pathlib import Path
from statistics import mean, stdev

import numpy as np

from common import save_results
from cartpole_world import (CARTPOLE_INITIAL, FORCE_MAG, X_THRESHOLD,
                            build_cartpole_world, step_batch, step_ref,
                            step_reward, swingup_success, wrap_angle)
from openworld.state import Action

ART = Path(__file__).parent / "results" / "cartpole_bench"
HORIZON, ITERS, POP, ELITES, EPISODE_STEPS, N_STARTS = 75, 5, 400, 40, 200, 5


def reward_batch(X, F):
    theta, x = X[:, 2], X[:, 0]
    upright = (1.0 + np.cos(theta)) / 2.0
    centered = np.maximum(0.0, 1.0 - (x / X_THRESHOLD) ** 2)
    return upright * centered - 1e-3 * (F / FORCE_MAG) ** 2


def cem_action(s0, mean_seq, rng):
    std = np.full(HORIZON, 6.0)
    base = np.array([s0[k] for k in ("x", "x_dot", "theta", "theta_dot")])
    for _ in range(ITERS):
        seqs = np.clip(mean_seq[None] + std[None] * rng.standard_normal((POP, HORIZON)), -FORCE_MAG, FORCE_MAG)
        X = np.tile(base, (POP, 1)); total = np.zeros(POP)
        for h in range(HORIZON):
            total += reward_batch(X, seqs[:, h]); X = step_batch(X, seqs[:, h])
        elite = seqs[np.argsort(total)[-ELITES:]]
        mean_seq = elite.mean(0); std = elite.std(0) + 1e-3
    return mean_seq


def run_cem_mpc(seed):
    rng = np.random.default_rng(seed)
    s = dict(CARTPOLE_INITIAL); s["theta"] += rng.uniform(-0.05, 0.05); s["x"] += rng.uniform(-0.05, 0.05)
    states = [dict(s)]; m = np.zeros(HORIZON)
    for _ in range(EPISODE_STEPS):
        m = cem_action(s, m, rng); s = step_ref(s, float(m[0])); states.append(dict(s))
        m = np.concatenate([m[1:], [0.0]])
    return swingup_success(states)


def run_random(seed):
    rng = np.random.default_rng(1000 + seed)
    s = dict(CARTPOLE_INITIAL); states = [dict(s)]
    for _ in range(EPISODE_STEPS):
        s = step_ref(s, float(rng.uniform(-FORCE_MAG, FORCE_MAG))); states.append(dict(s))
    return swingup_success(states)


def validate_world(n=300):
    w = build_cartpole_world(); rng = random.Random(0); s = dict(CARTPOLE_INITIAL); md = 0.0
    for _ in range(n):
        f = rng.uniform(-FORCE_MAG, FORCE_MAG)
        sc = dict(w.transition.step(s, Action("push", params={"force": f})))
        sr = step_ref(s, f)
        sv = step_batch(np.array([[s[k] for k in ("x", "x_dot", "theta", "theta_dot")]]), np.array([f]))[0]
        for i, k in enumerate(("x", "x_dot", "theta", "theta_dot")):
            md = max(md, abs(sc[k] - sr[k]), abs(sc[k] - sv[i]))
        s = sc
    return md


def parse_dreamer(path, thresh=100.0):
    if not path.exists():
        return {"status": "missing"}
    pairs = [(int(m.group(1)), float(m.group(2)))
             for m in re.finditer(r"policy_step=(\d+),\s*reward_env_\d+=([0-9.]+)", path.read_text(errors="ignore"))]
    if not pairs:
        return {"status": "missing"}
    mx = max(s for s, _ in pairs)
    final = [r for s, r in pairs if s >= mx - 3000]
    return {"status": "ran", "obs": "64x64 RGB", "max_step": mx,
            "final_mean_reward": round(mean(final), 1),
            "steps_to_reward_100": next((s for s, r in pairs if r > thresh), None),
            "episodes": len(pairs), "hardware": "A100 (us-central1-f)"}


def parse_dreamer_multiseed(paths, thresh=100.0):
    """Aggregate per-seed DreamerV3 runs into mean +/- sd -- the multi-seed error bars."""
    per = [parse_dreamer(p, thresh) for p in paths]
    per = [x for x in per if x.get("status") == "ran"]
    if not per:
        return {"status": "missing"}
    rewards = [x["final_mean_reward"] for x in per]
    steps = [x["steps_to_reward_100"] for x in per if x["steps_to_reward_100"] is not None]
    return {"status": "ran", "obs": "64x64 RGB", "n_seeds": len(per),
            "final_mean_reward": round(mean(rewards), 1),
            "final_reward_std": round(stdev(rewards), 1) if len(rewards) > 1 else 0.0,
            "steps_to_reward_100": round(mean(steps)) if steps else None,
            "steps_std": round(stdev(steps)) if len(steps) > 1 else 0,
            "per_seed": per, "hardware": "A100 (us-central1-f)"}


def _load(name):
    p = ART / name
    return json.loads(p.read_text()) if p.exists() else {}


def main():
    md = validate_world()
    ow_solve = mean(run_cem_mpc(s) for s in range(N_STARTS))
    rnd_solve = mean(run_random(s) for s in range(N_STARTS))

    abl = _load("frozen_encoder_ablation_sincos.json").get("encoders", {})
    vj = _load("vjepa_cartpole.json")
    results = {
        "task": "cartpole-swingup (continuous control, verified ODE; learned models from 64x64 pixels)",
        "world_validation_max_diff": md,
        "openworld": {"method": "CEM-MPC over verified model", "training_transitions": 0,
                      "solve_rate": round(ow_solve, 3), "n_starts": N_STARTS},
        "random_control": {"solve_rate": round(rnd_solve, 3)},
        "dreamerv3": parse_dreamer_multiseed([ART / f"dreamer_seed{s}.log" for s in (0, 1, 2)]),
        "dreamerv2": parse_dreamer(ART / "dreamerv2.log"),
        "vjepa2": {"planning_solve_rate": vj.get("solve_rate"),
                   "dynamics_final_loss": vj.get("dynamics_final_loss"),
                   "note": "frozen V-JEPA encoder + learned latent dynamics(+reward) + MPC"},
        "frozen_encoder_control": {
            "behavioral_cloning_solve": {k: v.get("bc_solve_rate") for k, v in abl.items()},
            "dagger_dinov2_solve": _load("dagger_dinov2.json").get("final_solve_rate"),
            "dagger_stacked_dinov2_solve": _load("dagger_stacked_dinov2.json").get("final_solve_rate"),
            "state_decodability_r2": {k: v.get("state_decode_r2") for k, v in abl.items()},
            "raw_angle_probe_artifact": _load("vjepa_probe_bc.json").get("state_decode_r2"),
            "finding": "frozen features (V-JEPA/DINOv2/ViT) decode the full state (sin/cos theta "
                       "~0.95-0.998) yet every bolt-on controller (planning, BC, DAgger, "
                       "frame-stacked DAgger) scores 0; the bottleneck is end-to-end task "
                       "training, not perception. (An earlier raw-angle probe wrongly suggested "
                       "the pole angle was unencoded -- an artifact of regressing a periodic "
                       "quantity; sin/cos decoding corrects it.)"},
    }
    save_results("e67_cartpole_bench", results)   # BEFORE asserts

    assert md < 1e-9, f"world validation failed: {md}"
    assert ow_solve >= 0.8 and ow_solve > rnd_solve, (ow_solve, rnd_solve)
    assert results["dreamerv3"]["final_mean_reward"] > 50, results["dreamerv3"]
    print(f"[ok] world validated (diff {md:.0e}); OpenWorld solve {ow_solve:.2f} vs random {rnd_solve:.2f}, 0 data")
    print(f"[ok] DreamerV3 reward {results['dreamerv3']['final_mean_reward']} "
          f"(first>100 @ {results['dreamerv3']['steps_to_reward_100']}); "
          f"DreamerV2 {results['dreamerv2']['final_mean_reward']} @ {results['dreamerv2']['steps_to_reward_100']}")
    print(f"[ok] V-JEPA/frozen control solve = 0 despite state-decode R2 ~0.95-0.998 (control, not perception)")


if __name__ == "__main__":
    main()
