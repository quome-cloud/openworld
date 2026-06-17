"""E65 - World-model head-to-head on a shared benchmark: MiniGrid DoorKey-6x6.

Three world models, same task, three species:

  * OpenWorld (verified symbolic code) -- plans over its exact transition with BFS.
    Zero training data; the transition is validated bit-for-bit against the real
    Farama `minigrid` environment (bench/validate_minigrid.py) so the task is
    genuinely shared.
  * DreamerV3 (learned, from pixels) -- a latent world model + actor-critic trained
    on 48x48 RGB frames on a dedicated A100. Parsed from the committed training log.
  * V-JEPA-2 (perceptual / latent video model) -- pretrained on internet video; we
    report a representation-drift metric, NOT a task success rate, because V-JEPA is
    a different (latent/perceptual) species with no symbolic policy of its own.

This script is deterministic and offline: OpenWorld's result is recomputed live
(BFS over the verified world); the learned/perceptual numbers are parsed from the
committed GPU artifacts under results/minigrid_bench/. save_results runs BEFORE the
asserts so a failed check never loses the run.
"""

import re
import statistics
from collections import deque
from pathlib import Path

from common import save_results
from minigrid_world import (MINIGRID_ACTIONS, MINIGRID_INITIAL,
                            build_minigrid_world, solved)
from openworld.state import Action

ART = Path(__file__).parent / "results" / "minigrid_bench"
SOLVE_REWARD = 0.05   # MiniGrid DoorKey: success episodes return ~0.9-0.97; 0 otherwise


def _skey(s):
    return tuple(sorted(s.items()))


def openworld_plan():
    """0-shot BFS over the verified world -> shortest action plan that solves it."""
    w = build_minigrid_world()
    start = dict(MINIGRID_INITIAL)
    seen = {_skey(start)}
    q = deque([(start, [])])
    while q:
        s, plan = q.popleft()
        if solved(s):
            return {"success": 1.0, "plan_length": len(plan), "plan": plan,
                    "training_transitions": 0, "verified": True,
                    "validated_vs_real_minigrid": "bit-exact, 600/600 steps "
                    "(bench/validate_minigrid.py)"}
        for a in MINIGRID_ACTIONS:
            ns = dict(w.transition.step(s, Action(a)))
            k = _skey(ns)
            if k not in seen:
                seen.add(k)
                q.append((ns, plan + [a]))
    return {"success": 0.0, "plan_length": None, "training_transitions": 0}


def parse_dreamer(log_path):
    """Sample-efficiency from the sheeprl training log (committed GPU artifact)."""
    text = log_path.read_text(errors="ignore")
    pairs = [(int(m.group(1)), float(m.group(2)))
             for m in re.finditer(r"policy_step=(\d+),\s*reward_env_\d+=([0-9.]+)", text)]
    if not pairs:
        return {"status": "missing", "note": "no episode records parsed"}
    steps = [s for s, _ in pairs]
    first = next((s for s, r in pairs if r > SOLVE_REWARD), None)

    def window(lo, hi):
        rs = [r for s, r in pairs if lo <= s < hi]
        return None if not rs else (round(sum(r > SOLVE_REWARD for r in rs) / len(rs), 3),
                                    round(statistics.mean(rs), 3), len(rs))

    curve = []
    for lo in range(0, max(steps) + 1, 25000):
        w = window(lo, lo + 25000)
        if w:
            curve.append({"step": lo + 25000, "solve_rate": w[0], "mean_reward": w[1], "n": w[2]})
    fin = window(max(0, max(steps) - 25000), max(steps) + 1)
    return {"status": "ran", "obs": "48x48x3 RGB", "max_step": max(steps),
            "steps_to_first_solve": first, "final_solve_rate": fin[0],
            "final_mean_reward": fin[1], "episodes_logged": len(pairs),
            "curve": curve, "seeds": 1, "hardware": "A100 (us-central1-f)"}


def read_vjepa(json_path):
    import json
    d = json.loads(json_path.read_text())
    return {"model": d.get("model"), "status": d.get("status"),
            "metric": "mean_consecutive_cosine",
            "value": d.get("mean_consecutive_cosine"),
            "embedding_dim": d.get("embedding_dim"), "frames": d.get("frames"),
            "note": "perceptual representation drift over a rollout; NOT a task "
                    "success rate -- V-JEPA is a different (latent/perceptual) species"}


def main():
    ow = openworld_plan()
    dv3 = parse_dreamer(ART / "dreamer.log")
    vj = read_vjepa(ART / "vjepa.json")
    results = {"task": "MiniGrid-DoorKey-6x6",
               "shared_benchmark": "OpenWorld transition validated bit-for-bit vs "
                                   "Farama minigrid; learned/perceptual models consume "
                                   "its rendered pixels",
               "openworld": ow, "dreamerv3": dv3, "vjepa2": vj}
    save_results("e65_minigrid_bench", results)   # BEFORE asserts

    # self-checks: the claims this experiment makes
    assert ow["success"] == 1.0 and ow["plan_length"] and ow["training_transitions"] == 0, ow
    assert dv3["status"] == "ran" and dv3["final_solve_rate"] >= 0.9, dv3
    assert dv3["steps_to_first_solve"] and dv3["steps_to_first_solve"] > 0, dv3
    assert vj["status"] == "ran" and vj["value"] is not None, vj
    print(f"[ok] OpenWorld: 0-shot solve, plan_length={ow['plan_length']}, 0 training")
    print(f"[ok] DreamerV3: first solve @ {dv3['steps_to_first_solve']} steps, "
          f"final solve_rate={dv3['final_solve_rate']}, mean_reward={dv3['final_mean_reward']}")
    print(f"[ok] V-JEPA-2: {vj['metric']}={vj['value']} (perceptual, different species)")


if __name__ == "__main__":
    main()
