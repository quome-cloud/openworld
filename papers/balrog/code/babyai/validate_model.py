"""Model-fidelity validation: lock-stepped rollouts of the symbolic model vs
the real env.

Two policies per episode:
  - random: uniform over the 6 actions (covers movement/pickup/drop/toggle
    edge cases, occlusion states, door mechanics)
  - guided: the planner's solution executed with 20% random action noise
    (drives trajectories into success regions so the verifier mirror is
    exercised where it matters)

Compared every step:
  full grid encode, agent pos, agent dir, carrying (type,color),
  reward, terminated, truncated, and obs['image'] vs render_view()
  (the observation model used by the clean arm).
"""

from __future__ import annotations

import json
import random
import sys
import time

import numpy as np

from balrog_env import TASKS, make_task_env
from symbolic_model import (
    extract_state, extract_instr, EpisodeModel, render_view, full_grid_encode,
)
from planner import solve, PlanFail


def compare(env, model, obs, reward, term, trunc, m_reward, m_term, m_trunc):
    """Return list of disagreement descriptions (empty if exact)."""
    u = env.unwrapped
    errs = []
    st = model.state
    if tuple(u.agent_pos) != st.agent_pos:
        errs.append(f"agent_pos {tuple(u.agent_pos)} vs {st.agent_pos}")
    if int(u.agent_dir) != st.agent_dir:
        errs.append(f"agent_dir {u.agent_dir} vs {st.agent_dir}")
    ec = (None if u.carrying is None else (u.carrying.type, u.carrying.color))
    mc = (None if st.carrying is None
          else (st.objs[st.carrying].type, st.objs[st.carrying].color))
    if ec != mc:
        errs.append(f"carrying {ec} vs {mc}")
    genc = u.grid.encode()
    mgenc = np.array(full_grid_encode(st), dtype=np.uint8)
    if not np.array_equal(genc, mgenc):
        errs.append("full grid encode mismatch")
    if not (abs(reward - m_reward) < 1e-9 and term == m_term and trunc == m_trunc):
        errs.append(f"outcome ({reward},{term},{trunc}) vs ({m_reward},{m_term},{m_trunc})")
    vimg = np.array(render_view(st), dtype=np.uint8)
    if not np.array_equal(obs["image"], vimg):
        errs.append("obs image mismatch")
    if int(obs["direction"]) != st.agent_dir:
        errs.append("obs direction mismatch")
    return errs


def run_episode(task, seed, policy, rng, log):
    env = make_task_env(task)
    obs, info = env.reset(seed=seed)
    st, id2oid = extract_state(env)
    instr = extract_instr(env, id2oid)
    model = EpisodeModel(st, instr, reset_verifier=False)  # env already resolved obj sets

    plan = []
    if policy == "guided":
        try:
            plan, _ = solve(model)
        except PlanFail:
            plan = []
    plan_i = 0

    steps = 0
    disagreements = 0
    while True:
        if policy == "guided" and plan_i < len(plan) and rng.random() > 0.2:
            a = plan[plan_i]
            plan_i += 1
        else:
            a = rng.randrange(6)
        obs, reward, term, trunc, info = env.step(a)
        m_reward, m_term, m_trunc = model.step(a)
        steps += 1
        errs = compare(env, model, obs, reward, term, trunc, m_reward, m_term, m_trunc)
        if errs:
            disagreements += len(errs)
            log(f"  DISAGREE task={task} seed={seed} step={steps} action={a}: {errs}")
        if term or trunc or m_term or m_trunc:
            break
    env.close()
    return steps, disagreements, bool(term and reward > 0)


def main():
    n_eps = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    out = {"episodes": [], "total_steps": 0, "total_disagreements": 0}
    t0 = time.time()

    def log(msg):
        print(msg, flush=True)

    for task_i, task in enumerate(TASKS):
        for ep in range(n_eps):
            for policy in ("random", "guided"):
                seed = 880000 + task_i * 1000 + ep * 10 + (0 if policy == "random" else 5)
                rng = random.Random(seed ^ 0xC0FFEE)
                steps, dis, succ = run_episode(task, seed, policy, rng, log)
                out["episodes"].append({
                    "task": task, "seed": seed, "policy": policy,
                    "steps": steps, "disagreements": dis, "env_success": succ,
                })
                out["total_steps"] += steps
                out["total_disagreements"] += dis
        log(f"{task}: done ({sum(e['steps'] for e in out['episodes'])} cumulative steps, "
            f"{out['total_disagreements']} disagreements)")

    out["wall_clock_s"] = round(time.time() - t0, 1)
    out["n_episodes"] = len(out["episodes"])
    with open("results/model_validation.json", "w") as f:
        json.dump(out, f, indent=1)
    log(f"TOTAL: {out['n_episodes']} episodes, {out['total_steps']} lock-stepped steps, "
        f"{out['total_disagreements']} disagreements, {out['wall_clock_s']}s")


if __name__ == "__main__":
    main()
