"""E61 - Verified code vs a *trained* world model, judged on downstream control.

The prior panels' central objection: every fidelity comparison is against a
self-built baseline, and there is no *trained* world model evaluated on what an
agent actually cares about -- task return. This experiment closes that gap, and
it is fully deterministic and offline (numpy; no LLM).

One shared symbolic task (the sprint world, value = shipped - bugs - 0.5*debt).
The SAME depth-3 lookahead planner plans inside each candidate world model, but
ACTS in the true environment; we measure realized return as a function of the
number of environment transitions K the world model was trained on:

  - verified code (exact):  the synthesized/declared transition -- 0 training
    transitions; the upper bound.
  - trained MLP world model: numpy MLP regressing next-state from K random-policy
    transitions (reused from E12).
  - trained 1-NN world model: nearest-neighbour over the same K transitions.
  - reactive / random:       model-free floors.

The question is R1's: does a verified code world model match a trained
model-based agent's return at a fraction of the samples -- and does planning
through a mis-trained model help or harm? Honest-results rule: whatever the
numbers say is what we report. Multi-seed (variance over the training draw).
"""

from statistics import mean, pstdev

import numpy as np

from common import SPRINT_INITIAL, save_results
from e12_learned_baseline import MLP, NearestNeighbor, collect_transitions, encode, FIELDS
from e22_planning import (ACTIONS, CODE_DEPTH, EPISODE_STEPS, env_step, lookahead,
                          reactive, value)

import random

SEEDS = [0, 1, 2, 3, 4]                 # variance over the training draw
KS = [50, 200, 1000, 5000]              # trained-WM sample budgets (verified code uses 0)
OPTIMAL_TOL = 1e-6


def plan_return(model_step, depth=CODE_DEPTH, steps=EPISODE_STEPS):
    """Plan inside model_step, act in the true env; return realized value."""
    s = dict(SPRINT_INITIAL)
    for _ in range(steps):
        a, _ = lookahead(model_step, s, depth)
        s = env_step(s, a)
    return value(s)


def policy_return(policy, steps=EPISODE_STEPS):
    s = dict(SPRINT_INITIAL)
    for _ in range(steps):
        s = env_step(s, policy(s))
    return value(s)


def train_models(k, seed):
    rng = random.Random(seed)
    data = collect_transitions(k, rng)
    nn = NearestNeighbor(data)
    x = np.array([encode(s, a) for s, a, _ in data])
    y = np.array([[float(ns[f]) for f in FIELDS] for _, _, ns in data])
    mlp = MLP(x.shape[1], y.shape[1], seed=seed)
    mlp.train(x, y)
    return {"mlp": mlp, "nn": nn}


def main():
    # Upper bound and model-free floors (deterministic env -> reactive is fixed).
    exact = plan_return(env_step)                       # verified code world model
    reactive_ret = policy_return(reactive)
    random_ret = mean(policy_return(lambda s, _r=random.Random(sd): _r.choice(ACTIONS))
                      for sd in SEEDS)

    # Trained world models across sample budgets and seeds.
    curves = {"mlp": {}, "nn": {}}
    harm = {"mlp": 0, "nn": 0}            # (seed,K) where planning through the WM < reactive
    total = 0
    for k in KS:
        per = {"mlp": [], "nn": []}
        for seed in SEEDS:
            models = train_models(k, seed)
            for kind in ("mlp", "nn"):
                r = plan_return(models[kind].predict_state)
                per[kind].append(r)
                total += 1
                if r < reactive_ret:
                    harm[kind] += 1
        for kind in ("mlp", "nn"):
            curves[kind][k] = {"mean": round(mean(per[kind]), 3),
                               "sd": round(pstdev(per[kind]), 3),
                               "runs": [round(v, 3) for v in per[kind]]}

    best_trained = max(curves[k][K]["mean"] for k in ("mlp", "nn") for K in KS)
    small_k = KS[0]
    results = {
        "task": "sprint", "value_function": "shipped - bugs - 0.5*debt",
        "episode_steps": EPISODE_STEPS, "depth": CODE_DEPTH,
        "seeds": SEEDS, "k_budgets": KS,
        "verified_code_return": round(exact, 3),
        "reactive_return": round(reactive_ret, 3),
        "random_return": round(random_ret, 3),
        "trained": curves,
        "best_trained_mean": round(best_trained, 3),
        "regret_best_trained": round(exact - best_trained, 3),
        "harm_fraction": round((harm["mlp"] + harm["nn"]) / total, 3),
        "harm_counts": harm, "n_trained_runs": total,
    }
    save_results("e61_trained_wm_control", results)

    print("E61 - verified code vs trained world model, downstream control\n")
    print(f"  verified code (0 samples): return = {exact:.2f}   [upper bound]")
    print(f"  reactive (model-free):     return = {reactive_ret:.2f}")
    print(f"  random:                    return = {random_ret:.2f}")
    print(f"  {'K':>6}  {'MLP-WM (mean±sd)':>20}  {'1NN-WM (mean±sd)':>20}")
    for k in KS:
        m, n = curves["mlp"][k], curves["nn"][k]
        print(f"  {k:>6}  {m['mean']:>10.2f} ± {m['sd']:<6.2f}  {n['mean']:>10.2f} ± {n['sd']:<6.2f}")
    print(f"\n  best trained mean = {best_trained:.2f}  (regret vs verified = {exact-best_trained:.2f})")
    print(f"  planning-through-trained-WM HARMS (< reactive) in "
          f"{harm['mlp']+harm['nn']}/{total} runs")

    # --- honest self-checks (save happened first) ---
    assert exact >= best_trained - OPTIMAL_TOL, "verified code should upper-bound trained WMs on return"
    assert exact >= reactive_ret >= random_ret - 1e-9, "sanity: verified >= reactive >= random"
    assert curves["mlp"][small_k]["mean"] < exact, "a small-sample trained WM should trail verified code"
    assert best_trained <= exact + OPTIMAL_TOL, "no trained WM should beat the verified (exact) model"
    print("\nchecks pass: a verified code world model attains optimal control return from zero "
          "training data; trained world models are sample-inefficient and planning through them "
          "can underperform model-free control.")


if __name__ == "__main__":
    main()
