"""E12 - A genuinely *learned* dynamics baseline (review item W2).

Trains real learned models on environment transitions from the sprint world:
a 2-layer MLP (numpy, full-batch gradient descent) and a 1-nearest-neighbor
memorizer, each on K in {100, 1000, 10000} transitions collected by a random
policy. Evaluates exact transition accuracy on the in-distribution and 10x
out-of-distribution probe suites and exact 20-step rollouts - the same
protocol the synthesized program is held to. The synthesized code needs ZERO
environment transitions (only the rule text); this experiment measures what
a trained model needs and what it still cannot do.
"""

import random

import numpy as np

from common import (
    SPRINT_ACTIONS, SPRINT_INITIAL, SPRINT_PROBES, SPRINT_PROBES_SCALED,
    save_results, sprint_ground_truth, wilson_ci,
)

FIELDS = ["backlog", "shipped", "bugs", "debt"]
ACTIONS = SPRINT_ACTIONS + ["noop"]
KS = [100, 1000, 10000]
ROLLOUT_STEPS = 20
N_SCRIPTS = 8
SEED = 23


def collect_transitions(k, rng):
    """Random-policy rollouts from the initial state (the data an agent gets)."""
    data = []
    state = dict(SPRINT_INITIAL)
    while len(data) < k:
        action = rng.choice(ACTIONS)
        next_state = sprint_ground_truth(state, {"name": action, "params": {}, "agent": None})
        data.append((dict(state), action, dict(next_state)))
        state = next_state
        if state["backlog"] == 0 and rng.random() < 0.3:  # episode reset
            state = dict(SPRINT_INITIAL)
    return data[:k]


def encode(state, action):
    onehot = [1.0 if action == a else 0.0 for a in ACTIONS]
    return [float(state[f]) for f in FIELDS] + onehot


class MLP:
    """Two hidden layers, ReLU, MSE on next-state; trained full-batch."""

    def __init__(self, n_in, n_out, hidden=64, seed=0):
        rng = np.random.RandomState(seed)
        self.w1 = rng.randn(n_in, hidden) * 0.1
        self.b1 = np.zeros(hidden)
        self.w2 = rng.randn(hidden, hidden) * 0.1
        self.b2 = np.zeros(hidden)
        self.w3 = rng.randn(hidden, n_out) * 0.1
        self.b3 = np.zeros(n_out)

    def forward(self, x):
        self.h1 = np.maximum(0, x @ self.w1 + self.b1)
        self.h2 = np.maximum(0, self.h1 @ self.w2 + self.b2)
        return self.h2 @ self.w3 + self.b3

    def train(self, x, y, epochs=3000, lr=1e-3):
        for _ in range(epochs):
            pred = self.forward(x)
            grad = 2 * (pred - y) / len(x)
            gw3 = self.h2.T @ grad
            gh2 = grad @ self.w3.T * (self.h2 > 0)
            gw2 = self.h1.T @ gh2
            gh1 = gh2 @ self.w2.T * (self.h1 > 0)
            gw1 = x.T @ gh1
            self.w3 -= lr * gw3; self.b3 -= lr * grad.sum(0)
            self.w2 -= lr * gw2; self.b2 -= lr * gh2.sum(0)
            self.w1 -= lr * gw1; self.b1 -= lr * gh1.sum(0)

    def predict_state(self, state, action):
        x = np.array([encode(state, action)])
        y = self.forward(x)[0]
        return {f: max(0, int(round(v))) for f, v in zip(FIELDS, y)}


class NearestNeighbor:
    def __init__(self, data):
        self.x = np.array([encode(s, a) for s, a, _ in data])
        self.y = [ns for _, _, ns in data]

    def predict_state(self, state, action):
        q = np.array(encode(state, action))
        idx = int(np.argmin(((self.x - q) ** 2).sum(axis=1)))
        return dict(self.y[idx])


def probe_exact(model, probes):
    hits = 0
    for state, action in probes:
        expected = sprint_ground_truth(dict(state), action.to_dict())
        if model.predict_state(dict(state), action.name) == expected:
            hits += 1
    return hits, len(probes)


def rollout_exact(model, rng):
    exact = 0
    for _ in range(N_SCRIPTS):
        state = dict(SPRINT_INITIAL)
        model_state = dict(SPRINT_INITIAL)
        ok = True
        for _ in range(ROLLOUT_STEPS):
            action = rng.choice(ACTIONS)
            state = sprint_ground_truth(state, {"name": action, "params": {}, "agent": None})
            model_state = model.predict_state(model_state, action)
            if model_state != state:
                ok = False
                break
        exact += ok
    return exact


def main():
    rows = []
    for k in KS:
        rng = random.Random(SEED)
        data = collect_transitions(k, rng)
        x = np.array([encode(s, a) for s, a, _ in data])
        y = np.array([[float(ns[f]) for f in FIELDS] for _, _, ns in data])

        mlp = MLP(x.shape[1], y.shape[1], seed=SEED)
        mlp.train(x, y)
        nn = NearestNeighbor(data)

        for name, model in (("mlp", mlp), ("knn1", nn)):
            in_hits, in_n = probe_exact(model, SPRINT_PROBES)
            ood_hits, ood_n = probe_exact(model, SPRINT_PROBES_SCALED)
            exact_rollouts = rollout_exact(model, random.Random(SEED + 1))
            rows.append({
                "model": name, "k_transitions": k,
                "probe_in_dist": in_hits / in_n,
                "probe_in_dist_ci": list(wilson_ci(in_hits, in_n)),
                "probe_ood_10x": ood_hits / ood_n,
                "probe_ood_ci": list(wilson_ci(ood_hits, ood_n)),
                "exact_rollouts": exact_rollouts,
                "n_rollouts": N_SCRIPTS,
            })
            print(f"  {name} K={k}: in-dist {in_hits}/{in_n}, "
                  f"OOD {ood_hits}/{ood_n}, exact rollouts {exact_rollouts}/{N_SCRIPTS}")

    save_results("e12_learned_baseline", {
        "seed": SEED, "ks": KS, "rollout_steps": ROLLOUT_STEPS,
        "rows": rows,
        "note": (
            "Synthesized code (E1/E11) achieves 1.0/1.0/all-exact with zero "
            "environment transitions, from rule text alone."
        ),
    })


if __name__ == "__main__":
    main()
