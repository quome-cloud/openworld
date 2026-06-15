"""E55 - Information geometry: identify the world by maximizing information.

A hidden world is one of K candidates; each probe yields a noisy outcome whose
distribution depends on the world. The agent holds a posterior over candidates
and must identify the true one in as few probes as possible. The information-
geometric criterion picks the probe with the highest expected information gain
(mutual information between the outcome and the world identity under the current
posterior) and updates by Bayes - choosing, adaptively, the probe that best
separates the worlds still in contention.

We compare it to random probing and to a version-space heuristic (pick the probe
that splits the plausible candidates into the most distinct most-likely outcomes,
ignoring probabilities - the E43 style). Expected-info-gain identifies the world
in fewer probes because it accounts for how much each probe actually tells you
given what you already believe. Deterministic/offline.
"""

import numpy as np

from openworld import bayes_update, expected_info_gain
from openworld.infogeom import entropy

from common import save_results

K = 10               # candidate worlds
M = 12               # probes
O = 4                # outcomes per probe
EPS = 0.05           # stop when posterior entropy (bits) below this
BUDGET = 40
SEED = 55


def make_Q(seed):
    """Q[probe] is a [K, O] matrix of outcome distributions (peaked, so probes
    are informative); drawn once and shared across strategies."""
    rng = np.random.RandomState(seed)
    return [rng.dirichlet(np.full(O, 0.3), size=K) for _ in range(M)]


def eig_probe(posterior, Q):
    return int(np.argmax([expected_info_gain(posterior, Q[p]) for p in range(M)]))


def heuristic_probe(posterior, Q, rng):
    """Version-space style: among plausible worlds, pick the probe that yields the
    most distinct most-likely outcomes (ignores probabilities)."""
    plausible = np.where(posterior > 0.01)[0]
    best, best_n = [], -1
    for p in range(M):
        n = len({int(np.argmax(Q[p][w])) for w in plausible})
        if n > best_n:
            best_n, best = n, [p]
        elif n == best_n:
            best.append(p)
    return rng.choice(best)


def run(strategy, Q, true, rng):
    posterior = np.full(K, 1.0 / K)
    curve = [entropy(posterior)]
    for step in range(BUDGET):
        if entropy(posterior) < EPS:
            break
        if strategy == "eig":
            p = eig_probe(posterior, Q)
        elif strategy == "random":
            p = rng.randint(M)
        else:
            p = heuristic_probe(posterior, Q, rng)
        outcome = rng.choice(O, p=Q[p][true])
        posterior = bayes_update(posterior, Q[p], outcome)
        curve.append(entropy(posterior))
    return step + 1, int(np.argmax(posterior) == true), curve


def main():
    strategies = ["eig", "random", "heuristic"]
    agg = {s: {"steps": [], "correct": []} for s in strategies}
    curves = {s: [] for s in strategies}
    for trial in range(K * 8):
        Q = make_Q(SEED + (trial % 5))
        true = trial % K
        for s in strategies:
            rng = np.random.RandomState(1000 + trial)
            steps, correct, curve = run(s, Q, true, rng)
            agg[s]["steps"].append(steps)
            agg[s]["correct"].append(correct)
            if trial < 40:
                curves[s].append(curve)

    def mean_curve(cl):
        m = max(len(c) for c in cl)
        padded = [c + [c[-1]] * (m - len(c)) for c in cl]
        return [round(float(np.mean([c[i] for c in padded])), 3) for i in range(m)]

    summary = {s: {"mean_steps": round(float(np.mean(agg[s]["steps"])), 2),
                   "accuracy": round(float(np.mean(agg[s]["correct"])), 3)}
               for s in strategies}
    results = {"n_worlds": K, "n_probes": M, "n_outcomes": O,
               "summary": summary,
               "entropy_curves": {s: mean_curve(curves[s]) for s in strategies}}
    save_results("e55_infogeom", results)

    print(f"E55 - information-geometric world identification "
          f"({K} worlds, {M} probes)\n")
    print(f"  {'strategy':<12}{'mean steps':>12}{'accuracy':>10}")
    for s in strategies:
        print(f"  {s:<12}{summary[s]['mean_steps']:>12.2f}{summary[s]['accuracy']:>10.2f}")

    # --- self-checks ---
    assert summary["eig"]["mean_steps"] < summary["random"]["mean_steps"], \
        "expected-info-gain should identify faster than random"
    assert summary["eig"]["mean_steps"] <= summary["heuristic"]["mean_steps"] + 1e-6, \
        "expected-info-gain should not be slower than the count heuristic"
    assert summary["eig"]["accuracy"] > 0.95, "EIG should reliably identify the world"
    print(f"\n  EIG identifies in {summary['eig']['mean_steps']} probes vs "
          f"{summary['heuristic']['mean_steps']} (heuristic) / "
          f"{summary['random']['mean_steps']} (random).")


if __name__ == "__main__":
    main()
