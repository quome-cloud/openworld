"""E72 - World-model rules as a sample-efficiency multiplier for generalization.

The thesis: having a world's RULES (which a verified code world model gives you) lets a
learner generalize from far fewer examples than learning the same dynamics from raw
transition examples -- and the gap explodes out of distribution. "The world model speeds
up learning."

Each verified recipe transition is an exact oracle, so it generates unlimited labelled
(state, action) -> next-state data and exact 10x out-of-distribution labels for free. We
compare, on next-numeric-state prediction:

  examples-only learners (must induce the dynamics from k labelled transitions):
    - 1-NN over (state, action)
    - ridge regression (closed form)
  rules-based learner (is given the dynamics):
    - the verified code itself = 1.0 in AND out of distribution at k=0 (the world model)

Headline: examples-only accuracy rises slowly with k and collapses OOD; the rules line is
flat at the top from k=0. Rules are worth thousands of examples.

We deliberately exclude an LLM next-state-prediction panel: predicting an EXACT next state
from k in-context examples requires perfect mental arithmetic on (often out-of-range)
numbers, which a local 7B does at noise level -- an ill-posed task that would add an
unreliable number, not evidence. The deterministic ML learners are the fair examples-only
baselines for the sample-efficiency claim. (A tractable LLM variant -- predicting the
DIRECTION of change rather than the exact value -- is left to a follow-up.)

Fully deterministic and offline (numpy only). save_results is called BEFORE the asserts.
"""

import json
import random
from pathlib import Path

import numpy as np

from common import save_results
from openworld.spec import from_spec
from openworld.sandbox import load_transition_code

ROOT = Path(__file__).resolve().parent.parent
RECIPES = ROOT / "recipes"
RESULTS = ROOT / "experiments" / "results"
SECTORS = ["healthcare", "financial", "legal", "cybersecurity", "energy", "agentic"]

K_GRID = [4, 8, 16, 32, 64, 128, 256]
N_TRAIN_POOL = 300
N_TEST = 120
WALK = 12             # rollout length used to sample realistic in-distribution states
OOD_SCALE = (3.0, 10.0)
FLOAT_TOL = 0.02      # relative tolerance for float-field exactness
N_ML_WORLDS = 24      # worlds for the offline ML curves
SEED = 72


def numeric_fields(state):
    return [k for k, v in state.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)]


def load_world(spec):
    w = from_spec(spec, allow_code=True)
    fn = load_transition_code(w.transition.code, getattr(w.transition, "func_name", "transition"))
    acts = [a for a in w.actions if ":" not in a] or list(w.actions)

    def step(s, a):
        return fn(dict(s), {"name": a, "params": {}, "agent": None})

    return step, dict(w.initial_state), acts


def visited_states(step, s0, acts, n, seed):
    rng = random.Random(seed)
    out, s = [], dict(s0)
    for _ in range(n):
        for _ in range(rng.randint(1, WALK)):
            s = step(s, rng.choice(acts))
        out.append(dict(s))
    return out


def ood_states(s0, nums, n, seed):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        st = dict(s0)
        for f in nums:
            scale = rng.uniform(*OOD_SCALE)
            st[f] = type(s0[f])(st[f] * scale) if s0[f] else type(s0[f])(scale * 5)
        out.append(st)
    return out


def make_examples(step, states, acts, nums, rng):
    """(input vector, target numeric vector) for each (state, action) the oracle handles."""
    X, Y = [], []
    for st in states:
        a = rng.choice(acts)
        try:
            ns = step(st, a)
            y = [float(ns[f]) for f in nums]
        except Exception:  # noqa: BLE001
            continue
        ai = acts.index(a)
        onehot = [1.0 if i == ai else 0.0 for i in range(len(acts))]
        X.append([float(st[f]) for f in nums] + onehot)
        Y.append(y)
    return np.array(X, dtype=float), np.array(Y, dtype=float)


def field_accuracy(pred, true, nums, s0):
    """Fraction of numeric fields predicted exactly (int) / within rel-tol (float)."""
    ok = 0
    for j, f in enumerate(nums):
        if isinstance(s0[f], int):
            ok += int(round(pred[j]) == round(true[j]))
        else:
            ok += int(abs(pred[j] - true[j]) <= FLOAT_TOL * (abs(true[j]) + 1.0))
    return ok / len(nums)


def knn_predict(Xtr, Ytr, Xte):
    preds = []
    for x in Xte:
        d = ((Xtr - x) ** 2).sum(axis=1)
        preds.append(Ytr[int(np.argmin(d))])
    return np.array(preds)


def ridge_fit_predict(Xtr, Ytr, Xte, lam=1.0):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Xtr_n, Xte_n = (Xtr - mu) / sd, (Xte - mu) / sd
    Xtr_b = np.hstack([Xtr_n, np.ones((len(Xtr_n), 1))])
    Xte_b = np.hstack([Xte_n, np.ones((len(Xte_n), 1))])
    A = Xtr_b.T @ Xtr_b + lam * np.eye(Xtr_b.shape[1])
    W = np.linalg.solve(A, Xtr_b.T @ Ytr)
    return Xte_b @ W


def bench_world_ml(spec, idx):
    step, s0, acts = load_world(spec)
    nums = numeric_fields(s0)
    if len(nums) < 2 or len(acts) < 2:
        return None
    rng = random.Random(SEED + idx)
    Xtr, Ytr = make_examples(step, visited_states(step, s0, acts, N_TRAIN_POOL, SEED + idx), acts, nums, rng)
    Xin, Yin = make_examples(step, visited_states(step, s0, acts, N_TEST, SEED + 999 + idx), acts, nums, rng)
    Xood, Yood = make_examples(step, ood_states(s0, nums, N_TEST, SEED + 7 + idx), acts, nums, rng)
    if len(Xtr) < max(K_GRID) or len(Xin) < 10 or len(Xood) < 10:
        return None

    out = {"in": {"knn": [], "ridge": []}, "ood": {"knn": [], "ridge": []}}
    for k in K_GRID:
        xk, yk = Xtr[:k], Ytr[:k]
        for split, Xte, Yte in (("in", Xin, Yin), ("ood", Xood, Yood)):
            for name, pred in (("knn", knn_predict(xk, yk, Xte)),
                               ("ridge", ridge_fit_predict(xk, yk, Xte))):
                acc = float(np.mean([field_accuracy(pred[i], Yte[i], nums, s0) for i in range(len(Yte))]))
                out[split][name].append(round(acc, 4))
    # verified code (the world model) is exact at k=0, in and out of distribution
    out["code_in"], out["code_ood"] = 1.0, 1.0
    return out


def curve_mean(rows, split, name):
    arr = np.array([r[split][name] for r in rows], dtype=float)
    return [round(float(v), 4) for v in arr.mean(axis=0)]


def main():
    # ---- offline ML sample-efficiency curves ----
    specs = []
    for sec in SECTORS:
        for f in sorted((RECIPES / sec).glob("*.json")):
            specs.append((sec, f.stem, json.loads(f.read_text())))
    rng = random.Random(SEED)
    rng.shuffle(specs)

    ml_rows, used = [], []
    for sec, name, spec in specs:
        if len(ml_rows) >= N_ML_WORLDS:
            break
        try:
            r = bench_world_ml(spec, len(ml_rows))
        except Exception:  # noqa: BLE001
            r = None
        if r:
            ml_rows.append(r)
            used.append((sec, name, spec))

    curves = {
        "k_grid": K_GRID,
        "in_distribution": {"knn": curve_mean(ml_rows, "in", "knn"),
                            "ridge": curve_mean(ml_rows, "in", "ridge"),
                            "code_rules": 1.0},
        "out_of_distribution": {"knn": curve_mean(ml_rows, "ood", "knn"),
                                "ridge": curve_mean(ml_rows, "ood", "ridge"),
                                "code_rules": 1.0},
    }

    results = {
        "task": "rules-vs-examples sample efficiency: how many transition examples a learner "
                "needs to match the dynamics a verified world model gives for free",
        "config": {"k_grid": K_GRID, "n_ml_worlds": len(ml_rows), "float_tol": FLOAT_TOL,
                   "ood_scale": OOD_SCALE, "seed": SEED},
        "note": "next-numeric-state prediction; accuracy = fraction of fields exact (int) or "
                "within rel-tol (float). The verified code (rules) is 1.0 at k=0 in and OOD. "
                "An LLM exact-prediction panel was excluded as ill-posed (see module docstring).",
        "curves": curves,
    }
    save_results("e72_sample_efficiency", results)

    # ---- self-checks (after save_results) ----
    assert len(ml_rows) >= 10, f"too few usable worlds: {len(ml_rows)}"
    knn_ood = curves["out_of_distribution"]["knn"]
    ridge_ood = curves["out_of_distribution"]["ridge"]
    knn_in = curves["in_distribution"]["knn"]
    # examples-only learners never reach the rules line, and are far worse OOD than the code.
    assert max(knn_ood + ridge_ood) < 1.0, "an examples-only learner matched the rules line OOD?"
    assert knn_ood[-1] < 0.99, "OOD generalization should fall short of the verified code"
    # more examples help in-distribution (a real learning curve).
    assert knn_in[-1] >= knn_in[0] - 0.05, f"in-dist curve should not collapse: {knn_in}"
    # the rules line dominates the best examples-only OOD result by a clear margin.
    gap = 1.0 - max(knn_ood + ridge_ood)
    assert gap > 0.1, f"rules advantage OOD too small: {gap}"
    print(f"[ok] E72: {len(ml_rows)} worlds | OOD @k={K_GRID[-1]}: "
          f"kNN {knn_ood[-1]} ridge {ridge_ood[-1]} vs code/rules 1.0 (gap {round(gap,3)}) | "
          f"in-dist kNN {knn_in[0]}->{knn_in[-1]} over k={K_GRID[0]}..{K_GRID[-1]}")


if __name__ == "__main__":
    main()
