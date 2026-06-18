"""E74 (offline core) - A diagnosis world FAMILY: does knowing many specialties let you
pick up a new one from far fewer examples?

E71 found no transfer across the 100 heterogeneous recipes -- because they share no task
structure. The fix (the user's design): a family of worlds that share a GOAL (diagnosis)
and an action grammar, varying only by specialty. Then "traverse a new specialty knowing
the others" becomes a real, measurable claim, with a clean ground-truth oracle (the hidden
disease), not a planner-normalized proxy.

Formalism. A specialty is a diagnosis POMDP: a hidden disease d ~ prior; binary features
(symptoms/tests) with disease-specific emission probabilities M[d, f]; an agent orders the
most-informative tests, maintains a Bayesian posterior over diseases, and commits when
confident. The diagnostic PROCEDURE (gather evidence -> infer -> decide) is SHARED across
specialties; only M (the symptom->disease map) varies. So "knowing other specialties" =
having a learned prior over what M looks like, which lets you infer a new specialty's M
from fewer labelled cases.

We hold out whole specialties (sklearn-style) and measure, on held-out specialties:
  - learning curve: diagnostic accuracy vs n labelled cases/disease, for a meta-learner
    (empirical-Bayes prior fit on TRAIN specialties) vs a from-scratch learner;
  - meta curve: held-out few-shot accuracy vs K (number of training specialties seen);
  - tests-to-diagnosis (the POMDP cost), vs an oracle that knows the true M (ceiling) and a
    prior-only baseline (floor).

Headline: the meta-learner reaches high held-out accuracy from far fewer cases -- the world
family makes a new specialty cheap to learn. Deterministic, numpy-only, offline. A sample
specialty is also instantiated as a verified OpenWorld world to ground the family in the
framework. save_results before asserts.
"""

import json
import math

import numpy as np

from common import save_results
from openworld import CodeTransition, World
from openworld.spec import from_spec, to_spec, validate_spec

N_SPEC = 120          # specialties in the family
N_TRAIN = 80          # specialties used to fit the meta-prior (rest held out)
D = 6                 # diseases per specialty
F = 12                # features (tests/symptoms) per specialty
N_SIGNATURE = 3       # characteristic features per disease
BASE_RATE = 0.10      # P(feature present | not a signature feature)
N_GRID = [1, 2, 4, 8, 16, 32]      # labelled cases per disease (the "examples" axis)
K_GRID = [2, 4, 8, 16, 32, 64]     # train specialties used to fit the meta-prior
N_EVAL_PATIENTS = 150
THRESHOLD = 0.9       # posterior confidence to commit a diagnosis
COMPONENT_STRENGTH = 6.0
SEED = 74


def make_specialty(seed):
    rng = np.random.RandomState(seed)
    M = np.full((D, F), BASE_RATE)
    for d in range(D):
        sig = rng.choice(F, N_SIGNATURE, replace=False)
        M[d, sig] = rng.uniform(0.70, 0.95, N_SIGNATURE)
    prior = rng.dirichlet(np.ones(D) * 2.0)
    return {"M": M, "prior": prior, "seed": seed}


def sample_patients(spec, n, rng):
    d = rng.choice(D, size=n, p=spec["prior"])
    x = (rng.rand(n, F) < spec["M"][d]).astype(int)
    return d, x


def labelled_counts(spec, n_per_disease, rng):
    """n_per_disease cases of each disease -> positive-feature counts[d, f]."""
    counts = np.zeros((D, F))
    for d in range(D):
        x = (rng.rand(n_per_disease, F) < spec["M"][d]).astype(int)
        counts[d] = x.sum(axis=0)
    return counts, n_per_disease


def estimate_scratch(counts, n):
    return (counts + 1.0) / (n + 2.0)            # Laplace, uninformed


def fit_meta_prior(train_specs):
    """Two-component empirical Bayes over M entries (base vs signature): the structure a
    learner distills from seeing many specialties."""
    entries = np.concatenate([s["M"].ravel() for s in train_specs])
    hi = entries[entries >= 0.4]
    lo = entries[entries < 0.4]
    mu_hi = float(hi.mean()) if len(hi) else 0.8
    mu_lo = float(lo.mean()) if len(lo) else BASE_RATE
    pi = float(len(hi) / len(entries))
    return {"mu_lo": mu_lo, "mu_hi": mu_hi, "pi": pi}


def estimate_meta(counts, n, prior):
    """Posterior-predictive mean under the 2-component prior, per entry."""
    s = COMPONENT_STRENGTH
    comps = [(1 - prior["pi"], prior["mu_lo"]), (prior["pi"], prior["mu_hi"])]
    M = np.zeros((D, F))
    for d in range(D):
        for f in range(F):
            k = counts[d, f]
            num = den = 0.0
            for w, mu in comps:
                a, b = mu * s, (1 - mu) * s
                # marginal likelihood of k successes in n under Beta(a,b) (Beta-Binomial,
                # constant binomial term cancels in the weight ratio so we drop it)
                ml = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                              + math.lgamma(a + k) + math.lgamma(b + n - k)
                              - math.lgamma(a + b + n))
                post_mean = (k + a) / (n + a + b)
                num += w * ml * post_mean
                den += w * ml
            M[d, f] = num / den if den > 0 else (k + 1) / (n + 2)
    return M


def diagnose(M_hat, prior, x_full, threshold=THRESHOLD):
    """Sequential POMDP policy: order the max-information test, update the posterior, stop
    when confident. Returns (predicted disease, #tests used)."""
    logp = np.log(prior + 1e-12)
    tested = np.zeros(F, dtype=bool)
    Mc = np.clip(M_hat, 1e-4, 1 - 1e-4)
    for t in range(F):
        post = np.exp(logp - logp.max())
        post /= post.sum()
        if post.max() >= threshold:
            break
        best_f, best_ig = -1, -1.0
        Hnow = -np.sum(post * np.log(post + 1e-12))
        for f in range(F):
            if tested[f]:
                continue
            p1 = float((post * Mc[:, f]).sum())
            ig = Hnow
            for o, po in ((1, p1), (0, 1 - p1)):
                if po <= 1e-9:
                    continue
                like = Mc[:, f] if o else (1 - Mc[:, f])
                pa = post * like
                pa /= pa.sum()
                ig -= po * (-np.sum(pa * np.log(pa + 1e-12)))
            if ig > best_ig:
                best_ig, best_f = ig, f
        if best_f < 0:
            break
        o = x_full[best_f]
        logp += np.log(Mc[:, best_f] if o else (1 - Mc[:, best_f]))
        tested[best_f] = True
    post = np.exp(logp - logp.max())
    post /= post.sum()
    return int(post.argmax()), int(tested.sum())


def eval_specialty(M_hat, spec, rng):
    d, x = sample_patients(spec, N_EVAL_PATIENTS, rng)
    correct, tests = 0, 0
    for i in range(N_EVAL_PATIENTS):
        pred, nt = diagnose(M_hat, spec["prior"], x[i])
        correct += int(pred == d[i])
        tests += nt
    return correct / N_EVAL_PATIENTS, tests / N_EVAL_PATIENTS


def ground_in_framework(spec):
    """Instantiate one specialty as a verified OpenWorld world to ground the family."""
    M = spec["M"].round(3).tolist()
    code = (
        "def transition(state, action):\n"
        f"    M = {M}\n"
        "    s = dict(state)\n"
        "    name = action['name']\n"
        "    if name.startswith('test_'):\n"
        "        f = int(name.split('_')[1])\n"
        "        d = s['_disease']\n"
        "        # reveal a deterministic thresholded reading (seed-free, replayable)\n"
        "        s.setdefault('evidence', {})[str(f)] = 1 if M[d][f] >= 0.5 else 0\n"
        "        s['n_tests'] = s.get('n_tests', 0) + 1\n"
        "    elif name.startswith('diagnose_'):\n"
        "        d = int(name.split('_')[1])\n"
        "        s['done'] = True\n"
        "        s['correct'] = (d == s['_disease'])\n"
        "    return s\n")
    actions = [f"test_{f}" for f in range(F)] + [f"diagnose_{d}" for d in range(D)]
    w = World(name="diagnosis", description="a diagnosis specialty (POMDP): order tests, then diagnose",
              initial_state={"_disease": 0, "evidence": {}, "n_tests": 0, "done": False, "correct": False},
              actions=actions, rules=["test_f reveals feature f; diagnose_d commits."],
              transition=CodeTransition(code))
    spec_json = to_spec(w)
    problems = validate_spec(spec_json)
    w2 = from_spec(spec_json, allow_code=True)
    s = dict(w.initial_state)
    rt_ok = (from_spec(to_spec(w), allow_code=True).transition.step(s, _A("test_0"))
             == w.transition.step(s, _A("test_0")))
    return problems, rt_ok


class _A:  # minimal Action shim
    def __init__(self, name):
        self.name = name

    def to_dict(self):
        return {"name": self.name, "params": {}, "agent": None}


def main():
    family = [make_specialty(SEED + i) for i in range(N_SPEC)]
    train_specs, test_specs = family[:N_TRAIN], family[N_TRAIN:]
    full_prior = fit_meta_prior(train_specs)

    rng = np.random.RandomState(SEED)

    # ---- prior-only floor and oracle ceiling on held-out specialties ----
    chance = float(np.mean([s["prior"].max() for s in test_specs]))
    oracle = np.mean([eval_specialty(s["M"], s, rng)[0] for s in test_specs])

    # ---- learning curve: meta vs scratch, held-out accuracy vs n cases/disease ----
    curve = {"n_grid": N_GRID, "meta_acc": [], "scratch_acc": [],
             "meta_tests": [], "scratch_tests": []}
    for n in N_GRID:
        ma, sa, mt, st = [], [], [], []
        for s in test_specs:
            counts, nn = labelled_counts(s, n, rng)
            acc_m, t_m = eval_specialty(estimate_meta(counts, nn, full_prior), s, rng)
            acc_s, t_s = eval_specialty(estimate_scratch(counts, nn), s, rng)
            ma.append(acc_m); sa.append(acc_s); mt.append(t_m); st.append(t_s)
        curve["meta_acc"].append(round(float(np.mean(ma)), 4))
        curve["scratch_acc"].append(round(float(np.mean(sa)), 4))
        curve["meta_tests"].append(round(float(np.mean(mt)), 3))
        curve["scratch_tests"].append(round(float(np.mean(st)), 3))

    # ---- meta curve: held-out few-shot accuracy (n=2) vs K training specialties ----
    n_few = 2
    meta_vs_k = {"k_grid": K_GRID, "acc": []}
    for K in K_GRID:
        prior_k = fit_meta_prior(train_specs[:K])
        accs = []
        for s in test_specs:
            counts, nn = labelled_counts(s, n_few, rng)
            accs.append(eval_specialty(estimate_meta(counts, nn, prior_k), s, rng)[0])
        meta_vs_k["acc"].append(round(float(np.mean(accs)), 4))

    problems, rt_ok = ground_in_framework(family[0])

    results = {
        "task": "diagnosis world family: meta-learning generalization across specialties "
                "(hold out whole specialties; clean diagnostic-accuracy oracle)",
        "config": {"n_specialties": N_SPEC, "n_train": N_TRAIN, "diseases": D, "features": F,
                   "signature": N_SIGNATURE, "n_grid": N_GRID, "k_grid": K_GRID,
                   "eval_patients": N_EVAL_PATIENTS, "seed": SEED},
        "prior_only_floor": round(chance, 4),
        "oracle_ceiling": round(float(oracle), 4),
        "learning_curve": curve,
        "meta_prior": {k: round(v, 4) for k, v in full_prior.items()},
        "meta_vs_k": meta_vs_k,
        "framework_grounding": {"validate_problems": problems, "roundtrip_ok": bool(rt_ok)},
        "note": "meta = empirical-Bayes prior over symptom->disease structure fit on TRAIN "
                "specialties; scratch = uninformed. Held-out specialties only.",
    }
    save_results("e74_diagnosis", results)

    # ---- self-checks (after save_results) ----
    assert not problems, f"diagnosis world fails validation: {problems}"
    assert rt_ok, "diagnosis world does not round-trip losslessly"
    # meta beats scratch at the smallest budget (knowing other specialties helps few-shot).
    assert curve["meta_acc"][0] > curve["scratch_acc"][0] + 0.02, \
        f"meta not better few-shot: {curve['meta_acc'][0]} vs {curve['scratch_acc'][0]}"
    # more cases help, and both stay under the oracle ceiling and above the floor.
    assert curve["meta_acc"][-1] >= curve["meta_acc"][0], "meta curve should not fall with data"
    assert oracle >= curve["meta_acc"][-1] - 1e-6, "no learner may exceed the oracle"
    assert curve["meta_acc"][0] > results["prior_only_floor"], "meta should beat prior-only"
    # the meta-learner reaches scratch's BEST accuracy from far fewer cases (sample efficiency)
    target = curve["scratch_acc"][-1]
    n_meta = next((N_GRID[i] for i, a in enumerate(curve["meta_acc"]) if a >= target), None)
    print(f"[ok] E74: family={N_SPEC} specialties (hold out {N_SPEC - N_TRAIN}) | "
          f"floor {results['prior_only_floor']} oracle {results['oracle_ceiling']}")
    print(f"  few-shot (n=1/disease): meta {curve['meta_acc'][0]} vs scratch {curve['scratch_acc'][0]}")
    print(f"  meta reaches scratch's best ({target}) at n={n_meta} (scratch needs {N_GRID[-1]})")
    print(f"  meta-vs-K acc @n=2: {meta_vs_k['acc']} over K={K_GRID}")
    print(f"  grounding: validate clean={not problems}, round-trip={rt_ok}")


if __name__ == "__main__":
    main()
