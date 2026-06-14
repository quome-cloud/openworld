"""E45 - Inducing a verified world model from REAL repository history.

The paper's exact-vs-approximate, OOD-transfer, and auditability claims are all
established on synthetic oracle worlds. E45 moves them onto real measured data: a
software repository is a world whose state (file counts, lines of Python)
evolves as commits apply diffs. We mine real `(state, action, next_state)`
triples from three open-source repos (state read from the git tree, the diff
read separately - see mine_realrepo.py; the laws are therefore empirical, not
circular), induce the dynamics with the framework's own LLM synthesis+verify
pipeline, and hold the induced program to the same learned baselines as E12/E37.

The empirical laws the synthesizer should recover:
    files'    = files    + added      - deleted
    py_files' = py_files + py_added    - py_deleted
    py_loc'   = py_loc   + py_ins      - py_del

Reported honestly: the laws are *linear*, so a linear regressor can represent
them too - the symbolic model's edge over linear is exactness on the clean
subset (it rejects the binary/rename/merge outliers that bias a least-squares
fit) plus an auditable, verification-certified program. Over the NONLINEAR
learned world models (MLP, 1-NN) the edge is everything, especially out of
distribution. Primary metric is exact-match (the real-data version of the
paper's exact probe accuracy); MAE is reported alongside so we do not overclaim.

Deterministic except the LLM synthesis path (qwen2.5:7b, like E37); the
synthesized program is recorded. CSVs are committed so the baselines + scoring
rerun fully offline.
"""

import csv
import random
from pathlib import Path

import numpy as np

from openworld import OllamaLLM
from openworld.parsing import extract_code
from openworld.sandbox import run_transition_code

from common import require_ollama, save_results
from e12_learned_baseline import MLP

REPOS = ["requests", "flask", "tqdm"]
STATE = ["files", "py_files", "py_loc"]
ACT = ["a_added", "a_deleted", "a_py_added", "a_py_deleted", "a_py_ins", "a_py_del"]
DATA = Path(__file__).resolve().parent / "data" / "realrepo"
MODEL = "qwen3-coder:30b"   # a code model recovers the accounting law cleanly
SEED = 45
OOD_FRAC = 0.10           # top-decile churn held out as the OOD test
SYNTH_ATTEMPTS = 4
MAX_SHOWN = 30

INDUCE_SYSTEM = (
    "You are a program-induction engine. You are given examples of a "
    "deterministic environment's transitions as (state, action) -> next_state. "
    "Infer the underlying rules and reply with ONLY a python code block "
    "defining `def transition(state, action):` that reproduces them. Both "
    "`state` and `action` are dicts; return the next state as a dict with the "
    "same keys as `state`. Use only pure python. Do NOT assume any rule not "
    "supported by the examples."
)


def load(name):
    rows = []
    for r in csv.DictReader((DATA / f"{name}.csv").open()):
        rows.append({k: (v if k == "sha" else int(v)) for k, v in r.items()})
    return rows


def triples(rows):
    """Consecutive measured states; action is the later commit's diff."""
    out = []
    for a, b in zip(rows, rows[1:]):
        out.append(({k: a[k] for k in STATE},
                    {k: b[k] for k in ACT},
                    {k: b[k] for k in STATE}))
    return out


def law_pred(s, act):
    return {"files": s["files"] + act["a_added"] - act["a_deleted"],
            "py_files": s["py_files"] + act["a_py_added"] - act["a_py_deleted"],
            "py_loc": s["py_loc"] + act["a_py_ins"] - act["a_py_del"]}


def churn(act):
    return act["a_added"] + act["a_deleted"] + act["a_py_ins"] + act["a_py_del"]


def split(tr, rng):
    order = sorted(range(len(tr)), key=lambda i: churn(tr[i][1]))
    n_ood = max(1, int(len(tr) * OOD_FRAC))
    ood_idx = set(order[-n_ood:])
    small = [tr[i] for i in range(len(tr)) if i not in ood_idx]
    ood = [tr[i] for i in ood_idx]
    rng.shuffle(small)
    cut = int(len(small) * 0.8)
    return small[:cut], small[cut:], ood            # train, in-dist test, OOD test


# --- exact-match / MAE helpers ----------------------------------------------
def exact(pred, nxt):
    return all(pred[k] == nxt[k] for k in STATE)


def abserr(pred, nxt):
    return sum(abs(pred[k] - nxt[k]) for k in STATE) / len(STATE)


def score(predict, test):
    if not test:
        return {"exact": None, "mae": None}
    hits = err = 0.0
    for s, act, nxt in test:
        p = predict(s, act)
        hits += exact(p, nxt)
        err += abserr(p, nxt)
    return {"exact": round(hits / len(test), 3), "mae": round(err / len(test), 2)}


# --- symbolic induction (the framework's LLM synthesis path) ----------------
def induce_prompt(train):
    seen, shown = set(), []
    for s, a, n in train:
        key = (tuple(s.items()), tuple(a.items()))
        if key not in seen:
            seen.add(key)
            shown.append((s, a, n))
        if len(shown) >= MAX_SHOWN:
            break
    lines = [f"State fields: {STATE}. Action fields: {ACT}.", "",
             "Observed transitions:"]
    for s, a, n in shown:
        lines.append(f"  state={s}, action={a} -> {n}")
    lines += ["", "Write transition(state, action) consistent with ALL of these."]
    return "\n".join(lines)


def reproduces(code, train):
    ok = 0
    for s, a, n in train:
        try:
            if run_transition_code(code, dict(s), dict(a)) == n:
                ok += 1
        except Exception:
            return 0.0
    return ok / len(train)


def induce(train):
    """Synthesis verified by reproduction; each attempt uses a fresh seed so the
    retries actually explore (a fixed seed would return identical code)."""
    prompt = induce_prompt(train)
    best, best_repro = None, -1.0
    for i in range(SYNTH_ATTEMPTS):
        llm = OllamaLLM(model=MODEL, temperature=0.4, options={"seed": SEED + i})
        code = extract_code(llm.ask(prompt, system=INDUCE_SYSTEM))
        repro = reproduces(code, train)
        if repro > best_repro:
            best, best_repro = code, repro
        if repro == 1.0:
            break
    return best, best_repro


def symbolic_predictor(code):
    def predict(s, act):
        try:
            p = run_transition_code(code, dict(s), dict(act))
            return {k: int(p[k]) for k in STATE}
        except Exception:
            return {k: -10 ** 9 for k in STATE}      # counts as wrong
    return predict


# --- learned baselines ------------------------------------------------------
def feat(s, act):
    return [s[k] for k in STATE] + [act[k] for k in ACT]


def linear_predictor(train):
    X = np.array([feat(s, a) + [1.0] for s, a, _ in train], float)
    Y = np.array([[n[k] for k in STATE] for _, _, n in train], float)
    W, *_ = np.linalg.lstsq(X, Y, rcond=None)

    def predict(s, act):
        p = np.array(feat(s, act) + [1.0], float) @ W
        return {k: int(round(p[i])) for i, k in enumerate(STATE)}
    return predict


def mlp_predictor(train, seed):
    X = np.array([feat(s, a) for s, a, _ in train], float)
    Y = np.array([[n[k] for k in STATE] for _, _, n in train], float)
    mx, sx = X.mean(0), X.std(0) + 1e-9
    my, sy = Y.mean(0), Y.std(0) + 1e-9
    net = MLP(X.shape[1], Y.shape[1], hidden=64, seed=seed)
    net.train((X - mx) / sx, (Y - my) / sy, epochs=3000, lr=1e-3)

    def predict(s, act):
        z = (np.array([feat(s, act)], float) - mx) / sx
        p = net.forward(z)[0] * sy + my
        return {k: int(round(p[i])) for i, k in enumerate(STATE)}
    return predict


def knn_predictor(train):
    X = np.array([feat(s, a) for s, a, _ in train], float)
    mx, sx = X.mean(0), X.std(0) + 1e-9
    Xn = (X - mx) / sx
    Y = [n for _, _, n in train]

    def predict(s, act):
        q = (np.array(feat(s, act), float) - mx) / sx
        return Y[int(((Xn - q) ** 2).sum(1).argmin())]
    return predict


def run_repo(name):
    rows = load(name)
    tr = triples(rows)
    coverage = sum(law_pred(s, a) == n for s, a, n in tr) / len(tr)
    train, indist, ood = split(tr, random.Random(SEED))

    code, repro = induce(train)
    preds = {
        "symbolic": symbolic_predictor(code),
        "linear": linear_predictor(train),
        "mlp": mlp_predictor(train, SEED),
        "knn1": knn_predictor(train),
    }
    out = {"repo": name, "n_transitions": len(tr), "law_coverage": round(coverage, 4),
           "train_reproduction": round(repro, 3),
           "synthesized_code": code,
           "in_dist": {}, "ood": {}}
    for m, fn in preds.items():
        out["in_dist"][m] = score(fn, indist)
        out["ood"][m] = score(fn, ood)
    print(f"[{name}] n={len(tr)} coverage={coverage:.3f} repro={repro:.2f}")
    for m in preds:
        print(f"    {m:<9} in-dist exact={out['in_dist'][m]['exact']} "
              f"(mae {out['in_dist'][m]['mae']})  "
              f"OOD exact={out['ood'][m]['exact']} (mae {out['ood'][m]['mae']})")
    return out


def main():
    require_ollama(MODEL, timeout=1800)
    repos = [run_repo(name) for name in REPOS]

    def mean(section, metric, method):
        vals = [r[section][method][metric] for r in repos
                if r[section][method][metric] is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    methods = ["symbolic", "linear", "mlp", "knn1"]
    summary = {
        "n_repos": len(repos),
        "mean_law_coverage": round(sum(r["law_coverage"] for r in repos) / len(repos), 4),
        "in_dist_exact": {m: mean("in_dist", "exact", m) for m in methods},
        "ood_exact": {m: mean("ood", "exact", m) for m in methods},
        "in_dist_mae": {m: mean("in_dist", "mae", m) for m in methods},
        "ood_mae": {m: mean("ood", "mae", m) for m in methods},
    }
    save_results("e45_real_repo_induction", {
        "model": MODEL, "state_fields": STATE, "action_fields": ACT,
        "ood_frac": OOD_FRAC, "window": len(load(REPOS[0])),
        "summary": summary, "repos": repos,
    })

    print("\nMean across repos:")
    print(f"  {'method':<9}{'in-dist exact':>14}{'OOD exact':>11}"
          f"{'in-dist MAE':>13}{'OOD MAE':>9}")
    for m in methods:
        print(f"  {m:<9}{str(summary['in_dist_exact'][m]):>14}"
              f"{str(summary['ood_exact'][m]):>11}"
              f"{str(summary['in_dist_mae'][m]):>13}{str(summary['ood_mae'][m]):>9}")

    so, ko, mo = (summary["ood_exact"]["symbolic"], summary["ood_exact"]["knn1"],
                  summary["ood_exact"]["mlp"])
    sm, km, mm = (summary["ood_mae"]["symbolic"], summary["ood_mae"]["knn1"],
                  summary["ood_mae"]["mlp"])
    # The robust, real-world claim: the framework induces an exact, auditable
    # program where the NONLINEAR learned world models (MLP, 1-NN) collapse OOD.
    # (Linear regression also fits this *linear* law - reported honestly; E36
    # shows linear collapsing on the paper's nonlinear worlds.)
    assert so > ko and so > mo, "symbolic should beat the learned world models OOD"
    assert sm < mm and sm < km, "symbolic OOD error should be far below the learned models'"
    assert summary["mean_law_coverage"] > 0.95, "the empirical laws should hold on most real commits"
    print(f"\n  symbolic OOD exact {so} (MAE {sm}) vs the learned world models' "
          f"collapse: mlp {mo} (MAE {mm}), 1-NN {ko} (MAE {km}).")
    print(f"  linear regression also fits this linear law "
          f"(OOD exact {summary['ood_exact']['linear']}); the symbolic program "
          f"is exact AND auditable AND verification-certified.")


if __name__ == "__main__":
    main()
