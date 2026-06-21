"""E80 rule-induction: world-time compute on boolean-concept families (the diagnosis archetype,
generalized).

A WORLD is a hidden boolean rule over binary features (a small DNF, e.g.
"TRUE iff (feat_2 and not feat_5) or feat_7"); an instance lists which features are present; the
answer is TRUE/FALSE. The rule is GIVEN in the prompt, so held-out rules are solvable by
*evaluating the boolean expression* -- the shared skill. Many random rules = many worlds; small
LLMs are imperfect at multi-literal boolean evaluation (headroom); exact labels.

Deterministic/offline (numpy). Emits the e80_common world format.
"""

import numpy as np

CONFIG = {"ladder": [2, 8, 32, 64, 128], "abl_n": 48,
          "abl_noise": [0.0, 0.15, 0.30, 0.45, 0.60, 0.80, 1.0],
          "cap": 60, "n_test": 40, "seeds": [0, 1], "base": "Qwen/Qwen2.5-0.5B-Instruct"}

NF = 8           # features feat_0..feat_7
N_WORLDS = 320


def _rule(seed):
    """A DNF: OR of 2 conjunctions, each of 1-2 literals (feature present/absent)."""
    rng = np.random.RandomState(seed)
    clauses = []
    for _ in range(2):
        lits = []
        for f in rng.choice(NF, int(rng.randint(1, 3)), replace=False):
            lits.append((int(f), bool(rng.randint(0, 2))))   # (feature, must_be_present)
        clauses.append(lits)
    return clauses


def _describe(clauses):
    def lit(f, p):
        return f"feat_{f} is present" if p else f"feat_{f} is absent"
    cs = ["(" + " and ".join(lit(f, p) for f, p in cl) + ")" for cl in clauses]
    return " or ".join(cs)


def _eval(clauses, present):
    return any(all((f in present) == p for f, p in cl) for cl in clauses)


def _instance(clauses, rng):
    present = set(int(f) for f in np.where(rng.rand(NF) < 0.5)[0])
    lab = "TRUE" if _eval(clauses, present) else "FALSE"
    plist = ", ".join(f"feat_{f}" for f in sorted(present)) or "(none)"
    q = (f"Rule: the label is TRUE iff {_describe(clauses)}. "
         f"This item has: {plist}. Is the label TRUE or FALSE? Reply with ONLY TRUE or FALSE.")
    return q, lab


def build_worlds():
    worlds = {}
    for i in range(N_WORLDS):
        clauses = _rule(4000 + i)
        rng = np.random.RandomState(7000 + i)
        rows, answers = [], set()
        for _ in range(800):
            q, a = _instance(clauses, rng)
            rows.append({"prompt": q, "label": a})
            answers.add(a)
        if len(answers) == 2:                       # both TRUE and FALSE occur (non-degenerate)
            worlds[f"concept_{i:03d}"] = {"classes": ["TRUE", "FALSE"], "rows": rows}
    return worlds


if __name__ == "__main__":
    w = build_worlds()
    print(f"rule-induction worlds: {len(w)}")
    for k in list(w)[:3]:
        ex = w[k]["rows"][0]
        print(f"  {k}: {ex['prompt'][:160]}... => {ex['label']}")
