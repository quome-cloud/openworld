"""E80 learner-invariance test: does the world-time-compute law hold for a NON-NEURAL learner?

If the law is about the learning PROBLEM (verified worlds) and not about neural networks, the
same signatures should appear for a symbolic learner. We use an enumerative program-synthesizer
over a small list DSL (programming-by-example: search programs consistent with the demos, apply
to held-out queries -- no gradients, no network). "World-time compute" becomes the SEARCH BUDGET
(number of programs enumerated). We test, on the real List Functions worlds:
  - clause 1 (scaling): held-out exact-match rises with search budget and SATURATES;
  - clause 4 (exactness gate): corrupting the demo labels collapses it to the floor;
  - cross-learner: the worlds the synthesizer solves should be the worlds the neural learner
    lifts (same problem-level identifiability/realizability), not a different set.

Pure stdlib, CPU, deterministic, offline. Run: python3 e80_symbolic.py [listfn_worlds.jsonl]
"""

import ast
import json
import random
import sys
from itertools import product
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUDGETS = [5, 20, 80, 300, 1200, 5000]   # programs enumerated (the non-neural compute axis)
N_DEMOS, N_EVAL = 6, 6


# ---- a small list DSL: each primitive maps list[int] -> list[int] ----
def _safe(f):
    def g(x):
        try:
            return f(list(x))
        except Exception:
            return None
    return g


PRIMS = {
    "id": _safe(lambda x: x), "rev": _safe(lambda x: x[::-1]),
    "sort": _safe(sorted), "sortd": _safe(lambda x: sorted(x, reverse=True)),
    "tail": _safe(lambda x: x[1:]), "init": _safe(lambda x: x[:-1]),
    "head1": _safe(lambda x: [x[0]]), "last1": _safe(lambda x: [x[-1]]),
    "uniq": _safe(lambda x: list(dict.fromkeys(x))),
    "incr": _safe(lambda x: [v + 1 for v in x]), "decr": _safe(lambda x: [v - 1 for v in x]),
    "dbl": _safe(lambda x: [v * 2 for v in x]), "neg": _safe(lambda x: [-v for v in x]),
    "sq": _safe(lambda x: [v * v for v in x]), "abs": _safe(lambda x: [abs(v) for v in x]),
    "evens": _safe(lambda x: [v for v in x if v % 2 == 0]),
    "odds": _safe(lambda x: [v for v in x if v % 2 == 1]),
    "pos": _safe(lambda x: [v for v in x if v > 0]),
    "nz": _safe(lambda x: [v for v in x if v != 0]),
    "sum": _safe(lambda x: [sum(x)]), "max": _safe(lambda x: [max(x)]),
    "min": _safe(lambda x: [min(x)]), "len": _safe(lambda x: [len(x)]),
    "take2": _safe(lambda x: x[:2]), "drop1": _safe(lambda x: x[1:]),
    "sortuniq": _safe(lambda x: sorted(set(x))),
}
NAMES = list(PRIMS)


def programs(max_depth=3):
    """Enumerate DSL programs (compositions) in increasing size -> list of (name, fn)."""
    progs = [(n, PRIMS[n]) for n in NAMES]              # depth 1
    for d in range(2, max_depth + 1):
        for combo in product(NAMES, repeat=d):
            fns = [PRIMS[c] for c in combo]

            def make(fns):
                def f(x):
                    for fn in reversed(fns):             # apply right-to-left (composition)
                        x = fn(x)
                        if x is None:
                            return None
                    return x
                return f
            progs.append(("∘".join(combo), make(fns)))
    return progs


PROGS = programs(3)


def synthesize(demos, budget):
    """First program (within budget) consistent with ALL demos, else None."""
    for name, fn in PROGS[:budget]:
        if all(fn(i) == o for i, o in demos):
            return fn
    return None


def parse_world(examples):
    out = []
    for e in examples:
        try:
            i, o = ast.literal_eval(e["input"]), ast.literal_eval(e["output"])
            if isinstance(i, list) and isinstance(o, list) and all(isinstance(v, int) for v in i + o):
                out.append((i, o))
        except (ValueError, SyntaxError):
            pass
    return out


def world_acc(pairs, budget, rng, corrupt=False):
    if len(pairs) < N_DEMOS + N_EVAL:
        return None
    idx = list(range(len(pairs)))
    rng.shuffle(idx)
    demos = [pairs[i] for i in idx[:N_DEMOS]]
    queries = [pairs[i] for i in idx[N_DEMOS:N_DEMOS + N_EVAL]]
    if corrupt:                                          # randomize demo outputs (wrong labels)
        outs = [d[1] for d in demos]
        rng.shuffle(outs)
        demos = [(d[0], o) for d, o in zip(demos, outs)]
    fn = synthesize(demos, budget)
    if fn is None:
        return 0.0
    return sum(fn(i) == o for i, o in queries) / len(queries)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/listfn_worlds.jsonl"
    by = {}
    for line in open(path):
        r = json.loads(line)
        by.setdefault(r["world"], []).append({"input": r["input"], "output": r["output"]})
    worlds = {w: parse_world(v) for w, v in by.items()}
    worlds = {w: v for w, v in worlds.items() if len(v) >= N_DEMOS + N_EVAL}
    names = sorted(worlds)
    random.Random(80).shuffle(names)                     # same selection as the neural run
    print(f"[symbolic] {len(names)} List Functions worlds (parsed as int-lists), |DSL programs|={len(PROGS)}")

    curve, corrupt = {}, {}
    per_world_solved = {}
    for B in BUDGETS:
        accs = [world_acc(worlds[w], B, random.Random(hash(w) % 2**32)) for w in names]
        accs = [a for a in accs if a is not None]
        curve[B] = sum(accs) / len(accs)
    for B in BUDGETS:
        ca = [world_acc(worlds[w], B, random.Random(hash(w) % 2**32), corrupt=True) for w in names]
        ca = [a for a in ca if a is not None]
        corrupt[B] = sum(ca) / len(ca)
    Bmax = BUDGETS[-1]
    for w in names:
        per_world_solved[w] = world_acc(worlds[w], Bmax, random.Random(hash(w) % 2**32))

    res = {"experiment": "symbolic-learner-invariance", "learner": "enumerative program synthesis",
           "n_worlds": len(names), "n_dsl_programs": len(PROGS), "budgets": BUDGETS,
           "scaling_curve": {str(b): round(curve[b], 3) for b in BUDGETS},
           "corrupt_curve": {str(b): round(corrupt[b], 3) for b in BUDGETS},
           "per_world_solved": {w: per_world_solved[w] for w in names}}

    # cross-learner: do symbolic-solved worlds match neural-lifted worlds?
    nf = HERE / "results" / "e80_text_listfn.json"
    if nf.exists():
        pw = json.load(open(nf)).get("per_world", {})
        zs, hv = pw.get("zeroshot", {}), pw.get("heavy", {})
        pairs = [(per_world_solved[w], hv[w] - zs[w]) for w in names
                 if w in hv and w in zs and per_world_solved.get(w) is not None
                 and hv[w] is not None and zs[w] is not None]
        if len(pairs) >= 5:
            import statistics as st
            a = [p[0] for p in pairs]
            b = [p[1] for p in pairs]
            def rank(z):
                order = sorted(range(len(z)), key=lambda i: z[i])
                r = [0] * len(z)
                for i, o in enumerate(order):
                    r[o] = i
                return r
            ra, rb = rank(a), rank(b)
            n = len(a)
            if st.pstdev(ra) > 0 and st.pstdev(rb) > 0:
                cov = sum((ra[i] - (n - 1) / 2) * (rb[i] - (n - 1) / 2) for i in range(n)) / n
                rho = cov / (st.pstdev(ra) * st.pstdev(rb))
                res["cross_learner_spearman_symbolic_vs_neural"] = round(rho, 3)
                res["cross_learner_n"] = n

    (HERE / "results" / "e80_symbolic.json").write_text(json.dumps(res, indent=2))
    print("scaling (acc vs search budget):", res["scaling_curve"])
    print("corrupt (labels shuffled):     ", res["corrupt_curve"])
    if "cross_learner_spearman_symbolic_vs_neural" in res:
        print(f"cross-learner Spearman(symbolic-solved, neural-lift) = "
              f"{res['cross_learner_spearman_symbolic_vs_neural']} (n={res['cross_learner_n']})")


if __name__ == "__main__":
    main()
