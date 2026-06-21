"""E80 math (GSM-style): world-time compute on arithmetic word-problem families.

A WORLD is a fixed multi-step arithmetic 'program' (a sequence of operations); instances sample
the numbers; the story narrates the steps; the answer is computed EXACTLY. Many distinct
step-sequences = many worlds that share one non-trivial skill ("follow the described arithmetic"),
where small LLMs are genuinely weak (real headroom). Train on many program-worlds, test on
STRICTLY held-out programs (unseen step-sequences) -> does the model learn the general
multi-step-arithmetic skill?

Deterministic/offline (numpy only). Emits worlds in the e80_common format
({name: {classes, rows:[{prompt,label}]}}). Answers are short integer strings (exact match).
"""

import itertools

import numpy as np

CONFIG = {"ladder": [2, 8, 32, 64, 128], "abl_n": 48,
          "abl_noise": [0.0, 0.15, 0.30, 0.45, 0.60, 0.80, 1.0],
          "cap": 60, "n_test": 40, "seeds": [0, 1], "base": "Qwen/Qwen2.5-0.5B-Instruct"}

# (name, story phrase, apply); multiply operands kept small so answers stay short.
OPS = [
    ("add", "add {n}", lambda v, n: v + n),
    ("sub", "subtract {n}", lambda v, n: v - n),
    ("mul", "multiply by {n}", lambda v, n: v * n),
]


def _schemas():
    """Distinct operation sequences (lengths 2..5) -> one per world."""
    seqs = []
    for L in (2, 3, 4, 5):
        seqs.extend(itertools.product(range(len(OPS)), repeat=L))
    return seqs


def _instance(schema, rng):
    x0 = int(rng.randint(2, 20))
    v = x0
    steps = []
    for op in schema:
        name, phrase, fn = OPS[op]
        n = int(rng.randint(2, 5)) if name == "mul" else int(rng.randint(2, 12))
        steps.append(phrase.format(n=n))
        v = fn(v, n)
    q = (f"You start with {x0}, then " + ", then ".join(steps)
         + ". What number do you have now? Reply with ONLY the integer.")
    return q, str(v)


def build_worlds():
    worlds = {}
    for si, sc in enumerate(_schemas()):
        rng = np.random.RandomState(1000 + si)
        rows, answers = [], set()
        for _ in range(CONFIG["cap"] * 3):
            q, a = _instance(sc, rng)
            rows.append({"prompt": q, "label": a})
            answers.add(a)
        # keep worlds with enough answer diversity (so corruption + the task are non-degenerate)
        if len(answers) >= 5:
            worlds[f"prog_{si:03d}"] = {"classes": sorted(answers), "rows": rows}
    return worlds


if __name__ == "__main__":
    w = build_worlds()
    sizes = sorted(len(v["rows"]) for v in w.values())
    print(f"math program-worlds: {len(w)} | rows/world {sizes[0]}..{sizes[-1]}")
    import random as _r
    for k in list(w)[:3] + [list(w)[-1]]:
        ex = w[k]["rows"][0]
        print(f"  {k}: {ex['prompt']}  => {ex['label']}  (#answers {len(w[k]['classes'])})")