"""E80 text-world core: world-time compute / test-time training on free-text I/O families
(List Functions, CLRS-Text) -- the same mechanism as e80_arc but for text instead of grids.

A WORLD is a set of {input, output} examples sharing a hidden rule (a list function; a classic
algorithm). Each example is verified/exact, so the corrupted-label ablation is meaningful. We
test the world-time-compute thesis the same way as on ARC: per-world test-time training (fit a
fresh adapter on a world's demonstrations, predict held-out queries of that world by exact
match), with light/heavy compute levels and a corrupted-demonstration control.

Stdlib only; the GPU runner (e80_text_ttt.py) imports this. Offline-validatable.
"""

import json
import re


def load_worlds(path, min_examples=12, max_worlds=None):
    """{world_name: [{input, output}, ...]} from a uniform JSONL of {world, input, output}."""
    by = {}
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        by.setdefault(r["world"], []).append({"input": r["input"], "output": r["output"]})
    names = sorted(k for k, v in by.items() if len(v) >= min_examples)
    if max_worlds:
        names = names[:max_worlds]
    return {k: by[k] for k in names}


def _norm(s):
    return re.sub(r"\s+", " ", str(s)).strip()


def match(pred, gold):
    """Exact match on whitespace-normalised text (full output, or the model's first line)."""
    g = _norm(gold)
    if not g:
        return False
    p = _norm(pred)
    if p == g:
        return True
    return _norm(str(pred).split("\n")[0]) == g


def build_prompt(instruction, demos, query_input):
    parts = [instruction, ""]
    for i, d in enumerate(demos, 1):
        parts += [f"Example {i}", "Input:", str(d["input"]), "Output:", str(d["output"]), ""]
    parts += ["Now solve.", "Input:", str(query_input), "Output:"]
    return "\n".join(parts)


def split_world(examples, n_pool, n_eval, rng):
    """Disjoint demonstration pool + held-out eval queries for one world."""
    idx = list(range(len(examples)))
    rng.shuffle(idx)
    pool = [examples[i] for i in idx[:n_pool]]
    qeval = [examples[i] for i in idx[n_pool:n_pool + n_eval]]
    return pool, qeval


def eval_cases(instruction, pool, qeval, n_ctx, rng, max_chars=9000):
    """Held-out eval prompts: each query gets n_ctx in-context demos from the pool."""
    cases = []
    for q in qeval:
        k = min(n_ctx, len(pool))
        demos = rng.sample(pool, k) if k else []
        prompt = build_prompt(instruction, demos, q["input"])
        if len(prompt) <= max_chars:
            cases.append({"prompt": prompt, "answer": q["output"]})
    return cases


def ttt_rows(instruction, pool, n_ctx, n_rows, rng, corrupt=False, max_chars=6000):
    """Leave-one-out training rows from a world's pool (the test queries are NOT here).
    corrupt=True swaps each target for a DIFFERENT pool output (wrong label, same structure)."""
    outs = [e["output"] for e in pool]
    distinct = list({o for o in outs})
    rows, tries = [], 0
    while len(rows) < n_rows and tries < n_rows * 6:
        tries += 1
        t = rng.randrange(len(pool))
        others = [i for i in range(len(pool)) if i != t]
        if not others:
            break
        k = min(n_ctx, len(others))
        demos = [pool[i] for i in rng.sample(others, k)]
        out = pool[t]["output"]
        if corrupt:
            choices = [o for o in distinct if o != out] or distinct
            out = rng.choice(choices)
        prompt = build_prompt(instruction, demos, pool[t]["input"])
        if len(prompt) + len(str(out)) <= max_chars:
            rows.append({"prompt": prompt, "completion": str(out)})
    return rows


if __name__ == "__main__":
    import random
    import statistics as st
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/listfn_worlds.jsonl"
    instr = "Infer the hidden rule from the examples and produce the output for the final input."
    W = load_worlds(path)
    sizes = [len(v) for v in W.values()]
    print(f"{path}: {len(W)} worlds; examples/world min {min(sizes)} "
          f"median {int(st.median(sizes))} max {max(sizes)}")
    rng = random.Random(0)
    name = sorted(W)[0]
    pool, qeval = split_world(W[name], n_pool=16, n_eval=8, rng=rng)
    ev = eval_cases(instr, pool, qeval, n_ctx=3, rng=rng)
    real = ttt_rows(instr, pool, n_ctx=3, n_rows=40, rng=rng)
    corr = ttt_rows(instr, pool, n_ctx=3, n_rows=40, rng=random.Random(0), corrupt=True)
    print(f"world {name}: pool {len(pool)} eval {len(ev)} | ttt rows real {len(real)} corrupt {len(corr)}")
    # match() sanity
    assert match("[1]", "[1]") and match("The output is [1]", "[1]") is False
    assert match("[1, 2]\nextra", "[1, 2]")
    diff = sum(1 for a, b in zip(real, corr) if a["completion"] != b["completion"])
    print(f"corrupt differs on {diff}/{len(real)} rows")
    print("eval prompt head:\n" + ev[0]["prompt"][:260])
    print("answer:", ev[0]["answer"][:60])
    # char budget across worlds
    plens = []
    for nm in list(W)[:40]:
        p, _ = split_world(W[nm], 16, 8, random.Random(1))
        plens += [len(r["prompt"]) + len(r["completion"]) for r in ttt_rows(instr, p, 3, 20, random.Random(2))]
    print(f"ttt row chars over 40 worlds: median {int(st.median(plens))} max {max(plens)}")
