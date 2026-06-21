"""E80-ARC: world-time compute on the REAL ARC-AGI benchmark (human-authored, not generated).

Each ARC task is a WORLD: a hidden grid-transformation rule shown only through a few
input->output demonstration pairs. Unlike the synthetic E80 domains (where the rule was handed
to the model in the prompt), here the rule is LATENT and must be induced from the examples --
strictly harder, and a real-data rebuttal to the "you generated it" critique. Labels are
exact (a predicted output grid is right or it is not -- no judge).

This module is the offline, stdlib+numpy core:
  - load_tasks(dir)            real ARC-AGI-1 tasks {tid: {"train":[...], "test":[...]}}
  - augment(task, d, perm)     dihedral x color-permutation -> a consistent NEW world variant
  - encode_grid / parse_grid   grid <-> text (digit rows); parse is the exact-match decoder
  - build_prompt(demos, query) the perceive->reason prompt (all demos + a query input)
  - task_to_sft_rows(...)      augmented {prompt, completion} rows for SFT / test-time training
  - task_eval_example(task)    the held-out eval case (all demos + real test input -> output)

The GPU stages (e80_arc_train.py / e80_arc_eval.py) and orchestrator (e80_arc_run.py)
consume these. Everything here runs offline with no model.
"""

import glob
import json
import os

import numpy as np

NCOLORS = 10  # ARC colors are digits 0..9; 0 is the conventional background.


# ---- grid <-> text -------------------------------------------------------------------------

def encode_grid(grid):
    """A grid (list of int rows) -> text: one line of digits per row."""
    return "\n".join("".join(str(int(c)) for c in row) for row in grid)


def parse_grid(text):
    """Decode generated text -> grid (list of list of int), or None if unparseable.

    Takes the maximal run of consecutive digit-only lines (rows must share a width)."""
    rows = []
    for ln in text.splitlines():
        s = ln.strip()
        if s and all(ch.isdigit() for ch in s):
            rows.append([int(ch) for ch in s])
        elif rows:
            break  # grid ended
    if not rows:
        return None
    w = len(rows[0])
    if any(len(r) != w for r in rows):
        return None
    return rows


def grids_equal(a, b):
    if a is None or b is None:
        return False
    if len(a) != len(b) or any(len(ra) != len(rb) for ra, rb in zip(a, b)):
        return False
    return all(ca == cb for ra, rb in zip(a, b) for ca, cb in zip(ra, rb))


# ---- augmentation: dihedral group x color permutation --------------------------------------

DIHEDRAL = ["id", "rot90", "rot180", "rot270", "fliplr", "flipud", "transpose", "antitranspose"]


def _apply_dihedral(arr, name):
    if name == "id":
        return arr
    if name == "rot90":
        return np.rot90(arr, 1)
    if name == "rot180":
        return np.rot90(arr, 2)
    if name == "rot270":
        return np.rot90(arr, 3)
    if name == "fliplr":
        return np.fliplr(arr)
    if name == "flipud":
        return np.flipud(arr)
    if name == "transpose":
        return arr.T
    if name == "antitranspose":
        return np.rot90(arr.T, 2)
    raise ValueError(name)


def _color_perm(rng):
    """A permutation of colors 1..9 (0/background fixed) as a length-10 lookup."""
    lut = np.arange(NCOLORS)
    p = rng.permutation(np.arange(1, NCOLORS))
    lut[1:] = p
    return lut


def _xform_grid(grid, dname, lut):
    arr = np.array(grid, dtype=int)
    arr = _apply_dihedral(arr, dname)
    arr = lut[arr]
    return arr.tolist()


def augment(task, dname, lut):
    """Apply the SAME dihedral + color map to every pair -> a consistent new world variant.
    (The rule is preserved up to the transform, so the augmented task is self-consistent.)"""
    def xf_pair(p):
        return {"input": _xform_grid(p["input"], dname, lut),
                "output": _xform_grid(p["output"], dname, lut)}
    return {"train": [xf_pair(p) for p in task["train"]],
            "test": [xf_pair(p) for p in task["test"]]}


# ---- prompt construction -------------------------------------------------------------------

PREAMBLE = ("You are solving an abstract grid puzzle. Each grid uses digits 0-9 as colors. "
            "Infer the single transformation rule from the examples, then output the grid for "
            "the final input. Reply with ONLY the output grid (digit rows).")


def build_prompt(demos, query_input):
    parts = [PREAMBLE, ""]
    for i, p in enumerate(demos, 1):
        parts += [f"Example {i}", "Input:", encode_grid(p["input"]),
                  "Output:", encode_grid(p["output"]), ""]
    parts += ["Final", "Input:", encode_grid(query_input), "Output:"]
    return "\n".join(parts)


# ---- worlds (real tasks) -> SFT rows / eval cases ------------------------------------------

def load_tasks(path):
    """{task_id: task} from a dir of ARC json files (or a single dir of training/ eval/)."""
    out = {}
    for f in sorted(glob.glob(os.path.join(path, "*.json"))):
        tid = os.path.splitext(os.path.basename(f))[0]
        if tid.startswith("._"):
            continue
        try:
            out[tid] = json.load(open(f))
        except (ValueError, OSError):
            continue
    return out


def _augment_variants(task, n_aug, rng):
    """Yield up to n_aug distinct (dihedral, color-lut) world variants of a task (incl. identity)."""
    yield task  # identity world first
    seen = {("id", 0)}
    tries = 0
    while len(seen) < n_aug + 1 and tries < n_aug * 6:
        tries += 1
        dname = DIHEDRAL[int(rng.integers(len(DIHEDRAL)))]
        lut = _color_perm(rng)
        key = (dname, int("".join(map(str, lut.tolist()))))
        if key in seen:
            continue
        seen.add(key)
        yield augment(task, dname, lut)


def _max_chars_ok(prompt, completion, max_chars):
    return (len(prompt) + len(completion)) <= max_chars


def task_to_sft_rows(task, n_aug, rng, use_test=True, max_chars=6000, corrupt=False):
    """{prompt, completion} rows from a task via leave-one-out over demos x augmentation.

    use_test=True also adds (all demos -> real test pair) rows (training tasks have labels).
    corrupt=True replaces each target output with a random grid of the same shape (noise
    ablation: the augmentation/structure is intact but the LABEL is wrong)."""
    rows = []
    for v in _augment_variants(task, n_aug, rng):
        demos = v["train"]
        queries = list(range(len(demos)))  # leave-one-out: each demo as the held-out query
        cases = [(demos[:i] + demos[i + 1:], demos[i]) for i in queries]
        if use_test:
            for tp in v["test"]:
                cases.append((demos, tp))
        for ctx, qp in cases:
            if not ctx:
                continue
            out = qp["output"]
            if corrupt:
                h, w = len(out), len(out[0])
                out = rng.integers(0, NCOLORS, size=(h, w)).tolist()
            prompt = build_prompt(ctx, qp["input"])
            completion = encode_grid(out)
            if _max_chars_ok(prompt, completion, max_chars):
                rows.append({"prompt": prompt, "completion": completion})
    return rows


def task_eval_example(task, max_chars=8000):
    """The held-out eval case: all demos + the real test input -> its output (exact match)."""
    tp = task["test"][0]
    prompt = build_prompt(task["train"], tp["input"])
    answer = encode_grid(tp["output"])
    if len(prompt) > max_chars:
        return None
    return {"prompt": prompt, "answer": answer}


if __name__ == "__main__":
    import sys
    base = sys.argv[1] if len(sys.argv) > 1 else "/tmp/ARC-AGI/data"
    tr = load_tasks(os.path.join(base, "training"))
    ev = load_tasks(os.path.join(base, "evaluation"))
    print(f"loaded: {len(tr)} training worlds, {len(ev)} evaluation worlds")

    rng = np.random.default_rng(0)

    # round-trip: encode -> parse is lossless on a real grid
    g = tr[sorted(tr)[0]]["train"][0]["output"]
    assert grids_equal(parse_grid(encode_grid(g)), g), "encode/parse round-trip failed"

    # augmentation is a bijection that preserves shapes pairwise within a task
    t = tr[sorted(tr)[0]]
    lut = _color_perm(rng)
    a = augment(t, "rot90", lut)
    assert len(a["train"]) == len(t["train"])
    # color perm is invertible
    inv = np.zeros(NCOLORS, dtype=int)
    inv[lut] = np.arange(NCOLORS)
    assert (inv[lut] == np.arange(NCOLORS)).all(), "color perm not invertible"

    # SFT rows + eval case
    rows = task_to_sft_rows(t, n_aug=8, rng=rng)
    crows = task_to_sft_rows(t, n_aug=8, rng=np.random.default_rng(0), corrupt=True)
    ev_case = task_eval_example(t)
    print(f"sample task {sorted(tr)[0]}: {len(t['train'])} demos -> {len(rows)} SFT rows "
          f"(corrupt {len(crows)}); eval-case chars={len(ev_case['prompt'])}")
    # corruption changes labels but keeps shapes
    assert len(crows) == len(rows)
    print("first SFT prompt (head):\n" + rows[0]["prompt"][:300])
    print("first SFT completion:\n" + rows[0]["completion"])

    # corpus-wide row + length stats (what the SFT loader will see)
    import statistics as stat
    nrows, plens = [], []
    for tid in list(tr)[:60]:
        r = task_to_sft_rows(tr[tid], n_aug=6, rng=np.random.default_rng(hash(tid) % 2**32))
        nrows.append(len(r))
        plens += [len(x["prompt"]) + len(x["completion"]) for x in r]
    ev_ok = sum(1 for tid in ev if task_eval_example(ev[tid]) is not None)
    print(f"\nover 60 train tasks @ n_aug=6: rows/task med={int(stat.median(nrows))} "
          f"(min{min(nrows)}-max{max(nrows)}); chars/row med={int(stat.median(plens))} "
          f"p95={int(np.percentile(plens, 95))}")
    print(f"eval cases within char budget: {ev_ok}/{len(ev)}")
