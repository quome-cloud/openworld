"""E84: ARC-AGI program synthesis — enumerative + LLM-abductive vs neural-TTT baseline.

Two strategies for finding the hidden grid-transformation rule from demonstrations:
  (a) Enumerative: bottom-up search over a geometric DSL; keep programs consistent with
      ALL demo pairs; scale budget (programs tried) to test the clause-1 scaling result.
  (b) Abductive/LLM: show Sonnet the demo pairs, ask it to WRITE candidate DSL programs;
      run each against demos; keep consistent ones; apply to test.

Corrupt-label control (required): randomize demo outputs -> no consistent program found ->
collapses to floor. This is the load-bearing proof that exact labels are doing the work.

Baseline comparison: e80_arc_ttt.py (neural TTT) — zeroshot=2.5%, heavy=10%, corrupt=2.5%.
Same 40-task split used for exact comparability.

Run:
  python3 e84_arc_synthesis.py [--data /root/ARC-AGI/data] [--budget 5000] [--n 40]
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path
from itertools import product as iproduct
from typing import Optional

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import e80_arc as A
from openworld.arc_dsl.primitives import PURE_PRIMS, get_parameterized_prims, Grid
from openworld.arc_dsl.program import Program, enumerate_programs

BUDGETS = [10, 50, 200, 800, 3000, 10000]   # programs tried (the non-neural compute axis)
N_EVAL = 40   # match neural-TTT split exactly
SEED = 84


def grids_eq(a, b):
    return A.grids_equal(a, b)


def parse_demo(demo_pair):
    """Convert ARC demo dict {'input': ..., 'output': ...} to (Grid, Grid) or None."""
    try:
        inp = demo_pair["input"]
        out = demo_pair["output"]
        if isinstance(inp[0], list) and isinstance(out[0], list):
            return (inp, out)
    except (KeyError, IndexError, TypeError):
        pass
    return None


def task_demos(task):
    """List of (input_grid, output_grid) training pairs."""
    pairs = []
    for ex in task.get("train", []):
        p = parse_demo(ex)
        if p:
            pairs.append(p)
    return pairs


def task_test_input(task):
    """The test input grid."""
    try:
        return task["test"][0]["input"]
    except (KeyError, IndexError):
        return None


def task_test_answer(task):
    """The test answer grid (for eval)."""
    try:
        return task["test"][0]["output"]
    except (KeyError, IndexError):
        return None


def consistent(prog, demos):
    """True if prog(inp) == out for ALL demos."""
    for inp, out in demos:
        pred = prog(inp)
        if pred is None or not grids_eq(pred, out):
            return False
    return True


# ---- (a) Enumerative synthesizer -------------------------------------------------------

def synthesize_enumerative(demos, budget, all_prims, rng=None):
    """Return (program, programs_tried) or (None, budget)."""
    tried = 0
    for prog in enumerate_programs(all_prims, max_depth=3):
        if tried >= budget:
            break
        tried += 1
        if consistent(prog, demos):
            return prog, tried
    return None, tried


def task_acc_enumerative(task, budget, corrupt=False, rng=None):
    """0.0 or 1.0 — exact-match on the test grid."""
    demos = task_demos(task)
    if not demos:
        return None
    if corrupt:
        outs = [o for _, o in demos]
        (rng or random.Random()).shuffle(outs)
        demos = [(i, o) for (i, _), o in zip(demos, outs)]
    all_prims = {**PURE_PRIMS, **get_parameterized_prims(demos)}
    prog, _ = synthesize_enumerative(demos, budget, all_prims, rng)
    if prog is None:
        return 0.0
    ans = task_test_answer(task)
    if ans is None:
        return None
    pred = prog(task_test_input(task))
    return 1.0 if grids_eq(pred, ans) else 0.0


# ---- (b) LLM-abductive proposer --------------------------------------------------------

LLM_SYSTEM = """You are a program synthesis assistant. Given ARC-AGI grid transformation demos,
write Python expressions using ONLY these DSL primitives to describe the transformation rule.

Available primitives (all take a grid and return a grid):
rotate_90, rotate_180, rotate_270, flip_lr, flip_ud, transpose, antitranspose,
gravity_down, gravity_up, gravity_right, gravity_left, crop_to_content,
mirror_h, mirror_v, invert_colors, outline, sort_rows,
make_recolor(from_c, to_c), make_translate(dr, dc), make_tile(nr, nc)

A "program" is a Python list of primitive names (strings) to apply in order, e.g.:
["rotate_90", "flip_lr"]
or with parameterized ones:
["recolor_0_to_1", "rotate_90"]

Respond with a JSON list of candidate programs (each program is a list of strings).
Give at most 5 candidates. Think step by step about what transforms input -> output.
"""


def format_grid(g):
    return "\n".join("".join(str(c) for c in row) for row in g)


def demos_to_prompt(demos):
    lines = ["Look at these grid transformation examples:"]
    for i, (inp, out) in enumerate(demos):
        lines.append(f"\nExample {i + 1}:")
        lines.append(f"Input:\n{format_grid(inp)}")
        lines.append(f"Output:\n{format_grid(out)}")
    lines.append("\nWhat DSL program (list of primitives) transforms input -> output?")
    lines.append("Return JSON: [[\"prim1\", \"prim2\", ...], ...]  (list of candidate programs)")
    return "\n".join(lines)


def call_llm_proposer(demos, model="claude-haiku-4-5-20251001"):
    """Ask the LLM to propose candidate DSL programs. Returns list of program name-lists."""
    try:
        import anthropic
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=model,
            max_tokens=512,
            system=LLM_SYSTEM,
            messages=[{"role": "user", "content": demos_to_prompt(demos)}]
        )
        text = msg.content[0].text.strip()
        # Extract JSON from response
        import re
        m = re.search(r'\[.*\]', text, re.DOTALL)
        if m:
            candidates = json.loads(m.group())
            if isinstance(candidates, list):
                return candidates
    except Exception as e:
        print(f"  [llm-proposer] failed: {e}", flush=True)
    return []


def task_acc_abductive(task, all_prims, corrupt=False, rng=None, model="claude-haiku-4-5-20251001"):
    """LLM proposes programs; we run each against demos; pick first consistent one."""
    demos = task_demos(task)
    if not demos:
        return None
    if corrupt:
        outs = [o for _, o in demos]
        (rng or random.Random()).shuffle(outs)
        demos = [(i, o) for (i, _), o in zip(demos, outs)]

    candidates = call_llm_proposer(demos, model=model)

    # Build programs from candidate name lists
    for candidate in candidates:
        if not isinstance(candidate, list):
            continue
        steps = []
        valid = True
        for name in candidate:
            if name in all_prims:
                steps.append((name, all_prims[name]))
            else:
                valid = False
                break
        if not valid or not steps:
            continue
        prog = Program(steps)
        if consistent(prog, demos):
            ans = task_test_answer(task)
            if ans is None:
                return None
            pred = prog(task_test_input(task))
            return 1.0 if grids_eq(pred, ans) else 0.0
    return 0.0  # No consistent program found


# ---- bootstrap CIs ---------------------------------------------------------------------

def bootstrap_ci(vals, n_boot=1000, ci=0.95, seed=42):
    rng = np.random.default_rng(seed)
    means = []
    n = len(vals)
    for _ in range(n_boot):
        sample = rng.choice(vals, size=n, replace=True)
        means.append(float(np.mean(sample)))
    lo = float(np.percentile(means, (1 - ci) / 2 * 100))
    hi = float(np.percentile(means, (1 + ci) / 2 * 100))
    return lo, hi


# ---- main ------------------------------------------------------------------------------

def ensure_arc_data(data_path):
    p = Path(data_path)
    if p.exists() and (p / "evaluation").exists():
        return str(p)
    # Auto-download
    import subprocess
    print("[e84] ARC data not found; cloning from GitHub...", flush=True)
    subprocess.run(["git", "clone", "--depth", "1",
                    "https://github.com/fchollet/ARC-AGI", "/tmp/arc-agi"], check=True)
    return "/tmp/arc-agi/data"


def main():
    ap = argparse.ArgumentParser(
        description="E84: ARC-AGI program synthesis — enumerative + LLM-abductive"
    )
    ap.add_argument("--data", default="/root/ARC-AGI/data",
                    help="Path to ARC data dir (containing evaluation/ subdir)")
    ap.add_argument("--budget", type=int, default=10000,
                    help="Max programs to try for enumerative arm")
    ap.add_argument("--n", type=int, default=N_EVAL,
                    help="Number of tasks to evaluate (default: 40, matching neural-TTT split)")
    ap.add_argument("--arm", choices=["enum", "abduct", "both"], default="both",
                    help="Which arms to run")
    ap.add_argument("--model", default="claude-haiku-4-5-20251001",
                    help="Model for LLM-abductive arm")
    ap.add_argument("--no-corrupt", action="store_true",
                    help="Skip corrupt-label control")
    args = ap.parse_args()

    data_path = ensure_arc_data(args.data)
    ev = A.load_tasks(str(Path(data_path) / "evaluation"))
    ev_ids = [t for t in sorted(ev) if A.task_eval_example(ev[t]) is not None][:args.n]
    print(f"[e84] {len(ev_ids)} ARC evaluation tasks", flush=True)

    res = {
        "experiment": "e84_arc_synthesis",
        "n_tasks": len(ev_ids),
        "baselines": {"zeroshot": 0.025, "heavy_ttt": 0.100, "corrupt_ttt": 0.025},
        "budgets": BUDGETS,
        "arms": {},
        "per_task": {},
        "scaling_curve": {},
    }

    def save():
        out = HERE / "results" / "e84_arc_synthesis.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(res, indent=2))

    save()  # save before asserts (repo convention)

    rng = random.Random(SEED)

    # ---- (a) Enumerative arm: scaling sweep ----
    if args.arm in ("enum", "both"):
        print("\n[e84] (a) Enumerative synthesizer — budget sweep", flush=True)
        enum_budgets = [b for b in BUDGETS if b <= args.budget]
        for B in enum_budgets:
            hits = []
            for tid in ev_ids:
                task = ev[tid]
                acc = task_acc_enumerative(task, B, corrupt=False,
                                           rng=random.Random(hash(tid) % 2 ** 32))
                hits.append(acc if acc is not None else 0.0)
            mu = float(np.mean(hits))
            lo, hi = bootstrap_ci(hits)
            res["scaling_curve"][str(B)] = {
                "acc": round(mu, 4),
                "ci_lo": round(lo, 4),
                "ci_hi": round(hi, 4),
            }
            print(f"  budget={B:6d}: acc={mu:.3f} [{lo:.3f}, {hi:.3f}]", flush=True)
            save()

        # Corrupt control for enumerative (at max budget)
        if not args.no_corrupt:
            corrupt_hits = []
            for tid in ev_ids:
                task = ev[tid]
                acc = task_acc_enumerative(task, args.budget, corrupt=True,
                                           rng=random.Random(hash(tid) % 2 ** 32))
                corrupt_hits.append(acc if acc is not None else 0.0)
            mu_c = float(np.mean(corrupt_hits))
            lo_c, hi_c = bootstrap_ci(corrupt_hits)
            res["arms"]["enum_corrupt"] = {
                "acc": round(mu_c, 4),
                "ci_lo": round(lo_c, 4),
                "ci_hi": round(hi_c, 4),
            }
            print(f"  corrupt control: acc={mu_c:.3f} [{lo_c:.3f}, {hi_c:.3f}]", flush=True)
            save()

        # Best budget result
        best_B = str(max(int(k) for k in res["scaling_curve"]))
        res["arms"]["enum_best"] = res["scaling_curve"][best_B]
        save()

    # ---- (b) Abductive / LLM proposer arm ----
    if args.arm in ("abduct", "both"):
        print(f"\n[e84] (b) LLM-abductive proposer ({args.model})", flush=True)
        abduct_hits = []
        for tid in ev_ids:
            task = ev[tid]
            demos = task_demos(task)
            all_prims = {**PURE_PRIMS, **get_parameterized_prims(demos)}
            acc = task_acc_abductive(task, all_prims, corrupt=False,
                                     rng=random.Random(hash(tid) % 2 ** 32), model=args.model)
            abduct_hits.append(acc if acc is not None else 0.0)
            res["per_task"].setdefault("abduct", {})[tid] = acc
            save()
        mu = float(np.mean(abduct_hits))
        lo, hi = bootstrap_ci(abduct_hits)
        res["arms"]["abduct"] = {"acc": round(mu, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4)}
        print(f"  abductive: acc={mu:.3f} [{lo:.3f}, {hi:.3f}]", flush=True)

        if not args.no_corrupt:
            corrupt_abduct = []
            for tid in ev_ids:
                task = ev[tid]
                demos = task_demos(task)
                all_prims = {**PURE_PRIMS, **get_parameterized_prims(demos)}
                acc = task_acc_abductive(task, all_prims, corrupt=True,
                                         rng=random.Random(hash(tid) % 2 ** 32), model=args.model)
                corrupt_abduct.append(acc if acc is not None else 0.0)
            mu_c = float(np.mean(corrupt_abduct))
            lo_c, hi_c = bootstrap_ci(corrupt_abduct)
            res["arms"]["abduct_corrupt"] = {
                "acc": round(mu_c, 4),
                "ci_lo": round(lo_c, 4),
                "ci_hi": round(hi_c, 4),
            }
            print(f"  abductive corrupt: acc={mu_c:.3f} [{lo_c:.3f}, {hi_c:.3f}]", flush=True)
            save()

    save()
    print("\n[e84] Results summary:")
    print(f"  Baselines (neural TTT): zeroshot=2.5%, heavy=10%, corrupt=2.5%")
    for arm, v in res["arms"].items():
        print(f"  {arm}: {v}")
    print(f"\nSaved to {HERE / 'results' / 'e84_arc_synthesis.json'}")


if __name__ == "__main__":
    main()
