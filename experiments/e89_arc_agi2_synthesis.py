"""E89 -- ARC-AGI-2 program synthesis: arbitrary LLM-written Python vs DSL (E84).

The E84 lesson: 0% with a 17-primitive geometric DSL because of a COVERAGE gap, not a
search gap. The LLM reasoned correctly but couldn't map reasoning to DSL primitives.

E89 tests the fix: drop the DSL. LLM writes arbitrary Python. Exact-match verify gate
still required -- accept only programs that match ALL demo pairs. Vote across survivors.

Pipeline:
  demos -> LLM writes "def transform(grid): ..." in free Python
        -> sandbox-execute against ALL demo pairs (subprocess + timeout, exact-match gate)
        -> vote survivors on test input
        -> score: 1.0 if vote matches answer, 0.0 otherwise.

Head-to-head vs E84 (same ARC-AGI-1 eval split, same metrics).
New: ARC-AGI-2 public eval set (120 tasks) -- purpose-built to resist DSL/brute-force solvers.

Run:
  python3 e89_arc_agi2_synthesis.py [--data /tmp/arc-agi-2/data/evaluation] [--n 120]
  python3 e89_arc_agi2_synthesis.py --n 40  # pilot (match E84 task count)
  python3 e89_arc_agi2_synthesis.py --corrupt  # corrupt-label control
"""
import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import e80_arc as A

ARC2_DEFAULT = "/tmp/arc-agi-2/data/evaluation"
RESULTS_OUT = HERE / "results" / "e89_arc_agi2.json"
SEED = 89
N_CANDIDATES = 8   # programs to request per task
SANDBOX_TIMEOUT = 10  # seconds per candidate execution

LLM_SYSTEM = """You are a program synthesis expert. Given ARC-AGI grid transformation examples,
write Python functions that implement the transformation rule.

A grid is a list of lists of integers (0-9). Your function must be named `transform` and take
a single argument (the input grid) and return the output grid.

Rules:
- Use ONLY the Python standard library (no imports beyond what you include inline).
- Do NOT use numpy, PIL, or any third-party library.
- The function must be self-contained.
- Return a list of lists of ints.

Respond with ONLY a JSON array of candidate Python function strings, like:
[
  "def transform(grid):\\n    return [row[::-1] for row in grid]",
  "def transform(grid):\\n    return grid[::-1]"
]

Give at most 8 candidates. Each should be a complete, runnable `def transform(grid): ...` string.
If you truly cannot infer a pattern, return an empty array: []
"""


def format_grid(g):
    return "\n".join("".join(str(c) for c in row) for row in g)


def demos_to_prompt(demos):
    lines = ["Study these input→output grid transformation examples:"]
    for i, (inp, out) in enumerate(demos):
        lines.append(f"\nExample {i + 1}:")
        lines.append(f"Input ({len(inp)} rows × {len(inp[0])} cols):\n{format_grid(inp)}")
        lines.append(f"Output ({len(out)} rows × {len(out[0])} cols):\n{format_grid(out)}")
    lines.append(
        "\nWrite Python `def transform(grid): ...` functions implementing the rule."
        "\nReturn ONLY a JSON array of function strings. No markdown fences, no explanation."
    )
    return "\n".join(lines)


def call_llm(demos, model="claude-haiku-4-5-20251001"):
    """Ask LLM for candidate Python transform functions. Returns list of code strings."""
    try:
        import anthropic
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=model,
            max_tokens=2048,
            system=LLM_SYSTEM,
            messages=[{"role": "user", "content": demos_to_prompt(demos)}],
        )
        text = msg.content[0].text.strip()
        # Extract outermost [...] block
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end > start:
            try:
                candidates = json.loads(text[start : end + 1])
                if isinstance(candidates, list):
                    return [c for c in candidates if isinstance(c, str) and "def transform" in c]
            except json.JSONDecodeError:
                pass
    except Exception as e:
        print(f"  [llm] failed: {e}", flush=True)
    return []


def sandbox_run(code, input_grid, timeout=SANDBOX_TIMEOUT):
    """Execute LLM code in subprocess. Returns predicted grid (list of lists) or None."""
    harness = textwrap.dedent(f"""
import json, sys

{code}

grid = json.loads(sys.argv[1])
try:
    result = transform(grid)
    print(json.dumps(result))
except Exception as e:
    print(f"ERROR: {{e}}", file=sys.stderr)
    sys.exit(1)
""")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(harness)
        fname = f.name
    try:
        proc = subprocess.run(
            [sys.executable, fname, json.dumps(input_grid)],
            capture_output=True, text=True, timeout=timeout
        )
        if proc.returncode != 0:
            return None
        out = proc.stdout.strip()
        result = json.loads(out)
        if isinstance(result, list) and all(isinstance(r, list) for r in result):
            return result
    except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception):
        return None
    finally:
        try:
            os.unlink(fname)
        except OSError:
            pass
    return None


def verify(code, demos):
    """True if transform(inp) == out for ALL demo pairs."""
    for inp, out in demos:
        pred = sandbox_run(code, inp)
        if pred is None or not A.grids_equal(pred, out):
            return False
    return True


def task_acc(task, corrupt=False, rng=None, model="claude-haiku-4-5-20251001"):
    """0.0 or 1.0. Returns None if task has no usable demos."""
    demos = [(ex["input"], ex["output"]) for ex in task.get("train", [])
             if "input" in ex and "output" in ex]
    if not demos:
        return None

    if corrupt:
        outs = [o for _, o in demos]
        (rng or random.Random()).shuffle(outs)
        demos = [(i, o) for (i, _), o in zip(demos, outs)]

    test_input = task.get("test", [{}])[0].get("input")
    test_answer = task.get("test", [{}])[0].get("output")
    if test_input is None or test_answer is None:
        return None

    candidates = call_llm(demos, model=model)
    verified = [c for c in candidates if verify(c, demos)]

    if not verified:
        return 0.0

    # Vote: each verified program predicts the test output; pick majority
    preds = []
    for code in verified:
        pred = sandbox_run(code, test_input)
        if pred is not None:
            preds.append(pred)

    if not preds:
        return 0.0

    # Majority vote (stringify for hashability)
    votes = {}
    for p in preds:
        key = json.dumps(p)
        votes[key] = votes.get(key, 0) + 1
    winner_key = max(votes, key=votes.__getitem__)
    winner = json.loads(winner_key)
    return 1.0 if A.grids_equal(winner, test_answer) else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=ARC2_DEFAULT)
    ap.add_argument("--n", type=int, default=120, help="# tasks to evaluate")
    ap.add_argument("--corrupt", action="store_true", help="corrupt-label control")
    ap.add_argument("--model", default="claude-haiku-4-5-20251001")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    tasks = A.load_tasks(args.data)
    task_ids = sorted(tasks)[:args.n]

    print(f"E89 ARC-AGI-2 Python synthesis | n={len(task_ids)} | corrupt={args.corrupt} | model={args.model}")

    results = {}
    solved = 0
    total = 0

    for i, tid in enumerate(task_ids):
        t0 = time.time()
        acc = task_acc(tasks[tid], corrupt=args.corrupt, rng=rng, model=args.model)
        elapsed = time.time() - t0
        if acc is not None:
            results[tid] = {"acc": acc, "time": round(elapsed, 2)}
            total += 1
            solved += int(acc > 0.5)
            pct = 100 * solved / total
            print(f"  [{i+1}/{len(task_ids)}] {tid}: {acc:.1f}  ({pct:.1f}% running)  {elapsed:.1f}s", flush=True)
        else:
            results[tid] = {"acc": None, "time": round(elapsed, 2)}
            print(f"  [{i+1}/{len(task_ids)}] {tid}: skipped (no demos)", flush=True)

    # Save before asserts
    summary = {
        "n_tasks": len(task_ids),
        "n_scored": total,
        "n_solved": solved,
        "pct_solved": round(100 * solved / total, 2) if total else 0.0,
        "corrupt": args.corrupt,
        "model": args.model,
        "tasks": results,
    }
    RESULTS_OUT.parent.mkdir(parents=True, exist_ok=True)
    label = "_corrupt" if args.corrupt else ""
    out_path = HERE / "results" / f"e89_arc_agi2{label}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved → {out_path}")
    print(f"FINAL: {solved}/{total} solved = {summary['pct_solved']}%")

    if not args.corrupt:
        print(f"\nE84 DSL baseline (same eval approach, ARC-AGI-1): 0.0%")
        print(f"E89 arbitrary Python (ARC-AGI-2): {summary['pct_solved']}%")


if __name__ == "__main__":
    main()
