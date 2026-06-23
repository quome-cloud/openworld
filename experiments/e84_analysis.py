"""E84 analysis helpers: per-task overlap heatmap + task categorization.

Reads e84_arc_synthesis.json and e80_arc_ttt.json to produce:
  - Per-task solve matrix: which arm solves which task (overlap/union/exclusive)
  - Task categorization: input→output size change, color count, demo count
  - Summary table ready for writeup

Run:
  python3 e84_analysis.py [--results-dir ./results] [--data /root/ARC-AGI/data]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import e80_arc as A


# ---- task categorization -------------------------------------------------------

def categorize_task(task):
    """Return a dict of properties useful for understanding why an arm fails/succeeds."""
    demos = task.get("train", [])
    if not demos:
        return {}
    sizes_in = [(len(ex["input"]), len(ex["input"][0])) for ex in demos]
    sizes_out = [(len(ex["output"]), len(ex["output"][0])) for ex in demos]
    test = task.get("test", [{}])[0]
    colors_in = set(c for ex in demos for row in ex["input"] for c in row)
    colors_out = set(c for ex in demos for row in ex["output"] for c in row)
    size_changes = [si != so for si, so in zip(sizes_in, sizes_out)]
    return {
        "n_demos": len(demos),
        "input_size": sizes_in[0] if sizes_in else None,
        "output_size": sizes_out[0] if sizes_out else None,
        "size_changes": any(size_changes),
        "n_input_colors": len(colors_in),
        "n_output_colors": len(colors_out),
        "new_colors_in_output": bool(colors_out - colors_in),
        "test_input_size": (len(test["input"]), len(test["input"][0])) if test.get("input") else None,
    }


# ---- overlap matrix ------------------------------------------------------------

def build_solve_matrix(e84, ttt):
    """
    Returns dict keyed by task_id: {arm -> 0/1} for all arms across both experiments.
    Arms: enum_best, abduct, zeroshot, light, heavy, corrupt (ttt)
    """
    matrix = {}

    # e84 arms
    e84_per = e84.get("per_task", {})
    for arm, per in e84_per.items():
        for tid, val in per.items():
            matrix.setdefault(tid, {})[f"e84_{arm}"] = int(val == 1.0) if val is not None else 0

    # e84 enumerative best: all 0 (stored in arms.enum_best, not per_task)
    # We know from the run it's 0 for all tasks — mark all as 0
    all_tids = set(matrix)

    # ttt arms
    ttt_per = ttt.get("per_task", {})
    for arm, per in ttt_per.items():
        for tid, val in per.items():
            if tid in all_tids or True:
                matrix.setdefault(tid, {})[f"ttt_{arm}"] = int(val == 1) if val is not None else 0

    return matrix


def summarize_overlap(matrix):
    """Print overlap statistics between e84 and ttt arms."""
    tids = sorted(matrix)
    arms = sorted({a for row in matrix.values() for a in row})

    print(f"\n{'Task':>12}", end="")
    for a in arms:
        print(f"  {a[:12]:>12}", end="")
    print()

    for tid in tids:
        row = matrix[tid]
        print(f"{tid:>12}", end="")
        for a in arms:
            v = row.get(a, 0)
            print(f"  {'✓' if v else '·':>12}", end="")
        print()

    print(f"\n{'TOTAL':>12}", end="")
    for a in arms:
        total = sum(row.get(a, 0) for row in matrix.values())
        print(f"  {total:>12}", end="")
    print()

    # Union and intersection between ttt_heavy and e84_abduct
    e84_solved = {tid for tid, row in matrix.items() if row.get("e84_abduct", 0) == 1}
    ttt_solved = {tid for tid, row in matrix.items() if row.get("ttt_heavy", 0) == 1}
    print(f"\nTTT-heavy solves: {len(ttt_solved)}")
    print(f"E84-abduct solves: {len(e84_solved)}")
    print(f"Union: {len(e84_solved | ttt_solved)}")
    print(f"Intersection: {len(e84_solved & ttt_solved)}")
    print(f"E84-only: {e84_solved - ttt_solved}")
    print(f"TTT-only: {ttt_solved - e84_solved}")

    return {"e84_solved": sorted(e84_solved), "ttt_solved": sorted(ttt_solved)}


def categorize_all(ev, tids):
    """Return task categories for all eval tasks."""
    cats = {}
    for tid in tids:
        if tid in ev:
            cats[tid] = categorize_task(ev[tid])
    return cats


def print_category_report(categories, matrix):
    """Show task properties for tasks solved vs unsolved."""
    size_changing = [tid for tid, c in categories.items() if c.get("size_changes")]
    same_size = [tid for tid, c in categories.items() if not c.get("size_changes")]
    print(f"\nSize-changing tasks: {len(size_changing)} ({size_changing[:5]}...)")
    print(f"Same-size tasks: {len(same_size)}")

    # Color analysis
    new_color_tasks = [tid for tid, c in categories.items() if c.get("new_colors_in_output")]
    print(f"Tasks with new output colors: {len(new_color_tasks)}")

    # Demo count distribution
    demo_counts = [c["n_demos"] for c in categories.values()]
    print(f"Demo count distribution: min={min(demo_counts)} max={max(demo_counts)} mean={np.mean(demo_counts):.1f}")


# ---- main -----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(HERE / "results"))
    ap.add_argument("--data", default="/tmp/arc-agi/data")
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    rd = Path(args.results_dir)
    e84_path = rd / "e84_arc_synthesis.json"
    ttt_path = rd / "e80_arc_ttt.json"

    if not e84_path.exists():
        print(f"[e84-analysis] {e84_path} not found — run e84_arc_synthesis.py first")
        sys.exit(1)
    if not ttt_path.exists():
        print(f"[e84-analysis] {ttt_path} not found — expected from e80_arc_ttt.py run")
        sys.exit(1)

    e84 = json.loads(e84_path.read_text())
    ttt = json.loads(ttt_path.read_text())

    # Load ARC data for categorization
    ev = {}
    try:
        ev = A.load_tasks(str(Path(args.data) / "evaluation"))
        print(f"[e84-analysis] Loaded {len(ev)} ARC eval tasks")
    except Exception as ex:
        print(f"[e84-analysis] Could not load ARC data: {ex} — skipping categorization")

    # Build solve matrix
    matrix = build_solve_matrix(e84, ttt)
    print(f"\n[e84-analysis] Solve matrix over {len(matrix)} tasks")

    # Summarize
    overlap = summarize_overlap(matrix)

    # Categorize tasks in matrix
    if ev:
        cats = categorize_all(ev, list(matrix))
        print_category_report(cats, matrix)

        # For TTT-solved tasks: show their properties
        ttt_solved_cats = {tid: cats[tid] for tid in overlap["ttt_solved"] if tid in cats}
        if ttt_solved_cats:
            print("\nProperties of TTT-solved tasks:")
            for tid, c in ttt_solved_cats.items():
                print(f"  {tid}: {c['input_size']} -> {c['output_size']} | {c['n_demos']} demos | "
                      f"size_changes={c['size_changes']} | new_colors={c['new_colors_in_output']}")

    if args.save:
        out = rd / "e84_overlap_analysis.json"
        out.write_text(json.dumps({"overlap": overlap, "matrix": matrix}, indent=2))
        print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
