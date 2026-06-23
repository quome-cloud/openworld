"""E84 Haiku diagnostic: inspect raw LLM responses for 5 specific ARC tasks."""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from openworld.arc_dsl.primitives import PURE_PRIMS, get_parameterized_prims
from openworld.arc_dsl.program import Program

ARC_DATA = Path("/tmp/arc-agi/data/evaluation")
RESULTS_OUT = HERE / "results" / "e84_haiku_inspection.json"

TASK_IDS = [
    "00576224",  # 2x2->6x6 size-changing, TTT-heavy solves
    "0607ce86",  # same-size, TTT-heavy solves
    "0e671a1a",  # same-size, new output colors, TTT-heavy solves
    "12422b43",  # same-size, TTT-heavy solves
    "009d5c81",  # same-size, no arm solves (TTT-unsolved representative)
]

MODEL = "claude-haiku-4-5-20251001"

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
    lines.append('Return JSON: [["prim1", "prim2", ...], ...]  (list of candidate programs)')
    return "\n".join(lines)


def load_task(task_id):
    p = ARC_DATA / f"{task_id}.json"
    return json.loads(p.read_text())


def task_demos(task):
    pairs = []
    for ex in task.get("train", []):
        try:
            inp = ex["input"]
            out = ex["output"]
            if isinstance(inp[0], list) and isinstance(out[0], list):
                pairs.append((inp, out))
        except (KeyError, IndexError, TypeError):
            pass
    return pairs


def grids_eq(a, b):
    if len(a) != len(b):
        return False
    for ra, rb in zip(a, b):
        if list(ra) != list(rb):
            return False
    return True


def consistent(prog, demos):
    for inp, out in demos:
        pred = prog(inp)
        if pred is None or not grids_eq(pred, out):
            return False
    return True


def call_llm(demos):
    import anthropic
    client = anthropic.Anthropic()
    prompt = demos_to_prompt(demos)
    msg = client.messages.create(
        model=MODEL,
        max_tokens=512,
        system=LLM_SYSTEM,
        messages=[{"role": "user", "content": prompt}]
    )
    return msg.content[0].text.strip()


def parse_candidates(raw_text):
    """Extract JSON candidate list from raw response."""
    m = re.search(r'\[.*\]', raw_text, re.DOTALL)
    if not m:
        return None
    try:
        candidates = json.loads(m.group())
        if isinstance(candidates, list):
            return candidates
    except json.JSONDecodeError:
        pass
    return None


def categorize(raw_text, candidates, all_prims, demos):
    """
    Categories:
    a = invalid DSL name (all candidates have at least one unknown primitive)
    b = valid JSON, all names known, but no program consistent with demos
    c = valid + consistent (unexpected win)
    d = JSON parse failure
    e = other / empty response
    """
    if candidates is None:
        # Check if it's a parse failure vs empty
        m = re.search(r'\[', raw_text)
        if m:
            return "d", "JSON found but failed to parse"
        return "e", "No JSON list found in response"

    if not isinstance(candidates, list) or len(candidates) == 0:
        return "e", "Parsed JSON but got empty or non-list"

    # Check each candidate
    n_valid_names = 0
    n_consistent = 0
    any_invalid_name = False
    invalid_name_examples = []

    for candidate in candidates:
        if not isinstance(candidate, list):
            continue
        # Check names
        bad_names = [n for n in candidate if n not in all_prims]
        if bad_names:
            any_invalid_name = True
            invalid_name_examples.extend(bad_names)
            continue
        # All names valid
        if not candidate:
            continue
        n_valid_names += 1
        steps = [(name, all_prims[name]) for name in candidate]
        prog = Program(steps)
        if consistent(prog, demos):
            n_consistent += 1

    if n_valid_names == 0:
        if any_invalid_name:
            bad_str = ", ".join(sorted(set(invalid_name_examples))[:5])
            return "a", f"All candidates have invalid primitive names. Examples: {bad_str}"
        return "e", "Candidates parsed but all are non-list or empty"

    if n_consistent > 0:
        return "c", f"{n_consistent}/{n_valid_names} candidates consistent with demos"

    return "b", f"0/{n_valid_names} candidates consistent with demos (names all valid)"


def print_ascii_grid(g, label=""):
    if label:
        print(f"  {label}:")
    for row in g:
        print("    " + " ".join(str(c) for c in row))


def main():
    results = {}

    for task_id in TASK_IDS:
        print(f"\n{'='*60}")
        print(f"Task: {task_id}")
        print(f"{'='*60}")

        task = load_task(task_id)
        demos = task_demos(task)
        if not demos:
            print("  ERROR: no demos found")
            continue

        inp0, out0 = demos[0]
        h_in, w_in = len(inp0), len(inp0[0])
        h_out, w_out = len(out0), len(out0[0])

        print(f"\nDemo 0 — Input ({h_in}x{w_in}):")
        print_ascii_grid(inp0)
        print(f"\nDemo 0 — Output ({h_out}x{w_out}):")
        print_ascii_grid(out0)

        all_prims = {**PURE_PRIMS, **get_parameterized_prims(demos)}
        print(f"\nTotal primitives available: {len(all_prims)}")

        print(f"\nCalling {MODEL}...")
        try:
            raw_response = call_llm(demos)
        except Exception as e:
            print(f"  API ERROR: {e}")
            results[task_id] = {
                "demo_0_input_shape": [h_in, w_in],
                "demo_0_output_shape": [h_out, w_out],
                "raw_response": f"API ERROR: {e}",
                "parsed_candidates": None,
                "category": "e",
                "category_detail": f"API call failed: {e}",
                "n_valid_names": 0,
                "n_consistent": 0,
            }
            continue

        print(f"\nRaw response (first 500 chars):")
        print(f"  {raw_response[:500]!r}")

        candidates = parse_candidates(raw_response)
        print(f"\nParsed candidates: {candidates}")

        # Count valid names and consistent programs
        n_valid_names = 0
        n_consistent = 0
        if candidates:
            for candidate in candidates:
                if not isinstance(candidate, list):
                    continue
                bad = [n for n in candidate if n not in all_prims]
                if bad:
                    continue
                if not candidate:
                    continue
                n_valid_names += 1
                steps = [(name, all_prims[name]) for name in candidate]
                prog = Program(steps)
                if consistent(prog, demos):
                    n_consistent += 1

        cat, detail = categorize(raw_response, candidates, all_prims, demos)

        print(f"\nCategory: {cat} — {detail}")
        print(f"n_valid_names: {n_valid_names}, n_consistent: {n_consistent}")

        results[task_id] = {
            "demo_0_input_shape": [h_in, w_in],
            "demo_0_output_shape": [h_out, w_out],
            "raw_response": raw_response,
            "parsed_candidates": candidates,
            "category": cat,
            "category_detail": detail,
            "n_valid_names": n_valid_names,
            "n_consistent": n_consistent,
        }

    # Save
    RESULTS_OUT.parent.mkdir(exist_ok=True)
    RESULTS_OUT.write_text(json.dumps(results, indent=2))
    print(f"\n\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    cat_counts = {}
    for tid, r in results.items():
        cat = r["category"]
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
        print(f"  {tid}: category={cat} | {r['category_detail']}")
    print(f"\nCategory distribution: {cat_counts}")
    print(f"\nSaved to: {RESULTS_OUT}")


if __name__ == "__main__":
    main()
