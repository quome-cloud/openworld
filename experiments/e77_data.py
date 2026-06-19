"""E77 (data) - split the verified coding-world family into train/test and build the SFT
set (the world-time-compute training signal for coding).

World-level split = TASK-level split here (whole tasks held out). SFT teaches the coding
skill by behavior-cloning verified solutions; eval is pass@1 on held-out tasks (and, via
e77_eval, on HumanEval/MBPP).

Reads experiments/results/e77_artifacts/tasks.jsonl; writes sft_train.jsonl (prompt ->
solution) and test_tasks.jsonl (prompt + tests for pass@1). Deterministic.
"""

import json
import random
from pathlib import Path

ART = Path(__file__).resolve().parent.parent / "experiments" / "results" / "e77_artifacts"
TEST_FRACTION = 0.25
SEED = 77

INSTR = "Implement the following Python function. Return ONLY the function definition, no explanation.\n\n"


def main():
    tasks = [json.loads(l) for l in (ART / "tasks.jsonl").read_text().splitlines() if l.strip()]
    rng = random.Random(SEED)
    rng.shuffle(tasks)
    n_test = max(1, round(len(tasks) * TEST_FRACTION))
    test, train = tasks[:n_test], tasks[n_test:]

    sft = [{"prompt": INSTR + t["prompt"], "completion": t["solution"]} for t in train]
    (ART / "sft_train.jsonl").write_text("\n".join(json.dumps(r) for r in sft) + "\n")

    test_rows = [{"id": t["name"], "prompt": INSTR + t["prompt"], "tests": t["tests"],
                  "kind": "synthetic"} for t in test]
    (ART / "test_tasks.jsonl").write_text("\n".join(json.dumps(r) for r in test_rows) + "\n")

    print(f"[e77-data] {len(tasks)} verified tasks -> {len(train)} train / {len(test)} test")
    print(f"  wrote sft_train.jsonl ({len(sft)} SFT) and test_tasks.jsonl ({len(test_rows)} held-out)")


if __name__ == "__main__":
    main()
