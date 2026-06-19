"""E77 (assemble) - coding-world results into one JSON: pass@1 (greedy) + pass@1/3/5
(sampled), base vs world-time-compute, on synthetic held-out and HumanEval.

Reads c_{sz}_{syn,he}_{base,ft}.json (greedy pass@1) and pk2_{sz}_{synthetic,humaneval}_{base,ft}.json
(pass@k). Writes experiments/results/e77_coding.json.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "experiments" / "results" / "e77_artifacts" / "results"
OUT = ROOT / "experiments" / "results" / "e77_coding.json"
SIZES = ["0.5B", "1.5B", "3B", "7B"]


def g(name, key="pass_at_1"):
    f = SRC / name
    return json.loads(f.read_text()).get(key) if f.exists() else None


def main():
    greedy = {"synthetic": {}, "humaneval": {}}
    for sz in SIZES:
        greedy["synthetic"][sz] = {"base": g(f"c_{sz}_syn_base.json"), "ft": g(f"c_{sz}_syn_ft.json")}
        greedy["humaneval"][sz] = {"base": g(f"c_{sz}_he_base.json"), "ft": g(f"c_{sz}_he_ft.json")}

    passk = {"synthetic": {}, "humaneval": {}}
    for sz in SIZES:
        row = {}
        for tag in ("base", "ft"):
            f = SRC / f"pk2_{sz}_synthetic_{tag}.json"
            if f.exists():
                d = json.loads(f.read_text())
                row[tag] = {k: d.get(f"pass_at_{k}") for k in (1, 3, 5)}
        if row:
            passk["synthetic"][sz] = row
    for sz in SIZES:
        row = {}
        for tag in ("base", "ft"):
            f = SRC / f"pk2_{sz}_humaneval_{tag}.json"
            if f.exists():
                d = json.loads(f.read_text())
                row[tag] = {k: d.get(f"pass_at_{k}") for k in (1, 3, 5)}
        if row:
            passk["humaneval"][sz] = row

    out = {
        "task": "coding world family (E77): pass@1 (greedy) + pass@1/3/5 (sampled, n=5, "
                "temp 0.8), base vs world-time-compute (fine-tuned on 164 verified coding worlds)",
        "model_family": "Qwen2.5-Instruct", "n_train_tasks": 164, "n_test_synthetic": 55,
        "n_humaneval": 164, "task_source": "LLM-authored (Gemini), verified by pytest oracle",
        "greedy_pass_at_1": greedy,
        "passk": passk,
        "note": "greedy pass@1 understates: in-domain pass@k helps every size; on HumanEval "
                "(OOD) ft hurts greedy pass@1 but transfers POSITIVELY at pass@5 (7B 0.866->0.909). "
                "164 worlds is below the E76 world-count threshold; scaling worlds is the path up.",
    }
    OUT.write_text(json.dumps(out, indent=2))
    print("[e77-assemble] greedy syn:", {s: greedy["synthetic"][s] for s in SIZES})
    print("  HumanEval pass@k 7B:", passk["humaneval"].get("7B"))


if __name__ == "__main__":
    main()
