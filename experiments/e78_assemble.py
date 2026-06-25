"""E78 (assemble) - fold the QLoRA eval into the canonical results JSON the paper reads.

Reads experiments/results/e78_artifacts/{e78_eval.json, data_meta.json} (the eval is produced
on the GPU box / Modal by e78_eval.py and pulled back into the artifacts dir) and writes
experiments/results/e78_world_model_qlora.json via save_results. Self-checks run AFTER saving
so a failed assert never loses the run.

The claim: a base qwen2.5 floors on Blocksworld/PlanBench (E78 runtime-tool arms A0/A1/A2 ~ 0),
but distilling the VERIFIED world model's BFS oracle into it via 4-bit QLoRA lifts planning
sharply on held-out instances -- scored by the same verified validator. test_long (horizons
longer than any trained) is reported honestly as the boundary on length extrapolation.
"""

import json
from pathlib import Path

from common import save_results

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "experiments" / "results" / "e78_artifacts"


def main():
    ev = json.loads((ART / "e78_eval.json").read_text())
    meta_path = ART / "data_meta.json"
    data_meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    s = ev["summary"]

    payload = {
        "task": "verified world model as TEACHER: 4-bit QLoRA distillation of the BFS oracle "
                "(which plans through the verified Blocksworld world model) into a small LLM, "
                "tested on held-out instances scored by the verified validator",
        "benchmark": "blocksworld (PlanBench-style), %d blocks" % data_meta.get("n_blocks", 4),
        "base_model": ev["base"],
        "precision": "4-bit QLoRA" if ev.get("load_4bit") else "bf16 LoRA",
        "n_test": ev["n"],
        "data": {k: data_meta.get(k) for k in
                 ("n_blocks", "n_train", "train_horizons", "test_id_horizons",
                  "test_long_horizons", "counts")},
        "base": s["base"],
        "ft": s.get("ft"),
        "mcnemar_ft_vs_base": ev.get("mcnemar_ft_vs_base"),
        "decoding": "greedy (pass@1)",
        "scorer": "verified bw.validate_plan (every action legal AND goal reached)",
    }
    save_results("e78_world_model_qlora", payload)

    base_id = s["base"]["by_split"]["test_id"]["valid_rate"]
    base_long = s["base"]["by_split"]["test_long"]["valid_rate"]
    print(f"[e78-assemble] base: id={base_id:.3f} long={base_long:.3f}")
    if s.get("ft"):
        ft_id = s["ft"]["by_split"]["test_id"]["valid_rate"]
        ft_long = s["ft"]["by_split"]["test_long"]["valid_rate"]
        mp = ev.get("mcnemar_ft_vs_base", {})
        print(f"[e78-assemble] ft  : id={ft_id:.3f} long={ft_long:.3f}  "
              f"McNemar p={mp.get('p')}")

    # ---- self-checks AFTER saving ----
    assert base_id < 0.35, f"base should floor on PlanBench, got id={base_id}"
    if s.get("ft"):
        assert ft_id > base_id, f"QLoRA should beat base in-distribution: {ft_id} vs {base_id}"
        assert ev["mcnemar_ft_vs_base"]["p"] < 0.05, "ft-vs-base should be significant"
    print("[e78-assemble] self-checks passed")


if __name__ == "__main__":
    main()
