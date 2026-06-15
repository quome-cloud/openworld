#!/usr/bin/env python3
"""Confound check #1: is the MLX 4-bit base being UNDERSOLD vs Ollama?

Runs the SAME held-out instances single-shot through Ollama qwen2.5:1.5b (the
original bench runtime) so we can compare to the MLX-4bit base (0/10). If Ollama
solves meaningfully more, our MLX eval handicaps the base — and the LoRA's "win"
might be teaching output format/quant, not repair skill. Runs in the repo .venv
(has openworld + the Ollama LLM). Greedy (temp 0) to match the MLX eval.
"""
import argparse
import json
from pathlib import Path

from openworld.swebench import load_dataset, solve_single_shot
from openworld.llm import OllamaLLM

DATASETS = [
    "datasets/openworld-swebench/tasks.jsonl",
    "datasets/openworld-swebench-staged/tasks.jsonl",
]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--heldout", default="sft/v1/heldout_instances.json")
    ap.add_argument("--model", default="qwen2.5:1.5b")
    args = ap.parse_args(argv)

    by_id = {}
    for p in DATASETS:
        if Path(p).exists():
            for inst in load_dataset(Path(p)):
                by_id[inst.instance_id] = inst

    held = [i for i in json.loads(Path(args.heldout).read_text()) if i in by_id]
    llm = OllamaLLM(model=args.model, temperature=0.0, options={"seed": 41})

    solved = {}
    for iid in held:
        r = solve_single_shot(by_id[iid], llm)
        solved[iid] = bool(r["solved"])
        print(f"  [ollama-base] {iid}: {'SOLVED' if solved[iid] else 'no'}")
    n = sum(solved.values())
    print(f"\n[ollama-base] {args.model} solved {n}/{len(held)} single-shot (greedy)")
    print("compare to MLX-4bit base 0/10 -> if this is higher, the MLX eval "
          "undersells the base and the harness needs fixing before trusting Δ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
