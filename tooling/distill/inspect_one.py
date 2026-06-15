#!/usr/bin/env python3
"""Eyeball check (#3): show base vs distilled raw output for one held-out instance.

Confirms the distilled model produces a genuine fix (valid module that passes the
hidden tests), not a lucky format alignment or garbage. Runs in .venv-distill.
"""
import argparse
from pathlib import Path

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

from openworld.swebench import (
    load_dataset, run_instance_tests, _base_prompt, extract_code, SYSTEM_PROMPT,
)

MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
DATASETS = ["datasets/openworld-swebench/tasks.jsonl",
            "datasets/openworld-swebench-staged/tasks.jsonl"]


def _gen(model, tok, inst):
    prompt = tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": _base_prompt(inst, inst.buggy_source)
          + "\nProvide the corrected module."}],
        add_generation_prompt=True, tokenize=False)
    out = generate(model, tok, prompt=prompt, max_tokens=1024,
                   sampler=make_sampler(temp=0.0), verbose=False)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="openworld-swebench-005-interval-merge-touching")
    ap.add_argument("--adapter", default="tooling/distill/adapters/qwen1.5b-v1")
    args = ap.parse_args(argv)

    by_id = {}
    for p in DATASETS:
        if Path(p).exists():
            for inst in load_dataset(Path(p)):
                by_id[inst.instance_id] = inst
    inst = by_id[args.instance]

    print(f"### INSTANCE: {args.instance}")
    print(f"\n--- ISSUE ---\n{inst.issue}")
    print(f"\n--- REFERENCE (correct) SOURCE ---\n{inst.reference_source}")

    for label, adapter in (("BASE", None), ("DISTILLED", args.adapter)):
        model, tok = load(MODEL, adapter_path=adapter)
        raw = _gen(model, tok, inst)
        patch = extract_code(raw)
        res = run_instance_tests(patch, inst)
        print(f"\n========== {label} RAW OUTPUT ==========\n{raw}")
        print(f"\n----- {label} extracted patch solves? {res['solved']} "
              f"(f2p failed={res['fail_to_pass']['failed']}, "
              f"p2p failed={res['pass_to_pass']['failed']}) -----")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
