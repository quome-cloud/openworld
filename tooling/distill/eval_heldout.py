#!/usr/bin/env python3
"""P4: single-shot eval of base vs LoRA-distilled qwen2.5-1.5b on HELD-OUT instances.

The honest test of the flywheel: on instances NOT in the training split, does the
distilled model solve more single-shot than the base model? Both run through the
SAME runtime (MLX) so the comparison is apples-to-apples; tests are the exact
hidden suites from the openworld-swebench harness (reused, pure, no Ollama).

Runs in .venv-distill (mlx-lm). Reuses openworld.swebench pure helpers (importable
from repo root; the framework is zero-dependency so the import is stdlib-only).
"""
import argparse
import json
from pathlib import Path

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

from openworld.swebench import (
    load_dataset, run_instance_tests, _base_prompt, extract_code, SYSTEM_PROMPT,
)

MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
DATASETS = [
    "datasets/openworld-swebench/tasks.jsonl",
    "datasets/openworld-swebench-staged/tasks.jsonl",
]


def _instances():
    by_id = {}
    for p in DATASETS:
        if Path(p).exists():
            for inst in load_dataset(Path(p)):
                by_id[inst.instance_id] = inst
    return by_id


def _solve(model, tok, inst, sampler, max_tokens):
    """One single-shot attempt: base prompt -> patch -> run hidden tests."""
    prompt = tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": _base_prompt(inst, inst.buggy_source)
          + "\nProvide the corrected module."}],
        add_generation_prompt=True, tokenize=False)
    out = generate(model, tok, prompt=prompt, max_tokens=max_tokens,
                   sampler=sampler, verbose=False)
    patch = extract_code(out)
    res = run_instance_tests(patch, inst)
    return bool(res["solved"])


def _eval_variant(label, adapter, held, by_id, max_tokens):
    model, tok = load(MODEL, adapter_path=adapter)
    sampler = make_sampler(temp=0.0)          # greedy = deterministic single-shot
    solved = {}
    for iid in held:
        solved[iid] = _solve(model, tok, by_id[iid], sampler, max_tokens)
        print(f"  [{label}] {iid}: {'SOLVED' if solved[iid] else 'no'}")
    n = sum(solved.values())
    print(f"[{label}] solved {n}/{len(held)} single-shot")
    return solved


def _mcnemar(base, dist, held):
    # paired discordant counts: b = base-only solved, c = distilled-only solved
    b = sum(1 for i in held if base[i] and not dist[i])
    c = sum(1 for i in held if dist[i] and not base[i])
    return b, c


def main(argv=None):
    global MODEL
    ap = argparse.ArgumentParser(prog="eval_heldout")
    ap.add_argument("--heldout", default="sft/v1/heldout_instances.json")
    ap.add_argument("--adapter", default="tooling/distill/adapters/qwen1.5b-v1")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--out", default="tooling/distill/eval/heldout_v1.json")
    ap.add_argument("--model", default=MODEL, help="base MLX model (override for 3B student etc.)")
    args = ap.parse_args(argv)
    MODEL = args.model

    held = json.loads(Path(args.heldout).read_text())
    by_id = _instances()
    held = [i for i in held if i in by_id]      # only ids we can load
    print(f"[eval] {len(held)} held-out instances\n")

    print("=== BASE qwen2.5-1.5b (no adapter) ===")
    base = _eval_variant("base", None, held, by_id, args.max_tokens)
    print("\n=== DISTILLED qwen2.5-1.5b (+LoRA) ===")
    dist = _eval_variant("distilled", args.adapter, held, by_id, args.max_tokens)

    nb, nd = sum(base.values()), sum(dist.values())
    b, c = _mcnemar(base, dist, held)
    flipped_win = [i for i in held if dist[i] and not base[i]]
    flipped_lose = [i for i in held if base[i] and not dist[i]]

    result = {
        "model": MODEL, "n_heldout": len(held),
        "base_solved": nb, "distilled_solved": nd,
        "delta": nd - nb,
        "mcnemar_discordant": {"base_only": b, "distilled_only": c},
        "distilled_wins": flipped_win, "distilled_regressions": flipped_lose,
        "per_instance": {i: {"base": base[i], "distilled": dist[i]} for i in held},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    print("\n=== RESULT ===")
    print(f"base:      {nb}/{len(held)} single-shot")
    print(f"distilled: {nd}/{len(held)} single-shot   (Δ {nd - nb:+d})")
    print(f"discordant: distilled-only={c}, base-only={b} "
          f"(McNemar exact p needs scipy; report counts)")
    if flipped_win:
        print(f"distilled newly solves: {flipped_win}")
    if flipped_lose:
        print(f"distilled regressions:  {flipped_lose}")
    print(f"[out] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
