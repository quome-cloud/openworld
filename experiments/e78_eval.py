"""E78 (QLoRA eval) - does verified-planner distillation make qwen2.5 plan Blocksworld?

For every held-out instance we prompt the model for a plan, parse it, and REPLAY it through
the verified world model (bw.validate_plan = the ground-truth oracle); valid = every action
legal AND the goal reached. We report base vs the QLoRA adapter, paired per-instance, split by:
  test_id   - held-out instances at TRAINED horizons (in-distribution generalization)
  test_long - LONGER horizons than any trained (length extrapolation: algorithm vs memorization)

When --adapter is given we load the base once and toggle the adapter on/off per instance
(peft disable_adapter), so base and fine-tuned are scored on the SAME generations setup and
the comparison is exactly paired (McNemar). Greedy decoding (pass@1).

  python e78_eval.py --base Qwen/Qwen2.5-7B-Instruct --adapter e78_adapter \
      --tasks test.jsonl --load_4bit --out e78_eval.json
"""

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import blocksworld as bw
from common import mcnemar_p, wilson_ci
from e78_world_model_tool import parse_plan

HORIZONS = [2, 4, 6, 8, 10, 12]


def _rate(flags):
    n, k = len(flags), sum(flags)
    lo, hi = wilson_ci(k, n) if n else (0.0, 0.0)
    return {"valid_rate": round(k / n, 4) if n else 0.0, "n": n,
            "n_valid": k, "ci": [round(lo, 4), round(hi, 4)]}


def _strata(records, arm):
    """Aggregate one arm's per-instance validity overall, by split, and by horizon."""
    flags = [r[arm] for r in records]
    by_split, by_h = {}, {}
    for sp in ("test_id", "test_long"):
        by_split[sp] = _rate([r[arm] for r in records if r["split"] == sp])
    for L in HORIZONS:
        sub = [r[arm] for r in records if r["optimal_len"] == L]
        if sub:
            by_h[str(L)] = _rate(sub)
    return {"overall": _rate(flags), "by_split": by_split, "by_horizon": by_h}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--tasks", default="test.jsonl")
    ap.add_argument("--load_4bit", action="store_true")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0, help="eval only first N (smoke)")
    ap.add_argument("--out", default="e78_eval.json")
    args = ap.parse_args()

    tasks = [json.loads(l) for l in open(args.tasks) if l.strip()]
    if args.limit:
        tasks = tasks[:args.limit]
    print(f"[e78-eval] {len(tasks)} instances; base={args.base} adapter={args.adapter}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(args.base, quantization_config=bnb,
                                                     device_map={"": 0})
    else:
        model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16,
                                                     device_map={"": 0})
    has_adapter = bool(args.adapter)
    if has_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    @torch.no_grad()
    def gen(prompt, use_adapter):
        # toggle the LoRA adapter: disabled => base weights, enabled => fine-tuned. Same loaded
        # model, so base vs FT differ only by the adapter (and decoding is identical/greedy).
        ctx = model.disable_adapter() if (has_adapter and not use_adapter) else nullcontext()
        text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       tokenize=False, add_generation_prompt=True)
        enc = tok(text, return_tensors="pt").to(model.device)
        with ctx:
            out = model.generate(**enc, do_sample=False, max_new_tokens=args.max_new_tokens,
                                 pad_token_id=tok.eos_token_id)
        return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)

    arms = ["base"] + (["ft"] if has_adapter else [])
    records = []
    for t in tasks:
        rec = {"split": t["split"], "optimal_len": t["optimal_len"]}
        for arm in arms:
            txt = gen(t["prompt"], use_adapter=(arm == "ft"))
            v = bw.validate_plan(t["init"], t["goal"], parse_plan(txt))
            rec[arm] = bool(v["valid"])
            rec[f"{arm}_reached"] = bool(v["reached"])
        records.append(rec)

    result = {"base": args.base, "adapter": args.adapter, "n": len(records),
              "load_4bit": args.load_4bit,
              "summary": {arm: _strata(records, arm) for arm in arms}}
    if has_adapter:
        b = sum(1 for r in records if r["ft"] and not r["base"])
        c = sum(1 for r in records if r["base"] and not r["ft"])
        result["mcnemar_ft_vs_base"] = {"ft_only": b, "base_only": c, "p": round(mcnemar_p(b, c), 6)}
    result["per_instance"] = records
    Path(args.out).write_text(json.dumps(result, indent=2))

    for arm in arms:
        s = result["summary"][arm]
        print(f"  {arm:4} overall {s['overall']['valid_rate']:.3f}  "
              f"id={s['by_split']['test_id']['valid_rate']:.3f}  "
              f"long={s['by_split']['test_long']['valid_rate']:.3f}", flush=True)
    if has_adapter:
        print(f"  McNemar ft_vs_base: {result['mcnemar_ft_vs_base']}", flush=True)
    print(f"[e78-eval] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
