"""Generic batched eval for E80 short-answer domains (integers, item numbers, TRUE/FALSE,
class strings). Left-padded greedy generation; a prediction counts if the answer matches the
last number / a token / the normalized output. Drop-in for e74_eval across domains.

  python3 e80_eval.py --base <model> --test test.jsonl --out eval.json [--adapter adp] [--eval_batch 256]
"""

import argparse
import json
import re
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def matches(gen, answer):
    g = (gen or "").strip().lower()
    a = str(answer).strip().lower()
    if not a:
        return False
    if a == g or a in g.split():
        return True
    if a in ("true", "false") and re.search(rf"\b{a}\b", g):
        return True
    if a.lstrip("-").isdigit():                       # numeric: compare the last integer emitted
        nums = re.findall(r"-?\d+", g)
        return bool(nums) and nums[-1] == a
    return a in g                                     # short class string fallback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--out", default="eval.json")
    ap.add_argument("--eval_batch", type=int, default=256)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.test) if l.strip()]
    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16,
                                                 device_map={"": 0})
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    @torch.no_grad()
    def gen_batch(prompts):
        texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                         add_generation_prompt=True) for p in prompts]
        enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
        out = model.generate(**enc, max_new_tokens=12, do_sample=False, pad_token_id=tok.eos_token_id)
        g = out[:, enc["input_ids"].shape[1]:]
        return [tok.decode(x, skip_special_tokens=True) for x in g]

    preds = []
    for i in range(0, len(rows), args.eval_batch):
        preds.extend(gen_batch([r["prompt"] for r in rows[i:i + args.eval_batch]]))

    correct = 0
    by = defaultdict(lambda: [0, 0])
    for r, p in zip(rows, preds):
        ok = int(matches(p, r["answer"]))
        correct += ok
        sp = r.get("specialty", "all")
        by[sp][0] += ok
        by[sp][1] += 1
    out = {"base": args.base, "adapter": args.adapter, "n_cases": len(rows),
           "accuracy": round(correct / len(rows), 4),
           "per_world_accuracy": {k: round(v[0] / v[1], 4) for k, v in sorted(by.items())}}
    open(args.out, "w").write(json.dumps(out, indent=2))
    print(f"[e80-eval] {args.out}: accuracy {out['accuracy']} over {len(rows)} cases", flush=True)


if __name__ == "__main__":
    main()
