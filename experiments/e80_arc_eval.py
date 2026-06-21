"""E80-ARC eval: generate the full output grid and score by EXACT grid match.

Reads a jsonl of {"prompt","answer"[, "tid"]} (answer = the target grid text), generates
greedily (left-padded batches), parses the model's grid, and counts a hit only if every cell
matches. Writes accuracy + per-task hits. No partial credit -- ARC is exact.

  python3 e80_arc_eval.py --base Qwen/Qwen2.5-7B-Instruct --test eval.jsonl --out e.json \
      [--adapter adp] [--max_new_tokens 1024] [--eval_batch 8]
"""

import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from e80_arc import grids_equal, parse_grid


def load_model(base, adapter=None, four_bit=True):
    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    kw = dict(device_map={"": 0})
    if four_bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    else:
        kw["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(base, **kw)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return tok, model


@torch.no_grad()
def generate(tok, model, prompts, max_new_tokens=1024, batch=8):
    outs = []
    for i in range(0, len(prompts), batch):
        chunk = prompts[i:i + batch]
        texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                         add_generation_prompt=True) for p in chunk]
        enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        g = gen[:, enc["input_ids"].shape[1]:]
        outs += [tok.decode(x, skip_special_tokens=True) for x in g]
    return outs


def score(preds, rows):
    hits, per = 0, []
    for r, p in zip(rows, preds):
        ok = grids_equal(parse_grid(p), parse_grid(r["answer"]))
        hits += int(ok)
        per.append({"tid": r.get("tid"), "hit": int(ok)})
    return hits / max(1, len(rows)), per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--out", default="arc_eval.json")
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--eval_batch", type=int, default=8)
    ap.add_argument("--no_4bit", action="store_true")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.test) if l.strip()]
    tok, model = load_model(args.base, args.adapter, four_bit=not args.no_4bit)
    preds = generate(tok, model, [r["prompt"] for r in rows], args.max_new_tokens, args.eval_batch)
    acc, per = score(preds, rows)
    json.dump({"base": args.base, "adapter": args.adapter, "n": len(rows),
               "accuracy": round(acc, 4), "per_task": per}, open(args.out, "w"), indent=2)
    print(f"[arc-eval] {args.out}: exact-match {acc:.4f} over {len(rows)} tasks", flush=True)


if __name__ == "__main__":
    main()
