"""E80 proxy harness: measure the CHEAP, training-free predictors of whether world-time compute
will pay off, so we can forecast a domain instead of guessing it.

For each world we record, using only BASE-model inference (no fine-tuning):
  - k-shot in-context accuracy at k in {ks} -> the in-context SLOPE sigma (identifiability x
    realizability: does showing more demos help at all?),
  - headroom = 1 - (accuracy at the largest k),
  - depth = median output length (a proxy for required reasoning steps).
Joined offline (e80_law.py) with the MEASURED per-world TTT lift (heavy - zero-shot from the
e80_*_ttt runs), these test the law:  lift ~ sigma * headroom * depth-discount.

Consumes the uniform worlds JSONL ({world,input,output}); ARC grids are pre-converted to the
same format. Partial results upload after every world.
  python3 e80_proxy.py --worlds listfn_worlds.jsonl --domain listfn --bucket gs://.../lf-proxy
"""

import argparse
import json
import random
import subprocess
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import e80_text_world as T

HERE = Path(__file__).resolve().parent
BASE = "Qwen/Qwen2.5-7B-Instruct"


@torch.no_grad()
def gen_batch(tok, model, prompts, max_new_tokens, bs=8):
    outs = []
    for i in range(0, len(prompts), bs):
        chunk = prompts[i:i + bs]
        texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                         add_generation_prompt=True) for p in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=4096).to(model.device)
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        g = out[:, enc["input_ids"].shape[1]:]
        outs += [tok.decode(x, skip_special_tokens=True) for x in g]
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", required=True)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--instruction", default="Infer the hidden rule from the examples and "
                    "produce the output for the final input.")
    ap.add_argument("--bucket", default="")
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--ks", default="0,1,2,4,8")
    ap.add_argument("--n_pool", type=int, default=16)
    ap.add_argument("--n_eval", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    args = ap.parse_args()
    ks = [int(k) for k in args.ks.split(",")]

    worlds = T.load_worlds(args.worlds, min_examples=args.n_pool + args.n_eval)
    names = sorted(worlds)
    random.Random(80).shuffle(names)
    names = names[:args.n]
    print(f"[proxy/{args.domain}] {len(names)} worlds, k-shot {ks}", flush=True)

    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16,
                                                 device_map={"": 0}).eval()

    res = {"experiment": f"proxy-{args.domain}", "base": BASE, "ks": ks, "per_world": {}}

    def upload():
        out = HERE / "results" / f"e80_proxy_{args.domain}.json"
        out.write_text(json.dumps(res, indent=2))
        if args.bucket:
            subprocess.run(["gcloud", "storage", "cp", str(out),
                            f"{args.bucket}/e80_proxy_{args.domain}.json"], check=False)

    for nm in names:
        pool, qeval = T.split_world(worlds[nm], args.n_pool, args.n_eval,
                                    random.Random(hash(nm) % 2**32))
        if not qeval:
            continue
        depth = sorted(len(str(q["output"])) for q in qeval)[len(qeval) // 2]
        kacc = {}
        for k in ks:
            cases = T.eval_cases(args.instruction, pool, qeval, k, random.Random(7))
            if not cases:
                continue
            preds = gen_batch(tok, model, [c["prompt"] for c in cases], args.max_new_tokens)
            kacc[str(k)] = sum(T.match(p, c["answer"]) for p, c in zip(preds, cases)) / len(cases)
        res["per_world"][nm] = {"kacc": kacc, "depth": depth}
        upload()
    print(f"[proxy/{args.domain}] done ({len(res['per_world'])} worlds)", flush=True)


if __name__ == "__main__":
    main()
