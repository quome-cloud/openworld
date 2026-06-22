"""E80 likelihood proxy: a stronger, training-free predictor of world-time-compute payoff.

The exact-match in-context slope (e80_proxy.py) goes blind where base accuracy is already high
(it saturates). This proxy instead measures the model's TEACHER-FORCED log-probability of the
CORRECT output given k demonstrations -- which keeps discriminating past the exact-match ceiling
and needs NO autoregressive generation (one forward pass per query), so it is fast on every
domain including ARC grids.

Per world we record mean answer log-prob at k in {ks}; the law uses:
  ll_slope = ll(k_max) - ll(k_0)  (identifiability: do demos raise the answer's likelihood?)
  ll_base  = ll(k_0)              (zero-shot familiarity)
plus depth (output length). Joined offline with the measured TTT lift in e80_law.py.

  python3 e80_proxy_ll.py --worlds listfn_worlds.jsonl --domain listfn --bucket gs://.../ll
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
def answer_logprob(tok, model, prompt, answer, max_len=4096):
    """Mean per-token log-prob of `answer` given `prompt` (teacher-forced, one forward pass)."""
    full = tok.apply_chat_template([{"role": "user", "content": prompt},
                                    {"role": "assistant", "content": answer}], tokenize=False)
    pre = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                  add_generation_prompt=True)
    ids = tok(full, return_tensors="pt", truncation=True, max_length=max_len).input_ids
    n_pre = len(tok(pre, truncation=True, max_length=max_len).input_ids)
    if ids.shape[1] <= n_pre + 1:
        return None
    ids = ids.to(model.device)
    logits = model(ids).logits[0]                       # [L, V]
    logp = torch.log_softmax(logits[:-1], dim=-1)       # predict token t from t-1
    tgt = ids[0, 1:]
    ans = range(max(0, n_pre - 1), tgt.shape[0])        # answer-token positions
    vals = [logp[i, tgt[i]].item() for i in ans]
    return sum(vals) / len(vals) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", required=True)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--instruction", default="Infer the hidden rule from the examples and "
                    "produce the output for the final input.")
    ap.add_argument("--bucket", default="")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--ks", default="0,1,2,4,8")
    ap.add_argument("--n_pool", type=int, default=16)
    ap.add_argument("--n_eval", type=int, default=8)
    args = ap.parse_args()
    ks = [int(k) for k in args.ks.split(",")]

    worlds = T.load_worlds(args.worlds, min_examples=args.n_pool + args.n_eval)
    names = sorted(worlds)
    random.Random(80).shuffle(names)
    names = names[:args.n]
    print(f"[ll/{args.domain}] {len(names)} worlds, ks {ks}", flush=True)

    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16,
                                                 device_map={"": 0}).eval()

    res = {"experiment": f"ll-{args.domain}", "base": BASE, "ks": ks, "per_world": {}}

    def upload():
        out = HERE / "results" / f"e80_ll_{args.domain}.json"
        out.write_text(json.dumps(res, indent=2))
        if args.bucket:
            subprocess.run(["gcloud", "storage", "cp", str(out),
                            f"{args.bucket}/e80_ll_{args.domain}.json"], check=False)

    for nm in names:
        pool, qeval = T.split_world(worlds[nm], args.n_pool, args.n_eval,
                                    random.Random(hash(nm) % 2**32))
        if not qeval:
            continue
        depth = sorted(len(str(q["output"])) for q in qeval)[len(qeval) // 2]
        kll = {}
        for k in ks:
            cases = T.eval_cases(args.instruction, pool, qeval, k, random.Random(7))
            lps = [lp for c in cases
                   if (lp := answer_logprob(tok, model, c["prompt"], c["answer"])) is not None]
            if lps:
                kll[str(k)] = sum(lps) / len(lps)
        res["per_world"][nm] = {"kll": kll, "depth": depth}
        upload()
    print(f"[ll/{args.domain}] done ({len(res['per_world'])} worlds)", flush=True)


if __name__ == "__main__":
    main()
