"""E80 world-count: the optimal number of simulated worlds per task.

The world-time-compute curve saturates, so the pragmatic question -- how many world simulations
should a reader build? -- has an answer: the knee of acc vs the number of simulated worlds.
For a sample of held-out worlds we sweep the per-world simulation count (augmented training
instances), fixing the optimizer budget, and record held-out exact-match. Offline we fit
  acc(n) = base + (C-base)(1 - e^{-n/n*})
and report N90 = n*-ln(10), the worlds needed for 90% of the asymptotic gain.

Reuses e80_text_ttt's in-process per-world TTT (7B loaded once). Partial results upload per step.
  python3 e80_worldcount.py --worlds listfn_worlds.jsonl --domain listfn --bucket gs://.../wc
"""

import argparse
import json
import random
import subprocess
from pathlib import Path

import e80_text_world as T
import e80_text_ttt as TT

HERE = Path(__file__).resolve().parent
SWEEP = [2, 4, 8, 16, 32, 64, 128]   # number of simulated worlds (augmented instances) per task
STEPS = 80                            # fixed optimizer budget, so we isolate world COUNT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", required=True)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--instruction", required=True)
    ap.add_argument("--bucket", default="")
    ap.add_argument("--n", type=int, default=25, help="held-out worlds sampled")
    ap.add_argument("--n_pool", type=int, default=16)
    ap.add_argument("--n_eval", type=int, default=8)
    ap.add_argument("--n_ctx", type=int, default=3)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    args = ap.parse_args()

    worlds = T.load_worlds(args.worlds, min_examples=args.n_pool + args.n_eval)
    names = sorted(worlds)
    random.Random(80).shuffle(names)
    names = names[:args.n]
    print(f"[worldcount/{args.domain}] {len(names)} worlds, sweep {SWEEP}", flush=True)

    import torch
    from peft import get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained(TT.BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    base = AutoModelForCausalLM.from_pretrained(TT.BASE, quantization_config=bnb, device_map={"": 0})
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    model = get_peft_model(base, TT.LORA)

    splits = {nm: T.split_world(worlds[nm], args.n_pool, args.n_eval,
                                random.Random(hash(nm) % 2**32)) for nm in names}
    res = {"experiment": f"worldcount-{args.domain}", "base": TT.BASE, "sweep": SWEEP,
           "steps": STEPS, "per_n": {}}

    def upload():
        out = HERE / "results" / f"e80_worldcount_{args.domain}.json"
        out.write_text(json.dumps(res, indent=2))
        if args.bucket:
            subprocess.run(["gcloud", "storage", "cp", str(out),
                            f"{args.bucket}/e80_worldcount_{args.domain}.json"], check=False)

    # zero-shot base (in-context, no fit)
    accs0 = []
    with model.disable_adapter():
        for nm in names:
            pool, qeval = splits[nm]
            cases = T.eval_cases(args.instruction, pool, qeval, args.n_ctx, random.Random(7))
            accs0.append(TT.world_acc(model, tok, cases, args.max_new_tokens))
    res["per_n"]["0"] = sum(a for a in accs0 if a is not None) / len(accs0)
    upload()
    print(f"[worldcount/{args.domain}] base {res['per_n']['0']:.3f}", flush=True)

    for n_rows in SWEEP:
        accs = []
        for nm in names:
            try:
                pool, qeval = splits[nm]
                rows = T.ttt_rows(args.instruction, pool, args.n_ctx, n_rows,
                                  random.Random(hash(nm) % 2**32))
                TT.reset_adapter(model)
                TT.ttt_fit(model, tok, rows, STEPS)
                cases = T.eval_cases(args.instruction, pool, qeval, args.n_ctx, random.Random(7))
                accs.append(TT.world_acc(model, tok, cases, args.max_new_tokens))
            except Exception as e:  # noqa: BLE001
                print(f"[n={n_rows} {nm}] FAILED {e}", flush=True)
        good = [a for a in accs if a is not None]
        res["per_n"][str(n_rows)] = sum(good) / len(good) if good else None
        upload()
        print(f"[worldcount/{args.domain}] n={n_rows}: {res['per_n'][str(n_rows)]}", flush=True)

    print(f"[worldcount/{args.domain}] done\n{json.dumps(res['per_n'], indent=2)}", flush=True)


if __name__ == "__main__":
    main()
