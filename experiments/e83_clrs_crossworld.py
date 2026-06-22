"""E83: real-data CROSS-WORLD transfer on CLRS-Text (the novel-mode result the review asked for).

E80 showed PER-WORLD test-time training on CLRS (fit a fresh adapter on each algorithm's own
demos). This tests the harder, load-bearing claim: world-time compute as CROSS-WORLD transfer.
We split the 29 CLRS algorithms (worlds) into a TRAIN pool and a HELD-OUT set, fine-tune ONE
LoRA on the pooled train algorithms' (demos->output) rows, and evaluate on the HELD-OUT
algorithms whose examples were never trained on. Arms per held-out algorithm:
  - zeroshot : base model, in-context demos only (no training)
  - crossworld: base + the cross-world LoRA (trained on OTHER algorithms)   [the claim]
  - corrupt  : base + a LoRA trained on the same pool with WRONG labels      [exactness control]
The claim is crossworld > zeroshot on held-out algorithms (transfer of "infer the algorithm
from demos"), with corrupt < crossworld. Bootstrap CIs over held-out worlds. Partial results
upload after each arm.

  python3 e83_clrs_crossworld.py --worlds clrs_worlds.jsonl --bucket gs://openworld-bench/e83-clrs
"""

import argparse
import json
import random
import subprocess
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import e80_text_world as T

HERE = Path(__file__).resolve().parent
BASE = "Qwen/Qwen2.5-7B-Instruct"
MAXLEN = 2048
INSTR = ("Infer the hidden algorithm from the examples and produce the output for the final "
         "input. Reply with only the output.")
LORA = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])


def reset_adapter(model):
    """Re-initialize a fresh LoRA adapter in place (no re-wrapping of the base -> no OOM)."""
    if "default" in getattr(model, "peft_config", {}):
        model.delete_adapter("default")
    model.add_adapter("default", LORA)
    model.set_adapter("default")


def build_train_rows(worlds, train_names, n_ctx, rows_per_world, rng, corrupt=False):
    """Pool (demos->output) rows across the TRAIN algorithms (held-out algos excluded)."""
    rows = []
    for nm in train_names:
        pool, _ = T.split_world(worlds[nm], n_pool=16, n_eval=8, rng=rng)
        rows += T.ttt_rows(INSTR, pool, n_ctx=n_ctx, n_rows=rows_per_world, rng=rng,
                           corrupt=corrupt)
    rng.shuffle(rows)
    return rows


def sft(model, tok, rows, steps, lr=1e-4, bs=2):
    model.config.use_cache = False
    model.train()
    seqs = []
    for r in rows:
        text = tok.apply_chat_template(
            [{"role": "user", "content": r["prompt"]},
             {"role": "assistant", "content": r["completion"]}], tokenize=False)
        ids = tok(text, truncation=True, max_length=MAXLEN)["input_ids"]
        if len(ids) >= 8:
            seqs.append(ids)
    if not seqs:
        return
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    rng = random.Random(0)
    done = 0
    while done < steps:
        rng.shuffle(seqs)
        for i in range(0, len(seqs), bs):
            batch = seqs[i:i + bs]
            m = max(len(s) for s in batch)
            inp = torch.full((len(batch), m), tok.pad_token_id, dtype=torch.long)
            att = torch.zeros((len(batch), m), dtype=torch.long)
            for j, s in enumerate(batch):
                inp[j, :len(s)] = torch.tensor(s)
                att[j, :len(s)] = 1
            inp, att = inp.to(model.device), att.to(model.device)
            lab = inp.clone()
            lab[att == 0] = -100
            model(input_ids=inp, attention_mask=att, labels=lab).loss.backward()
            opt.step()
            opt.zero_grad()
            done += 1
            if done >= steps:
                break
    model.config.use_cache = True
    model.eval()
    try:
        model.gradient_checkpointing_disable()
    except Exception:
        pass


@torch.no_grad()
def predict(model, tok, prompt, mnt=64):
    model.config.use_cache = True
    model.eval()
    text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                   add_generation_prompt=True)
    enc = tok(text, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    gen = model.generate(**enc, max_new_tokens=mnt, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(gen[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)


def world_acc(model, tok, cases):
    if not cases:
        return None
    return sum(T.match(predict(model, tok, c["prompt"]), c["answer"]) for c in cases) / len(cases)


def boot_ci(vals, n=5000, seed=0):
    vals = [v for v in vals if v is not None]
    if not vals:
        return (None, None)
    rng = random.Random(seed)
    k = len(vals)
    means = sorted(sum(vals[rng.randrange(k)] for _ in range(k)) / k for _ in range(n))
    return round(means[int(0.025 * n)], 4), round(means[int(0.975 * n)], 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", required=True)
    ap.add_argument("--bucket", default="")
    ap.add_argument("--n_holdout", type=int, default=8)
    ap.add_argument("--rows_per_world", type=int, default=30)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--n_ctx", type=int, default=3)
    ap.add_argument("--seed", type=int, default=83)
    args = ap.parse_args()

    worlds = T.load_worlds(args.worlds, min_examples=24)
    names = sorted(worlds)
    rng = random.Random(args.seed)
    shuffled = names[:]
    rng.shuffle(shuffled)
    holdout = sorted(shuffled[:args.n_holdout])
    train_names = sorted(shuffled[args.n_holdout:])
    print(f"[e83-clrs] {len(names)} algorithms: {len(train_names)} train / {len(holdout)} held-out",
          flush=True)
    print(f"  held-out: {holdout}", flush=True)

    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)

    # fixed eval cases per held-out algorithm (same across arms)
    ev = {}
    for nm in holdout:
        r = random.Random(hash(nm) % 2**32)
        pool, qeval = T.split_world(worlds[nm], n_pool=16, n_eval=8, rng=r)
        ev[nm] = T.eval_cases(INSTR, pool, qeval, n_ctx=args.n_ctx, rng=r)

    res = {"experiment": "e83-clrs-crossworld", "base": BASE,
           "n_train_algorithms": len(train_names), "n_holdout_algorithms": len(holdout),
           "holdout": holdout, "train_algorithms": train_names,
           "rows_per_world": args.rows_per_world, "steps": args.steps,
           "arms": {}, "per_world": {}, "ci": {}}

    def upload():
        out = HERE / "results" / "e83_clrs_crossworld.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=2))
        if args.bucket:
            subprocess.run(["gcloud", "storage", "cp", str(out),
                            f"{args.bucket}/e83_clrs_crossworld.json"], check=False)

    def score_arm(model, name):
        pw = {nm: world_acc(model, tok, ev[nm]) for nm in holdout}
        vals = [v for v in pw.values() if v is not None]
        res["per_world"][name] = {k: (round(v, 4) if v is not None else None) for k, v in pw.items()}
        res["arms"][name] = round(sum(vals) / len(vals), 4) if vals else None
        res["ci"][name] = boot_ci(vals)
        print(f"[{name}] acc={res['arms'][name]} ci={res['ci'][name]}", flush=True)
        upload()

    # ---- zero-shot (base, no adapter) ----
    base = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map={"": 0})
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    model = get_peft_model(base, LORA)
    with model.disable_adapter():
        score_arm(model, "zeroshot")

    # ---- cross-world: fresh LoRA trained on the pooled TRAIN algorithms, eval held-out ----
    train_rows = build_train_rows(worlds, train_names, args.n_ctx, args.rows_per_world,
                                  random.Random(args.seed), corrupt=False)
    print(f"[e83] {len(train_rows)} cross-world train rows", flush=True)
    reset_adapter(model)
    sft(model, tok, train_rows, steps=args.steps)
    score_arm(model, "crossworld")
    torch.cuda.empty_cache()

    # ---- corrupt control: fresh adapter on WRONG labels (reset in place, do NOT re-wrap) ----
    corrupt_rows = build_train_rows(worlds, train_names, args.n_ctx, args.rows_per_world,
                                    random.Random(args.seed), corrupt=True)
    reset_adapter(model)
    sft(model, tok, corrupt_rows, steps=args.steps)
    score_arm(model, "corrupt")
    torch.cuda.empty_cache()

    print(f"[e83] done\n{json.dumps(res['arms'], indent=2)}", flush=True)


if __name__ == "__main__":
    main()
