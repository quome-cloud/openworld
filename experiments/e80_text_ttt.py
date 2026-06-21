"""E80 text test-time training: world-time compute per world on free-text I/O benchmarks
(List Functions, CLRS-Text). For each held-out world we fit a FRESH LoRA on that world's
demonstrations (its eval queries strictly held out) and predict them by exact match. Arms:
zero-shot (in-context demos, no training) -> light TTT -> heavy TTT (more world-time compute),
plus a corrupted-demonstration ablation (labels randomised). The 7B base is loaded ONCE
(4-bit); a named LoRA is reset per world. Partial results upload to GCS after every world.

  python3 e80_text_ttt.py --worlds listfn_worlds.jsonl --domain listfn \
      --instruction "Infer the hidden function ..." --bucket gs://openworld-bench/lf-ttt
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
LORA = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
LEVELS = [("light", 40, 40), ("heavy", 120, 100)]   # (name, n_rows, train_steps)
ABL = ("corrupt", 120, 100)


def reset_adapter(model):
    if "default" in getattr(model, "peft_config", {}):
        model.delete_adapter("default")
    model.add_adapter("default", LORA)
    model.set_adapter("default")


def ttt_fit(model, tok, rows, steps, lr=1e-4, bs=2):
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


@torch.no_grad()
def predict(model, tok, prompt, max_new_tokens):
    model.config.use_cache = True
    model.eval()
    text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                   add_generation_prompt=True)
    enc = tok(text, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(gen[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)


def world_acc(model, tok, cases, mnt):
    if not cases:
        return None
    hits = sum(T.match(predict(model, tok, c["prompt"], mnt), c["answer"]) for c in cases)
    return hits / len(cases)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", required=True)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--instruction", required=True)
    ap.add_argument("--bucket", default="")
    ap.add_argument("--n", type=int, default=60, help="max held-out worlds")
    ap.add_argument("--n_pool", type=int, default=16)
    ap.add_argument("--n_eval", type=int, default=8)
    ap.add_argument("--n_ctx", type=int, default=3)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    args = ap.parse_args()

    worlds = T.load_worlds(args.worlds, min_examples=args.n_pool + args.n_eval)
    names = sorted(worlds)
    random.Random(80).shuffle(names)
    names = names[:args.n]
    print(f"[text-ttt/{args.domain}] {len(names)} held-out worlds", flush=True)

    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    base = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map={"": 0})
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    model = get_peft_model(base, LORA)

    res = {"experiment": f"text-ttt-{args.domain}", "base": BASE, "n_worlds": len(names),
           "levels": [dict(name=n, n_rows=r, steps=s) for n, r, s in LEVELS + [ABL]],
           "arms": {}, "per_world": {}}

    def upload():
        for arm, accs in res["per_world"].items():
            done = [a for a in accs.values() if a is not None]
            res["arms"][arm] = {"acc": round(sum(done) / len(done), 4) if done else None,
                                "n_done": len(done)}
        out = HERE / "results" / f"e80_text_{args.domain}.json"
        out.write_text(json.dumps(res, indent=2))
        if args.bucket:
            subprocess.run(["gcloud", "storage", "cp", str(out),
                            f"{args.bucket}/e80_text_{args.domain}.json"], check=False)

    # per-world splits, fixed across arms
    splits = {}
    for nm in names:
        splits[nm] = T.split_world(worlds[nm], args.n_pool, args.n_eval, random.Random(hash(nm) % 2**32))

    # zero-shot (in-context demos, adapters disabled)
    res["per_world"]["zeroshot"] = {}
    with model.disable_adapter():
        for nm in names:
            pool, qeval = splits[nm]
            cases = T.eval_cases(args.instruction, pool, qeval, args.n_ctx, random.Random(7))
            res["per_world"]["zeroshot"][nm] = world_acc(model, tok, cases, args.max_new_tokens)
            upload()
    print(f"[zeroshot] {res['arms']['zeroshot']}", flush=True)

    for name, n_rows, steps in LEVELS + [ABL]:
        corrupt = (name == ABL[0])
        res["per_world"][name] = {}
        for nm in names:
            try:
                pool, qeval = splits[nm]
                rows = T.ttt_rows(args.instruction, pool, args.n_ctx, n_rows,
                                  random.Random(hash(nm) % 2**32), corrupt=corrupt)
                reset_adapter(model)
                ttt_fit(model, tok, rows, steps)
                cases = T.eval_cases(args.instruction, pool, qeval, args.n_ctx, random.Random(7))
                acc = world_acc(model, tok, cases, args.max_new_tokens)
            except Exception as e:  # noqa: BLE001
                print(f"[{name} {nm}] FAILED {e}", flush=True)
                acc = None
            res["per_world"][name][nm] = acc
            upload()
        print(f"[{name}] {res['arms'][name]}", flush=True)

    print(f"[text-ttt/{args.domain}] done\n" + json.dumps(res["arms"], indent=2), flush=True)


if __name__ == "__main__":
    main()
