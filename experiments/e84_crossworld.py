"""E84 cross-world transfer (the NOVEL axis, on a REAL coherent family).

Unlike per-world test-time training (e80_text_ttt, which fits a fresh adapter on EACH held-out
world), this trains ONE adapter on a DISJOINT set of held-IN worlds, freezes it, and evaluates
held-OUT worlds zero-shot. The adapter never sees the eval worlds -- so any lift is genuine
cross-world / family transfer (Route 3), the axis E83 (CLRS, skill-disjoint) could not show.

Arms (all evaluated on the SAME held-out worlds, identical zero-shot eval protocol):
  base        -- no adapter (the floor),
  crossworld  -- adapter trained on held-in worlds with EXACT labels,
  corrupt     -- adapter trained on held-in worlds with CORRUPTED labels (control: if the
                 transfer needs exact labels, this collapses back to base).

A positive (crossworld - base) gap whose CI excludes zero, with corrupt ~= base, is cross-world
transfer on a real, human-authored domain -- the result that distinguishes world-time compute
from "self-training on our own worlds."

  python3 e84_crossworld.py --worlds data/listfn_worlds.jsonl --domain listfn \
      --instruction "Infer the hidden rule from the examples ..." --bucket gs://... --seed 0
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


def reset_adapter(model):
    if "default" in getattr(model, "peft_config", {}):
        model.delete_adapter("default")
    model.add_adapter("default", LORA)
    model.set_adapter("default")


def fit(model, tok, rows, steps, seed, lr=1e-4, bs=2):
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
    rng = random.Random(seed)
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
def predict(model, tok, prompt, mnt=64):
    model.config.use_cache = True
    model.eval()
    text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                   add_generation_prompt=True)
    enc = tok(text, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    gen = model.generate(**enc, max_new_tokens=mnt, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(gen[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)


def world_acc(model, tok, cases, mnt=64):
    if not cases:
        return None
    return sum(T.match(predict(model, tok, c["prompt"], mnt), c["answer"]) for c in cases) / len(cases)


def build_train_rows(worlds, names, instruction, rows_per_world, n_ctx, rng, corrupt=False):
    pool_rows = []
    for nm in names:
        pool_rows += T.ttt_rows(instruction, worlds[nm], n_ctx, rows_per_world, rng, corrupt=corrupt)
    rng.shuffle(pool_rows)
    return pool_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", required=True)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--instruction", required=True)
    ap.add_argument("--bucket", default="")
    ap.add_argument("--n_train_worlds", type=int, default=128)
    ap.add_argument("--n_eval_worlds", type=int, default=60)
    ap.add_argument("--rows_per_world", type=int, default=20)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--n_pool", type=int, default=16)
    ap.add_argument("--n_eval", type=int, default=8)
    ap.add_argument("--n_ctx", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    worlds = T.load_worlds(args.worlds, min_examples=args.n_pool + args.n_eval)
    names = sorted(worlds)
    random.Random(80 + args.seed).shuffle(names)
    train_names = names[:args.n_train_worlds]
    eval_names = names[args.n_train_worlds:args.n_train_worlds + args.n_eval_worlds]
    assert not (set(train_names) & set(eval_names)), "train/eval worlds must be disjoint"
    print(f"[crossworld/{args.domain}] seed={args.seed} train={len(train_names)} "
          f"eval={len(eval_names)} (disjoint)", flush=True)

    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    base = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map={"": 0})
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    model = get_peft_model(base, LORA)

    splits = {nm: T.split_world(worlds[nm], args.n_pool, args.n_eval, random.Random(hash(nm) % 2**32))
              for nm in eval_names}

    def eval_arm():
        accs = {}
        for nm in eval_names:
            pool, qeval = splits[nm]
            cases = T.eval_cases(args.instruction, pool, qeval, args.n_ctx, random.Random(7))
            accs[nm] = world_acc(model, tok, cases)
        return accs

    res = {"experiment": f"crossworld-{args.domain}", "base": BASE, "seed": args.seed,
           "n_train_worlds": len(train_names), "n_eval_worlds": len(eval_names),
           "steps": args.steps, "rows_per_world": args.rows_per_world,
           "per_world": {}, "arms": {}}

    def upload():
        for arm, accs in res["per_world"].items():
            d = [a for a in accs.values() if a is not None]
            res["arms"][arm] = {"acc": round(sum(d) / len(d), 4) if d else None, "n_done": len(d)}
        out = HERE / "results" / f"e84_crossworld_{args.domain}_seed{args.seed}.json"
        out.write_text(json.dumps(res, indent=2))
        if args.bucket:
            subprocess.run(["gcloud", "storage", "cp", str(out),
                            f"{args.bucket}/e84_crossworld_{args.domain}_seed{args.seed}.json"], check=False)

    # ARM 1: base (no adapter) -- the floor
    with model.disable_adapter():
        res["per_world"]["base"] = eval_arm()
        upload()
    print(f"[base] {res['arms']['base']}", flush=True)

    # ARM 2: crossworld -- adapter trained on held-in worlds with EXACT labels
    rows = build_train_rows(worlds, train_names, args.instruction, args.rows_per_world,
                            args.n_ctx, random.Random(1000 + args.seed), corrupt=False)
    print(f"[crossworld] training on {len(rows)} pooled rows from {len(train_names)} held-in worlds",
          flush=True)
    reset_adapter(model)
    fit(model, tok, rows, args.steps, seed=args.seed)
    res["per_world"]["crossworld"] = eval_arm()
    upload()
    print(f"[crossworld] {res['arms']['crossworld']}", flush=True)

    # ARM 3: corrupt -- adapter trained on held-in worlds with CORRUPTED labels (control)
    rows_c = build_train_rows(worlds, train_names, args.instruction, args.rows_per_world,
                              args.n_ctx, random.Random(2000 + args.seed), corrupt=True)
    reset_adapter(model)
    fit(model, tok, rows_c, args.steps, seed=args.seed)
    res["per_world"]["corrupt"] = eval_arm()
    upload()
    print(f"[corrupt] {res['arms']['corrupt']}", flush=True)

    print(f"[crossworld/{args.domain}] done\n" + json.dumps(res["arms"], indent=2), flush=True)


if __name__ == "__main__":
    main()
