"""E80-ARC test-time training (the headline): world-time compute spent learning EACH real ARC
world at inference. For every held-out evaluation task we fit a FRESH LoRA on that task's own
augmented demonstrations (leave-one-out over the demos x dihedral x color), then predict its
held-out test grid by exact match. The base model never saw the task.

This is the purest statement of the thesis -- compute spent simulating a world yields
generalization within it -- and the proven-effective ARC recipe. We report:
  - zero-shot base (no world-time compute),
  - TTT at two compute levels (light/heavy augmentation = more world-time compute),
  - a corrupted-demo ablation (labels randomized): if TTT only helps with VERIFIED labels,
    the corrupt arm collapses back to ~zero-shot -- the mechanism is learning the real rule.

The 7B base is loaded ONCE (4-bit); a named LoRA is reset per task (no reload). Partial
results upload to GCS after every task.
  python3 e80_arc_ttt.py --data /root/ARC-AGI/data --bucket gs://openworld-bench/arc-ttt
"""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig)

import e80_arc as A

HERE = Path(__file__).resolve().parent
BASE = "Qwen/Qwen2.5-7B-Instruct"
N_TTT = 40            # held-out eval tasks given test-time training
MAXLEN = 1536
LEVELS = [("light", 4, 40), ("heavy", 16, 100)]   # (name, n_aug, train_steps)
ABL = ("corrupt", 16, 100)
SEED = 0  # set from --seed; varies LoRA init, shuffle, augmentation
LORA = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])


def reset_adapter(model):
    if "default" in getattr(model, "peft_config", {}):
        model.delete_adapter("default")
    model.add_adapter("default", LORA)
    model.set_adapter("default")


def ttt_fit(model, tok, rows, steps, lr=1e-4, bs=2):
    """Manual LoRA SGD on full-text (prompt+completion) sequences for `steps` optimizer steps."""
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
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    rng = np.random.default_rng(SEED)
    done = 0
    while done < steps:
        rng.shuffle(seqs)
        for i in range(0, len(seqs), bs):
            batch = seqs[i:i + bs]
            m = max(len(s) for s in batch)
            pad = tok.pad_token_id
            inp = torch.full((len(batch), m), pad, dtype=torch.long)
            att = torch.zeros((len(batch), m), dtype=torch.long)
            for j, s in enumerate(batch):
                inp[j, :len(s)] = torch.tensor(s)
                att[j, :len(s)] = 1
            inp, att = inp.to(model.device), att.to(model.device)
            lab = inp.clone()
            lab[att == 0] = -100
            loss = model(input_ids=inp, attention_mask=att, labels=lab).loss
            loss.backward()
            opt.step()
            opt.zero_grad()
            done += 1
            if done >= steps:
                break


@torch.no_grad()
def predict(model, tok, prompt, max_new_tokens=1024):
    model.config.use_cache = True
    model.eval()
    text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                   add_generation_prompt=True)
    enc = tok(text, return_tensors="pt").to(model.device)
    gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(gen[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--bucket", default="")
    ap.add_argument("--n", type=int, default=N_TTT)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    global SEED
    SEED = args.seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    print(f"[arc-ttt] seed={SEED}", flush=True)

    ev = A.load_tasks(str(Path(args.data) / "evaluation"))
    ev_ids = [t for t in sorted(ev) if A.task_eval_example(ev[t]) is not None][:args.n]
    print(f"[arc-ttt] test-time training on {len(ev_ids)} held-out ARC worlds", flush=True)

    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    base = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map={"": 0})
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    model = get_peft_model(base, LORA)

    res = {"experiment": "arc-ttt", "base": BASE, "n_tasks": len(ev_ids),
           "levels": [dict(name=n, n_aug=a, steps=s) for n, a, s in LEVELS + [ABL]],
           "arms": {}, "per_task": {}}

    def upload():
        for arm, hits in res["per_task"].items():
            done = [h for h in hits.values() if h is not None]
            res["arms"][arm] = {"acc": round(sum(done) / len(done), 4) if done else None,
                                "n_done": len(done)}
        out = HERE / "results" / f"e80_arc_ttt_seed{SEED}.json"
        res["seed"] = SEED
        out.write_text(json.dumps(res, indent=2))
        if args.bucket:
            subprocess.run(["gcloud", "storage", "cp", str(out),
                            f"{args.bucket}/e80_arc_ttt_seed{SEED}.json"], check=False)

    # zero-shot baseline (adapters disabled = raw base)
    res["per_task"]["zeroshot"] = {}
    with model.disable_adapter():
        for tid in ev_ids:
            case = A.task_eval_example(ev[tid])
            pred = predict(model, tok, case["prompt"])
            ok = int(A.grids_equal(A.parse_grid(pred), A.parse_grid(case["answer"])))
            res["per_task"]["zeroshot"][tid] = ok
            upload()
    print(f"[zeroshot] {res['arms']['zeroshot']}", flush=True)

    # TTT arms: light, heavy, corrupt
    for name, n_aug, steps in LEVELS + [ABL]:
        res["per_task"][name] = {}
        corrupt = (name == ABL[0])
        for tid in ev_ids:
            try:
                task = ev[tid]
                rows = A.task_to_sft_rows(task, n_aug=n_aug,
                                          rng=np.random.default_rng((hash(tid) + SEED * 1000003) % 2**32),
                                          use_test=False, corrupt=corrupt)
                reset_adapter(model)
                ttt_fit(model, tok, rows, steps)
                case = A.task_eval_example(task)
                pred = predict(model, tok, case["prompt"])
                ok = int(A.grids_equal(A.parse_grid(pred), A.parse_grid(case["answer"])))
            except Exception as e:  # noqa: BLE001
                print(f"[{name} {tid}] FAILED {e}", flush=True)
                ok = None
            res["per_task"][name][tid] = ok
            upload()
        print(f"[{name}] {res['arms'][name]}", flush=True)

    print("[arc-ttt] done\n" + json.dumps(res["arms"], indent=2), flush=True)


if __name__ == "__main__":
    main()
