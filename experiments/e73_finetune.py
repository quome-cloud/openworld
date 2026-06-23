"""E73 (fine-tune stage) - LoRA-SFT a small LLM on planner-labeled, rule-grounded decisions
from a domain's TRAIN worlds. trl-free: a manual PEFT + torch training loop (the same proven
loop used in e80/e82), so it does not depend on trl/transformers version compatibility.

Runs on a GPU box. Consumes a JSONL of {prompt, completion}; writes a LoRA adapter that
e74_eval.py loads via PeftModel.from_pretrained.

  python e73_finetune.py --data sft_train.jsonl --out e73_adapter \
      --base Qwen/Qwen2.5-1.5B-Instruct --epochs 3
"""

import argparse
import json
import math
import random

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="sft_train.jsonl")
    ap.add_argument("--out", default="e73_adapter")
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=2)  # accepted for compat; loop uses batch
    ap.add_argument("--load_4bit", action="store_true", help="QLoRA: 4-bit NF4 base (for 14B/32B)")
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--max_steps", type=int, default=-1, help="cap optimizer steps (>0 overrides epochs)")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=73)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rows = [json.loads(l) for l in open(args.data) if l.strip()]
    print(f"[e73-ft] {len(rows)} SFT examples from {args.data}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16,
                                 bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(args.base, quantization_config=bnb,
                                                     device_map={"": 0})
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16,
                                                     device_map={"": 0})
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                      task_type="CAUSAL_LM", target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    model = get_peft_model(model, lora)
    model.config.use_cache = False
    model.train()

    # tokenize to chat-template sequences (full-text LM loss, padding masked) -- mirrors e80
    seqs = []
    for r in rows:
        text = tok.apply_chat_template(
            [{"role": "user", "content": r["prompt"]},
             {"role": "assistant", "content": r["completion"]}], tokenize=False)
        ids = tok(text, truncation=True, max_length=args.max_length)["input_ids"]
        if len(ids) >= 8:
            seqs.append(ids)
    if not seqs:
        raise SystemExit("[e73-ft] no usable sequences")

    bs = args.batch
    steps_per_epoch = math.ceil(len(seqs) / bs)
    total_steps = (args.max_steps if args.max_steps and args.max_steps > 0
                   else int(args.epochs * steps_per_epoch))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    rng = random.Random(args.seed)
    done = 0
    print(f"[e73-ft] {len(seqs)} seqs, bs={bs}, total_steps={total_steps}", flush=True)
    while done < total_steps:
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
            if done >= total_steps:
                break

    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"[e73-ft] saved LoRA adapter to {args.out} ({done} steps)", flush=True)


if __name__ == "__main__":
    main()
