"""E78 (QLoRA fine-tune) - 4-bit QLoRA-SFT a small LLM on verified-planner labels.

Behavior-cloning of the BFS oracle (which plans through the VERIFIED Blocksworld world model)
into qwen2.5. Consumes experiments/results/e78_artifacts/sft_train.jsonl ({prompt, completion}
where completion is the optimal PLAN text) and writes a LoRA adapter. The matching eval is
e78_eval.py; both share the prompt format from e78_world_model_tool.build_prompt.

Same QLoRA recipe as E73/E74 (4-bit NF4 base + LoRA on attention projections), but written
against the plain transformers Trainer (not trl) so it is robust to trl's API churn.

  python e78_finetune.py --data sft_train.jsonl --out e78_adapter \
      --base Qwen/Qwen2.5-7B-Instruct --load_4bit --epochs 3
"""

import argparse
import json

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForLanguageModeling, Trainer, TrainingArguments)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="sft_train.jsonl")
    ap.add_argument("--out", default="e78_adapter")
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--load_4bit", action="store_true", help="QLoRA: 4-bit NF4 base")
    ap.add_argument("--seed", type=int, default=78)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.data) if l.strip()]
    print(f"[e78-ft] {len(rows)} SFT examples from {args.data}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def to_text(r):
        msgs = [{"role": "user", "content": r["prompt"]},
                {"role": "assistant", "content": r["completion"]}]
        return tok.apply_chat_template(msgs, tokenize=False)

    ds = Dataset.from_dict({"text": [to_text(r) for r in rows]})
    ds = ds.map(lambda b: tok(b["text"], truncation=True, max_length=args.max_len),
                batched=True, remove_columns=["text"])
    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16,
                                 bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.base, quantization_config=bnb, device_map={"": 0})
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.base, torch_dtype=torch.bfloat16, device_map={"": 0})
    model.config.use_cache = False

    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    targs = TrainingArguments(
        output_dir=args.out, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch, gradient_accumulation_steps=args.grad_accum,
        learning_rate=2e-4, lr_scheduler_type="cosine", warmup_ratio=0.05,
        logging_steps=10, save_strategy="no", bf16=True,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=args.seed, report_to=[])

    trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=collator)
    trainer.train()
    trainer.model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"[e78-ft] saved LoRA adapter to {args.out}", flush=True)


if __name__ == "__main__":
    main()
