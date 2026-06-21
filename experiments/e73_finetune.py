"""E73 (fine-tune stage) - LoRA-SFT a small LLM on the planner-labeled, rule-grounded
decisions from a domain's TRAIN worlds (behavior cloning of a model-based planner).

Runs on a GPU box (isolated venv). Consumes experiments/results/e73_artifacts/sft_train.jsonl
and writes a LoRA adapter. Small by design (qwen 1.5B, LoRA) so it co-exists with other
GPU jobs in spare VRAM. The matching eval is e73_eval.py.

  python e73_finetune.py --data sft_train.jsonl --out e73_adapter \
      --base Qwen/Qwen2.5-1.5B-Instruct --epochs 3
"""

import argparse
import inspect
import json

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="sft_train.jsonl")
    ap.add_argument("--out", default="e73_adapter")
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--load_4bit", action="store_true", help="QLoRA: 4-bit NF4 base (for 14B/32B)")
    ap.add_argument("--max_length", type=int, default=1024, help="SFT max sequence length")
    ap.add_argument("--max_steps", type=int, default=-1, help="cap optimizer steps (>0 overrides epochs)")
    ap.add_argument("--seed", type=int, default=73)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.data) if l.strip()]
    print(f"[e73-ft] {len(rows)} SFT examples from {args.data}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def fmt(r):
        msgs = [{"role": "user", "content": r["prompt"]},
                {"role": "assistant", "content": r["completion"]}]
        return {"text": tok.apply_chat_template(msgs, tokenize=False)}

    ds = Dataset.from_list([fmt(r) for r in rows])

    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        from peft import prepare_model_for_kbit_training
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

    cfg_kwargs = dict(
        output_dir=args.out, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch, gradient_accumulation_steps=args.grad_accum,
        learning_rate=2e-4, lr_scheduler_type="cosine", warmup_ratio=0.05,
        logging_steps=10, save_strategy="no", bf16=True, gradient_checkpointing=True,
        seed=args.seed, report_to=[])
    if args.max_steps and args.max_steps > 0:
        cfg_kwargs["max_steps"] = args.max_steps
    # trl renamed the max-sequence-length arg across versions (max_seq_length <-> max_length)
    # and dataset_text_field is not accepted in every version; pass only what this trl supports.
    _sft = inspect.signature(SFTConfig.__init__).parameters
    if "max_length" in _sft:
        cfg_kwargs["max_length"] = args.max_length
    elif "max_seq_length" in _sft:
        cfg_kwargs["max_seq_length"] = args.max_length
    if "dataset_text_field" in _sft:
        cfg_kwargs["dataset_text_field"] = "text"
    cfg = SFTConfig(**cfg_kwargs)

    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, peft_config=lora)
    trainer.train()
    trainer.model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"[e73-ft] saved LoRA adapter to {args.out}", flush=True)


if __name__ == "__main__":
    main()
