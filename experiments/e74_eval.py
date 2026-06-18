"""E74 (LLM stage, eval) - diagnostic accuracy on HELD-OUT specialties, base vs fine-tuned.

Pure text task (no world stepping): for each held-out-specialty patient, prompt the model,
parse the predicted disease, compare to the answer. Run once for the base model and once with
the LoRA adapter to isolate the fine-tune's effect.

  python e74_eval.py --test test_dx.jsonl --base Qwen/Qwen2.5-1.5B-Instruct \
      [--adapter e74_adapter] --out eval_dx_ft.json
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="test_dx.jsonl")
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--load_4bit", action="store_true", help="4-bit NF4 base (for 14B/32B)")
    ap.add_argument("--out", default="eval_dx.json")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.test) if l.strip()]
    tok = AutoTokenizer.from_pretrained(args.base)
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(args.base, quantization_config=bnb,
                                                     device_map={"": 0})
    else:
        model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16,
                                                     device_map={"": 0})
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    @torch.no_grad()
    def predict(prompt):
        text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       tokenize=False, add_generation_prompt=True)
        enc = tok(text, return_tensors="pt").to(model.device)
        out = model.generate(**enc, max_new_tokens=8, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        txt = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        m = re.search(r"disease[_ ]?(\d+)", txt.lower())
        return f"disease_{m.group(1)}" if m else txt.strip().split()[0] if txt.strip() else ""

    correct = 0
    by_spec = defaultdict(lambda: [0, 0])
    for r in rows:
        ok = int(predict(r["prompt"]) == r["answer"])
        correct += ok
        by_spec[r["specialty"]][0] += ok
        by_spec[r["specialty"]][1] += 1
    per_spec = {str(k): round(v[0] / v[1], 4) for k, v in sorted(by_spec.items())}

    out = {"base": args.base, "adapter": args.adapter, "n_cases": len(rows),
           "n_held_out_specialties": len(by_spec),
           "accuracy": round(correct / len(rows), 4),
           "per_specialty_accuracy": per_spec}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[e74-eval] {args.out}: accuracy {out['accuracy']} over {len(rows)} held-out cases "
          f"({len(by_spec)} specialties)", flush=True)


if __name__ == "__main__":
    main()
