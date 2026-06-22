"""E81 Phase 3: world-time compute ACROSS language frames (composite programming worlds).

Each composite world is a programming problem; its sub-worlds are the problem in N languages --
reference frames of one principle. We test whether world-time compute is frame-covariant: for a
held-out problem and target language, fit a fresh LoRA on that problem's solutions in the OTHER
languages (the principle seen in other frames), then generate the target-language solution and
execute its tests. Arms:
  - zeroshot : generate the target solution with no adaptation (base code model);
  - frame-TTT: fit on the problem's other-language (prompt->solution) pairs, then generate target;
  - corrupt  : fit on MISMATCHED pairs (a prompt with another problem-language's solution = wrong
               frame mapping) -- the exactness/coherence control.
Exact label = the target language's tests pass (verified oracle). Needs GPU + language runtimes
(g++, node, javac); uses the 4 validated frames (Go assembly excluded).

  python3 e81_wtc.py --bucket gs://openworld-bench/e81wtc --n 20
"""

import argparse
import json
import random
import re
import subprocess
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import e81_progworlds as P

HERE = Path(__file__).resolve().parent
BASE = "Qwen/Qwen2.5-Coder-7B-Instruct"
LANGS = ["python", "cpp", "java", "js", "go"]   # 5 verified language frames
LORA = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
TTT_STEPS = 60


def assemble_gen(lang, sw, gen):
    """Full program from a generated body that CONTINUES sw['prompt'] (HumanEval convention)."""
    code = gen
    if "```" in code:                              # strip markdown fences if present
        m = re.findall(r"```[a-zA-Z]*\n(.*?)```", code, re.DOTALL)
        if m:
            code = m[0]
    if lang == "python":
        ep = P._entry_point(sw["prompt"], sw["declaration"])
        return f"{sw['prompt']}{code}\n{sw['test']}\ncheck({ep})\n"
    return f"{sw['prompt']}{code}\n{sw['test']}\n"


def reset_adapter(model):
    if "default" in getattr(model, "peft_config", {}):
        model.delete_adapter("default")
    model.add_adapter("default", LORA)
    model.set_adapter("default")


def ttt_fit(model, tok, pairs, steps, lr=1e-4):
    model.config.use_cache = False
    model.train()
    seqs = []
    for pr, sol in pairs:
        text = tok.apply_chat_template([{"role": "user", "content": pr},
                                        {"role": "assistant", "content": sol}], tokenize=False)
        ids = tok(text, truncation=True, max_length=2048)["input_ids"]
        if len(ids) >= 8:
            seqs.append(ids)
    if not seqs:
        return
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    rng = random.Random(0)
    done = 0
    while done < steps:
        rng.shuffle(seqs)
        for s in seqs:
            inp = torch.tensor([s]).to(model.device)
            model(input_ids=inp, labels=inp).loss.backward()
            opt.step()
            opt.zero_grad()
            done += 1
            if done >= steps:
                break


@torch.no_grad()
def gen_body(model, tok, prompt, max_new_tokens=384):
    model.config.use_cache = True
    model.eval()
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)


def main():
    global BASE
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", default="")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--base", default=BASE)
    args = ap.parse_args()
    BASE = args.base

    comp = P.build_worlds()
    pids = list(comp)
    random.Random(81).shuffle(pids)
    pids = pids[:args.n]
    print(f"[e81-wtc] {len(pids)} held-out composite worlds, frames {LANGS}", flush=True)

    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    base = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map={"": 0})
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    model = get_peft_model(base, LORA)

    res = {"experiment": "e81-wtc-frame-invariance", "base": BASE, "langs": LANGS,
           "n_worlds": len(pids), "ttt_steps": TTT_STEPS, "arms": {}, "per": {}}
    for arm in ("zeroshot", "frame_ttt", "corrupt"):
        res["per"][arm] = {"pass": 0, "n": 0}

    def upload():
        for a, v in res["per"].items():
            res["arms"][a] = round(v["pass"] / v["n"], 4) if v["n"] else None
        out = HERE / "results" / "e81_wtc.json"
        out.write_text(json.dumps(res, indent=2))
        if args.bucket:
            subprocess.run(["gcloud", "storage", "cp", str(out), f"{args.bucket}/e81_wtc.json"],
                           check=False)

    for pi, pid in enumerate(pids):
        for tgt in LANGS:
            srcs = [s for s in LANGS if s != tgt]
            sw_t = comp[pid][tgt]
            # zero-shot
            ok = P.run_one(tgt, assemble_gen(tgt, sw_t, gen_body(model, tok, sw_t["prompt"])))[0]
            res["per"]["zeroshot"]["pass"] += int(ok); res["per"]["zeroshot"]["n"] += 1
            # frame-TTT: fit on this problem's solutions in the OTHER languages
            pairs = [(comp[pid][s]["prompt"], comp[pid][s]["canonical_solution"]) for s in srcs]
            reset_adapter(model); ttt_fit(model, tok, pairs, TTT_STEPS)
            ok = P.run_one(tgt, assemble_gen(tgt, sw_t, gen_body(model, tok, sw_t["prompt"])))[0]
            res["per"]["frame_ttt"]["pass"] += int(ok); res["per"]["frame_ttt"]["n"] += 1
            # corrupt: mismatched (prompt of lang s, solution of a DIFFERENT lang) = wrong frame
            cpairs = [(comp[pid][s]["prompt"], comp[pid][srcs[(i + 1) % len(srcs)]]["canonical_solution"])
                      for i, s in enumerate(srcs)]
            reset_adapter(model); ttt_fit(model, tok, cpairs, TTT_STEPS)
            ok = P.run_one(tgt, assemble_gen(tgt, sw_t, gen_body(model, tok, sw_t["prompt"])))[0]
            res["per"]["corrupt"]["pass"] += int(ok); res["per"]["corrupt"]["n"] += 1
        upload()
        print(f"[{pi + 1}/{len(pids)}] arms={res['arms']}", flush=True)

    print(f"[e81-wtc] done\n{json.dumps(res['arms'], indent=2)}", flush=True)


if __name__ == "__main__":
    main()
