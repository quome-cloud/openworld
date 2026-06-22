"""E82: the hybrid loop -- a verified world model as both data generator and inference oracle.

Demonstrates all three routes in one experiment, with an OPEN-WEIGHT model and zero human labels:
  1) TOOL generates exact data: sample solutions from the base model, keep ONLY those whose tests
     pass (rejection sampling against the verified oracle = the world model).
  2) TTT amortizes: QLoRA-fine-tune on the verified-passing solutions -> base pass@1 -> amortized.
  3) HYBRID at inference: the amortized model generates; the tool verifies; on failure, retry
     (verify-and-retry). The tool bootstraps the data, training amortizes it, the tool backstops.

Substrate: HumanEval-X Python (the test suite is a perfect exact verifier). Open-weight
Qwen2.5-Coder-7B. GPU + python only (no extra runtimes). Partial results upload as it runs.
  python3 e82_hybrid.py --bucket gs://openworld-bench/e82 --n_train 80 --n_eval 40
"""

import argparse
import json
import random
import subprocess
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import e81_progworlds as P

HERE = Path(__file__).resolve().parent
BASE = "Qwen/Qwen2.5-Coder-7B-Instruct"
LANG = "python"
LORA = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])


def trunc_py(body):
    """Keep the function body; stop at the first dedented line (end of the function)."""
    out = []
    for ln in body.split("\n"):
        if ln.strip() and not ln[0].isspace() and out:
            break
        out.append(ln)
    return "\n".join(out)


@torch.no_grad()
def gen(model, tok, prompt, n=1, temp=0.0, max_new_tokens=320):
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=1536).to(model.device)
    kw = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.eos_token_id)
    if temp and temp > 0:
        kw.update(do_sample=True, temperature=temp, top_p=0.95, num_return_sequences=n)
    else:
        kw.update(do_sample=False)
    out = model.generate(**enc, **kw)
    g = out[:, enc["input_ids"].shape[1]:]
    return [trunc_py(tok.decode(x, skip_special_tokens=True)) for x in g]


def passes(sw, body):
    return P.run_one(LANG, P.assemble(LANG, sw, body))[0]


def sft(model, tok, pairs, epochs=3, lr=1e-4):
    # Train in the SAME raw completion format gen() uses (prompt -> body): NO chat template,
    # else the adapter is tuned for chat-format inputs but always fed raw prompts -> degenerate.
    # Loss is masked to the solution tokens only (standard SFT).
    model.config.use_cache = False
    model.train()
    seqs = []
    for pr, sol in pairs:
        pids = tok(pr, truncation=True, max_length=1024)["input_ids"]
        sids = tok(sol, truncation=True, max_length=512, add_special_tokens=False)["input_ids"]
        if sids and len(pids) + len(sids) >= 8:
            seqs.append((pids + sids, len(pids)))
    if not seqs:
        return
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    rng = random.Random(0)
    for _ in range(epochs):
        rng.shuffle(seqs)
        for ids, npr in seqs:
            inp = torch.tensor([ids]).to(model.device)
            lab = inp.clone()
            lab[0, :npr] = -100                      # loss only on the solution tokens
            model(input_ids=inp, labels=lab).loss.backward()
            opt.step()
            opt.zero_grad()
    # restore inference state for the amortized/hybrid generation that follows
    model.config.use_cache = True
    model.eval()
    try:
        model.gradient_checkpointing_disable()
    except Exception:
        pass


def main():
    global BASE
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", default="")
    ap.add_argument("--n_train", type=int, default=80)
    ap.add_argument("--n_eval", type=int, default=40)
    ap.add_argument("--k_boot", type=int, default=8)
    ap.add_argument("--r_hybrid", type=int, default=5)
    ap.add_argument("--base", default=BASE)
    args = ap.parse_args()
    BASE = args.base

    comp = P.build_worlds()
    pids = list(comp)
    random.Random(82).shuffle(pids)
    train_ids = pids[:args.n_train]
    eval_ids = pids[args.n_train:args.n_train + args.n_eval]
    print(f"[e82] {len(train_ids)} bootstrap / {len(eval_ids)} held-out problems (python)", flush=True)

    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    base = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map={"": 0})
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    model = get_peft_model(base, LORA)

    res = {"experiment": "e82-hybrid", "base": BASE, "lang": LANG,
           "n_train": len(train_ids), "n_eval": len(eval_ids), "k_boot": args.k_boot,
           "r_hybrid": args.r_hybrid, "results": {}}

    def upload():
        out = HERE / "results" / "e82_hybrid.json"
        out.write_text(json.dumps(res, indent=2))
        if args.bucket:
            subprocess.run(["gcloud", "storage", "cp", str(out), f"{args.bucket}/e82_hybrid.json"],
                           check=False)

    # ---- base pass@1 (adapter disabled) ----
    with model.disable_adapter():
        base_pass = sum(passes(comp[p][LANG], gen(model, tok, comp[p][LANG]["prompt"])[0])
                        for p in eval_ids) / len(eval_ids)
        res["results"]["base_pass1"] = round(base_pass, 4); upload()
        print(f"[base pass@1] {base_pass:.3f}", flush=True)

        # ---- Route 1 as data generator: rejection-sample TRAIN, keep passing (exact data) ----
        pairs = []
        for i, p in enumerate(train_ids):
            sw = comp[p][LANG]
            for body in gen(model, tok, sw["prompt"], n=args.k_boot, temp=0.8):
                if passes(sw, body):
                    pairs.append((sw["prompt"], body))
                    break                                  # one verified solution per problem
            if i % 10 == 0:
                res["results"]["verified_pairs"] = len(pairs); upload()
        res["results"]["verified_pairs"] = len(pairs)
        print(f"[bootstrap] {len(pairs)} verified solutions from {len(train_ids)} problems", flush=True)

    # ---- TTT amortize on self-generated verified data ----
    sft(model, tok, pairs, epochs=3)

    # ---- amortized pass@1 + hybrid (verify-and-retry) ----
    amort, hyb_pass, hyb_calls = 0, 0, 0
    for p in eval_ids:
        sw = comp[p][LANG]
        if passes(sw, gen(model, tok, sw["prompt"])[0]):    # greedy, pass@1
            amort += 1
        # hybrid: greedy first, then sampled retries verified by the tool
        ok, calls = False, 0
        cand = gen(model, tok, sw["prompt"])[0]; calls += 1
        if passes(sw, cand):
            ok = True
        else:
            for _ in range(args.r_hybrid - 1):
                cand = gen(model, tok, sw["prompt"], temp=0.8)[0]; calls += 1
                if passes(sw, cand):
                    ok = True; break
        hyb_pass += int(ok); hyb_calls += calls
    res["results"]["amortized_pass1"] = round(amort / len(eval_ids), 4)
    res["results"]["hybrid_pass"] = round(hyb_pass / len(eval_ids), 4)
    res["results"]["hybrid_avg_verifier_calls"] = round(hyb_calls / len(eval_ids), 2)
    upload()
    print(f"[e82] done\n{json.dumps(res['results'], indent=2)}", flush=True)


if __name__ == "__main__":
    main()
