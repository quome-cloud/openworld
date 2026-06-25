"""E91 -- GRPO verifier-RL for the code-world-model synthesizer.

The exact-match verification gate is a perfect OUTCOME-VERIFIABLE reward, so we can RL-tune the
synthesizer (GRPO / DAPO-style) instead of just prompting it: sample K candidate predict() codes,
score each by held-out exact-match, use group-relative advantage to update. Tests whether verifier-RL
lifts a LOCAL model (qwen) toward the frontier -- on-thesis with world-time compute (train on the
verified signal). Reads PRE-COLLECTED transitions (no arc-agi here -> no py-version clash with torch).

  python3 e91_grpo_synth.py --transitions /tmp/arc3_trans --base Qwen/Qwen2.5-Coder-7B-Instruct
"""
import argparse
import glob
import json
import random
import re
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

HERE = Path(__file__).resolve().parent
LORA = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
PROMPT = """You are given transitions from a deterministic 64x64 grid game (colors 0-15).
Each transition: an action (int) maps the current grid to the next grid. Background color = {bg}.
Most cells are unchanged; only a few change. Write:

    def predict(frame, action):  # frame: np.ndarray (64,64) int; returns np.ndarray (64,64)

that reproduces the next grid EXACTLY. Use numpy (np). Return ONLY a ```python code block.

Transitions (action -> changed cells [row,col,new_color]):
{demos}
"""


def deltas(f, n):
    f, n = np.asarray(f), np.asarray(n)
    rs, cs = np.where(f != n)
    return [[int(r), int(c), int(n[r, c])] for r, c in zip(rs, cs)]


def bg_of(g):
    v, c = np.unique(np.asarray(g), return_counts=True)
    return int(v[np.argmax(c)])


def demo_str(t, cap=80):
    d = deltas(t["frame"], t["next"])
    return f"action {t['action']} -> {str(d[:cap]) + (' ...' if len(d) > cap else '')}"


def build_prompt(tok, trans, n_demo=12):
    bg = bg_of(trans[0]["frame"])
    demos = "\n".join(demo_str(t) for t in trans[:n_demo])
    msg = [{"role": "user", "content": PROMPT.format(bg=bg, demos=demos)}]
    return tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)


def extract_code(text):
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.S)
    return m.group(1).strip() if m else text.strip()


def verify(code, trans):
    ns = {"np": np, "numpy": np}
    try:
        exec(compile(code, "<v>", "exec"), ns)  # noqa: S102
        pred = ns["predict"]
    except Exception:  # noqa: BLE001
        return 0.0
    ok = 0
    for t in trans:
        try:
            out = np.asarray(pred(np.asarray(t["frame"]), t["action"]))
            ok += int(out.shape == (64, 64) and np.array_equal(out, np.asarray(t["next"])))
        except Exception:  # noqa: BLE001
            pass
    return ok / len(trans) if trans else 0.0


def load_games(d):
    games = {}
    for f in glob.glob(str(Path(d) / "*.json")):
        tr = json.load(open(f))
        if isinstance(tr, list) and len(tr) >= 20:
            games[Path(f).stem] = tr
    return games


@torch.no_grad()
def fidelity(model, tok, games, n=None):
    model.eval()
    vals = []
    for g, tr in list(games.items())[:n] if n else games.items():
        cut = len(tr) * 3 // 4
        text = build_prompt(tok, tr[:cut])
        enc = tok(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
        out = model.generate(**enc, do_sample=False, max_new_tokens=400, pad_token_id=tok.eos_token_id)
        code = extract_code(tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True))
        vals.append(verify(code, tr[cut:]))
    return float(np.mean(vals)) if vals else 0.0


def seq_logprob(model, prompt_ids, gen, pad_id):
    K = gen.shape[0]
    full = torch.cat([prompt_ids.repeat(K, 1), gen], dim=1)
    logits = model(full).logits[:, :-1]
    logp = torch.log_softmax(logits.float(), dim=-1)
    tgt = full[:, 1:]
    tok_lp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    P = prompt_ids.shape[1]
    gen_lp = tok_lp[:, P - 1:]
    mask = (gen != pad_id).float()
    return (gen_lp * mask).sum(1) / mask.sum(1).clamp(min=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transitions", required=True)
    ap.add_argument("--base", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--group", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bucket", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    games = load_games(args.transitions)
    names = sorted(games)
    random.Random(args.seed).shuffle(names)
    train_g = names[: max(1, len(names) * 7 // 10)]
    eval_g = {k: games[k] for k in names[len(train_g):]}
    print(f"[e91] {len(train_g)} train games, {len(eval_g)} eval games", flush=True)

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    base = AutoModelForCausalLM.from_pretrained(args.base, quantization_config=bnb, device_map={"": 0})
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    model = get_peft_model(base, LORA)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    rng = random.Random(args.seed)

    fid0 = fidelity(model, tok, eval_g)
    print(f"[e91] eval fidelity BEFORE: {fid0:.4f}", flush=True)

    rewards_hist = []
    done = 0
    while done < args.steps:
        g = rng.choice(train_g)
        tr = games[g]
        cut = len(tr) * 3 // 4
        train_tr, held = tr[:cut], tr[cut:]
        text = build_prompt(tok, train_tr)
        enc = tok(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
        model.eval()
        with torch.no_grad():
            out = model.generate(**enc, do_sample=True, temperature=args.temp, top_p=0.95,
                                 num_return_sequences=args.group, max_new_tokens=400,
                                 pad_token_id=tok.eos_token_id)
        gen = out[:, enc.input_ids.shape[1]:]
        texts = tok.batch_decode(gen, skip_special_tokens=True)
        rewards = torch.tensor([verify(extract_code(t), held) for t in texts], dtype=torch.float32)
        if rewards.std() < 1e-4:        # DAPO: skip zero-variance groups (no learning signal)
            continue
        adv = ((rewards - rewards.mean()) / (rewards.std() + 1e-6)).to(model.device)
        model.train()
        logp = seq_logprob(model, enc.input_ids, gen, tok.pad_token_id)
        loss = -(adv.detach() * logp).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step(); opt.zero_grad()
        done += 1
        rewards_hist.append(float(rewards.mean()))
        if done % 20 == 0:
            print(f"[e91] step {done}: group-mean-reward {np.mean(rewards_hist[-20:]):.3f}", flush=True)

    fid1 = fidelity(model, tok, eval_g)
    res = {"experiment": "grpo-synth", "base": args.base, "steps": done,
           "eval_fidelity_before": round(fid0, 4), "eval_fidelity_after": round(fid1, 4),
           "lift": round(fid1 - fid0, 4), "n_train_games": len(train_g), "n_eval_games": len(eval_g)}
    print(f"[e91] eval fidelity AFTER: {fid1:.4f} (lift {fid1 - fid0:+.4f})", flush=True)
    out = Path(args.out) if args.out else HERE / "results" / "e91_grpo_synth.json"
    out.write_text(json.dumps(res, indent=2))
    if args.bucket:
        import subprocess
        subprocess.run(["gcloud", "storage", "cp", str(out), f"{args.bucket}/e91_grpo_synth.json"], check=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
