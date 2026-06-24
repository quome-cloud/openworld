"""E85 trajectory-vs-answer ablation: is the verified TRAJECTORY (s,a,s') a better training unit
than the verified ANSWER (s0 -> final state)?  This isolates the paper's claimed differentiator
from answer-level self-training (STaR/ReST/RFT): we train on the same worlds, same token budget,
and vary only the unit.

Domain: synthetic ITERATED worlds. Each world is a deterministic update rule f over a length-L
integer list mod M (a composition of simple ops: add-c, rotate-k, swap(i,j), inc-pos). A
trajectory is s0 -> s1 -> ... with s_{t+1} = f(s_t). Worlds are split disjointly into held-in
(train) and held-out (eval) -- cross-world, like E84.

Two cross-world adapters, trained on the SAME held-in worlds:
  ANSWER     -- rows present in-context (x -> f^H(x)) demos and ask the H-step output for a query.
  TRAJECTORY -- rows present in-context (x -> f(x)) demos and ask the 1-step output for a query.
Trained at horizon H_train. Evaluated on held-out worlds by predicting f^{Htest}(s0):
  answer adapter: asked the Htest-step query directly (Htest-step demos in context).
  trajectory adapter: rolled out Htest single steps (1-step demos in context), final state checked.
At Htest = H_train both see matched demos; at Htest > H_train (extrapolation) only the per-step
unit can compose to the longer horizon. A trajectory>answer gap, growing with horizon, measures
the value of the (s,a,s') unit over the answer unit.

  python3 e85_traj_vs_answer.py --bucket gs://openworld-bench/e85 --seed 0
"""
import argparse
import json
import random
import subprocess
from pathlib import Path

try:  # heavy GPU deps -- guarded so the pure world logic is importable/testable on CPU
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from _adapter_ckpt import load_or_train
    LORA = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
except ImportError:
    torch = None

HERE = Path(__file__).resolve().parent
BASE = "Qwen/Qwen2.5-7B-Instruct"
MAXLEN = 1536
L, M = 4, 10                      # state: length-4 list mod 10
INSTR = ("Each example maps an input list of integers to an output list by a fixed hidden rule. "
         "Infer the rule from the examples and give the output for the final input.")


# ---------- synthetic iterated worlds ----------
def make_rule(rng):
    """A world = composition of two simple deterministic ops over a length-L list mod M."""
    ops = []
    for _ in range(2):
        kind = rng.choice(["add", "rot", "swap", "inc"])
        if kind == "add":
            ops.append(("add", rng.randrange(1, M)))
        elif kind == "rot":
            ops.append(("rot", rng.randrange(1, L)))
        elif kind == "swap":
            i, j = rng.sample(range(L), 2)
            ops.append(("swap", i, j))
        else:
            ops.append(("inc", rng.randrange(L)))
    return ops


def step(state, ops):
    s = list(state)
    for op in ops:
        if op[0] == "add":
            s = [(x + op[1]) % M for x in s]
        elif op[0] == "rot":
            k = op[1]
            s = s[k:] + s[:k]
        elif op[0] == "swap":
            i, j = op[1], op[2]
            s[i], s[j] = s[j], s[i]
        else:
            i = op[1]
            s[i] = (s[i] + 1) % M
    return s


def rollout(state, ops, h):
    s = list(state)
    for _ in range(h):
        s = step(s, ops)
    return s


def fmt(s):
    return "[" + ", ".join(str(x) for x in s) + "]"


def rand_state(rng):
    return [rng.randrange(M) for _ in range(L)]


def build_prompt(demos, query):
    parts = [INSTR, ""]
    for k, (a, b) in enumerate(demos, 1):
        parts += [f"Example {k}", "Input:", fmt(a), "Output:", fmt(b), ""]
    parts += ["Now solve.", "Input:", fmt(query), "Output:"]
    return "\n".join(parts)


def train_rows(worlds, names, h, rows_per_world, n_ctx, rng):
    """Cross-world training rows: in-context h-step demos + a held-in query -> its h-step output."""
    rows = []
    for nm in names:
        ops = worlds[nm]
        for _ in range(rows_per_world):
            pool = [(s0, rollout(s0, ops, h)) for s0 in (rand_state(rng) for _ in range(n_ctx + 1))]
            demos, (q, ans) = pool[:n_ctx], pool[n_ctx]
            rows.append({"prompt": build_prompt(demos, q), "completion": fmt(ans)})
    rng.shuffle(rows)
    return rows


# ---------- LoRA train / predict ----------
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


def predict(model, tok, prompt, mnt=24):
    model.config.use_cache = True
    model.eval()
    text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                   add_generation_prompt=True)
    enc = tok(text, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    with torch.no_grad():
        gen = model.generate(**enc, max_new_tokens=mnt, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(gen[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)


def parse(text):
    import re
    m = re.search(r"\[([0-9,\s]+)\]", str(text))
    if not m:
        return None
    try:
        return [int(x) for x in m.group(1).split(",")]
    except ValueError:
        return None


def eval_answer(model, tok, worlds, names, h, n_ctx, n_q, rng):
    """Predict f^h(s0) directly; in-context demos are h-step pairs."""
    hits = tot = 0
    for nm in names:
        ops = worlds[nm]
        for _ in range(n_q):
            pool = [(s0, rollout(s0, ops, h)) for s0 in (rand_state(rng) for _ in range(n_ctx + 1))]
            demos, (q, ans) = pool[:n_ctx], pool[n_ctx]
            pred = parse(predict(model, tok, build_prompt(demos, q)))
            hits += int(pred == ans)
            tot += 1
    return hits / tot if tot else None


def eval_traj(model, tok, worlds, names, h, n_ctx, n_q, rng):
    """Predict f^h(s0) by rolling out h single steps; demos are 1-step pairs."""
    hits = tot = 0
    for nm in names:
        ops = worlds[nm]
        for _ in range(n_q):
            q = rand_state(rng)
            ans = rollout(q, ops, h)
            demos = [(s0, step(s0, ops)) for s0 in (rand_state(rng) for _ in range(n_ctx))]
            cur, ok = list(q), True
            for _ in range(h):
                nxt = parse(predict(model, tok, build_prompt(demos, cur)))
                if nxt is None or len(nxt) != L:
                    ok = False
                    break
                cur = nxt
            hits += int(ok and cur == ans)
            tot += 1
    return hits / tot if tot else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", default="")
    ap.add_argument("--n_worlds", type=int, default=160)
    ap.add_argument("--n_train_worlds", type=int, default=100)
    ap.add_argument("--n_eval_worlds", type=int, default=40)
    ap.add_argument("--h_train", type=int, default=3)
    ap.add_argument("--h_tests", type=str, default="3,6")
    ap.add_argument("--rows_per_world", type=int, default=24)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--n_ctx", type=int, default=4)
    ap.add_argument("--n_q", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    grng = random.Random(85 + args.seed)
    worlds = {f"w{i}": make_rule(grng) for i in range(args.n_worlds)}
    names = list(worlds)
    grng.shuffle(names)
    train_names = names[:args.n_train_worlds]
    eval_names = names[args.n_train_worlds:args.n_train_worlds + args.n_eval_worlds]
    h_tests = [int(x) for x in args.h_tests.split(",")]
    print(f"[e85] seed={args.seed} train={len(train_names)} eval={len(eval_names)} "
          f"h_train={args.h_train} h_tests={h_tests}", flush=True)

    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    base = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map={"": 0})
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    model = get_peft_model(base, LORA)

    res = {"experiment": "traj-vs-answer", "base": BASE, "seed": args.seed,
           "n_train_worlds": len(train_names), "n_eval_worlds": len(eval_names),
           "h_train": args.h_train, "h_tests": h_tests, "results": {}}

    def upload():
        out = HERE / "results" / f"e85_traj_vs_answer_seed{args.seed}.json"
        out.write_text(json.dumps(res, indent=2))
        if args.bucket:
            subprocess.run(["gcloud", "storage", "cp", str(out),
                            f"{args.bucket}/e85_traj_vs_answer_seed{args.seed}.json"], check=False)

    # ANSWER adapter: train on h_train-step (s0 -> f^h) pairs
    rows_a = train_rows(worlds, train_names, args.h_train, args.rows_per_world, args.n_ctx,
                        random.Random(1000 + args.seed))
    print(f"[answer] training on {len(rows_a)} rows", flush=True)
    load_or_train(model, f"e85_s{args.seed}_answer", args.bucket, reset_adapter,
                  lambda: fit(model, tok, rows_a, args.steps, seed=args.seed))
    for h in h_tests:
        acc = eval_answer(model, tok, worlds, eval_names, h, args.n_ctx, args.n_q,
                          random.Random(7 + h))
        res["results"].setdefault(f"h{h}", {})["answer"] = acc
        print(f"[answer] h={h}: {acc}", flush=True)
        upload()

    # TRAJECTORY adapter: train on 1-step (s -> f(s)) pairs
    rows_t = train_rows(worlds, train_names, 1, args.rows_per_world, args.n_ctx,
                        random.Random(2000 + args.seed))
    print(f"[traj] training on {len(rows_t)} rows", flush=True)
    load_or_train(model, f"e85_s{args.seed}_traj", args.bucket, reset_adapter,
                  lambda: fit(model, tok, rows_t, args.steps, seed=args.seed))
    for h in h_tests:
        acc = eval_traj(model, tok, worlds, eval_names, h, args.n_ctx, args.n_q,
                        random.Random(7 + h))
        res["results"].setdefault(f"h{h}", {})["trajectory"] = acc
        print(f"[traj] h={h}: {acc}", flush=True)
        upload()

    print("[e85] done\n" + json.dumps(res["results"], indent=2), flush=True)


if __name__ == "__main__":
    main()
