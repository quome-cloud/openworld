"""E77 (eval) - pass@1 on held-out coding worlds and real benchmarks (HumanEval / MBPP),
base vs world-time-compute (fine-tuned on traversed coding worlds).

For each task we prompt the model to implement the function, extract the code, and run it
against the task's tests in a sandboxed subprocess (tests = ground-truth oracle). pass@1 =
fraction of tasks whose generated code passes ALL tests. Run base, then with the adapter, to
isolate the world-time-compute effect; and on HumanEval/MBPP to measure real-benchmark
transfer.

Runs on the GPU box (transformers). Code execution is sandboxed (subprocess + timeout).

  python e77_eval.py --base Qwen/Qwen2.5-1.5B-Instruct [--adapter e77_ad] \
      --source synthetic --tasks test_tasks.jsonl --out e77_eval_ft.json
  python e77_eval.py --base ... --source humaneval --path benchmarks/humaneval.jsonl --out ...
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path


def pass_at_k(n, c, k):
    """Unbiased pass@k estimator (Chen et al. 2021): 1 - C(n-c,k)/C(n,k)."""
    if k > n:
        return None
    if n - c < k:
        return 1.0
    p = 1.0
    for i in range(k):
        p *= (n - c - i) / (n - i)
    return 1.0 - p

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

INSTR = "Implement the following Python function. Return ONLY the function definition, no explanation.\n\n"
TIMEOUT_S = 12


def extract_code(text):
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    code = m.group(1) if m else text
    return code.strip()


def run_program(program):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=True) as f:
        f.write(program)
        f.flush()
        try:
            r = subprocess.run([sys.executable, f.name], capture_output=True,
                               text=True, timeout=TIMEOUT_S)
            return r.returncode == 0
        except Exception:  # noqa: BLE001
            return False


def load_tasks(args):
    """Return list of (id, model_prompt, make_program(code)->str)."""
    out = []
    if args.source == "synthetic":
        for l in Path(args.tasks).read_text().splitlines():
            if not l.strip():
                continue
            t = json.loads(l)
            tests = "\n".join(t["tests"])
            out.append((t["id"], t["prompt"], lambda code, tests=tests: code + "\n" + tests))
    elif args.source == "humaneval":
        for l in Path(args.path).read_text().splitlines():
            if not l.strip():
                continue
            t = json.loads(l)
            prompt = INSTR + t["prompt"]
            test, ep = t["test"], t["entry_point"]
            # concatenate (NOT str.format) -- HumanEval tests contain literal {..} dict braces
            out.append((t["task_id"], prompt,
                        lambda code, test=test, ep=ep: code + "\n" + test + f"\ncheck({ep})\n"))
    elif args.source == "mbpp":
        data = json.loads(Path(args.path).read_text())
        for t in data:
            tests = "\n".join(t.get("test_list", []))
            imports = "\n".join(t.get("test_imports", []) or [])
            prompt = INSTR + t["prompt"] + "\nYour function must pass:\n" + tests
            out.append((str(t["task_id"]), prompt,
                        lambda code, im=imports, te=tests: im + "\n" + code + "\n" + te))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--source", choices=["synthetic", "humaneval", "mbpp"], default="synthetic")
    ap.add_argument("--tasks", default="test_tasks.jsonl")
    ap.add_argument("--path", default=None)
    ap.add_argument("--load_4bit", action="store_true")
    ap.add_argument("--n_samples", type=int, default=1, help="samples/task; 1=greedy pass@1, >1=pass@k")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--out", default="e77_eval.json")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base)
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(args.base, quantization_config=bnb, device_map={"": 0})
    else:
        model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16, device_map={"": 0})
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    @torch.no_grad()
    def gen(prompt, n):
        text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       tokenize=False, add_generation_prompt=True)
        enc = tok(text, return_tensors="pt").to(model.device)
        kw = dict(max_new_tokens=512, pad_token_id=tok.eos_token_id)
        if n == 1:
            out = model.generate(**enc, do_sample=False, **kw)
        else:
            out = model.generate(**enc, do_sample=True, temperature=args.temperature,
                                 top_p=0.95, num_return_sequences=n, **kw)
        L = enc["input_ids"].shape[1]
        return [tok.decode(o[L:], skip_special_tokens=True) for o in out]

    tasks = load_tasks(args)
    ks = [k for k in (1, 3, 5) if k <= args.n_samples]
    sums = {k: 0.0 for k in ks}
    per = []
    for tid, prompt, make_prog in tasks:
        comps = gen(prompt, args.n_samples)
        c = sum(run_program(make_prog(extract_code(x))) for x in comps)
        for k in ks:
            sums[k] += pass_at_k(args.n_samples, c, k)
        per.append({"id": tid, "c": c, "n": args.n_samples})
    result = {"base": args.base, "adapter": args.adapter, "source": args.source,
              "n_tasks": len(tasks), "n_samples": args.n_samples, "temperature": args.temperature,
              **{f"pass_at_{k}": round(sums[k] / len(tasks), 4) if tasks else None for k in ks},
              "per_task": per}
    Path(args.out).write_text(json.dumps(result, indent=2))
    summary = " ".join(f"pass@{k}={result.get(f'pass_at_{k}')}" for k in ks)
    print(f"[e77-eval] {args.source}: {summary} over {len(tasks)} tasks "
          f"(n={args.n_samples}, adapter={args.adapter})", flush=True)


if __name__ == "__main__":
    main()
