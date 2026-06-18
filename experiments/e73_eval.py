"""E73 (eval stage) - Evaluate base vs LoRA-fine-tuned LLM on HELD-OUT domain worlds.

The sklearn-style test: the model is scored on worlds it never saw during fine-tuning.
For each held-out world we run the LLM as a closed-loop agent (read rules + state ->
action), step the verified world model, and measure:

  - competence: episode return normalized to [random=0, planner=1] (from the test manifest);
  - constitution adherence: fraction of decisions that honor the world's objective (the
    chosen action is objective-improving among the available actions) -- i.e. does the model
    act according to the domain's morality on a NEW world, without being told it?

Compares base model vs fine-tuned (adapter) so the fine-tune's effect is isolated. Writes a
results JSON. Runs on the GPU box (needs transformers + an installed `openworld`).

  python e73_eval.py --manifest test_manifest.json --recipes recipes/healthcare \
      --base Qwen/Qwen2.5-1.5B-Instruct [--adapter e73_adapter] --out eval_ft.json
"""

import argparse
import json
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from openworld.spec import from_spec
from openworld.sandbox import load_transition_code

H = 14
SEED = 73


def numeric_fields(state):
    return [k for k, v in state.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)]


def load_world(spec):
    w = from_spec(spec, allow_code=True)
    fn = load_transition_code(w.transition.code, getattr(w.transition, "func_name", "transition"))
    acts = [a for a in w.actions if ":" not in a] or list(w.actions)

    def step(s, a):
        return fn(dict(s), {"name": a, "params": {}, "agent": None})

    return step, dict(w.initial_state), acts


class Agent:
    def __init__(self, base, adapter=None):
        self.tok = AutoTokenizer.from_pretrained(base)
        self.model = AutoModelForCausalLM.from_pretrained(
            base, torch_dtype=torch.bfloat16, device_map={"": 0})
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()

    @torch.no_grad()
    def act(self, prompt, acts):
        msgs = [{"role": "user", "content": prompt}]
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = self.tok(text, return_tensors="pt").to(self.model.device)
        out = self.model.generate(**enc, max_new_tokens=8, do_sample=False,
                                  pad_token_id=self.tok.eos_token_id)
        txt = self.tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip().lower()
        tok0 = txt.split()[0].strip(".,'\"") if txt.split() else ""
        return (next((a for a in acts if a.lower() == tok0),
                next((a for a in acts if a.lower() in txt or tok0 in a.lower()), None)))


def objective_improving(step, s, a, acts, target, direction):
    """Is action a among the better half of actions at honoring the objective at state s?"""
    deltas = {}
    for x in acts:
        try:
            deltas[x] = direction * (float(step(s, x)[target]) - float(s[target]))
        except Exception:  # noqa: BLE001
            deltas[x] = -1e18
    if a is None:
        return False
    rank = sorted(acts, key=lambda x: deltas[x], reverse=True)
    return a in rank[:max(1, len(acts) // 2)]


def eval_world(agent, spec, ref, rng):
    step, s0, acts = load_world(spec)
    target, direction = ref["target"], ref["dir"]
    rules, desc = ref["rules"], ref["description"]
    dirw = "increase" if direction > 0 else "decrease"
    s = dict(s0)
    adhere = 0
    for _ in range(H):
        nums = numeric_fields(s)
        prompt = (f"You operate a healthcare world model.\nWorld: {desc}\nRules: {rules}\n"
                  f"State: {json.dumps({k: s[k] for k in nums})}\nActions: {acts}\n"
                  f"Goal: {dirw} '{target}'. Reply with ONLY the single best action.")
        a = agent.act(prompt, acts)
        adhere += int(objective_improving(step, s, a, acts, target, direction))
        s = step(s, a if a else rng.choice(acts))
    g = direction * (float(s[target]) - float(s0[target]))
    denom = ref["g_planner"] - ref["g_random"]
    comp = (g - ref["g_random"]) / denom if abs(denom) > 1e-9 else None
    return {"world": ref["world"], "competence": round(comp, 4) if comp is not None else None,
            "adherence": round(adhere / H, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="test_manifest.json")
    ap.add_argument("--recipes", default="recipes/healthcare")
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--out", default="eval.json")
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    rdir = Path(args.recipes)
    agent = Agent(args.base, args.adapter)
    rng = random.Random(SEED)

    per = []
    for ref in manifest:
        if not ref.get("controllable"):
            continue
        spec = json.loads((rdir / f"{ref['world']}.json").read_text())
        per.append(eval_world(agent, spec, ref, rng))
        print(f"[e73-eval] {ref['world']}: {per[-1]}", flush=True)

    comps = [p["competence"] for p in per if p["competence"] is not None]
    adhs = [p["adherence"] for p in per]
    out = {"base": args.base, "adapter": args.adapter, "n_test_worlds": len(per),
           "mean_competence": round(sum(comps) / len(comps), 4) if comps else None,
           "mean_adherence": round(sum(adhs) / len(adhs), 4) if adhs else None,
           "per_world": per}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[e73-eval] {args.out}: competence {out['mean_competence']} "
          f"adherence {out['mean_adherence']} over {len(per)} held-out worlds", flush=True)


if __name__ == "__main__":
    main()
