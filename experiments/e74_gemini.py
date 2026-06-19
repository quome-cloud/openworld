"""E74 (frontier check) - does the world-time-compute story hold for frontier Gemini models?

We cannot fine-tune a closed API model, so the API-feasible analogue of world-time compute
is IN-CONTEXT traversal: prepend k diagnosis demos drawn from TRAIN specialties (teaching
the matching skill), then test on HELD-OUT specialties. We also measure plain zero-shot, to
see where Gemini Pro/Flash sit relative to the local-model curve and the oracle.

For each model x family (easy E74 / hard E75) we report:
  - base    : zero-shot held-out diagnostic accuracy
  - fewshot : with k train-specialty demos in context (in-context world-time compute)
  - the lift (fewshot - base), with a bootstrap CI over specialties

Stdlib only (urllib REST). Reads GEMINI_API_KEY from .env (gitignored). Subsamples to bound
cost; wraps every call in try/except so rate limits/errors are misses, not crashes.
Save before asserts.
"""

import json
import random
import re
import time
import urllib.request
import urllib.error
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "experiments" / "results"

MODELS = ["gemini-2.5-flash", "gemini-2.5-pro"]
FAMILIES = {"easy": "e74_artifacts", "hard": "e75_artifacts"}
N_EVAL = 80          # held-out cases sampled per (model, family)
K_FEWSHOT = 8        # in-context train-specialty demos
SLEEP_S = 0.6        # politeness between calls
SEED = 74


def load_key():
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line.startswith("GEMINI_API_KEY=") and not line.startswith("#"):
                return line.split("=", 1)[1].strip()
    import os
    return os.environ.get("GEMINI_API_KEY", "")


def gemini(model, prompt, key, retries=4):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
    # Flash can turn thinking off (fast/cheap, 1-token answer); Pro only runs in thinking
    # mode, so give it output-token room for the (hidden) reasoning plus the short answer.
    if "flash" in model:
        gencfg = {"temperature": 0, "maxOutputTokens": 32, "thinkingConfig": {"thinkingBudget": 0}}
    else:
        gencfg = {"temperature": 0, "maxOutputTokens": 2048}
    payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": gencfg}
    body = json.dumps(payload).encode("utf-8")
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                d = json.loads(r.read().decode("utf-8"))
            return d["candidates"][0]["content"]["parts"][0]["text"]
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < retries - 1:
                time.sleep(2 ** attempt * 2)
                continue
            return None
        except Exception:  # noqa: BLE001
            return None
    return None


def parse_disease(txt):
    if not txt:
        return ""
    m = re.search(r"disease[_ ]?(\d+)", txt.lower())
    return f"disease_{m.group(1)}" if m else ""


def boot_ci(xs, n=4000, seed=0):
    if not xs:
        return [None, None]
    rng = random.Random(seed)
    k = len(xs)
    m = sorted(sum(xs[rng.randrange(k)] for _ in range(k)) / k for _ in range(n))
    return [round(m[int(0.025 * n)], 4), round(m[int(0.975 * n)], 4)]


def eval_model_family(model, art, key, rng):
    test = [json.loads(l) for l in (RESULTS / art / "test_dx.jsonl").read_text().splitlines() if l.strip()]
    demos_all = [json.loads(l) for l in (RESULTS / art / "sft_train_dx.jsonl").read_text().splitlines() if l.strip()]
    sample = test if len(test) <= N_EVAL else [test[i] for i in rng.sample(range(len(test)), N_EVAL)]
    demos = rng.sample(demos_all, K_FEWSHOT)
    shot_block = "\n\n".join(f"{d['prompt']}\n{d['completion']}" for d in demos)

    per = defaultdict(lambda: {"base": [0, 0], "few": [0, 0]})  # specialty -> hits/total
    n_api_fail = 0
    for r in sample:
        sp = r["specialty"]
        # base (zero-shot)
        ans = parse_disease(gemini(model, r["prompt"], key))
        time.sleep(SLEEP_S)
        if ans == "":
            n_api_fail += 1
        per[sp]["base"][0] += int(ans == r["answer"]); per[sp]["base"][1] += 1
        # few-shot (in-context traversal)
        fp = (f"Here are worked diagnosis examples from other specialties:\n\n{shot_block}\n\n"
              f"Now this case:\n{r['prompt']}")
        ans2 = parse_disease(gemini(model, fp, key))
        time.sleep(SLEEP_S)
        per[sp]["few"][0] += int(ans2 == r["answer"]); per[sp]["few"][1] += 1

    base_by_sp = [v["base"][0] / v["base"][1] for v in per.values() if v["base"][1]]
    few_by_sp = [v["few"][0] / v["few"][1] for v in per.values() if v["few"][1]]
    gain_by_sp = [f - b for b, f in zip(base_by_sp, few_by_sp)]
    n = sum(v["base"][1] for v in per.values())
    return {
        "n_cases": n, "n_specialties": len(per), "n_api_fail": n_api_fail,
        "base": round(sum(base_by_sp) / len(base_by_sp), 4) if base_by_sp else None,
        "base_ci": boot_ci(base_by_sp),
        "fewshot": round(sum(few_by_sp) / len(few_by_sp), 4) if few_by_sp else None,
        "fewshot_ci": boot_ci(few_by_sp),
        "incontext_gain": round(sum(gain_by_sp) / len(gain_by_sp), 4) if gain_by_sp else None,
        "gain_ci": boot_ci(gain_by_sp),
    }


def main():
    key = load_key()
    if not key:
        raise SystemExit("Set GEMINI_API_KEY in .env (get one at https://aistudio.google.com/apikey)")
    rng = random.Random(SEED)
    oracle = {fam: json.loads((RESULTS / "e74_diagnosis.json").read_text())["oracle_ceiling"]
              if fam == "easy" else 0.69 for fam in FAMILIES}  # hard oracle ~0.69 (E75 design)

    results = {"task": "frontier-model (Gemini) check of world-time compute via in-context "
                       "traversal, zero-shot baseline, on held-out diagnosis specialties",
               "config": {"models": MODELS, "n_eval": N_EVAL, "k_fewshot": K_FEWSHOT, "seed": SEED},
               "oracle_ceiling": oracle, "runs": {}}
    for model in MODELS:
        results["runs"][model] = {}
        for fam, art in FAMILIES.items():
            if not (RESULTS / art / "test_dx.jsonl").exists():
                continue
            r = eval_model_family(model, art, key, rng)
            results["runs"][model][fam] = r
            print(f"[gemini] {model} / {fam}: base {r['base']} -> fewshot {r['fewshot']} "
                  f"(in-context gain {r['incontext_gain']}, CI {r['gain_ci']}); "
                  f"oracle {oracle[fam]}; api_fail {r['n_api_fail']}/{r['n_cases']}", flush=True)
    (RESULTS / "e74_gemini.json").write_text(json.dumps(results, indent=2))
    print(f"[gemini] saved {RESULTS / 'e74_gemini.json'}")


if __name__ == "__main__":
    main()
