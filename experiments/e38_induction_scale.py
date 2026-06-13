"""E38 - Does induction-from-traces improve with generator capability?

E37 showed that on equal information (traces only, no rule text) a 7B model
induces dynamics that are scale-invariant but imperfect (~0.43 probe
accuracy), well below the rule-text anchor (1.00). A natural objection: is
0.43 a capability ceiling of the 7B, or fundamental? E38 reruns E37's
equal-information induction across a generator ladder and reports whether the
rules-vs-traces gap closes with scale.

Same protocol as E37 (reuse its machinery): the model is given only observed
transitions, must induce transition() verified by reproduction, then scored
on the in-dist and 10x OOD probe suites. Reasoning models' <think> blocks are
stripped before code extraction.

Resumable: finished models are skipped. Uses the same big models as E35, so
run it when the sprint ladder has freed the model server (avoids VRAM
thrashing between large models).
"""

import json
import random
import re
from pathlib import Path

from openworld import OllamaLLM

from common import (
    SPRINT_PROBES, SPRINT_PROBES_SCALED, require_ollama, save_results,
)
from e37_induction import (
    KS, REPLICATES, SEED, collect, distinct, induce_from_traces,
    probe_acc_code, probe_acc_knn, probe_acc_mlp, train_mlp,
)

MODELS = ["qwen2.5:7b", "qwen3-coder:30b", "gpt-oss:20b"]
# deepseek-r1:14b excluded: its reasoning-trace length makes trace-induction
# impractical on local hardware (timed out at 300s; ~27 min on the first of 6
# calls at 1800s, VRAM-swapping). A methods limitation, not a result.
EXCLUDED = ["deepseek-r1:14b (reasoning-trace length impractical for trace induction on local hardware)"]
RESULTS_PATH = Path(__file__).resolve().parent / "results" / "e38_induction_scale.json"
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def induce_stripped(llm, traces):
    """E37 induction but tolerant of reasoning-model <think> blocks."""
    orig_ask = llm.ask

    def ask(prompt, system=None, **kw):
        return _THINK.sub("", orig_ask(prompt, system=system, **kw))

    llm.ask = ask
    return induce_from_traces(llm, traces)


def run_model(model):
    rows = []
    for k in KS:
        for rep in range(REPLICATES):
            r = random.Random(SEED + rep)
            traces = collect(k, r)
            # reasoning models (deepseek-r1) think long; give them headroom
            llm = OllamaLLM(model=model, temperature=0.4, timeout=1800,
                            options={"seed": SEED + rep})
            code, repro = induce_stripped(llm, traces)
            rows.append({
                "k": k, "replicate": rep, "train_reproduction": repro,
                "probe_in_dist": probe_acc_code(code, SPRINT_PROBES),
                "probe_ood_10x": probe_acc_code(code, SPRINT_PROBES_SCALED),
            })
            print(f"  [{model}] k={k} rep={rep}: repro={repro:.2f} "
                  f"in-dist={rows[-1]['probe_in_dist']:.2f} "
                  f"ood={rows[-1]['probe_ood_10x']:.2f}")
    big = [x for x in rows if x["k"] == max(KS)]
    return {
        "model": model,
        "rows": rows,
        "mean_in_dist_bigK": sum(x["probe_in_dist"] for x in big) / len(big),
        "mean_ood_bigK": sum(x["probe_ood_10x"] for x in big) / len(big),
    }


def main():
    ladder = {}
    if RESULTS_PATH.exists():
        ladder = {m["model"]: m for m in json.loads(RESULTS_PATH.read_text())["ladder"]}
        print(f"[resume] done: {list(ladder)}")
    for model in MODELS:
        if model in ladder:
            continue
        require_ollama(model, timeout=1800)
        print(f"[{model}] inducing from traces")
        ladder[model] = run_model(model)
    save_results("e38_induction_scale", {
        "ks": KS, "replicates": REPLICATES,
        "anchor_note": "rule-text synthesis scores 1.00/1.00 (E37)",
        "excluded_models": EXCLUDED,
        "ladder": [ladder[m] for m in MODELS if m in ladder],
    })
    print(f"\n{'model':<18} in-dist(bigK)  ood(bigK)")
    for m in ladder.values():
        print(f"{m['model']:<18} {m['mean_in_dist_bigK']:>11.2f}  {m['mean_ood_bigK']:>9.2f}")


if __name__ == "__main__":
    main()
