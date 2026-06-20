"""E78 (GPU loop) - runs ON the A100 box. Consumes e78_artifacts/manifest.json, fine-tunes
a LoRA adapter per entry (e73_finetune.py) and evaluates it on the fixed hard held-out test
(e74_eval.py), aggregates per-seed held-out accuracies into
experiments/results/e78_worldtime_power.json, and uploads the (growing) JSON to GCS after
every run so a spot preemption never loses completed work. Wraps each run in try/except: a
failure records null and the loop continues.

  python e78_run.py [--bucket gs://openworld-bench/<run-id>] [--epochs 2]
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from e75_data import D, F, SEED as E75_SEED, N_TRAIN_SPEC, N_TEST_SPEC, make_specialty, oracle_and_floor

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ART = ROOT / "experiments" / "results" / "e78_artifacts"
OUT = ROOT / "experiments" / "results" / "e78_worldtime_power.json"
TMP = ART / "_tmp"


def sh(cmd):
    print("  $ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def eval_acc(base, test, adapter=None):
    TMP.mkdir(parents=True, exist_ok=True)
    out = TMP / "eval.json"
    cmd = [sys.executable, str(HERE / "e74_eval.py"), "--base", base,
           "--test", str(test), "--out", str(out)]
    if adapter:
        cmd += ["--adapter", str(adapter)]
    sh(cmd)
    return json.loads(out.read_text())["accuracy"]


def finetune(base, data, seed, epochs):
    adapter = TMP / "adapter"
    if adapter.exists():
        shutil.rmtree(adapter)
    sh([sys.executable, str(HERE / "e73_finetune.py"), "--base", base,
        "--data", str(data), "--out", str(adapter), "--epochs", str(epochs),
        "--seed", str(73 + seed)])
    return adapter


def mean_ci(vals):
    """Mean and a t-based 95% CI across seeds (small-sample honest)."""
    v = [x for x in vals if x is not None]
    if not v:
        return {"mean": None, "ci": [None, None], "seeds": vals}
    a = np.array(v, float)
    m, sd, n = a.mean(), a.std(ddof=1) if len(a) > 1 else 0.0, len(a)
    # t_0.975 for df=1,2 -> 12.71, 4.30; fall back to 1.96 for larger n
    tcrit = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78}.get(n - 1, 1.96)
    half = tcrit * sd / np.sqrt(n) if n > 1 else 0.0
    return {"mean": round(m, 4), "ci": [round(m - half, 4), round(m + half, 4)],
            "std": round(sd, 4), "n_seeds": n, "seeds": vals}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", default="")
    ap.add_argument("--epochs", type=float, default=2.0)
    args = ap.parse_args()

    manifest = json.loads((ART / "manifest.json").read_text())
    test = ART / "test_dx.jsonl"

    # oracle ceiling / prior-only floor over the fixed E75 held-out specialties
    rng = np.random.RandomState(E75_SEED)
    test_specs = [make_specialty(E75_SEED + N_TRAIN_SPEC + i) for i in range(N_TEST_SPEC)]
    o, fl = zip(*(oracle_and_floor(s, 300, rng) for s in test_specs))
    oracle, floor = round(float(np.mean(o)), 4), round(float(np.mean(fl)), 4)

    res = {
        "task": "world-time compute, powered: multi-seed world-count scaling (with CIs) + "
                "verified-vs-noisy label ablation, on held-out hard diagnosis specialties",
        "config": {"seeds": sorted({m["seed"] for m in manifest}), "epochs": args.epochs,
                   "test_cases": sum(1 for _ in test.read_text().splitlines() if _.strip())},
        "oracle_ceiling": oracle, "prior_only_floor": floor,
        "base_acc": {}, "scaling_raw": {}, "ablation_raw": {},
    }

    def upload():
        OUT.write_text(json.dumps(res, indent=2))
        if args.bucket:
            try:
                subprocess.run(["gcloud", "storage", "cp", str(OUT),
                                f"{args.bucket}/e78_worldtime_power.json"], check=False)
            except Exception as e:  # noqa: BLE001
                print(f"  [upload] skipped: {e}", flush=True)

    # base (zero-shot) accuracy per distinct base model
    for base in sorted({m["base"] for m in manifest}):
        try:
            res["base_acc"][base] = eval_acc(base, test)
            print(f"[base] {base}: {res['base_acc'][base]}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[base] {base} FAILED: {e}", flush=True)
            res["base_acc"][base] = None
        upload()

    # every fine-tune + eval
    for i, m in enumerate(manifest):
        tag = (f"{m['kind']}/{m['size']}/seed{m['seed']}/"
               + (f"N{m['N']}" if m["kind"] == "scaling" else m["cond"]))
        print(f"\n=== [{i + 1}/{len(manifest)}] {tag} ({m['n_examples']} ex) ===", flush=True)
        acc = None
        try:
            adapter = finetune(m["base"], ART / m["file"], m["seed"], args.epochs)
            acc = eval_acc(m["base"], test, adapter)
            print(f"  -> held-out acc {acc}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED: {e}", flush=True)
        if m["kind"] == "scaling":
            d = res["scaling_raw"].setdefault(m["size"], {}).setdefault(str(m["N"]), {})
            d[str(m["seed"])] = acc
        else:
            d = res["ablation_raw"].setdefault(m["cond"], {})
            d[str(m["seed"])] = acc
            res["ablation_raw"].setdefault("_noise", {})[m["cond"]] = m["noise"]
            res["ablation_raw"].setdefault("_realized_flip", {})[m["cond"]] = m["realized_flip"]
        upload()

    # summaries with CIs across seeds
    res["scaling"] = {}
    for size, byN in res["scaling_raw"].items():
        res["scaling"][size] = {n: mean_ci(list(d.values())) for n, d in byN.items()}
    res["ablation"] = {}
    for cond, d in res["ablation_raw"].items():
        if cond.startswith("_"):
            continue
        res["ablation"][cond] = mean_ci(list(d.values()))
    upload()
    print("\n[e78-run] done. summary:", flush=True)
    print(json.dumps({"oracle": oracle, "floor": floor, "base": res["base_acc"],
                      "ablation": res["ablation"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
