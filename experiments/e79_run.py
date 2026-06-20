"""E79 (GPU loop) - world-as-a-judge de-risk: fine-tune + eval each (arm, N, seed) entry and
aggregate held-out accuracy by arm and world count, so we can read off whether CURATED
(judge) selection beats RANDOM at matched N. Uploads partial results to GCS after every run
(spot-preemption safety). Mirrors e78_run.

  python3 e79_run.py [--bucket gs://openworld-bench/<run-id>] [--epochs 2]
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from e75_data import SEED as E75_SEED, N_TRAIN_SPEC, N_TEST_SPEC, make_specialty, oracle_and_floor

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ART = ROOT / "experiments" / "results" / "e79_artifacts"
OUT = ROOT / "experiments" / "results" / "e79_worldjudge.json"
TMP = ART / "_tmp"


def sh(cmd):
    print("  $ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def eval_acc(base, test, adapter=None):
    TMP.mkdir(parents=True, exist_ok=True)
    out = TMP / "eval.json"
    cmd = [sys.executable, str(HERE / "e74_eval.py"), "--base", base,
           "--test", str(test), "--out", str(out), "--eval_batch", "256"]
    if adapter:
        cmd += ["--adapter", str(adapter)]
    sh(cmd)
    return json.loads(out.read_text())["accuracy"]


def finetune(base, data, seed, epochs):
    adapter = TMP / "adapter"
    if adapter.exists():
        shutil.rmtree(adapter)
    # large batch + no grad-accum: a 0.5B/1.5B LoRA barely touches a 40GB A100 at bs=8.
    sh([sys.executable, str(HERE / "e73_finetune.py"), "--base", base, "--data", str(data),
        "--out", str(adapter), "--epochs", str(epochs), "--seed", str(79 + seed),
        "--batch", "64", "--grad_accum", "1"])
    return adapter


def mean_ci(vals):
    v = [x for x in vals if x is not None]
    if not v:
        return {"mean": None, "ci": [None, None], "seeds": vals}
    a = np.array(v, float)
    m, sd, n = a.mean(), (a.std(ddof=1) if len(a) > 1 else 0.0), len(a)
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

    rng = np.random.RandomState(E75_SEED)
    test_specs = [make_specialty(E75_SEED + N_TRAIN_SPEC + i) for i in range(N_TEST_SPEC)]
    o, fl = zip(*(oracle_and_floor(s, 300, rng) for s in test_specs))
    oracle, floor = round(float(np.mean(o)), 4), round(float(np.mean(fl)), 4)

    res = {"task": "world-as-a-judge: curated vs diversity vs random world selection at "
                   "matched N, held-out hard diagnosis generalization",
           "config": {"arms": sorted({m["arm"] for m in manifest}),
                      "N_points": sorted({m["N"] for m in manifest}),
                      "seeds": sorted({m["seed"] for m in manifest}), "epochs": args.epochs},
           "oracle_ceiling": oracle, "prior_only_floor": floor,
           "sel_oracle_mean": {}, "base_acc": None, "raw": {}}

    def upload():
        OUT.write_text(json.dumps(res, indent=2))
        if args.bucket:
            subprocess.run(["gcloud", "storage", "cp", str(OUT),
                            f"{args.bucket}/e79_worldjudge.json"], check=False)

    base = manifest[0]["base"]
    try:
        res["base_acc"] = eval_acc(base, test)
        print(f"[base] {base}: {res['base_acc']}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[base] FAILED: {e}", flush=True)
    upload()

    for i, m in enumerate(manifest):
        tag = f"{m['arm']}/N{m['N']}/seed{m['seed']}"
        res["sel_oracle_mean"][f"{m['arm']}/N{m['N']}"] = m["sel_oracle_mean"]
        print(f"\n=== [{i + 1}/{len(manifest)}] {tag} ===", flush=True)
        acc = None
        try:
            adapter = finetune(m["base"], ART / m["file"], m["seed"], args.epochs)
            acc = eval_acc(m["base"], test, adapter)
            print(f"  -> held-out acc {acc}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED: {e}", flush=True)
        res["raw"].setdefault(m["arm"], {}).setdefault(str(m["N"]), {})[str(m["seed"])] = acc
        upload()

    res["summary"] = {arm: {n: mean_ci(list(d.values())) for n, d in byN.items()}
                      for arm, byN in res["raw"].items()}
    upload()
    print("\n[e79-run] done. curated vs random by N:", flush=True)
    print(json.dumps({"oracle": oracle, "base": res["base_acc"], "summary": res["summary"],
                      "sel_oracle_mean": res["sel_oracle_mean"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
