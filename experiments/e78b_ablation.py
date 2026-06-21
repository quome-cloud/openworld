"""E78b - EXTENDED verified-vs-noisy label ablation: trace the full dose-response of held-out
generalization vs training-label corruption (0/15/30/45/60/80%) at fixed N=256, so we can see
exactly where label exactness starts to matter (the 0/15/30 sweep was too coarse -- 15% washed
out, only 30% moved). Same worlds/patients/prompts across all levels; only the labels differ;
held-out test labels are always exact. Self-contained: data-gen + GPU loop + GCS upload.

  python3 e78b_ablation.py [--bucket gs://openworld-bench/<run>] [--epochs 2]
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from e75_data import (SEED as E75_SEED, N_TRAIN_SPEC, N_TEST_SPEC, TRAIN_PATIENTS,
                      make_specialty, profiles, oracle_and_floor)
from e78_data import sample_patients, row_of, noisy_label, write_jsonl

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ART = ROOT / "experiments" / "results" / "e78b_artifacts"
OUT = ROOT / "experiments" / "results" / "e78b_ablation.json"
TMP = ART / "_tmp"

NOISE = [0.0, 0.15, 0.30, 0.45, 0.60, 0.80, 1.0]   # 100% = every label a (wrong) confusable disease
SEEDS = [0, 1, 2]
N = 256
SEED_BLOCK = 10000          # same train-world block as the E78 ablation (worlds match)
BASE = "Qwen/Qwen2.5-0.5B-Instruct"


def sh(cmd):
    print("  $ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def eval_acc(base, test, adapter=None):
    TMP.mkdir(parents=True, exist_ok=True)
    o = TMP / "eval.json"
    cmd = [sys.executable, str(HERE / "e74_eval.py"), "--base", base, "--test", str(test),
           "--out", str(o), "--eval_batch", "256"]
    if adapter:
        cmd += ["--adapter", str(adapter)]
    sh(cmd)
    return json.loads(o.read_text())["accuracy"]


def finetune(base, data, seed, epochs):
    a = TMP / "adapter"
    if a.exists():
        shutil.rmtree(a)
    sh([sys.executable, str(HERE / "e73_finetune.py"), "--base", base, "--data", str(data),
        "--out", str(a), "--epochs", str(epochs), "--seed", str(178 + seed),
        "--batch", "16", "--grad_accum", "1"])
    return a


def mean_ci(vals):
    v = [x for x in vals if x is not None]
    if not v:
        return {"mean": None, "ci": [None, None], "seeds": vals}
    a = np.array(v, float)
    m, sd, n = a.mean(), (a.std(ddof=1) if len(a) > 1 else 0.0), len(a)
    t = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78}.get(n - 1, 1.96)
    h = t * sd / np.sqrt(n) if n > 1 else 0.0
    return {"mean": round(m, 4), "ci": [round(m - h, 4), round(m + h, 4)],
            "std": round(sd, 4), "n_seeds": n, "seeds": vals}


def generate():
    """Same patients per world across all noise levels; only labels differ."""
    ART.mkdir(parents=True, exist_ok=True)
    e75test = ROOT / "experiments" / "results" / "e75_artifacts" / "test_dx.jsonl"
    assert e75test.exists(), "run e75_data.py first"
    shutil.copy(e75test, ART / "test_dx.jsonl")

    manifest = []
    for s in SEEDS:
        rng = np.random.RandomState(E75_SEED + 200 + s)
        specs = [make_specialty(SEED_BLOCK * (s + 1) + i) for i in range(N)]
        per_spec = []
        for spec in specs:
            prof = profiles(spec["M"])
            M = np.clip(spec["M"], 1e-4, 1 - 1e-4)
            logM, log1M = np.log(M), np.log(1 - M)
            logpri = np.log(spec["prior"] + 1e-12)
            pts = sample_patients(spec, TRAIN_PATIENTS, rng)   # shared across noise levels
            per_spec.append((prof, logpri, logM, log1M, pts))
        for p in NOISE:
            nrng = np.random.RandomState(7000 + int(p * 100) + s)  # independent noise stream
            rows, flips, tot = [], 0, 0
            for prof, logpri, logM, log1M, pts in per_spec:
                for d, x in pts:
                    lab = d if p == 0 else noisy_label(None, d, x, p, nrng, logpri, logM, log1M)
                    rows.append(row_of(prof, x, lab))
                    flips += int(lab != d)
                    tot += 1
            tag = f"n{int(p * 100):02d}"
            f = ART / f"abl_{tag}_s{s}.jsonl"
            write_jsonl(f, rows)
            manifest.append({"noise": p, "seed": s, "file": f.name,
                             "realized_flip": round(flips / tot, 4), "base": BASE})
    (ART / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[e78b] generated {len(manifest)} SFT sets "
          f"({len(NOISE)} noise x {len(SEEDS)} seeds) at N={N}")
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", default="")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--gen_only", action="store_true")
    args = ap.parse_args()

    manifest = generate()
    if args.gen_only:
        for m in manifest:
            print(f"  noise {m['noise']:>4} seed {m['seed']}: realized flip {m['realized_flip']}")
        return

    test = ART / "test_dx.jsonl"
    rng = np.random.RandomState(E75_SEED)
    tspecs = [make_specialty(E75_SEED + N_TRAIN_SPEC + i) for i in range(N_TEST_SPEC)]
    o, fl = zip(*(oracle_and_floor(s, 300, rng) for s in tspecs))
    res = {"task": "extended verified-vs-noisy label ablation (dose-response) at N=256",
           "config": {"noise": NOISE, "seeds": SEEDS, "N": N, "epochs": args.epochs},
           "oracle_ceiling": round(float(np.mean(o)), 4),
           "prior_only_floor": round(float(np.mean(fl)), 4),
           "realized_flip": {}, "raw": {}}

    def upload():
        OUT.write_text(json.dumps(res, indent=2))
        if args.bucket:
            subprocess.run(["gcloud", "storage", "cp", str(OUT),
                            f"{args.bucket}/e78b_ablation.json"], check=False)

    for i, m in enumerate(manifest):
        tag = f"n{int(m['noise'] * 100):02d}"
        res["realized_flip"][tag] = m["realized_flip"]
        print(f"\n=== [{i + 1}/{len(manifest)}] noise={m['noise']} seed={m['seed']} "
              f"(flip {m['realized_flip']}) ===", flush=True)
        acc = None
        try:
            adapter = finetune(m["base"], ART / m["file"], m["seed"], args.epochs)
            acc = eval_acc(m["base"], test, adapter)
            print(f"  -> held-out acc {acc}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED: {e}", flush=True)
        res["raw"].setdefault(tag, {})[str(m["seed"])] = acc
        upload()

    res["summary"] = {tag: mean_ci(list(d.values())) for tag, d in res["raw"].items()}
    upload()
    print("\n[e78b] done. dose-response (held-out acc vs label corruption):", flush=True)
    print(json.dumps(res["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
