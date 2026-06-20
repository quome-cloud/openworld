"""E80 (engine) - generic world-time-compute MECHANISM test on a real domain.

A domain module (e80_<domain>.py) exposes:
  build_worlds() -> {world_name: {"classes": [str,...], "rows": [{"prompt","label"}, ...]}}
  CONFIG = {"ladder": [..N..], "abl_n": N, "abl_noise": [0.0,..,1.0], "cap": int,
            "n_test": int, "seeds": [..], "base": "Qwen/Qwen2.5-0.5B-Instruct"}

We run two tests, reusing e73_finetune/e74_eval, on STRICTLY held-out worlds:
  (a) world-count ladder  -- does held-out accuracy rise with the number of train worlds?
  (b) verified-vs-noisy label ablation -- does it collapse as the REAL labels are corrupted?

Partial results upload to GCS after every run. Usage on the box:
  python3 e80_common.py --domain genomics --bucket gs://openworld-bench/e80-genomics
"""

import argparse
import importlib
import json
import math
import random
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ART = ROOT / "experiments" / "results" / "e80_artifacts"
TMP = ART / "_tmp"


def sh(cmd):
    print("  $ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True)


def _emit_sft(worlds, names, cap, rng, noise=0.0):
    rows = []
    for nm in names:
        w = worlds[nm]
        idx = list(range(len(w["rows"])))
        rng.shuffle(idx)
        for i in idx[:cap]:
            lab = w["rows"][i]["label"]
            if noise > 0 and rng.random() < noise and len(w["classes"]) > 1:
                lab = rng.choice([c for c in w["classes"] if c != w["rows"][i]["label"]])
            rows.append({"prompt": w["rows"][i]["prompt"], "completion": lab})
    rng.shuffle(rows)
    return rows


def _emit_test(worlds, names, cap, rng):
    rows = []
    for nm in names:
        w = worlds[nm]
        idx = list(range(len(w["rows"])))
        rng.shuffle(idx)
        for i in idx[:cap]:
            rows.append({"prompt": w["rows"][i]["prompt"], "answer": w["rows"][i]["label"],
                         "specialty": nm})
    return rows


def _write(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def _finetune(base, data, seed, epochs):
    adapter = TMP / "adapter"
    if adapter.exists():
        shutil.rmtree(adapter)
    sh([sys.executable, str(HERE / "e73_finetune.py"), "--base", base, "--data", str(data),
        "--out", str(adapter), "--epochs", str(epochs), "--seed", str(800 + seed),
        "--batch", "16", "--grad_accum", "1"])
    return adapter


def _eval(base, test, adapter=None):
    out = TMP / "eval.json"
    cmd = [sys.executable, str(HERE / "e74_eval.py"), "--base", base, "--test", str(test),
           "--out", str(out), "--eval_batch", "256"]
    if adapter:
        cmd += ["--adapter", str(adapter)]
    sh(cmd)
    return json.loads(out.read_text())["accuracy"]


def _mean_ci(vals):
    v = [x for x in vals if x is not None]
    if not v:
        return {"mean": None, "ci": [None, None], "seeds": vals}
    m = sum(v) / len(v)
    n = len(v)
    sd = (sum((x - m) ** 2 for x in v) / (n - 1)) ** 0.5 if n > 1 else 0.0
    t = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78}.get(n - 1, 1.96)
    h = t * sd / math.sqrt(n) if n > 1 else 0.0
    return {"mean": round(m, 4), "ci": [round(m - h, 4), round(m + h, 4)],
            "std": round(sd, 4), "n_seeds": n, "seeds": [round(x, 4) if x is not None else None for x in vals]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--bucket", default="")
    ap.add_argument("--epochs", type=float, default=2.0)
    args = ap.parse_args()
    TMP.mkdir(parents=True, exist_ok=True)

    mod = importlib.import_module(f"e80_{args.domain}")
    cfg = mod.CONFIG
    worlds = mod.build_worlds()
    names = sorted(worlds)
    print(f"[e80/{args.domain}] {len(names)} worlds; "
          f"sizes min/median/max "
          f"{min(len(worlds[n]['rows']) for n in names)}/"
          f"{sorted(len(worlds[n]['rows']) for n in names)[len(names)//2]}/"
          f"{max(len(worlds[n]['rows']) for n in names)}", flush=True)

    # fixed held-out worlds (split by world, no leakage)
    split_rng = random.Random(80)
    order = names[:]
    split_rng.shuffle(order)
    n_test = min(cfg["n_test"], len(order) // 3)
    test_names = order[:n_test]
    train_pool = order[n_test:]
    cap = cfg["cap"]
    ladder = [n for n in cfg["ladder"] if n <= len(train_pool)]
    abl_n = min(cfg["abl_n"], len(train_pool))

    res = {"domain": args.domain, "n_worlds": len(names), "n_test_worlds": n_test,
           "n_train_pool": len(train_pool), "config": {k: cfg[k] for k in
           ("ladder", "abl_n", "abl_noise", "cap", "seeds")},
           "base_acc": None, "ladder_raw": {}, "ablation_raw": {}}
    base = cfg["base"]

    def upload():
        out = ROOT / "experiments" / "results" / f"e80_{args.domain}.json"
        # finalize summaries
        res["ladder"] = {str(n): _mean_ci(list(d.values())) for n, d in res["ladder_raw"].items()}
        res["ablation"] = {t: _mean_ci(list(d.values())) for t, d in res["ablation_raw"].items()}
        out.write_text(json.dumps(res, indent=2))
        if args.bucket:
            subprocess.run(["gcloud", "storage", "cp", str(out),
                            f"{args.bucket}/e80_{args.domain}.json"], check=False)

    # fixed test set (held-out worlds)
    test_rng = random.Random(81)
    test_file = TMP / "test.jsonl"
    _write(test_file, _emit_test(worlds, test_names, cap, test_rng))

    try:
        res["base_acc"] = _eval(base, test_file)
        print(f"[base] {res['base_acc']}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[base] FAILED {e}", flush=True)
    upload()

    # (a) world-count ladder
    for seed in cfg["seeds"]:
        rng = random.Random(100 + seed)
        pool = train_pool[:]
        rng.shuffle(pool)
        for n in ladder:
            tag = str(n)
            acc = None
            try:
                _write(TMP / "sft.jsonl", _emit_sft(worlds, pool[:n], cap, random.Random(200 + seed)))
                acc = _eval(base, test_file, _finetune(base, TMP / "sft.jsonl", seed, args.epochs))
                print(f"[ladder seed{seed} N={n}] {acc}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[ladder seed{seed} N={n}] FAILED {e}", flush=True)
            res["ladder_raw"].setdefault(tag, {})[str(seed)] = acc
            upload()

    # (b) verified-vs-noisy ablation at fixed world count
    for seed in cfg["seeds"]:
        rng = random.Random(300 + seed)
        pool = train_pool[:]
        rng.shuffle(pool)
        abl_names = pool[:abl_n]
        for p in cfg["abl_noise"]:
            tag = f"n{int(p * 100):02d}"
            acc = None
            try:
                _write(TMP / "sft.jsonl",
                       _emit_sft(worlds, abl_names, cap, random.Random(400 + seed), noise=p))
                acc = _eval(base, test_file, _finetune(base, TMP / "sft.jsonl", seed, args.epochs))
                print(f"[ablation seed{seed} p={p}] {acc}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[ablation seed{seed} p={p}] FAILED {e}", flush=True)
            res["ablation_raw"].setdefault(tag, {})[str(seed)] = acc
            upload()

    upload()
    print(f"[e80/{args.domain}] done.\n", json.dumps({"base": res["base_acc"],
          "ladder": res.get("ladder"), "ablation": res.get("ablation")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
