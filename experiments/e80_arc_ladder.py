"""E80-ARC ladder (cross-task transfer): does training on MORE real ARC worlds build a
transferable induction skill that helps on STRICTLY held-out ARC worlds?

For N in a world-count ladder, LoRA-SFT Qwen-7B on augmented rows from N training tasks, then
exact-match eval on a fixed held-out set of EVALUATION tasks (never trained, novel rules). A
rising curve = cross-world transfer; flat = ARC's per-task novelty defeats pure transfer (the
honest, expected-hard baseline that motivates test-time training, e80_arc_ttt.py).

Partial results upload to GCS after every rung. Runs as a subprocess driver on the box.
  python3 e80_arc_ladder.py --data /root/ARC-AGI/data --bucket gs://openworld-bench/arc-ladder
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

import e80_arc as A

HERE = Path(__file__).resolve().parent
TMP = HERE / "results" / "arc_artifacts" / "_tmp"
BASE = "Qwen/Qwen2.5-7B-Instruct"
LADDER = [25, 50, 100, 200, 400]
ROWS_PER_TASK = 24      # cap augmented rows/task so the top rung stays tractable
N_AUG = 8
N_EVAL = 100            # fixed held-out evaluation tasks
MAXLEN = 2048


def sh(cmd):
    print("  $ " + " ".join(map(str, cmd)), flush=True)
    subprocess.run(cmd, check=True)


def write(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def finetune(data, out, seed):
    if out.exists():
        shutil.rmtree(out)
    sh([sys.executable, str(HERE / "e73_finetune.py"), "--base", BASE, "--data", str(data),
        "--out", str(out), "--epochs", "2", "--batch", "4", "--grad_accum", "4",
        "--max_length", str(MAXLEN), "--load_4bit", "--seed", str(seed)])
    return out


def evaluate(test_file, adapter=None):
    out = TMP / "eval.json"
    cmd = [sys.executable, str(HERE / "e80_arc_eval.py"), "--base", BASE, "--test", str(test_file),
           "--out", str(out), "--max_new_tokens", "1024", "--eval_batch", "8"]
    if adapter:
        cmd += ["--adapter", str(adapter)]
    sh(cmd)
    return json.loads(out.read_text())["accuracy"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="ARC-AGI data dir (has training/ and evaluation/)")
    ap.add_argument("--bucket", default="")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    TMP.mkdir(parents=True, exist_ok=True)

    train = A.load_tasks(str(Path(args.data) / "training"))
    ev = A.load_tasks(str(Path(args.data) / "evaluation"))
    train_ids = sorted(train)
    rng = np.random.default_rng(100 + args.seed)
    rng.shuffle(train_ids)

    # fixed held-out eval set (real eval tasks within the char budget)
    ev_ids = [t for t in sorted(ev) if A.task_eval_example(ev[t]) is not None][:N_EVAL]
    eval_rows = [dict(A.task_eval_example(ev[t]), tid=t) for t in ev_ids]
    eval_file = TMP / "eval_set.jsonl"
    write(eval_file, eval_rows)
    print(f"[arc-ladder] {len(train_ids)} train worlds, eval on {len(eval_rows)} held-out worlds",
          flush=True)

    res = {"experiment": "arc-ladder", "base": BASE, "n_eval": len(eval_rows),
           "ladder": LADDER, "rows_per_task": ROWS_PER_TASK, "seed": args.seed,
           "base_acc": None, "rung_acc": {}}

    def upload():
        out = HERE / "results" / "e80_arc_ladder.json"
        out.write_text(json.dumps(res, indent=2))
        if args.bucket:
            subprocess.run(["gcloud", "storage", "cp", str(out),
                            f"{args.bucket}/e80_arc_ladder.json"], check=False)

    try:
        res["base_acc"] = evaluate(eval_file)
        print(f"[base zero-shot] {res['base_acc']}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[base] FAILED {e}", flush=True)
    upload()

    for n in [n for n in LADDER if n <= len(train_ids)]:
        acc = None
        try:
            rows = []
            for tid in train_ids[:n]:
                r = A.task_to_sft_rows(train[tid], n_aug=N_AUG,
                                       rng=np.random.default_rng(hash(tid) % 2**32))
                rng.shuffle(r)
                rows += r[:ROWS_PER_TASK]
            rng.shuffle(rows)
            write(TMP / "sft.jsonl", rows)
            print(f"[rung N={n}] {len(rows)} SFT rows", flush=True)
            adapter = finetune(TMP / "sft.jsonl", TMP / "adapter", 800 + args.seed)
            acc = evaluate(eval_file, adapter)
            print(f"[rung N={n}] exact-match {acc}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[rung N={n}] FAILED {e}", flush=True)
        res["rung_acc"][str(n)] = acc
        upload()

    print("[arc-ladder] done\n" + json.dumps(
        {"base": res["base_acc"], "rungs": res["rung_acc"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
