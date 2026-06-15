#!/usr/bin/env python3
"""Format harvested verified traces into an SFT training set + held-out eval split.

NOT part of the zero-dependency `openworld` framework — this is experiment-side
tooling for the verified-trace distillation flywheel. Stdlib only.

Input: one or more `*.traces.jsonl` produced by `bench ... run --log-traces DIR`
(each line = one attempt: instance_id, condition, prompt, completion, patch,
passed, seed, model, ...).

What it does:
  1. Keep only `passed == True` records — every kept patch provably passed the
     hidden tests, so the labels are clean (the whole point of verified traces).
  2. Split instances into TRAIN vs HELD-OUT *by whole instance* (deterministic
     hash split) so eval measures generalization, never memorization.
  3. For each TRAIN instance, emit SFT pairs whose INPUT is that instance's
     canonical single-shot (base) prompt and whose TARGET is each distinct
     verified patch found for it — including patches the teacher only reached
     *in-world*. That is the distillation move: iterate-to-find at harvest time,
     answer single-shot after training.

Outputs (in --out-dir):
  train.jsonl          one {"messages":[...], "instance_id":...} per SFT pair
  heldout_instances.json   the eval instance ids (no patches — run the model)
  manifest.json        counts, split, source sha256s (reproducibility)
"""
import argparse
import glob
import hashlib
import json
from pathlib import Path


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def _heldout(instance_ids, frac, salt="owsb-distill-v1"):
    """Deterministic, reproducible instance-level split (no RNG state).

    Rank instances by sha256(salt+id); the lowest `frac` fraction are held out.
    Same inputs -> same split forever; independent of file/seed order.
    """
    ordered = sorted(
        instance_ids,
        key=lambda i: hashlib.sha256(f"{salt}:{i}".encode()).hexdigest(),
    )
    n_hold = max(1, round(len(ordered) * frac))
    return set(ordered[:n_hold])


def main(argv=None):
    ap = argparse.ArgumentParser(prog="format_traces")
    ap.add_argument("traces", nargs="+",
                    help="trace jsonl files or globs (bench --log-traces output)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--heldout-frac", type=float, default=0.3,
                    help="fraction of instances reserved for eval (default 0.3)")
    ap.add_argument("--split-salt", default="owsb-distill-v1",
                    help="salt for the deterministic instance split (vary it to "
                         "test split robustness)")
    args = ap.parse_args(argv)

    files = []
    for pat in args.traces:
        files.extend(sorted(glob.glob(pat)))
    files = [Path(f) for f in files]
    if not files:
        raise SystemExit("no trace files matched")

    records = []
    for f in files:
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))

    # canonical base (single-shot) prompt per instance — present for every
    # instance because single-shot is always logged (pass or fail).
    base_prompt = {}
    system_by_inst = {}
    for r in records:
        if r["condition"] == "single_shot" and r["instance_id"] not in base_prompt:
            base_prompt[r["instance_id"]] = r["prompt"]
            system_by_inst[r["instance_id"]] = r.get("system", "")

    # distinct verified patches per instance (both conditions)
    verified = {}  # instance_id -> set(patch)
    for r in records:
        if r["passed"] and r["patch"].strip():
            verified.setdefault(r["instance_id"], set()).add(r["patch"])

    all_instances = sorted({r["instance_id"] for r in records})
    heldout = _heldout(all_instances, args.heldout_frac, salt=args.split_salt)
    train_instances = [i for i in all_instances if i not in heldout]

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    n_pairs = 0
    train_only_no_patch = []
    with (out / "train.jsonl").open("w", encoding="utf-8") as fh:
        for inst in train_instances:
            patches = verified.get(inst)
            if not patches:
                train_only_no_patch.append(inst)
                continue
            for patch in sorted(patches):
                target = f"```python\n{patch}\n```"
                ex = {
                    "instance_id": inst,
                    "messages": [
                        {"role": "system", "content": system_by_inst.get(inst, "")},
                        {"role": "user", "content": base_prompt[inst]},
                        {"role": "assistant", "content": target},
                    ],
                }
                fh.write(json.dumps(ex) + "\n")
                n_pairs += 1

    (out / "heldout_instances.json").write_text(
        json.dumps(sorted(heldout), indent=2), encoding="utf-8")

    manifest = {
        "source_traces": [{"path": str(f), "sha256": _sha256_file(f)} for f in files],
        "n_records": len(records),
        "n_instances_total": len(all_instances),
        "heldout_frac": args.heldout_frac,
        "n_heldout": len(heldout),
        "n_train_instances": len(train_instances),
        "n_train_instances_with_patches": len(train_instances) - len(train_only_no_patch),
        "train_instances_without_verified_patch": sorted(train_only_no_patch),
        "n_sft_pairs": n_pairs,
        "split_salt": args.split_salt,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[traces] {len(records)} records from {len(files)} file(s)")
    print(f"[split]  {len(all_instances)} instances -> "
          f"{len(train_instances)} train / {len(heldout)} held-out (eval)")
    print(f"[sft]    {n_pairs} verified pairs from "
          f"{manifest['n_train_instances_with_patches']} train instances")
    if train_only_no_patch:
        print(f"[warn]   {len(train_only_no_patch)} train instances had NO verified "
              f"patch (teacher never solved them): {train_only_no_patch}")
    print(f"[out]    {out}/train.jsonl, heldout_instances.json, manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
