#!/usr/bin/env python3
"""Project our SFT pairs into the MLX-LM LoRA data layout.

Reads sft/<v>/train.jsonl (lines = {"messages":[...], "instance_id":...}) and
writes <out>/train.jsonl + <out>/valid.jsonl as MLX chat-format records
({"messages":[...]} only — MLX is strict about extra keys). Deterministic
valid split (every Nth line) so the run is reproducible. Stdlib only.
"""
import argparse
import json
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser(prog="to_mlx_data")
    ap.add_argument("sft_train", help="sft/<v>/train.jsonl")
    ap.add_argument("--out", required=True, help="MLX data dir to write")
    ap.add_argument("--valid-every", type=int, default=8,
                    help="put every Nth pair in valid.jsonl (default 8)")
    args = ap.parse_args(argv)

    rows = [json.loads(l) for l in Path(args.sft_train).read_text().splitlines() if l.strip()]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train, valid = [], []
    for i, r in enumerate(rows):
        rec = {"messages": r["messages"]}
        (valid if (i + 1) % args.valid_every == 0 else train).append(rec)
    if not valid:                      # tiny sets: guarantee a non-empty valid
        valid.append(train.pop())

    for name, recs in (("train", train), ("valid", valid)):
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
            for rec in recs:
                fh.write(json.dumps(rec) + "\n")

    print(f"[mlx-data] {len(train)} train / {len(valid)} valid -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
