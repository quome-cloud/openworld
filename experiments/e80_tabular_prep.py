"""E80 tabular prep: build a real tabular domain (OpenML-CC18) in the uniform world format, so
the world-time-compute law can be tested OUT-OF-SAMPLE on a modeling problem maximally different
from grids/lists/algorithms/vision.

Each dataset = a world; each row = an example: input = "feat=val, ..." (capped features),
output = the class label. Emits {world,input,output} JSONL consumable by e80_text_ttt / e80_proxy_ll.

  python3 e80_tabular_prep.py --out tabular_worlds.jsonl --n_datasets 30 --rows 60
"""

import argparse
import json
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tabular_worlds.jsonl")
    ap.add_argument("--n_datasets", type=int, default=30)
    ap.add_argument("--rows", type=int, default=60)
    ap.add_argument("--max_feats", type=int, default=12)
    args = ap.parse_args()

    import openml
    suite = openml.study.get_suite(99)            # OpenML-CC18
    ids = list(suite.data)
    random.Random(80).shuffle(ids)

    rng = random.Random(80)
    rows_out, kept = [], 0
    for did in ids:
        if kept >= args.n_datasets:
            break
        try:
            ds = openml.datasets.get_dataset(did, download_data=True,
                                             download_qualities=False, download_features_meta_data=False)
            X, y, _, names = ds.get_data(target=ds.default_target_attribute)
            if y is None or X is None or len(X) < args.rows:
                continue
            feats = [c for c in X.columns][:args.max_feats]
            idx = list(range(len(X)))
            rng.shuffle(idx)
            n = 0
            for i in idx[:args.rows * 2]:
                vals = []
                for f in feats:
                    v = X.iloc[i][f]
                    try:
                        v = round(float(v), 3)
                    except (ValueError, TypeError):
                        v = str(v)
                    vals.append(f"{f}={v}")
                lab = str(y.iloc[i])
                rows_out.append({"world": f"tab_{ds.name[:20]}", "input": ", ".join(vals),
                                 "output": lab})
                n += 1
                if n >= args.rows:
                    break
            kept += 1
            print(f"  {ds.name}: {n} rows, {len(feats)} feats", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  dataset {did} skipped: {e}", flush=True)
    open(args.out, "w").write("\n".join(json.dumps(r) for r in rows_out) + "\n")
    from collections import Counter
    c = Counter(r["world"] for r in rows_out)
    print(f"tabular worlds: {len(c)}; total rows {len(rows_out)} -> {args.out}")


if __name__ == "__main__":
    main()
