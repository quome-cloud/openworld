#!/usr/bin/env python3
"""run_eval.py — E-market v2 pipeline runner.

RESEARCH ONLY — paper trading / analysis. No live trading.

Stages (order matters for pre-registration discipline):
  --stage features : compute + cache the feature matrix (data/features.parquet)
  --stage dev      : DEV sanity evaluation (walk-forward predictions on
                     2025-02-01..2026-01-31, pre-cutoff). Purpose: verify the
                     pipeline computes what the causal stories say, and give
                     an order-of-magnitude read. NOT the scored result.
  --stage post     : POST-CUTOFF scored evaluation (2026-02-01..end), full
                     gate: net LS Sharpe, 500-shuffle cross-sectional
                     permutation p, benchmark variant, bootstrap CI, per
                     variant: FULL, SOLO_<fam>, DROP_<fam>.
                     Refuses to run unless results/predictions_registered.json
                     exists (pre-committed P-numbered predictions).
Variants:
  FULL      — all features, walk-forward ridge
  SOLO_<F>  — only family F's features (per-family gate verdict)
  DROP_<F>  — all minus family F (drop-one ablation: marginal contribution)
"""
from __future__ import annotations

import argparse, json, pathlib, sys, time

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from panel import DATA, DEV_END, POST_START, load_panel  # noqa: E402
import features as F  # noqa: E402
from model import walk_forward_predict  # noqa: E402
from gate import evaluate, ls_returns, sharpe  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

DEV_EVAL_START = "2025-02-01"


def compute_features(panel) -> dict[str, dict]:
    cache = DATA / "features.parquet"
    meta_f = DATA / "features_meta.json"
    if cache.exists():
        stacked = pd.read_parquet(cache)
        meta = json.load(open(meta_f))
        out = {}
        for name, fam in meta.items():
            df = stacked[stacked["feature"] == name].pivot(
                index="date", columns="symbol", values="value")
            df = df.reindex(index=panel.dates, columns=panel.symbols)
            out[name] = {"df": df, "family": fam}
        return out
    t0 = time.time()
    out = F.compute_all(panel)
    frames = []
    for name, d in out.items():
        long = d["df"].stack(future_stack=True).rename("value").reset_index()
        long.columns = ["date", "symbol", "value"]
        long["feature"] = name
        frames.append(long.dropna(subset=["value"]))
    pd.concat(frames, ignore_index=True).to_parquet(cache, index=False)
    json.dump({k: v["family"] for k, v in out.items()}, open(meta_f, "w"), indent=2)
    print(f"features computed in {time.time()-t0:.0f}s -> {cache}")
    return out


def variants(feats: dict[str, dict]) -> dict[str, dict[str, pd.DataFrame]]:
    fams = sorted({d["family"] for d in feats.values()})
    v = {"FULL": {n: d["df"] for n, d in feats.items()}}
    for fam in fams:
        v[f"SOLO_{fam}"] = {n: d["df"] for n, d in feats.items() if d["family"] == fam}
        v[f"DROP_{fam}"] = {n: d["df"] for n, d in feats.items() if d["family"] != fam}
    return v


def run_stage(stage: str, n_perm: int) -> None:
    panel = load_panel()
    print(f"panel: {panel.close.shape[1]} symbols x {len(panel.dates)} days "
          f"({panel.dates[0].date()}..{panel.dates[-1].date()})")
    feats = compute_features(panel)
    vs = variants(feats)

    if stage == "dev":
        start, end, tag = DEV_EVAL_START, str(DEV_END.date()), "DEV"
    else:
        start, end, tag = str(POST_START.date()), None, "POST"
        if not (RESULTS / "predictions_registered.json").exists():
            sys.exit("REFUSING to run POST stage: pre-register predictions first "
                     "(results/predictions_registered.json missing).")

    all_res, importances = {}, {}
    for name, fdfs in vs.items():
        t0 = time.time()
        preds, imp, lambdas = walk_forward_predict(panel, fdfs, start, end)
        res = evaluate(preds, panel.ret, f"{tag}:{name}", n_perm=n_perm)
        res["lambdas"] = lambdas
        res["importance_mean"] = imp.mean().to_dict() if not imp.empty else {}
        all_res[name] = res
        importances[name] = imp.to_dict() if not imp.empty else {}
        if stage == "post":
            preds.to_parquet(RESULTS / f"preds_post_{name}.parquet")
        print(f"{tag} {name}: net_sharpe={res['net_sharpe']:.2f} "
              f"p={res['permutation']['p_value']:.3f} "
              f"turn={res['avg_daily_turnover']:.2f} ({time.time()-t0:.0f}s)", flush=True)

    outf = RESULTS / f"gate_{stage}.json"
    json.dump({"stage": tag, "n_perm": n_perm, "results": all_res,
               "importances": importances,
               "generated_utc": pd.Timestamp.utcnow().isoformat()},
              open(outf, "w"), indent=2, default=float)
    print(f"wrote {outf}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["features", "dev", "post"])
    ap.add_argument("--n-perm", type=int, default=None)
    a = ap.parse_args()
    if a.stage == "features":
        compute_features(load_panel())
    else:
        run_stage(a.stage, a.n_perm or (200 if a.stage == "dev" else 500))
