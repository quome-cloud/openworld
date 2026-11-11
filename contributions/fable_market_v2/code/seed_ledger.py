#!/usr/bin/env python3
"""seed_ledger.py — Prospective pre-registered prediction ledger (v2 leg 4).

RESEARCH ONLY — paper predictions; nothing here is tradeable advice.

For each strategy variant (FULL + each SOLO family), generate predictions
from data through the LAST cached close and record the top/bottom decile
tickers for the next 5 trading days. Each prediction file is hash-committed:
the ledger row carries its sha256, so post-hoc edits are detectable.

Resolution (someone else's cycle): after >=5 trading days, compute each
ticker's realized close-to-close return over the 5 sessions following
`asof_date`, equal-weight top decile minus bottom decile, compare to 0 and to
the EW-universe benchmark; append outcome to the ledger.
"""
from __future__ import annotations

import datetime, hashlib, json, pathlib, sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from panel import load_panel  # noqa: E402
from run_eval import compute_features, variants  # noqa: E402
from model import walk_forward_predict  # noqa: E402
from gate import decile_weights  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
PROS = ROOT / "prospective"
PROS.mkdir(exist_ok=True)


def main() -> None:
    panel = load_panel()
    feats = compute_features(panel)
    vs = variants(feats)
    asof = panel.dates[-1]
    rows = []
    for name in ["FULL"] + sorted(k for k in vs if k.startswith("SOLO_")):
        preds, _, _ = walk_forward_predict(panel, vs[name],
                                           str(asof.date()), str(asof.date()))
        row = preds.loc[asof]
        w = decile_weights(row.to_numpy(dtype=float))
        syms = preds.columns
        top = sorted(syms[w > 0])
        bot = sorted(syms[w < 0])
        payload = {
            "variant": name,
            "asof_date": str(asof.date()),
            "horizon_trading_days": 5,
            "claim": "equal-weight(top) minus equal-weight(bottom) 5-session "
                     "close-to-close return > 0",
            "top_decile": top,
            "bottom_decile": bot,
            "generated_utc": datetime.datetime.utcnow().isoformat() + "Z",
        }
        pf = PROS / f"pred_{name}_{asof.date()}.json"
        pf.write_text(json.dumps(payload, indent=2))
        sha = hashlib.sha256(pf.read_bytes()).hexdigest()
        rows.append({"registered_utc": payload["generated_utc"],
                     "variant": name, "asof_date": str(asof.date()),
                     "horizon_trading_days": 5, "n_top": len(top),
                     "n_bottom": len(bot), "prediction_file": pf.name,
                     "sha256": sha, "resolved": "", "realized_ls_return": "",
                     "outcome": ""})
        print(f"{name}: {len(top)} long / {len(bot)} short, sha256={sha[:16]}...")
    ledger = PROS / "ledger.csv"
    df = pd.DataFrame(rows)
    if ledger.exists():
        df = pd.concat([pd.read_csv(ledger), df], ignore_index=True)
    df.to_csv(ledger, index=False)
    print(f"ledger: {ledger} ({len(df)} rows)")


if __name__ == "__main__":
    main()
