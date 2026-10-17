#!/usr/bin/env python3
"""make_report.py — Fill the report placeholders from results JSONs.
RESEARCH ONLY. Idempotent: rewrites the section between placeholder markers.
"""
from __future__ import annotations

import json, pathlib, sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import features as F  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
REPORT = ROOT / "FABLE_MARKET_V2_REPORT.md"


def feature_dictionary_md() -> str:
    rows = F.dictionary_rows()
    lines = ["| feature | family | module | summary |", "|---|---|---|---|"]
    for r in rows:
        lines.append(f"| `{r['feature']}` | {r['family']} | `{r['module']}` | {r['summary']} |")
    return "\n".join(lines)


def results_table_md(path: pathlib.Path) -> str:
    data = json.load(open(path))
    res = data["results"]
    order = ["FULL"] + sorted(k for k in res if k.startswith("SOLO_")) + \
            sorted(k for k in res if k.startswith("DROP_"))
    lines = [
        "| variant | net Sharpe | 95% CI | gross Sharpe | perm p (gross) | "
        "relabel p (net) | long−bench net Sharpe | turn/day | PASS |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for k in order:
        r = res[k]
        pm = r["permutation"]
        ci = r["bootstrap_ci_net_sharpe"]
        lines.append(
            f"| {k} | {r['net_sharpe']:+.2f} | [{ci['ci_lo']:+.2f}, {ci['ci_hi']:+.2f}] "
            f"| {r['gross_sharpe']:+.2f} | {pm['p_value']:.3f} "
            f"| {pm.get('p_net_static_relabel', float('nan')):.3f} "
            f"| {r['long_minus_bench_net_sharpe']:+.2f} "
            f"| {r['avg_daily_turnover']:.2f} | {'**PASS**' if r['PASS'] else 'fail'} |")
    w = res["FULL"]["window"]
    hdr = (f"Window: {w['start']} → {w['end']} ({w['n_days']} pnl days). "
           f"n_perm={data['n_perm']}. EW-universe benchmark annualized return: "
           f"{res['FULL']['bench_ann_return']:+.1%}.\n\n")
    return hdr + "\n".join(lines)


def importance_table_md(path: pathlib.Path) -> str:
    data = json.load(open(path))
    imp = data["results"]["FULL"].get("importance_mean", {})
    if not imp:
        return ""
    lines = ["", "Mean standardized ridge coefficients, FULL model (units: xs-demeaned "
             "5d forward return per 1-sigma of feature):", "",
             "| feature | mean coef |", "|---|---|"]
    for k, v in sorted(imp.items(), key=lambda kv: -abs(kv[1])):
        lines.append(f"| `{k}` | {v:+.2e} |")
    return "\n".join(lines)


def fill(marker: str, content: str) -> None:
    text = REPORT.read_text()
    tag = f"<!-- {marker} -->"
    assert tag in text, f"missing {tag}"
    text = text.replace(tag, tag + "\n\n" + content, 1)
    REPORT.write_text(text)


if __name__ == "__main__":
    which = sys.argv[1]
    if which == "dictionary":
        fill("FEATURE_DICTIONARY", feature_dictionary_md())
    elif which == "dev":
        fill("DEV_RESULTS", results_table_md(ROOT / "results" / "gate_dev.json")
             + importance_table_md(ROOT / "results" / "gate_dev.json"))
    elif which == "post":
        fill("POST_RESULTS", results_table_md(ROOT / "results" / "gate_post.json")
             + importance_table_md(ROOT / "results" / "gate_post.json"))
    print("filled", which)
