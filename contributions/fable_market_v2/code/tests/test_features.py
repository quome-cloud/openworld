#!/usr/bin/env python3
"""Unit tests for the feature library, on synthetic panels with planted
structure. Each test checks (a) the feature detects the structure its causal
story claims, and (b) no lookahead: truncating future data does not change
past feature values.

Run: python3 -m pytest code/tests/test_features.py -q  (or python3 <file>)
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from panel import Panel  # noqa: E402
from features import sll_sector_leadlag, peer_momentum, reversal_volume  # noqa: E402
from features import pead_earnings_drift  # noqa: E402

RNG = np.random.default_rng(7)


def make_panel(n_days=900, n_stocks=40, seed=7) -> Panel:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    syms = [f"S{i:03d}" for i in range(n_stocks)]
    ret = pd.DataFrame(rng.normal(0, 0.01, (n_days, n_stocks)), index=dates, columns=syms)
    close = 100 * (1 + ret).cumprod()
    volume = pd.DataFrame(rng.integers(1e5, 2e5, (n_days, n_stocks)).astype(float),
                          index=dates, columns=syms)
    sector = pd.Series([f"SEC{i % 4}" for i in range(n_stocks)], index=syms)
    return Panel(close=close, volume=volume, ret=ret, sector=sector)


def _truncation_invariance(module, panel, cut=650, tol=1e-10):
    """Feature at dates <= cut must not change when future rows are removed."""
    full = module.compute(panel)
    trunc_panel = Panel(close=panel.close.iloc[:cut], volume=panel.volume.iloc[:cut],
                        ret=panel.ret.iloc[:cut], sector=panel.sector)
    trunc = module.compute(trunc_panel)
    for name in full:
        a = full[name].iloc[:cut - 30]  # margin: same-month refit boundary
        b = trunc[name].iloc[:cut - 30]
        diff = (a - b).abs().max().max()
        assert not np.isfinite(diff) or diff < tol, f"{name}: lookahead, diff={diff}"


def test_sll_detects_planted_leadlag():
    panel = make_panel()
    # Plant: sector SEC1's return follows SEC0's previous-day return.
    sec0 = panel.sector.index[panel.sector == "SEC0"]
    sec1 = panel.sector.index[panel.sector == "SEC1"]
    driver = panel.ret[sec0].mean(axis=1)
    panel.ret.loc[:, sec1] = (panel.ret[sec1].values
                              + 0.8 * driver.shift(1).fillna(0).values[:, None])
    panel.close.loc[:, :] = 100 * (1 + panel.ret).cumprod()
    feats = sll_sector_leadlag.compute(panel)
    f = feats["sll_daily"]
    # Realized next-day relative return per stock:
    rel = panel.ret.sub(panel.ret.mean(axis=1), axis=0)
    ic = f.shift(1).iloc[400:].corrwith(rel.iloc[400:]).mean()
    assert ic > 0.05, f"SLL failed to detect planted lead-lag, mean IC={ic:.4f}"
    _truncation_invariance(sll_sector_leadlag, panel)


def test_peer_detects_planted_diffusion():
    panel = make_panel()
    # Plant: stocks S000..S009 share a common factor; S000 loads on it partly
    # contemporaneously (so the correlation graph can find its peers) and
    # partly with a 5-day lag (the diffusion the feature should exploit).
    n = len(panel.ret)
    factor = RNG.normal(0, 0.02, n)
    for i in range(1, 10):
        panel.ret.iloc[:, i] += factor
    panel.ret.iloc[:, 0] += 0.5 * factor + 0.8 * np.roll(factor, 5)
    panel.ret.iloc[:5, 0] = 0.0
    panel.close.loc[:, :] = 100 * (1 + panel.ret).cumprod()
    feats = peer_momentum.compute(panel)
    f = feats["peer_mom_gap"]["S000"]
    fwd5 = (1 + panel.ret["S000"]).rolling(5).apply(np.prod, raw=True).shift(-5) - 1
    mkt5 = (1 + panel.ret.mean(axis=1)).rolling(5).apply(np.prod, raw=True).shift(-5) - 1
    # gap at t should predict S000's catch-up over the next 5 days
    ic = f.iloc[300:-5].corr((fwd5 - mkt5).iloc[300:-5])
    assert ic > 0.05, f"PEER failed to detect planted diffusion, IC={ic:.4f}"
    _truncation_invariance(peer_momentum, panel)


def test_rev_lowvol_isolates_low_volume_reversal():
    panel = make_panel()
    n, k = panel.ret.shape
    # Plant: negative AR(5) in half the stocks ONLY on low-volume weeks.
    # Construct: every 20 days, stock j gets a +3% shock; if the shock week
    # volume is low, it reverts fully over the next 5 days.
    for j in range(0, k, 2):
        for t0 in range(100 + j, n - 10, 40):
            panel.volume.iloc[t0 - 4:t0 + 1, j] = 8e4      # low volume week
            panel.ret.iloc[t0, j] += 0.03                   # shock
            panel.ret.iloc[t0 + 1:t0 + 6, j] -= 0.006       # reversion
    panel.close.loc[:, :] = 100 * (1 + panel.ret).cumprod()
    feats = reversal_volume.compute(panel)
    f = feats["rev_lowvol"]
    fwd5 = (1 + panel.ret).rolling(5).apply(np.prod, raw=True).shift(-5) - 1
    rel5 = fwd5.sub(fwd5.mean(axis=1), axis=0)
    ic = f.iloc[100:-10].corrwith(rel5.iloc[100:-10]).mean()
    assert ic > 0.03, f"REV failed to detect planted low-vol reversal, IC={ic:.4f}"
    _truncation_invariance(reversal_volume, panel)


def test_pead_event_mapping(tmp_path=None):
    panel = make_panel(n_days=200, n_stocks=3)
    # Synthetic earnings cache in a temp dir.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        old = pead_earnings_drift.EARNINGS_DIR
        pead_earnings_drift.EARNINGS_DIR = pathlib.Path(td)
        try:
            d40 = panel.dates[40]
            # AMC announcement on date index 40 -> effective at index 41
            pd.DataFrame({"timestamp": [f"{d40.date()} 16:00:00-05:00"],
                          "eps_estimate": [1.0], "eps_reported": [1.2],
                          "surprise_pct": [20.0]}).to_csv(
                pathlib.Path(td) / "S000.csv", index=False)
            # BMO announcement on date index 80 -> effective same day
            d80 = panel.dates[80]
            pd.DataFrame({"timestamp": [f"{d80.date()} 08:00:00-05:00"],
                          "eps_estimate": [1.0], "eps_reported": [0.5],
                          "surprise_pct": [-50.0]}).to_csv(
                pathlib.Path(td) / "S001.csv", index=False)
            f = pead_earnings_drift.compute(panel)["pead_drift"]
        finally:
            pead_earnings_drift.EARNINGS_DIR = old
    assert f.iloc[40]["S000"] == 0.0, "AMC event must not be tradeable same day"
    assert abs(f.iloc[41]["S000"] - 0.2 * (1 - 1 / 60)) < 1e-9
    assert f.iloc[41 + 60]["S000"] == 0.0, "drift must expire after 60 days"
    assert f.iloc[80]["S001"] < 0, "BMO event effective same day"
    assert (f["S002"] == 0).all(), "no events -> all zero"
    print("PEAD mapping ok")


if __name__ == "__main__":
    for fn in [test_sll_detects_planted_leadlag, test_peer_detects_planted_diffusion,
               test_rev_lowvol_isolates_low_volume_reversal, test_pead_event_mapping]:
        fn()
        print(f"PASS {fn.__name__}")
