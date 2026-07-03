#!/usr/bin/env python3
"""
verify.py — Perturbation gate for world model verification.

RESEARCH ONLY — NOT INVESTMENT ADVICE.

## Gate design (v2 — benchmark-adjusted, per A001 E1/E2 pilots)

The naive perturbation gate (v1) failed-open in bull regimes: strategies that are
"mostly long" beat zero on shuffled returns just from drift, not from timing skill.
This is the E1/E2 finding: 74% of pure-noise trials pass a naive Sharpe>0 gate.

Fix: score EXCESS over buy-and-hold, not absolute Sharpe. Under this metric, a
strategy that is "just buy and hold" has zero excess — the drift is priced out.
A model must show GENUINE TIMING SKILL above the passive benchmark.

Per-strategy permutation p-value:
  The model must beat its own shuffled-data excess-Sharpe distribution, not just
  a fixed threshold. This catches models that "look good" only because the fixed
  bar is too low relative to the noise floor for that particular strategy.

Demeaned-returns variant (E2b):
  Also run the gate on returns with the mean removed. This makes the null
  distribution truly zero-drift and is the strictest version of the test.
"""

import sys
import importlib.util
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def _sharpe(ret: pd.Series, annualize: float = 252.0) -> float:
    """Annualized Sharpe ratio."""
    std = ret.std()
    if std == 0 or np.isnan(std):
        return float("nan")
    return float(ret.mean() / std * np.sqrt(annualize))


def _strategy_net_returns(
    signals: pd.Series,
    returns: pd.Series,
    cost_bps: float = 10.0,
) -> pd.Series:
    """Apply signals to returns with transaction costs."""
    sig, ret = signals.align(returns, join="inner")
    strat = sig * ret
    costs = sig.diff().abs().fillna(0) * (cost_bps / 10_000.0)
    return strat - costs


def _bh_net_returns(returns: pd.Series, cost_bps: float = 10.0) -> pd.Series:
    """Buy-and-hold net returns (long always, one entry cost)."""
    bh = returns.copy()
    cost_series = pd.Series(0.0, index=returns.index)
    cost_series.iloc[0] = cost_bps / 10_000.0  # one entry trade
    return bh - cost_series


def _rebuild_ohlcv_with_shuffled_returns(
    X_val: pd.DataFrame,
    y_shuffled: pd.Series,
) -> pd.DataFrame:
    """
    Rebuild an OHLCV DataFrame with shuffled returns reconstructed as prices.
    This ensures world models using price levels (EMA, SMA, RSI, Bollinger)
    see the shuffled data consistently.
    """
    X_copy = X_val.copy()
    if "close" in X_copy.columns:
        close_0 = float(X_copy["close"].iloc[0])
        shuffled_arr = y_shuffled.reindex(X_copy.index).fillna(0).values
        new_close = pd.Series(
            close_0 * np.cumprod(1 + shuffled_arr),
            index=X_copy.index,
        )
        X_copy["close"] = new_close
        spread = (X_val["high"] - X_val["low"]) / X_val["close"].replace(0, np.nan)
        spread = spread.fillna(0.01)
        X_copy["high"] = new_close * (1 + spread / 2)
        X_copy["low"] = new_close * (1 - spread / 2)
        X_copy["open"] = new_close.shift(1).fillna(new_close.iloc[0])
    return X_copy


# ---------------------------------------------------------------------------
# Main gate
# ---------------------------------------------------------------------------


def perturbation_gate(
    world_model_fn: Callable[[pd.DataFrame], pd.Series],
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_shuffles: int = 50,
    significance: float = 0.05,
    cost_bps: float = 10.0,
    rng_seed: int = 42,
    demeaned: bool = False,
) -> dict:
    """
    Benchmark-adjusted perturbation gate (v2).

    Tests whether a world model shows genuine TIMING SKILL above buy-and-hold
    rather than just harvesting market drift.

    Scoring metric: excess Sharpe = model Sharpe − buy-and-hold Sharpe,
    measured on the SAME return series (real or shuffled).

    Per-strategy permutation p-value: the model's real excess Sharpe must
    exceed its own null distribution (shuffled excess Sharpes) at (1-significance)
    confidence.

    Args:
        world_model_fn: Callable taking OHLCV DataFrame, returning pd.Series signals.
        X_val:          OHLCV DataFrame for the validation period.
        y_val:          Daily returns (pd.Series) for validation.
        n_shuffles:     Permutation iterations for null distribution.
        significance:   Max fraction of null excess Sharpes >= real excess Sharpe.
                        If more than this fraction, the model fails (no genuine edge).
        cost_bps:       Transaction cost per trade in bps.
        rng_seed:       Fixed seed for reproducibility.
        demeaned:       If True, demean y_val before gate (E2b variant — zero-drift null).

    Returns dict with:
        passed (bool)
        real_excess_sharpe (float)   — model Sharpe − BH Sharpe on real data
        real_model_sharpe (float)
        real_bh_sharpe (float)
        null_excess_sharpe_mean (float)
        null_excess_sharpe_std (float)
        null_excess_sharpes (list)
        permutation_pvalue (float)   — fraction of null >= real excess Sharpe
        n_shuffles (int)
        demeaned (bool)
        verdict (str)
    """
    rng = np.random.default_rng(rng_seed)

    y_eval = y_val.copy()
    if demeaned:
        y_eval = y_eval - y_eval.mean()

    # Real data: model vs buy-and-hold
    signals = world_model_fn(X_val)
    model_ret = _strategy_net_returns(signals, y_eval, cost_bps=cost_bps)
    bh_ret = _bh_net_returns(y_eval, cost_bps=cost_bps)
    bh_ret = bh_ret.reindex(model_ret.index).fillna(0)

    real_model_sharpe = _sharpe(model_ret)
    real_bh_sharpe = _sharpe(bh_ret)
    real_excess = real_model_sharpe - real_bh_sharpe

    # Null distribution: shuffle y_eval, score excess on each
    y_arr = y_eval.values.copy()
    null_excess_sharpes = []

    for _ in range(n_shuffles):
        shuffled = rng.permutation(y_arr)
        y_shuffled = pd.Series(shuffled, index=y_eval.index, name=y_eval.name)

        # Rebuild OHLCV with shuffled prices
        X_shuffled = _rebuild_ohlcv_with_shuffled_returns(X_val, y_shuffled)
        null_signals = world_model_fn(X_shuffled)

        null_model_ret = _strategy_net_returns(null_signals, y_shuffled, cost_bps=cost_bps)
        null_bh_ret = _bh_net_returns(y_shuffled, cost_bps=cost_bps)
        null_bh_ret = null_bh_ret.reindex(null_model_ret.index).fillna(0)

        null_model_sharpe = _sharpe(null_model_ret)
        null_bh_sharpe = _sharpe(null_bh_ret)
        null_excess = null_model_sharpe - null_bh_sharpe
        null_excess_sharpes.append(null_excess)

    null_arr = np.array(null_excess_sharpes)
    null_mean = float(np.nanmean(null_arr))
    null_std = float(np.nanstd(null_arr))

    # Per-strategy permutation p-value: fraction of null runs >= real excess
    pvalue = float(np.mean(null_arr >= real_excess))
    passed = pvalue <= significance

    if passed:
        verdict = (
            f"PASSED — excess Sharpe={real_excess:.3f} (model {real_model_sharpe:.3f} − BH {real_bh_sharpe:.3f}), "
            f"permutation p={pvalue:.3f} <= {significance}. "
            "Model shows genuine timing skill above buy-and-hold."
        )
    else:
        verdict = (
            f"FAILED — excess Sharpe={real_excess:.3f} (model {real_model_sharpe:.3f} − BH {real_bh_sharpe:.3f}), "
            f"permutation p={pvalue:.3f} > {significance}. "
            "Model does NOT beat buy-and-hold on shuffled data at required confidence. "
            "Likely harvesting drift, not genuine timing signal. REJECT."
        )

    return {
        "passed": passed,
        "real_excess_sharpe": real_excess,
        "real_model_sharpe": real_model_sharpe,
        "real_bh_sharpe": real_bh_sharpe,
        "null_excess_sharpe_mean": null_mean,
        "null_excess_sharpe_std": null_std,
        "null_excess_sharpes": null_arr.tolist(),
        "permutation_pvalue": pvalue,
        "n_shuffles": n_shuffles,
        "demeaned": demeaned,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Convenience: run both variants
# ---------------------------------------------------------------------------


def run_both_gate_variants(
    world_model_fn: Callable[[pd.DataFrame], pd.Series],
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_shuffles: int = 50,
    significance: float = 0.05,
    cost_bps: float = 10.0,
) -> dict:
    """
    Run E2a (drift-present) and E2b (demeaned, zero-drift) variants.
    A model must pass BOTH to clear the gate.
    """
    result_a = perturbation_gate(
        world_model_fn, X_val, y_val,
        n_shuffles=n_shuffles, significance=significance,
        cost_bps=cost_bps, demeaned=False,
    )
    result_b = perturbation_gate(
        world_model_fn, X_val, y_val,
        n_shuffles=n_shuffles, significance=significance,
        cost_bps=cost_bps, demeaned=True,
    )
    passed_both = result_a["passed"] and result_b["passed"]
    return {
        "passed": passed_both,
        "e2a": result_a,
        "e2b": result_b,
        "verdict": (
            f"E2a {'PASS' if result_a['passed'] else 'FAIL'} "
            f"| E2b {'PASS' if result_b['passed'] else 'FAIL'} "
            f"→ overall {'PASS' if passed_both else 'FAIL'}"
        ),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run benchmark-adjusted perturbation gate on a world model")
    parser.add_argument("--world_model", required=True, help="Path to world model .py file")
    parser.add_argument("--symbol", default="SPY", help="Symbol to run on (default: SPY)")
    parser.add_argument("--n_shuffles", type=int, default=50, help="Number of null shuffles")
    parser.add_argument("--significance", type=float, default=0.05, help="Max permutation p-value")
    parser.add_argument("--cost_bps", type=float, default=10.0, help="Transaction cost in bps")
    parser.add_argument("--demeaned", action="store_true", help="Use demeaned returns (E2b variant)")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).parent))
    from build_splits import load_split

    val_df = load_split("validation")
    sym_df = val_df[val_df["symbol"] == args.symbol].copy().sort_values("date").set_index("date")
    X_val = sym_df[["open", "high", "low", "close", "volume"]]
    y_val = sym_df["close"].pct_change().dropna()
    X_val = X_val.loc[y_val.index]

    p = Path(args.world_model)
    spec = importlib.util.spec_from_file_location("wm", p)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    wm_fn = module.get_signals

    print(f"Running benchmark-adjusted gate (n_shuffles={args.n_shuffles}, demeaned={args.demeaned})...")
    result = perturbation_gate(
        wm_fn, X_val, y_val,
        n_shuffles=args.n_shuffles,
        significance=args.significance,
        cost_bps=args.cost_bps,
        demeaned=args.demeaned,
    )
    print(f"\nVerdict: {result['verdict']}")
    print(f"Permutation p-value: {result['permutation_pvalue']:.3f}")
    print(f"Real excess Sharpe:  {result['real_excess_sharpe']:.3f}")
    print(f"Null excess Sh mean: {result['null_excess_sharpe_mean']:.3f} ± {result['null_excess_sharpe_std']:.3f}")
    print(f"Passed: {result['passed']}")
