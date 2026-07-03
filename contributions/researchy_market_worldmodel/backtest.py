#!/usr/bin/env python3
"""
backtest.py — Backtesting engine (validation split only).

RESEARCH ONLY — NOT INVESTMENT ADVICE.

Runs a world model's signals against the validation split and computes:
  - Sharpe ratio (annualized)
  - Total return
  - Max drawdown
  - Number of trades
  - Cost-adjusted Sharpe

IMPORTANT: This module only operates on the VALIDATION split.
The final_holdout split is enforced as locked by build_splits.py.
"""

import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Core backtest function
# ---------------------------------------------------------------------------


def backtest(
    signals: pd.Series,
    returns: pd.Series,
    cost_bps: float = 10.0,
    annualize: float = 252.0,
) -> dict:
    """
    Backtest a signal series against a return series.

    Args:
        signals:   pd.Series of {-1, 0, 1} indexed by date.
                   -1 = short/cash, 0 = flat/cash, 1 = long.
        returns:   pd.Series of daily returns (e.g., close.pct_change()),
                   indexed by date. Must overlap with signals.
        cost_bps:  Transaction cost per trade in basis points (default: 10bps).
        annualize: Annualization factor (252 for daily).

    Returns:
        dict with:
            sharpe (float)          — Annualized Sharpe (no cost)
            cost_adj_sharpe (float) — Annualized Sharpe after transaction costs
            total_return (float)    — Cumulative return over the period
            max_drawdown (float)    — Maximum drawdown (negative number)
            n_trades (int)          — Number of position changes
            avg_daily_return (float)
            volatility (float)      — Annualized daily return std
            win_rate (float)        — Fraction of trading days with positive return
            dates (dict)            — Start/end dates of the backtest
    """
    # Align by date
    sig, ret = signals.align(returns, join="inner")

    if len(sig) == 0:
        raise ValueError("No overlapping dates between signals and returns.")

    # Gross strategy returns
    gross_ret = sig * ret

    # Transaction costs: 1 trade = position changes by 1 unit
    n_trades = int(sig.diff().abs().fillna(0).sum())
    cost_series = sig.diff().abs().fillna(0) * (cost_bps / 10_000.0)
    net_ret = gross_ret - cost_series

    # Metrics — gross
    sharpe = _sharpe(gross_ret, annualize)

    # Metrics — cost-adjusted
    cost_adj_sharpe = _sharpe(net_ret, annualize)

    # Total return (cost-adjusted)
    total_return = float((1 + net_ret).prod() - 1)

    # Max drawdown
    cum = (1 + net_ret).cumprod()
    rolling_max = cum.cummax()
    drawdown = (cum - rolling_max) / rolling_max
    max_drawdown = float(drawdown.min())

    # Win rate (days with positive net return, excluding flat days)
    trading_days = net_ret[sig != 0]
    win_rate = float((trading_days > 0).mean()) if len(trading_days) > 0 else float("nan")

    return {
        "sharpe": sharpe,
        "cost_adj_sharpe": cost_adj_sharpe,
        "total_return": total_return,
        "max_drawdown": max_drawdown,
        "n_trades": n_trades,
        "avg_daily_return": float(net_ret.mean()),
        "volatility": float(net_ret.std() * np.sqrt(annualize)),
        "win_rate": win_rate,
        "cost_bps": cost_bps,
        "n_days": len(sig),
        "dates": {
            "start": str(sig.index.min().date()),
            "end": str(sig.index.max().date()),
        },
    }


def run_world_model_backtest(
    world_model_fn: Callable[[pd.DataFrame], pd.Series],
    symbol: str = "SPY",
    split: str = "validation",
    cost_bps: float = 10.0,
) -> dict:
    """
    Convenience: load a split, run a world model, and return backtest results.

    Args:
        world_model_fn: Callable that takes OHLCV DataFrame, returns signals.
        symbol:         Which symbol to backtest on.
        split:          Which split to use ('train' or 'validation').
                        'final_holdout' requires explicit unlock.
        cost_bps:       Transaction cost in bps.

    Returns:
        Backtest result dict.
    """
    if split == "final_holdout":
        print(
            "WARNING: You are about to load the final holdout split.\n"
            "This should only be done ONCE, after all strategy selection is complete.\n"
            "Ensure unlock_final_holdout() has been called in build_splits.py."
        )

    sys.path.insert(0, str(Path(__file__).parent))
    from build_splits import load_split

    df = load_split(split)
    sym_df = df[df["symbol"] == symbol].copy()
    if sym_df.empty:
        raise ValueError(f"No data for symbol {symbol} in {split} split.")

    sym_df = sym_df.sort_values("date").set_index("date")
    X = sym_df[["open", "high", "low", "close", "volume"]]
    returns = sym_df["close"].pct_change().dropna()
    X = X.loc[returns.index]

    signals = world_model_fn(X)
    result = backtest(signals, returns, cost_bps=cost_bps)
    result["symbol"] = symbol
    result["split"] = split
    return result


def print_backtest_results(result: dict, name: str = "") -> None:
    """Pretty-print backtest results."""
    label = f" ({name})" if name else ""
    print(f"\n{'='*55}")
    print(f"BACKTEST RESULTS{label}")
    print(f"{'='*55}")
    print(f"Split:              {result.get('split', 'N/A')} ({result.get('symbol', '')})")
    print(f"Dates:              {result['dates']['start']} to {result['dates']['end']}")
    print(f"Trading days:       {result['n_days']}")
    print(f"Trades:             {result['n_trades']}")
    print(f"---")
    print(f"Sharpe (gross):     {result['sharpe']:.4f}")
    print(f"Sharpe (cost-adj):  {result['cost_adj_sharpe']:.4f}  ← primary metric")
    print(f"Total return:       {result['total_return']:.2%}")
    print(f"Max drawdown:       {result['max_drawdown']:.2%}")
    print(f"Annualized vol:     {result['volatility']:.2%}")
    print(f"Win rate:           {result['win_rate']:.1%}")
    print(f"Cost (bps):         {result['cost_bps']}")


def _sharpe(ret: pd.Series, annualize: float) -> float:
    """Annualized Sharpe ratio."""
    std = ret.std()
    if std == 0 or np.isnan(std):
        return float("nan")
    return float(ret.mean() / std * np.sqrt(annualize))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import importlib.util

    parser = argparse.ArgumentParser(description="Run backtest on a world model")
    parser.add_argument("--world_model", required=True, help="Path to world model .py file")
    parser.add_argument("--symbol", default="SPY", help="Symbol to backtest on (default: SPY)")
    parser.add_argument(
        "--split",
        default="validation",
        choices=["train", "validation"],
        help="Which split to use (default: validation). final_holdout requires unlock.",
    )
    parser.add_argument("--cost_bps", type=float, default=10.0, help="Transaction cost in bps")
    args = parser.parse_args()

    # Load world model
    p = Path(args.world_model)
    if not p.exists():
        print(f"ERROR: World model not found: {p}")
        sys.exit(1)

    spec = importlib.util.spec_from_file_location("wm", p)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "get_signals"):
        print(f"ERROR: {p.name} must define get_signals(ohlcv) -> pd.Series")
        sys.exit(1)

    wm_name = getattr(module, "NAME", p.stem)
    print(f"Running backtest for world model: {wm_name}")

    result = run_world_model_backtest(
        world_model_fn=module.get_signals,
        symbol=args.symbol,
        split=args.split,
        cost_bps=args.cost_bps,
    )
    print_backtest_results(result, name=wm_name)

    # Check against validation criterion
    threshold = 0.5
    if args.split == "validation":
        if result["cost_adj_sharpe"] >= threshold:
            print(f"\nPASS: cost-adjusted Sharpe {result['cost_adj_sharpe']:.3f} >= {threshold}")
            print("Next step: run verify.py --world_model ... to check perturbation gate")
        else:
            print(f"\nFAIL: cost-adjusted Sharpe {result['cost_adj_sharpe']:.3f} < {threshold}")
            print("World model does not pass validation criterion.")
