#!/usr/bin/env python3
"""
run_experiment.py — Main orchestration script for the market world-model experiment.

RESEARCH ONLY — NOT INVESTMENT ADVICE.

Usage:
    python run_experiment.py              # run full pipeline
    python run_experiment.py --help       # show options

Pipeline:
    1. Check for raw data; download if missing
    2. Build train/val/holdout splits if missing
    3. Verify split checksums
    4. Discover world models in world_models/
    5. For each world model: run backtest on validation + perturbation gate
    6. Print multiple-testing ledger summary
    7. Remind user that final holdout is locked
"""

import sys
import json
import importlib.util
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------

try:
    import yfinance as yf  # noqa: F401
except ImportError:
    print("ERROR: yfinance not installed.")
    print("Install: pip install yfinance pandas numpy matplotlib")
    sys.exit(1)

import numpy as np  # noqa: F401 (ensure available)
import pandas as pd  # noqa: F401

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
WORLD_MODELS_DIR = BASE_DIR / "world_models"
RESULTS_DIR = BASE_DIR / "results"
RAW_DATA_PATH = DATA_DIR / "raw_ohlcv.csv"
CHECKSUMS_PATH = DATA_DIR / "split_checksums.json"
LEDGER_PATH = BASE_DIR / "MULTIPLE_TESTING_LEDGER.md"

sys.path.insert(0, str(BASE_DIR))

# ---------------------------------------------------------------------------
# Step functions
# ---------------------------------------------------------------------------


def step_download_data(force: bool = False) -> None:
    """Step 1: Download raw OHLCV data if not present."""
    if RAW_DATA_PATH.exists() and not force:
        print(f"[1/7] Raw data found: {RAW_DATA_PATH} (skip download)")
        return

    print("[1/7] Downloading raw OHLCV data...")
    from download_data import download_ohlcv
    download_ohlcv(force=force)
    print("[1/7] Done.\n")


def step_build_splits(force: bool = False) -> None:
    """Step 2: Build train/val/holdout splits."""
    splits_exist = all(
        (DATA_DIR / f).exists()
        for f in ["train.csv", "validation.csv", "final_holdout.csv"]
    )
    if splits_exist and CHECKSUMS_PATH.exists() and not force:
        print("[2/7] Splits already exist (skip rebuild)")
        return

    print("[2/7] Building splits...")
    from build_splits import build_splits
    build_splits(force=force)
    print("[2/7] Done.\n")


def step_verify_checksums() -> None:
    """Step 3: Verify split checksums."""
    print("[3/7] Verifying split checksums...")
    from build_splits import verify_splits_intact
    verify_splits_intact()
    print("[3/7] Checksums OK.\n")


def step_discover_world_models(skip_placeholder: bool = True) -> list[Path]:
    """Step 4: Discover world model files."""
    wm_files = sorted(WORLD_MODELS_DIR.glob("wm_*.py"))
    if skip_placeholder:
        wm_files = [f for f in wm_files if "placeholder" not in f.name]

    print(f"[4/7] Found {len(wm_files)} world model(s):")
    for f in wm_files:
        print(f"      {f.name}")
    if not wm_files:
        print("      (none — write a world model in world_models/wm_NN_name.py)")
    print()
    return wm_files


def step_evaluate_world_models(
    wm_files: list[Path],
    symbol: str = "SPY",
    cost_bps: float = 10.0,
    n_shuffles: int = 50,
) -> list[dict]:
    """Step 5: Backtest + benchmark-adjusted perturbation gate (E2a + E2b) on validation split."""
    if not wm_files:
        print("[5/7] No world models to evaluate.\n")
        return []

    from build_splits import load_split
    from backtest import backtest
    from verify import run_both_gate_variants

    print(f"[5/7] Evaluating {len(wm_files)} world model(s) on validation split ({symbol})...")
    print(f"      Gate: benchmark-adjusted excess Sharpe, per-strategy permutation p-value")
    print(f"      Both E2a (drift-present) and E2b (demeaned) variants required")
    print()

    # Load validation data once
    val_df = load_split("validation")
    sym_df = val_df[val_df["symbol"] == symbol].copy()
    if sym_df.empty:
        print(f"ERROR: No validation data for symbol {symbol}")
        return []

    sym_df = sym_df.sort_values("date").set_index("date")
    X_val = sym_df[["open", "high", "low", "close", "volume"]]
    y_val = sym_df["close"].pct_change().dropna()
    X_val = X_val.loc[y_val.index]

    results = []

    for i, wm_path in enumerate(wm_files, 1):
        print(f"  [{i}/{len(wm_files)}] {wm_path.name}")

        try:
            spec = importlib.util.spec_from_file_location("wm", wm_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            if not hasattr(module, "get_signals"):
                print(f"    SKIP: no get_signals() function")
                continue

            wm_name = getattr(module, "NAME", wm_path.stem)
            wm_fn = module.get_signals

            # Backtest on validation
            signals = wm_fn(X_val)
            bt = backtest(signals, y_val, cost_bps=cost_bps)

            # Benchmark-adjusted gate (E2a drift-present + E2b demeaned)
            gate = run_both_gate_variants(
                world_model_fn=wm_fn,
                X_val=X_val,
                y_val=y_val,
                n_shuffles=n_shuffles,
                significance=0.05,
                cost_bps=cost_bps,
            )

            # Excess Sharpe = model − buy-and-hold (primary metric, not absolute Sharpe)
            real_excess = gate["e2a"]["real_excess_sharpe"]

            # Deflated excess Sharpe (multiple-testing adjustment)
            n_tried = i
            deflated_excess = real_excess * np.sqrt(1.0 / n_tried)

            # Validation criterion: positive excess Sharpe AND both gate variants pass
            passes_excess = real_excess > 0
            passes_gate = gate["passed"]
            overall_pass = passes_excess and passes_gate

            result = {
                "name": wm_name,
                "file": wm_path.name,
                "val_sharpe_cost_adj": bt["cost_adj_sharpe"],
                "val_bh_sharpe": gate["e2a"]["real_bh_sharpe"],
                "val_excess_sharpe": real_excess,
                "gate_e2a_passed": gate["e2a"]["passed"],
                "gate_e2a_pvalue": gate["e2a"]["permutation_pvalue"],
                "gate_e2b_passed": gate["e2b"]["passed"],
                "gate_e2b_pvalue": gate["e2b"]["permutation_pvalue"],
                "perturbation_gate_passed": passes_gate,
                "passes_validation_criterion": overall_pass,
                "deflated_excess_sharpe": deflated_excess,
                "n_trades": bt["n_trades"],
                "total_return": bt["total_return"],
                "max_drawdown": bt["max_drawdown"],
            }
            results.append(result)

            status = "PASS" if overall_pass else "FAIL"
            print(f"    Model Sharpe: {bt['cost_adj_sharpe']:.3f}  BH Sharpe: {gate['e2a']['real_bh_sharpe']:.3f}  Excess: {real_excess:.3f}")
            print(f"    E2a p={gate['e2a']['permutation_pvalue']:.3f}  E2b p={gate['e2b']['permutation_pvalue']:.3f}  Gate: {'PASS' if passes_gate else 'FAIL'}  Overall: {status}")

            # Save per-model results
            RESULTS_DIR.mkdir(exist_ok=True)
            result_path = RESULTS_DIR / f"{wm_path.stem}_validation.json"
            with open(result_path, "w") as f:
                def _serial(v):
                    if isinstance(v, (np.floating, np.integer)):
                        return float(v)
                    return v
                json.dump({k: _serial(v) for k, v in result.items()}, f, indent=2)

        except NotImplementedError:
            print(f"    SKIP: get_signals() not implemented (placeholder)")
        except Exception as e:
            import traceback
            print(f"    ERROR: {e}")
            traceback.print_exc()

        print()

    return results


def step_print_ledger(results: list[dict], n_variants_total: int) -> None:
    """Step 6: Print multiple-testing ledger summary."""
    print(f"[6/7] Multiple-Testing Ledger Summary ({n_variants_total} total variants tried)")
    print(f"      Metric: EXCESS Sharpe (model − buy-and-hold); both E2a+E2b gate variants required")
    print()

    header = f"{'#':>3}  {'Name':<25}  {'Mod Sh':>7}  {'BH Sh':>7}  {'Excess':>7}  {'E2a p':>6}  {'E2b p':>6}  {'Gate':>5}  {'Pass?':>5}  {'Defl Ex':>7}"
    print(header)
    print("-" * len(header))

    for i, r in enumerate(results, 1):
        gate_str = "PASS" if r["perturbation_gate_passed"] else "FAIL"
        pass_str = "YES" if r["passes_validation_criterion"] else "NO"
        print(
            f"{i:>3}  {r['name']:<25}  "
            f"{r['val_sharpe_cost_adj']:>7.3f}  "
            f"{r.get('val_bh_sharpe', float('nan')):>7.3f}  "
            f"{r.get('val_excess_sharpe', float('nan')):>7.3f}  "
            f"{r.get('gate_e2a_pvalue', float('nan')):>6.3f}  "
            f"{r.get('gate_e2b_pvalue', float('nan')):>6.3f}  "
            f"{gate_str:>5}  {pass_str:>5}  "
            f"{r.get('deflated_excess_sharpe', float('nan')):>7.3f}"
        )

    if not results:
        print("  (no world models evaluated)")

    n_pass = sum(1 for r in results if r["passes_validation_criterion"])
    print()
    print(f"  {n_pass}/{len(results)} world models passed validation criterion (excess Sharpe > 0 AND both gate variants passed)")
    print()


def step_final_holdout_reminder() -> None:
    """Step 7: Remind user that final holdout is locked."""
    print("[7/7] FINAL HOLDOUT STATUS: LOCKED")
    print()
    print("  The final_holdout split (2025-04-01 to 2026-06-30) has NOT been evaluated.")
    print("  When you are ready for the single-shot final evaluation:")
    print()
    print("    from build_splits import unlock_final_holdout, load_split")
    print("    unlock_final_holdout(reason='All strategy selection complete')")
    print("    df = load_split('final_holdout')")
    print()
    print("  Pre-committed predictions are in predictions.json.")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Market World-Model Experiment — main orchestration script\nRESEARCH ONLY — NOT INVESTMENT ADVICE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--symbol", default="SPY", help="Primary symbol for evaluation (default: SPY)")
    parser.add_argument("--cost_bps", type=float, default=10.0, help="Transaction cost in bps (default: 10)")
    parser.add_argument("--n_shuffles", type=int, default=20, help="Perturbation gate shuffles (default: 20)")
    parser.add_argument("--force_download", action="store_true", help="Re-download raw data")
    parser.add_argument("--force_splits", action="store_true", help="Rebuild splits")
    parser.add_argument("--include_placeholder", action="store_true", help="Include wm_00_placeholder.py in evaluation")
    args = parser.parse_args()

    print()
    print("=" * 60)
    print("MARKET WORLD-MODEL EXPERIMENT")
    print("RESEARCH ONLY — NOT INVESTMENT ADVICE")
    print(f"Started: {datetime.utcnow().isoformat()}Z")
    print("=" * 60)
    print()

    step_download_data(force=args.force_download)
    step_build_splits(force=args.force_splits)
    step_verify_checksums()

    wm_files = step_discover_world_models(skip_placeholder=not args.include_placeholder)

    results = step_evaluate_world_models(
        wm_files=wm_files,
        symbol=args.symbol,
        cost_bps=args.cost_bps,
        n_shuffles=args.n_shuffles,
    )

    step_print_ledger(results, n_variants_total=len(wm_files))
    step_final_holdout_reminder()

    print("=" * 60)
    print(f"Done: {datetime.utcnow().isoformat()}Z")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
