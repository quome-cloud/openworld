#!/usr/bin/env python3
"""
build_splits.py — Create train/validation/holdout splits from raw OHLCV data.

RESEARCH ONLY — NOT INVESTMENT ADVICE.

Splits:
  TRAIN:         2010-01-01 to 2023-12-31
  VALIDATION:    2024-01-01 to 2025-03-31
  FINAL_HOLDOUT: 2025-04-01 to 2026-06-30

After splitting, checksums each file and saves to data/split_checksums.json.
FINAL_HOLDOUT is saved but must NOT be read during strategy development.
"""

import sys
import hashlib
import json
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent / "data"
RAW_PATH = DATA_DIR / "raw_ohlcv.csv"
CHECKSUMS_PATH = DATA_DIR / "split_checksums.json"

SPLITS = {
    "train": {
        "path": DATA_DIR / "train.csv",
        "start": "2010-01-01",
        "end": "2023-12-31",
        "description": "Training split — use for world model fitting",
    },
    "validation": {
        "path": DATA_DIR / "validation.csv",
        "start": "2024-01-01",
        "end": "2025-03-31",
        "description": "Validation split — use for strategy selection and perturbation gate",
    },
    "final_holdout": {
        "path": DATA_DIR / "final_holdout.csv",
        "start": "2025-04-01",
        "end": "2026-06-30",
        "description": "LOCKED — do not read during strategy development. Evaluate ONCE at the end.",
    },
}

# Guard: raise if someone tries to load final holdout without explicit unlock
_HOLDOUT_UNLOCKED = False


def unlock_final_holdout(reason: str) -> None:
    """
    Explicitly unlock the final holdout for evaluation.
    Call this ONCE, after all strategy selection is complete.

    Args:
        reason: Brief explanation of why holdout is being unlocked.
    """
    global _HOLDOUT_UNLOCKED
    print(f"\n{'='*60}")
    print("FINAL HOLDOUT UNLOCK")
    print(f"Reason: {reason}")
    print("This should be called EXACTLY ONCE, after strategy selection.")
    print(f"{'='*60}\n")
    _HOLDOUT_UNLOCKED = True


def load_split(split_name: str, verify_checksum: bool = True) -> pd.DataFrame:
    """
    Load a named split (train, validation, or final_holdout).

    The final_holdout split is blocked unless unlock_final_holdout() has been called.
    """
    if split_name not in SPLITS:
        raise ValueError(f"Unknown split: {split_name!r}. Choose from {list(SPLITS.keys())}")

    if split_name == "final_holdout" and not _HOLDOUT_UNLOCKED:
        raise RuntimeError(
            "HOLDOUT LOCKED: The final_holdout split must not be read during strategy development.\n"
            "Call unlock_final_holdout(reason=...) after all strategy selection is complete.\n"
            "This is enforced to prevent look-ahead bias on the final evaluation."
        )

    cfg = SPLITS[split_name]
    path = cfg["path"]

    if not path.exists():
        raise FileNotFoundError(
            f"Split file not found: {path}\n"
            "Run build_splits.py first to create splits from raw data."
        )

    if verify_checksum:
        verify_splits_intact(splits=[split_name])

    df = pd.read_csv(path, parse_dates=["date"])
    return df


def sha256_file(path: Path) -> str:
    """Compute SHA-256 of a file's contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_splits(
    raw_path: Path = RAW_PATH,
    force: bool = False,
) -> dict:
    """
    Load raw OHLCV data and create train/validation/final_holdout splits.

    Returns: dict mapping split_name -> pd.DataFrame
    """
    if not raw_path.exists():
        print(f"ERROR: Raw data not found at {raw_path}")
        print("Run download_data.py first.")
        sys.exit(1)

    # Check if splits already exist and checksums are valid
    if not force and CHECKSUMS_PATH.exists():
        all_exist = all(cfg["path"].exists() for cfg in SPLITS.values())
        if all_exist:
            try:
                verify_splits_intact()
                print("Splits already exist and checksums are valid. Use --force to rebuild.")
                return {name: pd.read_csv(cfg["path"], parse_dates=["date"]) for name, cfg in SPLITS.items()}
            except AssertionError:
                print("Checksum mismatch detected — rebuilding splits.")

    print(f"Loading raw data from {raw_path} ...")
    df = pd.read_csv(raw_path, parse_dates=["date"])
    print(f"  {len(df)} rows, {df['symbol'].nunique()} symbols")

    checksums = {}
    result = {}

    for split_name, cfg in SPLITS.items():
        start = pd.Timestamp(cfg["start"])
        end = pd.Timestamp(cfg["end"])
        mask = (df["date"] >= start) & (df["date"] <= end)
        split_df = df[mask].copy().reset_index(drop=True)
        split_df.to_csv(cfg["path"], index=False)

        checksum = sha256_file(cfg["path"])
        checksums[split_name] = {
            "path": str(cfg["path"].name),
            "sha256": checksum,
            "start": cfg["start"],
            "end": cfg["end"],
            "rows": len(split_df),
            "symbols": split_df["symbol"].nunique() if "symbol" in split_df.columns else None,
            "description": cfg["description"],
        }

        n_trading_days = split_df["date"].nunique()
        print(
            f"  {split_name:15s}: {len(split_df):6d} rows, "
            f"{n_trading_days} trading days, "
            f"sha256={checksum[:12]}..."
        )
        result[split_name] = split_df

    # Save checksums
    with open(CHECKSUMS_PATH, "w") as f:
        json.dump(checksums, f, indent=2)
    print(f"\nChecksums saved to {CHECKSUMS_PATH}")
    print(
        "\nNOTE: final_holdout is LOCKED for strategy development. "
        "Call unlock_final_holdout() only when ready for the single-shot final evaluation."
    )
    return result


def verify_splits_intact(splits: list[str] | None = None) -> None:
    """
    Verify that split CSV files match their stored checksums.

    Args:
        splits: List of split names to verify. Defaults to all splits.

    Raises:
        AssertionError if any checksum fails.
        FileNotFoundError if checksum file or split file is missing.
    """
    if not CHECKSUMS_PATH.exists():
        raise FileNotFoundError(
            f"Checksum file not found: {CHECKSUMS_PATH}\n"
            "Run build_splits.py first."
        )

    with open(CHECKSUMS_PATH) as f:
        stored = json.load(f)

    to_check = splits or list(SPLITS.keys())

    all_ok = True
    for split_name in to_check:
        if split_name not in stored:
            raise KeyError(f"Split {split_name!r} not found in checksum file.")

        path = SPLITS[split_name]["path"]
        if not path.exists():
            raise FileNotFoundError(f"Split file missing: {path}")

        expected = stored[split_name]["sha256"]
        actual = sha256_file(path)

        if actual != expected:
            print(f"  FAIL {split_name}: expected {expected[:12]}..., got {actual[:12]}...")
            all_ok = False
        else:
            print(f"  OK   {split_name}: sha256={actual[:12]}...")

    assert all_ok, "One or more split checksums failed. Data may have been modified."
    print("All checksums verified.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build train/validation/holdout splits")
    parser.add_argument("--force", action="store_true", help="Rebuild splits even if they exist")
    parser.add_argument("--verify", action="store_true", help="Only verify checksums, do not rebuild")
    args = parser.parse_args()

    if args.verify:
        print("Verifying split checksums...")
        verify_splits_intact()
    else:
        splits = build_splits(force=args.force)
        print("\nSplit row counts:")
        for name, df in splits.items():
            print(f"  {name}: {len(df)} rows")
