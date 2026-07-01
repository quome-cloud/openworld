#!/usr/bin/env python3
"""CLI wrapper for the E143 source-free transition miner."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.e143.transition_miner import main


if __name__ == "__main__":
    raise SystemExit(main())
