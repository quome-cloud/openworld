#!/usr/bin/env python3
"""CLI wrapper for the E145 source-free episodic-memory controller."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.e145.sourcefree_controller import main


if __name__ == "__main__":
    raise SystemExit(main())

