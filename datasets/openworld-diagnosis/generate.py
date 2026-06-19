"""Regenerate the openworld-diagnosis dataset deterministically.

Re-runs the committed, seeded parametric generators (no LLM/Claude Code) and rematerializes
the easy/, hard/, and world_count/ splits identically. Single source of truth: the
experiments/ generators -- this script just runs them and lays the files out as the dataset.

    python datasets/openworld-diagnosis/generate.py
"""

import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RES = ROOT / "experiments" / "results"
sys.path.insert(0, str(ROOT / "experiments"))

import e74_data  # noqa: E402  (easy family)
import e75_data  # noqa: E402  (hard family)
import e76_data  # noqa: E402  (world-count scaling; imports e75_data)

WORLD_COUNTS = e76_data.N_GRID


def cp(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst)


def main():
    e74_data.main()
    e75_data.main()
    e76_data.main()
    cp(RES / "e74_artifacts" / "sft_train_dx.jsonl", HERE / "easy" / "sft_train.jsonl")
    cp(RES / "e74_artifacts" / "test_dx.jsonl", HERE / "easy" / "test.jsonl")
    cp(RES / "e75_artifacts" / "sft_train_dx.jsonl", HERE / "hard" / "sft_train.jsonl")
    cp(RES / "e75_artifacts" / "test_dx.jsonl", HERE / "hard" / "test.jsonl")
    for n in WORLD_COUNTS:
        cp(RES / "e76_artifacts" / f"sft_train_N{n}.jsonl", HERE / "world_count" / f"sft_train_N{n}.jsonl")
    cp(RES / "e76_artifacts" / "test_dx.jsonl", HERE / "world_count" / "test.jsonl")
    print("regenerated openworld-diagnosis dataset (easy/, hard/, world_count/)")


if __name__ == "__main__":
    main()
