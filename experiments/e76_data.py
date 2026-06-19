"""E76 (data) - world-count scaling for world-time compute: does fine-tuning on MORE
traversed worlds keep improving held-out generalization, or does it saturate?

We fix the model and the (hard) family and vary only the number of TRAIN specialties the
model is fine-tuned on, from a handful up to many hundreds (generation is free -- the
diagnosis family is a parametric generator). The held-out TEST set is the SAME E75 hard
specialties throughout, so every point is comparable. Train seeds are disjoint from the
test specialties.

Writes experiments/results/e76_artifacts/sft_train_N{n}.jsonl for each n in N_GRID, plus a
copy of the fixed hard test set. Reuses the E75 hard-family generator so hardness matches.
Deterministic, offline.
"""

import json
import shutil
from pathlib import Path

import numpy as np

from e75_data import make_specialty, profiles, make_prompt, TRAIN_PATIENTS, D

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "experiments" / "results" / "e76_artifacts"
E75_TEST = ROOT / "experiments" / "results" / "e75_artifacts" / "test_dx.jsonl"

N_GRID = [8, 16, 32, 64, 128, 256, 512]
TRAIN_SEED_BASE = 3000           # disjoint from E75 seeds (75..194)
SEED = 76


def emit_specialty(spec, n, rng):
    prof = profiles(spec["M"])
    rows = []
    for _ in range(n):
        d = int(rng.choice(D, p=spec["prior"]))
        x = (rng.rand(len(spec["M"][0])) < spec["M"][d]).astype(int)
        present = [f"feat_{f}" for f in range(len(x)) if x[f]]
        rows.append({"prompt": make_prompt(prof, present), "completion": f"disease_{d}"})
    return rows


def main():
    ART.mkdir(parents=True, exist_ok=True)
    assert E75_TEST.exists(), "run e75_data.py first (need the fixed hard test set)"
    shutil.copy(E75_TEST, ART / "test_dx.jsonl")

    max_n = max(N_GRID)
    train_specs = [make_specialty(TRAIN_SEED_BASE + i) for i in range(max_n)]
    rng = np.random.RandomState(SEED)

    # cumulative SFT: the first n specialties (so larger n is a superset)
    for n in N_GRID:
        rows = []
        for s in train_specs[:n]:
            rows.extend(emit_specialty(s, TRAIN_PATIENTS, rng))
        (ART / f"sft_train_N{n}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        print(f"[e76-data] N={n:>4} specialties -> {len(rows)} SFT examples")
    print(f"  fixed hard test set copied from {E75_TEST.name}")


if __name__ == "__main__":
    main()
