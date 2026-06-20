"""E78 (data) - powering the world-time-compute spine against the two strongest reviewer
objections:

  (a) MULTI-SEED WORLD-COUNT SCALING. E76 reported a single-seed curve with no CIs. Here we
      regenerate the world-count ladder with disjoint TRAIN world samples per seed, so the
      held-out gain at each world count carries a bootstrap/seed CI.

  (b) VERIFIED-vs-NOISY LABEL ABLATION. The paper's causal claim is that label EXACTNESS --
      not task variety -- is what makes world-time compute work (and what distinguishes it
      from domain randomization). We hold the worlds, the patients, the prompts, and the
      example count FIXED, and vary ONLY the labels: 'verified' uses the exact ground-truth
      diagnosis (what a verified code world model yields); 'noisy@p' replaces a fraction p of
      labels with a CONFUSABLE disease drawn from the posterior over the OTHER diseases --
      the structured error a non-verified / learned world model makes when its rules are
      slightly wrong. The held-out TEST labels are always exact, so any gap isolates the
      effect of training-label exactness.

Reuses the E75 hard diagnosis family. Deterministic, offline (numpy only). Writes SFT jsonl
artifacts + manifest.json; the GPU loop (e78_run.py) consumes the manifest.
"""

import json
from pathlib import Path

import numpy as np

from e75_data import (D, F, TRAIN_PATIENTS, SEED as E75_SEED, make_specialty,
                      make_prompt, profiles)

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "experiments" / "results" / "e78_artifacts"
E75_TEST = ROOT / "experiments" / "results" / "e75_artifacts" / "test_dx.jsonl"

SEEDS = [0, 1, 2]                       # disjoint train-world blocks + fine-tune seeds
N_GRID = [8, 16, 32, 64, 128, 256, 512]
SECOND_SIZE_NS = [8, 128, 512]         # 1.5B confirmation grid (subset, to bound compute)
ABL_N = 256                            # ablation fixes the world count where verified is solid
ABL_NOISE = [0.15, 0.30]
SEED_BLOCK = 10000                     # seed s draws specialties [SEED_BLOCK*(s+1) + i]
BASE_SMALL = "Qwen/Qwen2.5-0.5B-Instruct"
BASE_MED = "Qwen/Qwen2.5-1.5B-Instruct"


def sample_patients(spec, n, rng):
    """Sample n (true_disease, presentation) patients from a specialty (the exact world)."""
    pts = []
    for _ in range(n):
        d = int(rng.choice(D, p=spec["prior"]))
        x = (rng.rand(F) < spec["M"][d]).astype(int)
        pts.append((d, x))
    return pts


def row_of(prof, x, label):
    present = [f"feat_{f}" for f in range(F) if x[f]]
    return {"prompt": make_prompt(prof, present), "completion": f"disease_{label}"}


def noisy_label(spec, d, x, p, rng, logpri, logM, log1M):
    """With prob p, replace the true disease d with a confusable one drawn from the
    posterior over the OTHER diseases (a structured, learned-world-model-style error)."""
    if rng.rand() >= p:
        return d
    lp = logpri + (x * logM + (1 - x) * log1M).sum(axis=1)
    post = np.exp(lp - lp.max())
    post[d] = 0.0
    s = post.sum()
    return int(rng.choice(D, p=post / s)) if s > 0 else d


def write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def main():
    ART.mkdir(parents=True, exist_ok=True)
    assert E75_TEST.exists(), "run e75_data.py first (need the fixed hard held-out test set)"
    import shutil
    shutil.copy(E75_TEST, ART / "test_dx.jsonl")

    manifest = []

    # ---- (a) multi-seed world-count scaling --------------------------------------------
    for size, base, grid in [("small", BASE_SMALL, N_GRID), ("med", BASE_MED, SECOND_SIZE_NS)]:
        for s in SEEDS:
            rng = np.random.RandomState(E75_SEED + s)
            specs = [make_specialty(SEED_BLOCK * (s + 1) + i) for i in range(max(grid))]
            # one fixed patient block per specialty -> larger N is a true superset
            per_spec = []
            for spec in specs:
                prof = profiles(spec["M"])
                pts = sample_patients(spec, TRAIN_PATIENTS, rng)
                per_spec.append([row_of(prof, x, d) for d, x in pts])
            for n in grid:
                rows = [r for block in per_spec[:n] for r in block]
                f = ART / f"scal_{size}_s{s}_N{n}.jsonl"
                write_jsonl(f, rows)
                manifest.append({"kind": "scaling", "size": size, "base": base,
                                 "seed": s, "N": n, "file": f.name, "n_examples": len(rows)})

    # ---- (b) verified-vs-noisy label ablation (small model) ----------------------------
    for s in SEEDS:
        rng = np.random.RandomState(E75_SEED + 100 + s)
        specs = [make_specialty(SEED_BLOCK * (s + 1) + i) for i in range(ABL_N)]
        verified, noisy = [], {p: [] for p in ABL_NOISE}
        flipped = {p: 0 for p in ABL_NOISE}
        total = 0
        for spec in specs:
            prof = profiles(spec["M"])
            M = np.clip(spec["M"], 1e-4, 1 - 1e-4)
            logM, log1M = np.log(M), np.log(1 - M)
            logpri = np.log(spec["prior"] + 1e-12)
            pts = sample_patients(spec, TRAIN_PATIENTS, rng)  # SAME patients across conditions
            for d, x in pts:
                verified.append(row_of(prof, x, d))
                total += 1
                for p in ABL_NOISE:
                    lab = noisy_label(spec, d, x, p, rng, logpri, logM, log1M)
                    noisy[p].append(row_of(prof, x, lab))
                    flipped[p] += int(lab != d)
        fv = ART / f"abl_s{s}_verified.jsonl"
        write_jsonl(fv, verified)
        manifest.append({"kind": "ablation", "size": "small", "base": BASE_SMALL, "seed": s,
                         "cond": "verified", "noise": 0.0, "realized_flip": 0.0,
                         "file": fv.name, "n_examples": len(verified)})
        for p in ABL_NOISE:
            tag = f"noisy{int(p * 100)}"
            fp = ART / f"abl_s{s}_{tag}.jsonl"
            write_jsonl(fp, noisy[p])
            manifest.append({"kind": "ablation", "size": "small", "base": BASE_SMALL,
                             "seed": s, "cond": tag, "noise": p,
                             "realized_flip": round(flipped[p] / total, 4),
                             "file": fp.name, "n_examples": len(noisy[p])})

    (ART / "manifest.json").write_text(json.dumps(manifest, indent=2))
    n_scal = sum(1 for m in manifest if m["kind"] == "scaling")
    n_abl = sum(1 for m in manifest if m["kind"] == "ablation")
    print(f"[e78-data] wrote {len(manifest)} SFT sets ({n_scal} scaling, {n_abl} ablation) "
          f"to {ART}")
    for m in manifest:
        if m["kind"] == "ablation":
            print(f"  ablation s{m['seed']} {m['cond']:>8}: {m['n_examples']} ex, "
                  f"realized flip {m['realized_flip']}")


if __name__ == "__main__":
    main()
