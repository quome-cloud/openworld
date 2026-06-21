"""E79 (data) - WORLD-AS-A-JUDGE: the inversion of agent-as-a-judge. Instead of judging an
agent's behavior INSIDE a world, we judge the WORLDS an agent trains on, and curate the set
predicted to most improve generalization to a target problem.

De-risk phase: does CURATED world selection beat RANDOM selection at matched world count N?
We draw a large pool of candidate diagnosis specialties and select N of them three ways:

  random     -- uniform N from the pool (the domain-randomization baseline).
  diversity  -- greedy max-min coverage of the signature-pattern space (a cheap, non-LLM
                complementarity heuristic; pick worlds least like those already chosen).
  judge      -- world-as-a-judge: score(world | selected) = identifiability * complementarity,
                where identifiability = the world's oracle separability (is its rule learnable)
                and complementarity = how much NEW signature coverage it adds. Greedily keep
                the highest-scoring worlds (the >cutoff "keep it" policy). A principled,
                offline, validatable stand-in for an LLM judge predicting generalization value.

All three train on EXACT (verified) labels and are evaluated on the SAME fixed hard held-out
test family, so any gap isolates the effect of WHICH worlds were chosen, at equal N. Reuses
the E75 family + E78 emit helpers. Deterministic, offline (numpy only).
"""

import json
from pathlib import Path

import numpy as np

from e75_data import D, F, TRAIN_PATIENTS, make_specialty, profiles, oracle_and_floor
from e78_data import sample_patients, row_of, write_jsonl

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "experiments" / "results" / "e79_artifacts"
E75_TEST = ROOT / "experiments" / "results" / "e75_artifacts" / "test_dx.jsonl"

POOL = 2000                      # candidate worlds (free: parametric generator)
POOL_SEED_BASE = 500000         # disjoint from e75 (75..), e76 (3000+), e78 (10000*)
N_POINTS = [40, 160, 640]       # de-risk ladder
SEEDS = [0, 1]                  # fine-tune seeds (random arm also varies its subset)
ARMS = ["random", "diversity", "judge"]
BASE = "Qwen/Qwen2.5-0.5B-Instruct"


def signature_masks(specs):
    """Binary D*F 'which (disease,feature) are signatures' fingerprint per world."""
    return np.stack([(s["M"] >= 0.5).astype(np.int8).flatten() for s in specs])


def greedy_select(masks, ident, n, mode):
    """Greedy curation. diversity = max-min Hamming coverage; judge = identifiability *
    marginal coverage. Deterministic; seeded only by the (fixed) pool."""
    P, dim = masks.shape
    start = int(np.argmax(ident))                 # begin from the most learnable world
    selected = [start]
    min_d = (masks != masks[start]).sum(axis=1).astype(float)
    while len(selected) < n:
        if mode == "diversity":
            score = min_d.copy()
        else:                                     # judge: learnable AND complementary
            score = ident * (min_d / dim)
        score[selected] = -1.0
        nxt = int(np.argmax(score))
        selected.append(nxt)
        min_d = np.minimum(min_d, (masks != masks[nxt]).sum(axis=1))
    return selected


def emit_for(specs, sel, data_seed):
    rng = np.random.RandomState(data_seed)
    rows = []
    for idx in sel:
        spec = specs[idx]
        prof = profiles(spec["M"])
        for d, x in sample_patients(spec, TRAIN_PATIENTS, rng):
            rows.append(row_of(prof, x, d))
    return rows


def main():
    ART.mkdir(parents=True, exist_ok=True)
    assert E75_TEST.exists(), "run e75_data.py first (need the fixed hard held-out test set)"
    import shutil
    shutil.copy(E75_TEST, ART / "test_dx.jsonl")

    specs = [make_specialty(POOL_SEED_BASE + i) for i in range(POOL)]
    masks = signature_masks(specs)
    orng = np.random.RandomState(POOL_SEED_BASE)
    ident = np.array([oracle_and_floor(s, 120, orng)[0] for s in specs])  # learnability proxy
    print(f"[e79-data] pool={POOL} worlds; identifiability(oracle) "
          f"mean {ident.mean():.3f} range [{ident.min():.3f},{ident.max():.3f}]")

    manifest = []
    sel_stats = {}
    for arm in ARMS:
        for n in N_POINTS:
            if arm == "diversity" or arm == "judge":
                sel = greedy_select(masks, ident, n, arm)
                f = ART / f"e79_{arm}_N{n}.jsonl"
                write_jsonl(f, emit_for(specs, sel, 920000 + n))
                sel_stats[f"{arm}/N{n}"] = round(float(ident[sel].mean()), 4)
                for seed in SEEDS:
                    manifest.append({"arm": arm, "N": n, "seed": seed, "base": BASE,
                                     "file": f.name, "sel_oracle_mean": sel_stats[f"{arm}/N{n}"]})
            else:  # random: a fresh subset per seed
                for seed in SEEDS:
                    sel = np.random.RandomState(910000 + seed * 100 + n).choice(
                        POOL, n, replace=False)
                    f = ART / f"e79_random_N{n}_s{seed}.jsonl"
                    write_jsonl(f, emit_for(specs, sel, 910000 + seed * 100 + n))
                    manifest.append({"arm": arm, "N": n, "seed": seed, "base": BASE,
                                     "file": f.name,
                                     "sel_oracle_mean": round(float(ident[sel].mean()), 4)})

    (ART / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[e79-data] wrote {len(manifest)} SFT sets to {ART}")
    for k, v in sel_stats.items():
        print(f"  curated {k}: selected-world oracle mean {v}")


if __name__ == "__main__":
    main()
