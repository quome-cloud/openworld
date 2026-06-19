"""E75 (data + headroom check) - a HARDER diagnosis world family, to test whether the
world-time-compute gain grows with model size when big models are not near-saturated.

E74's family is easy enough that base accuracy plateaus ~0.72 for 7B+ against a 0.855
oracle -- little headroom, so a flat gain curve can't distinguish "fixed top-up" from
"intelligence amplification." This family is deliberately harder:
  - more diseases (D), more features (F), more signature features per disease;
  - WEAKER signatures (lower P(feature | signature disease));
  - HIGHER base rate (features fire for non-signature diseases too) and heavy signature
    OVERLAP across diseases -> confusable specialties.
Same shared goal + action grammar (so world-time compute still applies), just lower
achievable accuracy -> real headroom for capable models to show a bigger lift, if there is
one. Same prompt/eval format as E74, so e73_finetune.py + e74_eval.py run unchanged.

This script writes the hard SFT/test sets and prints the oracle/floor headroom so we can
confirm the design before spending GPU. Deterministic, offline.
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "experiments" / "results" / "e75_artifacts"

N_SPEC = 120
N_TRAIN_SPEC = 60
N_TEST_SPEC = 20
# --- hardness knobs (vs E74: D=6, F=12, N_SIG=3, sig 0.70-0.95, base 0.10) ---
# Tuned so the full-information oracle sits ~0.70 (vs E74's 0.855): clear headroom for
# capable models, while the task stays learnable (good dynamic range to detect a
# size-dependent gain).
D, F, N_SIGNATURE = 10, 20, 4
SIG_LO, SIG_HI = 0.58, 0.78       # weaker signatures
BASE_RATE = 0.18                  # higher background -> more false positives
TRAIN_PATIENTS = 24
TEST_PATIENTS = 40
SEED = 75


def make_specialty(seed):
    rng = np.random.RandomState(seed)
    M = np.full((D, F), BASE_RATE)
    for d in range(D):
        # independent signature choice -> heavy overlap across diseases (confusable)
        sig = rng.choice(F, N_SIGNATURE, replace=False)
        M[d, sig] = rng.uniform(SIG_LO, SIG_HI, N_SIGNATURE)
    prior = rng.dirichlet(np.ones(D) * 2.0)
    return {"M": M, "prior": prior}


def profiles(M):
    return {d: [f"feat_{f}" for f in range(F) if M[d, f] >= 0.5] for d in range(D)}


def make_prompt(prof, present):
    lines = [f"disease_{d} typically presents: {', '.join(prof[d]) or '(no strong markers)'}"
             for d in range(D)]
    return ("You are a diagnostician. Disease profiles for this specialty:\n"
            + "\n".join(lines)
            + f"\nPatient presents with: {', '.join(present) or '(none)'}.\n"
            + "Which single disease best explains it? Reply with ONLY 'disease_k'.")


def emit(spec, n, rng):
    prof = profiles(spec["M"])
    rows = []
    for _ in range(n):
        d = int(rng.choice(D, p=spec["prior"]))
        x = (rng.rand(F) < spec["M"][d]).astype(int)
        present = [f"feat_{f}" for f in range(F) if x[f]]
        rows.append({"prompt": make_prompt(prof, present), "completion": f"disease_{d}",
                     "answer": f"disease_{d}"})
    return rows


def oracle_and_floor(spec, n, rng):
    """Full-information Bayes with the TRUE M (ceiling) and prior-only (floor)."""
    M = np.clip(spec["M"], 1e-4, 1 - 1e-4)
    logM, log1M, logpri = np.log(M), np.log(1 - M), np.log(spec["prior"] + 1e-12)
    correct = 0
    for _ in range(n):
        d = int(rng.choice(D, p=spec["prior"]))
        x = (rng.rand(F) < spec["M"][d]).astype(int)
        lp = logpri + (x * logM + (1 - x) * log1M).sum(axis=1)
        correct += int(lp.argmax() == d)
    return correct / n, float(spec["prior"].max())


def main():
    ART.mkdir(parents=True, exist_ok=True)
    family = [make_specialty(SEED + i) for i in range(N_SPEC)]
    train = family[:N_TRAIN_SPEC]
    test = family[N_TRAIN_SPEC:N_TRAIN_SPEC + N_TEST_SPEC]
    rng = np.random.RandomState(SEED)

    sft = []
    for s in train:
        sft.extend({"prompt": r["prompt"], "completion": r["completion"]}
                   for r in emit(s, TRAIN_PATIENTS, rng))
    (ART / "sft_train_dx.jsonl").write_text("\n".join(json.dumps(r) for r in sft) + "\n")

    test_rows = []
    for si, s in enumerate(test):
        for r in emit(s, TEST_PATIENTS, rng):
            test_rows.append({"prompt": r["prompt"], "answer": r["answer"], "specialty": si})
    (ART / "test_dx.jsonl").write_text("\n".join(json.dumps(r) for r in test_rows) + "\n")

    oracles, floors = zip(*(oracle_and_floor(s, 300, rng) for s in test))
    print(f"[e75-data] hard family: D={D} F={F} N_SIG={N_SIGNATURE} sig[{SIG_LO},{SIG_HI}] base={BASE_RATE}")
    print(f"  {len(sft)} SFT / {len(test_rows)} test cases ({N_TEST_SPEC} held-out specialties)")
    print(f"  oracle ceiling {np.mean(oracles):.3f}  prior-only floor {np.mean(floors):.3f}  "
          f"(E74 was 0.855 / 0.34) -> headroom {'OPENED' if np.mean(oracles) < 0.80 else 'still high'}")


if __name__ == "__main__":
    main()
