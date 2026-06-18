"""E74 (LLM stage, data gen) - turn the diagnosis world family into an SFT set + a held-out
test set for the LLM 'superphysician' experiment.

Each example is one patient case: the prompt states the specialty's disease profiles (its
symptom->disease structure, i.e. the world's rules) and the patient's observed features; the
completion is the correct disease. Fine-tuning on TRAIN specialties teaches the transferable
SKILL (match a patient to the profiles); we then test on HELD-OUT specialties (new profiles)
-- the sklearn-style, world-level split. Pure text: eval needs no world stepping.

Writes experiments/results/e74_artifacts/{sft_train_dx.jsonl, test_dx.jsonl}. Deterministic.
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "experiments" / "results" / "e74_artifacts"

N_SPEC = 120
N_TRAIN_SPEC = 60          # specialties contributing SFT data
N_TEST_SPEC = 20           # held-out specialties for eval
D, F, N_SIGNATURE, BASE_RATE = 6, 12, 3, 0.10
TRAIN_PATIENTS = 24        # SFT cases per train specialty
TEST_PATIENTS = 40         # eval cases per held-out specialty
SEED = 74


def make_specialty(seed):
    rng = np.random.RandomState(seed)
    M = np.full((D, F), BASE_RATE)
    for d in range(D):
        sig = rng.choice(F, N_SIGNATURE, replace=False)
        M[d, sig] = rng.uniform(0.70, 0.95, N_SIGNATURE)
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

    print(f"[e74-data] {len(sft)} SFT examples from {len(train)} train specialties; "
          f"{len(test_rows)} test cases from {len(test)} held-out specialties")
    print(f"  wrote {ART/'sft_train_dx.jsonl'} and {ART/'test_dx.jsonl'}")


if __name__ == "__main__":
    main()
