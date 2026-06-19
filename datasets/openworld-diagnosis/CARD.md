# OpenWorld-Diagnosis

A parametric **family of diagnosis world-models** for studying *world-time compute* —
whether fine-tuning a model on trajectories traversed through many verified world-models
makes it generalize to **unseen** worlds from fewer real examples (OpenWorld experiments
E74–E76; paper §"World-time compute").

## What it is

Each *specialty* is a diagnosis **POMDP**: a hidden disease `d`; binary features
(symptoms/tests) with disease-specific emission probabilities `M[d, f]`; the task is to read
the specialty's disease profiles and a patient's observed features and name the disease.
All specialties share one template (and action grammar); they differ only in their
symptom→disease structure `M` and disease prior. The diagnostic *skill* (match a patient to
the profiles) is therefore transferable across specialties, while each specialty's facts are
specialty-specific — which is exactly what makes "learn the skill on train specialties,
test on held-out specialties" a meaningful generalization measure.

## Provenance — generated, not authored

**These specialties are deterministic, seeded numpy parameterizations — NOT produced by
Claude Code or any LLM.** `make_specialty(seed)` draws a random emission matrix `M`
(disease × feature) and a Dirichlet prior from a fixed `RandomState(seed)`, so every
specialty is reproducible bit-for-bit from its integer seed. (This is distinct from the 100
*sector recipes* elsewhere in OpenWorld, which were authored by Claude Code from one-line
prompts. This family is synthetic-parametric on purpose: controlled, infinitely scalable,
and equipped with a clean ground-truth oracle.)

## Splits (scikit-learn style: whole worlds are held out)

The test set is composed of **entire specialties never seen in training** — generalization
means competence on *new worlds*, not new states of a familiar world.

| Family | Diseases × Features (sig) | Sig prob / base rate | Bayes oracle | Prior floor | Train / Test specialties |
|---|---|---|---|---|---|
| `easy/` (E74) | 6 × 12 (3) | 0.70–0.95 / 0.10 | ~0.86 | ~0.34 | 60 / 20 |
| `hard/` (E75) | 10 × 20 (4) | 0.58–0.78 / 0.18 | ~0.69 | ~0.23 | 60 / 20 |
| `world_count/` (E76) | hard params | — | ~0.69 | ~0.23 | 8…512 / 20 (fixed hard test) |

`world_count/` varies only the **number of train specialties** (`sft_train_N{8..512}.jsonl`)
against a fixed held-out hard test, to measure whether *more worlds traversed* keeps lifting
held-out accuracy. Train seeds are disjoint from the test specialties in every split.

## Schema (JSONL)

- `sft_train*.jsonl`: `{"prompt": <profiles + patient features + instruction>, "completion": "disease_k"}`
- `test.jsonl`: `{"prompt": ..., "answer": "disease_k", "specialty": <int test-specialty id>}`

The prompt is self-contained: it states the specialty's disease profiles and the patient's
present features, and asks for a single `disease_k`.

## How to use

Fine-tune on `sft_train`, evaluate diagnostic accuracy on `test` (held-out specialties).
Reference points: the **oracle** (full-information Bayes with the true `M`) is the ceiling;
**prior-only** (guess the most likely disease) is the floor. Both are computable offline.

## Reproduce

```
python datasets/openworld-diagnosis/generate.py
```

Deterministic; re-runs the committed seeded generators
(`experiments/e74_data.py`, `e75_data.py`, `e76_data.py`) and rematerializes every split
identically. The full meta-learning / oracle analysis is `experiments/e74_diagnosis.py`.

## Results & paper

Evaluation artifacts live in `experiments/results/` (`e74_scaling.json`,
`e74_diagnosis.json`, `e75_artifacts/`, `e76_*`, `e74_gemini.json`) and are written up in the
paper's "World-time compute" section.

## Caveats

Synthetic data — *not* real clinical records. The oracle is Bayes-optimal on the generated
`M`, so accuracies are bounded by the (deliberately tuned) task difficulty, not by medicine.

## License

MIT (same as the OpenWorld repository).
