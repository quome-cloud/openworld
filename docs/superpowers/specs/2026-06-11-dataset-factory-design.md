# The OpenWorld dataset factory and recipe standard: design

**Date:** 2026-06-11
**Status:** approved (design); sub-projects to be planned individually

## Goal

Scale from two hand-built SWE coding world-model datasets to an open-ended
family of them, and make every dataset + result reproducible the way the ML
field expects: pinned recipes, one-command regeneration, auto-generated
dataset cards, and a frozen result schema.

## The core insight

The validation gate built for openworld-swebench — *reference must solve both
suites; buggy must fail every `fail_to_pass` and pass every `pass_to_pass`;
output must rebuild byte-identically* — converts **untrusted instance
sources into trusted benchmark data**. Once the gate is the quality bar, the
provenance of an instance (a 7B model, a procedural template, a mined corpus,
a human) stops mattering. Scaling datasets therefore means scaling
*generators* behind one gate, not scaling authoring effort.

The week's parallel-development experience is the cautionary tale this design
answers: three concurrent implementations of the same spec produced schema
drift (`summary` vs `summaries`, `saw_regression` vs
`regression_failures_seen`), divergent runners, merge conflicts, and results
that could not be compared without archaeology. The recipe standard exists to
make that structurally impossible.

## 1. The dataset matrix

A dataset is a point in five axes:

| axis | values |
|---|---|
| source | hand, llm, parametric, mined |
| staging depth | 1 (atomic), 2, 3–5 (deep chains) |
| code scale | function, module, multi-module package |
| feedback richness | counts-only, +error strings (default), +traces |
| domain | data structures, parsers, state machines, numeric, schedulers, ... |

Naming: `owsb-<source-or-theme>-<distinguisher>-v<N>` (e.g.
`owsb-llm-3stage-module-v1`). Versions are immutable: a changed instance set
is a new version, never an in-place edit.

Initial roadmap (one recipe each):

| dataset | source | exercises |
|---|---|---|
| `owsb-atomic` (exists: v0 n=6, v1 n=20) | hand | baseline; capability ladder |
| `owsb-staged` (exists: n=15) | hand | 2-stage; in-world lift (E29) |
| `owsb-deep` | llm | 3–5 stage latent chains — does Δ grow with depth? |
| `owsb-param-*` | parametric | difficulty dials; scaling curves; volume |
| `owsb-quixbugs` | mined | external validity / decontamination anchor |
| `owsb-package` | llm | multi-module code scale |
| `owsb-feedback` | reuse staged | feedback-richness ablation (harness knob) |
| `owsb-flaky` | parametric | seeded stochastic tests (extends E21) |

## 2. Generator plugins

All generators emit the existing `SWEBenchInstance` schema and feed the same
gate.

- **`hand`** — today's `build_tasks.py` instance dicts, unchanged, as a
  first-class plugin.
- **`llm`** — archetype prompt templates parameterized by (domain,
  bug-family, staging-depth, code-scale). The generating model and its
  Ollama digest are pinned in the recipe. Expected workflow: generate K
  candidates, keep gate survivors; rejection is cheap, so low acceptance is
  acceptable. The generating model must be recorded so eval-model overlap
  can be disclosed on the card (don't benchmark a model on instances it
  generated without saying so).
- **`parametric`** — pure-Python families `make_instance(seed, difficulty)`
  emitting endless controlled variations (N interacting off-by-ones, k-state
  machines, etc.), in the spirit of E20's parametric rule worlds. The only
  source with exact difficulty knobs; powers scaling-law plots.
- **`mined`** — adapters from real bug corpora, vendored under their
  licenses, filtered for sandbox fit (restricted builtins + `math`; no
  imports). QuixBugs first (MIT, single-function Python, nearly
  sandbox-ready); BugsInPy/HumanEvalFix later if the sandbox-fit yield
  justifies it.

## 3. Validation gate v2

Today's invariants, plus:

1. **Staging verification** — generalize the staged set's `STAGE1_PATCHES`
   to an N-stage patch ladder: for each declared stage k, a provided
   stage-k patch must pass the tests of stages 1..k and fail stage k+1.
   This is what makes "staging depth" a verified property rather than an
   intention.
2. **Difficulty calibration** — a pinned calibration model (e.g.
   `qwen2.5:1.5b` at a recorded digest) runs single-shot on every candidate;
   instances solved above a recipe-set threshold are rejected to preserve
   headroom. Calibration results ship with the card.
3. **Leak check** — the issue text must not contain the fix (token-overlap
   check between issue and the buggy→reference diff).
4. **Dedup** — normalized-AST hash of buggy sources, checked across *all*
   datasets in the repo, so volume generation can't silently repeat itself.
5. **Determinism** — rebuild from the recipe is byte-identical (existing
   check, kept).

Gate pass/fail statistics per generator run are recorded and surfaced on the
dataset card.

## 4. The recipe standard

One YAML per dataset under `recipes/`, one runner:

```
python -m openworld.bench recipes/owsb-deep-v1.yaml {build|validate|run|card|all}
```

Recipe fields (frozen `schema_version: 1`):

- `dataset`: name, version, one-line description.
- `generator`: type (hand/llm/parametric/mined), type-specific config,
  `seed`, and for `llm` the model name + Ollama digest + quantization.
- `gate`: calibration model + digest, solve-rate threshold, staging depth,
  dedup scope.
- `harness`: the openworld git SHA (or package version) the dataset was
  built and evaluated with.
- `eval`: ladder model names + digests, budget, temperature, seed.
- `artifacts`: sha256 of `tasks.jsonl` and of the results files (filled in
  by the runner; the manifest is how drift is detected).

Outputs per run:

- `datasets/<name>/tasks.jsonl` — the artifact (committed).
- `datasets/<name>/CARD.md` — auto-generated dataset card: provenance,
  recipe hash, gate statistics, calibration results, license notes for
  mined sources, decontamination statement, known limitations.
- `datasets/<name>/results/<model>@<digest>.json` — one file per
  (model, recipe-hash), in the **frozen result schema**: per-instance paired
  records (always — exact McNemar tests must stay possible), aggregate rates
  with Wilson CIs, and the recipe hash they were produced under.

The two existing datasets are retrofitted into recipes first, which is also
the act that unifies the divergent runners (`run_comparison.py` variants and
the contextbench runner fold into `openworld.bench`).

### Reproducibility tiers (documented on every card)

- **Tier 0 — structural:** the mock path (`--mock`) runs in pytest on every
  commit; free and deterministic.
- **Tier 1 — artifact:** `bench build` regenerates `tasks.jsonl`
  byte-identically from the recipe.
- **Tier 2 — statistical:** rerunning `bench run` with the same Ollama
  digest reproduces results within the stated CIs (temperature noise is
  acknowledged, not hidden; absolute rates are documented as specific to the
  pinned quantized snapshots, as the paper already states).

## 5. Sub-project decomposition (build order)

Each is its own spec → plan → implementation cycle:

1. **`openworld.bench` runner + recipe schema**, retrofit `owsb-atomic` and
   `owsb-staged` (foundation; unifies runners and freezes the result
   schema).
2. **Gate v2** (N-stage verification, calibration, leak check, dedup).
3. **Parametric generator** + first `owsb-param-*` family (fastest volume;
   enables scaling curves).
4. **LLM generator** + `owsb-deep` (the scientifically interesting axis:
   Δ vs staging depth).
5. **QuixBugs adapter** + `owsb-quixbugs` (external validity).
6. **Feedback-richness knob** in the harness + `owsb-feedback` ablation.

**Prerequisite:** reconcile PR #4 (20-instance atomic) and the unmerged
staged branch into one canonical `openworld/swebench.py` before sub-project
1, so the factory builds on a single harness.

## Out of scope (this design)

- External publishing (HuggingFace Hub, leaderboard submissions, Docker
  images) — deliberately deferred; the recipe standard is designed so this
  can be added later without rework.
- Multi-file/multi-repo instances with real package installs (breaks the
  zero-dependency sandbox; revisit only after `owsb-package`).
- Non-Python languages.
