# E45 - Inducing a verified world model from real repository history

**Date:** 2026-06-13
**Status:** approved (design); pending spec review

## Goal

Translate the paper's central result - verified-code world models are *exact*,
transfer OOD, and stay auditable where learned models approximate - from
synthetic oracle worlds onto **real, measured data**. We treat a software
repository as a world whose state evolves as commits apply diffs, induce its
dynamics with the framework's own LLM synthesis+verify pipeline from real
commit traces, and show the induced program recovers an exact empirical law
that holds on held-out and out-of-distribution real commits, where learned
baselines do not.

This is experiment #1 of the "real-world translation" thread; its boundary
findings (where the exact law fails on real data) motivate the residual-hybrid
(#2) and misspecification-detection (#3) follow-ons.

## The world

State at commit `t`, **measured independently from the repository tree** (this
independence is the crux - see Anti-circularity):

- `files`  - number of tracked blobs in the tree
- `py_files` - number of `.py` blobs
- `test_files` - number of blobs under a test path (`tests/`, `test_*`, `*_test`)
- `loc` - total newline count across text blobs (binary blobs excluded)

The **action** is the commit's real diff, measured from git independently of the
tree counts:

- `added`, `deleted`, `renamed`, `modified` - file-status counts (`name-status`)
- `insertions`, `deletions` - line churn (`numstat`)
- partitioned variants for the `.py` / test subsets

## The empirical laws to induce

Because state is measured independently of the action, these are real,
falsifiable claims about git - not identities-by-construction:

- `files_{t+1}   = files_t   + added - deleted`        (renames net zero)
- `py_files_{t+1}= py_files_t + py_added - py_deleted`
- `loc_{t+1}     = loc_t      + insertions - deletions` (text blobs only)

A capable local model, given real `(state, action, next_state)` triples and the
framework's verify-by-reproduction gate, should synthesize exactly these. The
induced transition is then evaluated on held-out and OOD commits.

## Anti-circularity (the experiment is worthless without this)

State MUST be measured from the repository tree (`git ls-tree -r <sha>` for
counts; blob-SHA-cached `git cat-file` newline counts for `loc`), NEVER by
cumulatively summing the diffs. Then "next state = state + diff" is a genuine
empirical law the framework discovers. The spec-review and self-checks must
confirm state and action come from independent git queries.

## Conditions / baselines (predict next_state from state+action features)

- **Symbolic (ours)** - `synthesize_transition(llm, ...)` over real training
  triples, verified by reproduction; the framework's faithful LLM path. The
  synthesized program is recorded in the results.
- **Linear regression** - least squares (numpy). NB: the law is linear, so this
  will fit it *closely* - the honest contrast is exactness, not MAE (below).
- **MLP** - small numpy MLP (reuse E36/E37 helper).
- **1-NN memorizer** - nearest training transition.

## Splits

- **In-distribution**: chronological train/test split on identity-holding
  commits.
- **OOD (large churn)**: hold out the top-decile commits by total churn; train
  only on smaller commits. 1-NN/MLP should degrade here; symbolic is invariant.

## Metrics

- **Exact-match accuracy** (primary, matches the paper's "exact probe
  accuracy"): fraction of held-out commits whose next_state is predicted
  *exactly* (integer-equal on every dimension). The symbolic program is exact
  on identity-holding commits (1.00); a fitted linear model with coefficients
  ~0.999 essentially never lands on the exact integers (~0.0), even though its
  MAE is small. This metric is *the* real-world version of the paper's thesis.
- **MAE** (secondary, fairness): per-dimension mean absolute error. Reported so
  we do not overclaim - linear MAE will be small.
- **Coverage / boundary**: fraction of all real commits on which the exact law
  holds; characterization of the violations (binary files via numstat `-`,
  renames, merge commits, line-ending churn). Reported honestly, not hidden.

## Cross-repo replication

Three real repos for a replication claim (not n=1): **requests**, **flask**,
**tqdm** (an independent, non-Pallets repo). Each mined to a bounded window
(most recent ~600 commits) for tractable independent `loc` measurement.

## Reproducibility

- `experiments/data/realrepo/<repo>.csv` - committed per-commit traces (state +
  action), so the experiment reruns fully offline; only the LLM synthesis path
  needs Ollama and its result is recorded in the committed JSON (same pattern as
  E37/E38).
- `experiments/mine_realrepo.py` - the miner that regenerates the CSVs from a
  fresh clone (needs network); documents exact repos, the commit window, and the
  git commands used.
- The analysis (`experiments/e45_real_repo_induction.py`) reads only the CSVs +
  records the synthesis result; learned baselines are deterministic numpy.

## Deliverables

- `experiments/mine_realrepo.py` + committed `experiments/data/realrepo/*.csv`.
- `experiments/e45_real_repo_induction.py` (+ `results/e45_real_repo_induction.json`).
- Figure (exact-match symbolic vs learned, in-dist vs OOD, across repos) + table
  (per-repo exact-match & MAE & coverage) via `scripts/make_paper_assets.py`.
- Paper subsection in a new "Real-world translation" area; `\NumExperiments`
  43 -> 44.
- PR (based on the E44 branch; merge after PR #26).

## Honest boundaries

- Recovering an accounting law is not deep ML - that is the point: the framework
  captures *exact* structure that learned models approximate, on real data, with
  OOD transfer and an auditable, verification-certified program.
- The interesting non-law dimensions (e.g. churn or test-count growth that is
  *not* a clean function of the action) are where pure symbolic hits its
  boundary; we report this rather than engineer it away, setting up #2/#3.
- Results are over a bounded recent window of three Python repos; we do not
  claim every repo or every state variable obeys a clean law.
