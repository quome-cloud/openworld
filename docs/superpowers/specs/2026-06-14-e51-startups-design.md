# E51 - Startup growth world model (YC-style batch)

**Date:** 2026-06-14
**Status:** approved (design)

## Goal

The E48 composite machinery applied to a cohort of startups: model what drives
growth, the power law of outcomes, and how predictable winners are - and make it
usable on a real startup call transcript for an investor.

## Model

A batch = `CompositeWorld` of `N` startup child-worlds. Per-startup factors: team,
market (TAM), product-market fit (PMF), grit, capital. Monthly dynamics: growth
is multiplicative in PMF (no fit -> no growth) x market x team x grit; revenue
compounds; burn depletes runway; traction-gated fundraising extends it;
idiosyncratic shocks (~1%/mo, factor-independent) inject tail luck; death when
cash runs out without revenue. Batch `Aggregator`s: total value, survival rate,
top-decile value share. Deterministic/offline/self-checking.

## Results

1. **Value-of-factor (causal)**: lift each factor to the 90th percentile across
   the batch; PMF moves total value most, capital (even doubled) barely moves it.
2. **Power law of returns**: a minority survive and the top decile captures the
   large majority of value (Gini high).
3. **Counterfactual attribution**: no-PMF batch collapses; doubling capital at
   low PMF does not - spend cannot substitute for fit.
4. **Honest predictability**: month-6 traction is informative but not decisive
   (Spearman < 1; a chunk of eventual winners were not early leaders) - more
   predictable than markets (E50), still luck-laden in the tail.

## Investor use (transcript -> diagnostic)

`datasets/openworld-startup/`: a sample pitch-call transcript and
`investor_diagnostic.py` - an LLM `TextPerceptor` extracts the factors from a
call transcript (e.g. a Google Meet export), the model forward-simulates THIS
startup many times, and it reports the factor read, the outcome distribution
(survival, median/p90), and the **binding constraint** (which factor, lifted,
moves the expected outcome most). Honest caveats: noisy one-call LLM read;
stylized model; not investment advice.

## Deliverables

`experiments/e51_startups.py` (+ results); `datasets/openworld-startup/`
(sample transcript + investor_diagnostic.py); figure + table + paper subsection
`sec:startups`; `\NumExperiments` -> 50. PR targets `main`.
