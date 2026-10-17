# E-market v2 — Cross-Sectional World-Model Synthesis at Max Reasoning Tier

**Status: IN PROGRESS — sections below the pre-registration line are frozen before the POST-CUTOFF gate runs.**

RESEARCH ONLY. Paper trading and analysis exclusively — no real capital, no brokerage
interaction, nothing here is investment advice. Compliance standing rule per T385.

Synthesis model: Fable 5 (max reasoning). Date: 2026-07-06.
Work dir: `work/fable_market_v2/`. Prior art: T374 / `work/market_sandbox/` /
openworld PR #171 (`contributions/researchy_market_worldmodel/`).

---

## 1. v1 diagnosis recap (why v2 exists)

T374/v1 tested 7 single-asset daily-timing strategies (RSI, SMA, momentum, vol
regime, breadth, 2 sentiment variants) on SPY with a benchmark-adjusted excess
Sharpe + permutation gate. Result: **0/7 passed**; 3/5 price strategies had
negative out-of-sample excess; the best (WM04, RSI+SMA200) had positive excess
in both validation (+1.99) and holdout (+1.88) but permutation p=0.52 — with
312 daily observations the test cannot separate a real 2-Sharpe edge from luck.

Four diagnosed failure causes → four v2 corrections:

| # | v1 failure | v2 correction |
|---|---|---|
| 1 | Statistical power: 1 asset × 312 days | Cross-sectional panel: ~490 S&P names × ~600 days; long-short decile portfolios; within-date permutation null |
| 2 | Hypothesis class: decades-decayed textbook indicators | Structural world models: sector lead-lag graph, peer-linkage diffusion, volume-conditioned liquidity provision, earnings-event drift |
| 3 | Synthesis tier: Sonnet-synthesized strategies | Fable-max synthesis (this document) |
| 4 | Contamination: backtests overlap the synthesis model's training data | Scored window is exclusively POST-CUTOFF (2026-02-01 →); pre-cutoff data is DEV-labeled, used for fitting and sanity checks only |

Operator restructure (2026-07-06, supersedes strategy-family framing where in
conflict): the core artifact is a **feature library** (each signal a documented
equation with a unit test), a **transparent model layer** (walk-forward ridge
with feature-importance readout), and **drop-one ablations** so the result
identifies *which* structure carries signal.

## 2. Architecture

```
data/            cached raw downloads (re-runnable offline)
code/features/   feature library — 1 module per family, docstring = equation +
                 units + causal story + data deps; synthetic-data unit tests
code/model.py    walk-forward cross-sectional ridge (monthly refits, h-day
                 embargo, time-ordered CV for lambda from a fixed grid)
code/gate.py     LS decile portfolio (overlapping 5d tranches, 10 bps/side),
                 permutation + bootstrap machinery
code/run_eval.py DEV and POST stages; POST refuses to run without the
                 pre-registered prediction file
```

Universe: S&P 500 constituents (Wikipedia, fetched 2026-07-06 18:54 UTC, GICS
sectors from the same table; provenance in `data/universe_provenance.json`).
Prices: Yahoo Finance daily auto-adjusted OHLCV via yfinance 1.5.1, 2023-06-01
onward (`data/fetch_log.json` for timestamps + per-batch status). Earnings:
yfinance `get_earnings_dates` per symbol (`data/earnings_log.json`).

Target: 5-day forward close-to-close return, cross-sectionally demeaned per
date. Positions formed from features computable at close *t* earn returns from
*t+1*; earnings timestamps map after-close announcements to the next trading
day.

**Splits.** DEV = through 2026-01-31 (synthesis model's knowledge cutoff:
January 2026). POST-CUTOFF (scored) = 2026-02-01 → 2026-07-06. Walk-forward
fits may consume all data before each monthly refit boundary (that is fitting,
not scoring); every reported table is labeled DEV or POST.

## 3. Feature dictionary

<!-- FEATURE_DICTIONARY -->

| feature | family | module | summary |
|---|---|---|---|
| `sll_daily` | SLL | `sll_sector_leadlag.py` | FEATURE FAMILY: SLL — Sector lead-lag graph. |
| `sll_weekly` | SLL | `sll_sector_leadlag.py` | FEATURE FAMILY: SLL — Sector lead-lag graph. |
| `peer_mom_gap` | PEER | `peer_momentum.py` | FEATURE FAMILY: PEER — Correlation-graph peer momentum gap. |
| `rev_5d` | REV | `reversal_volume.py` | FEATURE FAMILY: REV — Short-term reversal conditioned on volume. |
| `rev_lowvol` | REV | `reversal_volume.py` | FEATURE FAMILY: REV — Short-term reversal conditioned on volume. |
| `pead_drift` | PEAD | `pead_earnings_drift.py` | FEATURE FAMILY: PEAD — Post-earnings-announcement drift. |

Causal stories (written before any fitting; full statements in the module
docstrings):

- **SLL — sector lead-lag graph.** Information diffuses gradually across
  economically linked sectors (Hong-Stein 1999; Menzly-Ozbas 2010): a ridge-
  estimated 11×11 map from lagged sector returns to next-period sector
  relative returns, walk-forward, at daily and weekly lags. Claim: the map is
  stable enough out-of-sample that member stocks of a predicted-to-outperform
  sector outperform peers over the next week.
- **PEER — correlation-graph peer momentum gap.** Investors under-attend to
  news arriving via a firm's economic neighbors (Cohen-Frazzini 2008). Proxy
  the linkage graph by trailing 126d return correlation (top-10 peers, refit
  monthly, graph frozen before each month). Claim: when peers rallied over the
  past month and the stock didn't, it catches up.
- **REV — volume-conditioned short-term reversal.** Liquidity provision: 5-day
  moves on unusually low volume are flow, not news, and revert (Nagel 2012);
  high-volume moves are information and do not. The volume interaction — not
  plain reversal — is the world-model claim; plain `rev_5d` is included so the
  model and ablation can separate the two.
- **PEAD — post-earnings-announcement drift.** Prices underreact to EPS
  surprises; drift persists for weeks after the announcement (Ball-Brown 1968;
  Bernard-Thomas 1989). Surprise% at the last announcement, linearly decayed
  over 60 trading days from the first tradeable close.

Unit tests (`code/tests/test_features.py`): each family is verified on a
synthetic panel with planted structure (planted sector lead-lag recovered;
planted peer diffusion recovered; planted low-volume-only reversal recovered;
PEAD event-to-date mapping incl. AMC/BMO handling and 60d expiry), plus a
truncation-invariance check (feature values at t unchanged when future rows
are deleted) as the no-lookahead test. All pass.

## 4. Model layer

Walk-forward ridge regression, refit at each month start on all panel days
whose 5-day forward-return window closes strictly before that month (embargo),
lambda from {1e2,1e3,1e4,1e5} by time-ordered 5-fold CV inside the training
window. Features cross-sectionally z-scored per date, missing = 0 (neutral).
The model is linear in meaningful features: the standardized coefficients ARE
the world-model readout, reported per refit.

Variants evaluated: `FULL` (all features), `SOLO_<family>` (that family alone —
this is the per-family gate verdict), `DROP_<family>` (drop-one ablation —
marginal contribution within the full model).

## 5. Gate definition (frozen before POST run)

Per variant, on POST-CUTOFF daily net LS returns (10 bps per side on turnover,
overlapping 5-day tranches, decile long-short, equal-weight within decile):

- (ii) net annualized LS Sharpe > 0;
- (iii) within-date cross-sectional permutation test, 500 shuffles, full
  pipeline re-run per shuffle (preserves market factor + prediction
  distribution, kills cross-sectional information), compared on GROSS
  Sharpe: p < 0.05. *Null-design correction, made at DEV calibration before
  pre-registration:* within-date shuffling destroys signal persistence, so
  shuffled books turn over ~fully daily; on NET returns the null centers near
  Sharpe −5.5 (measured on random predictions, 0.75 daily turnover), which
  would let any persistent-but-useless signal pass on cost savings alone.
  Gross-vs-gross removes that asymmetry; costs are enforced by (ii). A
  turnover-preserving robustness null (static ticker relabeling, compared on
  NET Sharpe) is reported alongside;
- (iv) benchmark variant reported: long-only top decile minus equal-weight
  universe, net;
- (v) everything after costs. Plus circular block bootstrap (block 10, 2000
  resamples) 95% CI on net Sharpe, and the permutation-null-implied minimum
  detectable effect.

**PASS = (ii) and (iii) jointly, after costs, POST window only.** 0/N passing
is a reportable result.

---

## 6. DEV sanity results (pre-cutoff — NOT scored)

<!-- DEV_RESULTS -->

Window: 2025-02-04 → 2026-01-30 (249 pnl days). n_perm=200. EW-universe benchmark annualized return: +16.2%.

| variant | net Sharpe | 95% CI | gross Sharpe | perm p (gross) | relabel p (net) | long−bench net Sharpe | turn/day | PASS |
|---|---|---|---|---|---|---|---|---|
| FULL | -1.84 | [-3.62, -0.30] | -0.58 | 0.706 | 0.035 | -1.24 | 0.67 | fail |
| SOLO_PEAD | +1.24 | [-0.73, +2.98] | +1.66 | 0.055 | 0.055 | +1.41 | 0.13 | fail |
| SOLO_PEER | -0.64 | [-2.62, +1.09] | -0.08 | 0.547 | 0.114 | +0.40 | 0.34 | fail |
| SOLO_REV | -0.98 | [-2.74, +0.23] | -0.18 | 0.567 | 0.005 | -0.20 | 0.70 | fail |
| SOLO_SLL | -1.57 | [-3.22, +0.02] | -0.41 | 0.706 | 0.005 | -1.66 | 0.71 | fail |
| DROP_PEAD | -1.94 | [-3.79, -0.29] | -0.67 | 0.746 | 0.025 | -1.28 | 0.68 | fail |
| DROP_PEER | -1.58 | [-3.28, +0.05] | -0.39 | 0.647 | 0.025 | -0.98 | 0.69 | fail |
| DROP_REV | -1.04 | [-2.32, +0.15] | +0.02 | 0.517 | 0.010 | -1.41 | 0.63 | fail |
| DROP_SLL | -0.76 | [-2.58, +0.62] | +0.06 | 0.527 | 0.010 | -0.15 | 0.61 | fail |
Mean standardized ridge coefficients, FULL model (units: xs-demeaned 5d forward return per 1-sigma of feature):

| feature | mean coef |
|---|---|
| `sll_weekly` | -3.38e-04 |
| `rev_5d` | +2.92e-04 |
| `peer_mom_gap` | -2.27e-04 |
| `sll_daily` | +1.25e-04 |
| `pead_drift` | +7.86e-05 |
| `rev_lowvol` | +4.81e-05 |

DEV reading (sanity only, pre-cutoff, unscored): the pipeline behaves as
designed — turnover, cost drag and null spreads are in the expected ranges;
PEAD is the only family with a positive net read; the ridge coefficient signs
already show two causal stories inverted in-sample (`sll_weekly`,
`peer_mom_gap` negative). CV selects the heaviest shrinkage (lambda=1e5) at
every refit — the model layer itself is reporting that these features barely
predict. Note on the relabel-p column: several negative-net variants show low
relabel p; that null preserves costs and turnover but breaks sector alignment,
so a low value means "better aligned than a random relabeling of itself," not
a viable edge — the primary gate is columns 2 and 5.

## 7. Pre-registered predictions (frozen before the POST gate)

<!-- PREDICTIONS -->

Registered 2026-07-06T19:55Z in `results/predictions_registered.json`
(sha256 `60259ca1726822ddaefb1f85751ac9b70aeb33be9128c9f4fd95627cd1fa2526`),
after the DEV sanity run and strictly before any post-cutoff evaluation
(`run_eval.py --stage post` refuses to run if the file is absent). Resolution
appended after the POST run:

| id | claim | conf | outcome |
|---|---|---|---|
| P1 | 0 of 4 SOLO families PASS the gate on POST | 0.80 | **TRUE** (0/4) |
| P2 | FULL does not PASS on POST | 0.85 | **TRUE** (net −0.15, p=0.389) |
| P3 | SOLO_PEAD highest net Sharpe among SOLO variants | 0.55 | **TRUE** (+0.66 vs +0.57 / −0.49 / −1.86) |
| P4 | SOLO_PEAD net Sharpe > 0 on POST | 0.55 | **TRUE** (+0.66) |
| P5 | ≥3 of 5 non-PEAD variants negative net on POST | 0.70 | **TRUE** (4/5: FULL −0.15, SLL −0.49, REV −1.86, DROP_PEAD −0.64; PEER +0.57) |
| P6 | DROP_PEAD net < FULL net (PEAD adds value at the margin) | 0.55 | **TRUE** (−0.64 < −0.15) |
| P7 | Permutation null gross-Sharpe std in [1.0, 2.5]; underpowered for sub-2-Sharpe edges | 0.80 | **TRUE** (std 1.53–1.59 across variants) |

7/7 resolved TRUE. The null outcome was the modal pre-registered expectation,
not a post-hoc rationalization.

---

## 8. POST-CUTOFF scored results (2026-02-01 → 2026-07-06)

<!-- POST_RESULTS -->

Window: 2026-02-03 → 2026-07-06 (105 pnl days). n_perm=500. EW-universe benchmark annualized return: +20.9%.

| variant | net Sharpe | 95% CI | gross Sharpe | perm p (gross) | relabel p (net) | long−bench net Sharpe | turn/day | PASS |
|---|---|---|---|---|---|---|---|---|
| FULL | -0.15 | [-3.81, +3.64] | +0.58 | 0.389 | 0.178 | +0.98 | 0.44 | fail |
| SOLO_PEAD | +0.66 | [-2.67, +5.64] | +0.98 | 0.269 | 0.283 | -0.36 | 0.13 | fail |
| SOLO_PEER | +0.57 | [-2.85, +3.97] | +1.07 | 0.261 | 0.094 | +1.29 | 0.35 | fail |
| SOLO_REV | -1.86 | [-4.81, +1.15] | -0.96 | 0.752 | 0.236 | -1.04 | 0.69 | fail |
| SOLO_SLL | -0.49 | [-3.15, +2.76] | +0.39 | 0.437 | 0.040 | +0.63 | 0.71 | fail |
| DROP_PEAD | -0.64 | [-4.37, +3.33] | +0.16 | 0.461 | 0.206 | +0.81 | 0.52 | fail |
| DROP_PEER | -1.25 | [-4.11, +3.32] | -0.56 | 0.637 | 0.381 | -0.54 | 0.45 | fail |
| DROP_REV | +0.42 | [-2.67, +3.76] | +1.05 | 0.295 | 0.126 | +1.22 | 0.41 | fail |
| DROP_SLL | +0.17 | [-3.41, +3.94] | +0.79 | 0.343 | 0.208 | +0.94 | 0.33 | fail |
Mean standardized ridge coefficients, FULL model (units: xs-demeaned 5d forward return per 1-sigma of feature):

| feature | mean coef |
|---|---|
| `peer_mom_gap` | -3.52e-04 |
| `pead_drift` | +2.55e-04 |
| `sll_weekly` | -1.83e-04 |
| `rev_5d` | +1.54e-04 |
| `rev_lowvol` | -3.98e-06 |
| `sll_daily` | -7.86e-07 |

**Verdict: 0/9 variants pass** (net Sharpe > 0 AND gross permutation p < 0.05
after 10 bps/side). Per-family verdicts (SOLO gates): SLL fail, PEER fail,
REV fail, PEAD fail. This extends v1's single-asset null (0/7) to the
cross-section at max synthesis tier: with ~52,000 post-cutoff panel
observations compressed into 105 daily long-short returns, none of the four
structural world models is distinguishable from within-date luck.

What the transparent layers add beyond the binary verdict:

- **Drop-one ablation:** PEAD and PEER are the only families whose removal
  hurts the full model (DROP_PEAD −0.64 and DROP_PEER −1.25 vs FULL −0.15),
  while dropping REV or SLL *helps* (+0.42, +0.17). The event-anchored and
  linkage-diffusion structures carry whatever weak signal exists; the
  price-shape families subtract value out-of-sample.
- **Consistency across windows:** PEAD is the top SOLO family in both DEV
  (+1.24, unscored) and POST (+0.66) with the lowest turnover (0.13/day) —
  directionally consistent with Bernard-Thomas underreaction, but at p=0.269
  it is exactly the situation v1 ended in: a positive point estimate the
  window cannot certify (bootstrap CI [−2.67, +5.64]).
- **Model-layer self-report:** time-ordered CV chose the maximum shrinkage
  (lambda=1e5) at all 18 refits — the ridge itself measures the feature
  matrix as barely predictive.
- One secondary stat (SOLO_SLL relabel p=0.040) crosses 0.05; with 18
  secondary p-values reported, one sub-0.05 value is expected multiplicity
  noise, and the variant's net Sharpe is negative anyway.

**Minimum detectable effect (quantified):** the permutation null gross-Sharpe
std is ~1.55 annualized on the 105-day window (analytic sqrt(252/105)=1.55
agrees); the one-sided 5% threshold sits at null q95 ≈ +2.4 to +2.65 gross
Sharpe. For ~80% power the true edge would need an annualized gross Sharpe
near 3.4. Realistic published cross-sectional anomalies run 0.5–1.5 net —
**this 5-month window cannot certify any realistically-sized edge**; it can
only reject the large ones v1-style tooling hopes for. Detecting a true 0.75
Sharpe at 5%/80% power needs ~T=(1.645+0.84)²/0.75² years ≈ 11 years
out-of-sample — which is exactly why the prospective ledger leg exists.

## 9. Prospective ledger seed

<!-- LEDGER -->

Seeded 2026-07-06 in `prospective/ledger.csv` + `prospective/pred_*.json`:
for each of FULL and the four SOLO families, the top- and bottom-decile
tickers (49 long / 49 short each) as of the 2026-07-06 close, claim =
equal-weight top minus bottom 5-session close-to-close return > 0. Each
prediction file's sha256 is recorded in its ledger row at registration time,
so post-hoc edits are detectable. Resolution protocol is in
`code/seed_ledger.py`'s docstring; resolving is a future cycle's job
(≥ 2026-07-13). One week is one Bernoulli draw per family — the ledger only
becomes informative as rows accumulate; the seed makes the commitments
verifiable, not the conclusions.

## 10. Limitations

<!-- LIMITATIONS -->

1. **Power, again, honestly quantified:** 5 months post-cutoff is ~105
   long-short observations however wide the panel is — breadth raises the
   Sharpe of a true signal but the *certification* variance is set by time.
   MDE ≈ 2.5 gross Sharpe at p<0.05 (see section 8). The post-cutoff-only
   discipline and adequate power are in direct tension at this calendar age
   of the synthesis model's cutoff; only the prospective ledger escapes it.
2. **Universe survivorship:** constituents fetched 2026-07-06; names that
   left the index Feb–Jul 2026 are absent (~1–2% turnover), and the
   min-coverage filter (496/503 kept) uses full-window availability. Both
   bias the equal-weight benchmark slightly up and are second-order for
   within-universe long-short deciles.
3. **Execution realism:** 10 bps/side flat, close-to-close fills, no
   borrow costs or shorting constraints, no market impact. Fine for a
   research gate; not a tradeable claim (and none is being made).
4. **Yahoo data quality:** auto-adjusted OHLCV and earnings surprise fields
   are as-is; ~100 tickers needed retry passes (transient 429s, all
   recovered — `data/fetch_log.json`). Earnings consensus history may embed
   revisions.
5. **Contamination residue:** walk-forward fits consume pre-cutoff data the
   synthesis model has memorized. The *scored* returns are all post-cutoff,
   but hypothesis CLASS selection (which four structures to code) was made
   by a model that knows those structures' published in-sample track records.
   The pre-registration + post-cutoff split controls scoring contamination,
   not idea-selection contamination — nothing can, except prospective data.
6. **Six features is a deliberately small library** (auditability over
   coverage): sector graph at GICS granularity only, no supply-chain data
   (paywalled), no analyst revisions, no intraday. Null verdicts are about
   these implementations, not the mechanism space.
