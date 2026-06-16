# E61 — Historical World Cup Backtest (2010 / 2014 / 2018 / 2022)

**Date:** 2026-06-15
**Status:** Design — approved approach, pending spec review
**Experiment number:** e61 (last existing is e60)

## Purpose

Validate the OpenWorld Elo→Poisson world model — the same engine that powers the
2026 forecaster (`examples/worldcup2026.py`) — by **predicting four past men's
World Cups from pre-tournament information only**, then scoring against what
actually happened.

The headline question: *how good is our simulation at calling matches and
forecasting tournaments?* We answer it across four layers (match-level skill
first), and we are honest about where reality defied the model (Brazil 1–7 in
2014, Italy's group exits, Argentina 2022).

This is a backtest of the **model**, not of any one Elo provider. We therefore
compute our own leakage-free Elo from the full results history and cross-check it
against published ratings.

## Tournaments

2010 (South Africa), 2014 (Brazil), 2018 (Russia), 2022 (Qatar).

All four used the **identical format**, which is why one engine covers them:
- 32 teams, 8 groups (A–H) of 4, round-robin, top two advance.
- Fixed Round-of-16 bracket (1A–2B, 1C–2D, 1E–2F, 1G–2H, 1B–2A, 1D–2C, 1F–2E,
  1H–2G) → QF → SF → Final (third-place playoff ignored).
- Knockouts: 90 min → extra time → penalties.
- Minor tiebreaker nuance (2018 added fair-play points) is immaterial to the
  simulation and is not modelled.

64 matches per cup × 4 = **256 real matches** (48 group + 16 knockout each).

## Data

Three CSVs vendored into `datasets/openworld-football/` (CC-licensed Kaggle
sources; provenance + license recorded in a sibling `README.md`) so the
experiment runs **offline and reproducibly**:

| File | Source | Use |
|---|---|---|
| `results.csv` | martj42 *International football results 1872–2026* (~49k matches, all teams) | Elo engine input **and** ground-truth scores |
| `shootouts.csv` | same | who advanced when a knockout was drawn |
| `elo_ratings_wc2026.csv` | afonsofernandescruz *2026 WC historical Elo* (48 teams, 1901–2026) | **validation cross-check only** — confirms our computed Elo ≈ eloratings.net |

`former_names.csv` is used to unify historical team identities. The published
2026-scoped Elo file is *not* used as a model input (it omits any team that did
not qualify for 2026, e.g. Italy/Russia/Serbia); it serves only as a validation
target.

## Design

### 1. Elo engine (computed from `results.csv` — "Option B")

Replay every international chronologically with the published World Football Elo
update:

```
R'      = R + K · G · (W − Wₑ)
Wₑ      = 1 / (1 + 10^(−Δ/400))
Δ       = R_home + HA − R_away          # HA = +100, but 0 when neutral == TRUE
W       ∈ {1, 0.5, 0}                    # actual result for the team
G       = 1                if |margin| ≤ 1
        = 1.5              if |margin| == 2
        = (11 + |margin|)/8 if |margin| ≥ 3
K       = 60 World Cup finals
        = 50 other major finals (continental championship)
        = 40 World Cup / continental qualifiers + major-tournament minor
        = 30 other tournaments
        = 20 friendlies
        (keyed off the `tournament` column; table documented in code)
```

- All teams initialised at **1500**, full carry-over between matches.
- Matches sorted by date; identities normalised via `former_names.csv`.
- **No look-ahead:** a cup's pre-tournament ratings are frozen as of the day
  *before* that cup's opening match — exactly the information a real forecaster
  would have had.

This produces a rating for **every** team at **every** date as a by-product, so
the missing-team problem never arises.

### 2. Engine validation (credibility receipt)

Compare our computed ratings against `elo_ratings_wc2026.csv` for the overlapping
(2026-qualifier) teams at end-of-2009 / 2013 / 2017 / 2021. Report:
- Pearson correlation, Spearman rank correlation, and RMSE (in Elo points).

Acceptance signal (asserted): Spearman ≥ 0.8 and Pearson ≥ 0.8 on each snapshot —
demonstrating our reconstructed Elo tracks eloratings.net well enough to trust the
gap-filling teams. (Exact thresholds finalised against observed values; assertions
written conservatively so a genuine regression fails but normal variation passes.)

### 3. Model under test (unchanged; one minimal refactor)

The outcome model is the existing `sample_match` Elo→Poisson (mean
`TOTAL_GOALS=2.7`, `SUPREMACY=1.9`) plus the group/knockout rules from
`worldcup2026.py`. To reuse it with per-cup ratings without touching its behaviour,
extract a rating-parameterised core in `worldcup2026.py`:

- New pure helper `sample_goals_from_elo(elo_home, elo_away, rng, *, total_goals,
  supremacy) -> (hg, ag)`.
- `sample_match(home, away, rng)` becomes a thin wrapper that looks up `_eff_elo`
  and delegates. Behaviour and existing tests unchanged (the refactor is covered
  by the current `test_worldcup2026.py`).

The backtest builds a **32-team variant** (`examples/worldcup_backtest.py` helpers
or inline in the experiment) that supplies its own per-cup Elo dict, hosts, groups,
and the fixed R16 bracket, and calls the shared core. Hosts get the existing
`HOST_ADVANTAGE` bump (RSA 2010, BRA 2014, RUS 2018, QAT 2022).

### 4. Tournament structure: groups, bracket, and ground truth

- **Group draw (A–H + the 4 teams each):** hand-encoded per cup as a constant.
  The draw is *known pre-tournament information* (held months before the cup), so
  using it as a simulation input is principled — exactly as the 2026 forecaster
  hand-encodes its `GROUPS`. Encoding it explicitly fixes the correct group
  **letters**, which the R16 pairing (1A–2B, …) depends on. Each encoded group is
  **verified against `results.csv`**: all 6 of its round-robin pairings must appear
  as real group-stage matches in that cup — assert on mismatch (guards transcription
  errors).
- **R16 bracket:** the fixed FIFA pairing rule, applied to *our simulated*
  standings during Monte-Carlo, and to the *real* standings for the actual bracket.
- **Group-stage vs knockout split** in `results.csv`: the 48 matches whose
  (home, away) pair lies within one encoded group are group-stage; the remaining 16
  are knockout.
- **Knockout ground truth:** advancer is the higher score, or the `shootouts.csv`
  winner when drawn after extra time.

### 5. Measurement (four layers; match-level first)

**(a) Match-level skill — the backbone.**
- *Group matches (192 total):* score the full **W/D/L** (3-way) prediction.
  For each match compute model P(W)/P(D)/P(L) by sampling `sample_match` with the
  frozen pre-tournament Elo. Metrics: Brier score, hit-rate (argmax == actual),
  decisive hit-rate (winner called on non-draws), mean log-loss, mean probability
  on the true outcome, and **skill-vs-uniform** = 1 − Brier/Brier_uniform.
  Reuses the existing `evaluate_predictions` machinery, generalised over a match
  list. Pooled across cups **and** per cup.
- *Knockout matches (64 total):* score **advancement** (2-way) — model
  P(team advances) vs the real advancer. Metrics: accuracy, log-loss, Brier on the
  binary. (A knockout cannot be a draw, so W/D/L is ill-defined; advancement is the
  well-posed prediction.)

**(b) Tournament-level calibration.** 10k Monte-Carlo sims per cup from frozen
pre-tournament Elo (deterministic seed). Report, per cup and pooled:
- Title probability and rank we assigned to the **actual champion**, and to the
  actual finalists / semifinalists.
- Champion log-loss = −ln P(actual champion).
- Reach-round calibration: bucket teams by predicted reach-QF (and reach-SF)
  probability, compare to observed frequency across the 4 cups.

**(c) Beat-a-baseline.** A deterministic **chalk** bracket ("higher pre-tournament
Elo always advances; ties to higher Elo"). Report its match hit-rate and how many
knockout rounds it called correctly per cup, vs our model's calibrated forecast —
showing the simulation adds value beyond "favourite always wins".

**(d) Bracket vs reality.** Per cup, render the model's **modal simulated bracket**
beside the **actual bracket** as a self-contained SVG (adapting
`render_bracket_svg` to 32 teams), so the hits and the chaos are visible at a
glance.

### 6. Deliverables

- `experiments/e61_worldcup_backtest.py` — deterministic (fixed seeds), offline,
  `save_results()` **before** asserts, self-checking.
- `experiments/results/e61_worldcup_backtest.json` — engine-validation stats,
  per-cup + pooled match-level metrics, knockout-advancement metrics,
  tournament-calibration numbers, baseline comparison.
- Four bracket SVGs (`paper/figs/` or example output dir): simulated-vs-actual per
  cup.
- `datasets/openworld-football/` with the three CSVs + `README.md` (provenance,
  license).
- Minimal `worldcup2026.py` refactor (rating-parameterised core), existing tests
  green.
- Tests: `tests/test_e61_worldcup_backtest.py` — Elo monotonicity, no-look-ahead
  (a cup's ratings unchanged by appending post-cup matches), encoded-group
  verification (all 6 pairings present in real data), validation-correlation sign,
  metric determinism, baseline ≤ model on
  pooled skill (or documented if not).
- Paper integration per `CLAUDE.md`: entry in `make_paper_assets.py` `EXPERIMENTS`,
  a `fig_*`/`table_*` function + its `main()` call, macros before the `numbers.tex`
  write, and a `\NumExperiments` bump. LaTeX macro names letters-only.

### 7. Honest reporting

The writeup (script docstring + paper prose) states clearly:
- Where the model shines (strong-favourite group results; tournament-level
  calibration; reach-round buckets).
- Where it misses (Brazil 1–7 Germany 2014; defending-strength upsets; Italy's
  group exits; 2022 Argentina from a non-top Elo). Single deterministic outcomes
  are noisy — emphasise calibration and pooled skill over any one bracket.
- Any metric that comes out weak or flat is reported as-is, not tuned away.

## Non-goals (YAGNI)

- No third-place playoff modelling.
- No live/walk-forward in-tournament Elo updates (ratings frozen at kickoff, matching
  how the 2026 forecaster works and what "forecast the tournament" means).
- No women's World Cup, no pre-2010 cups (different formats / sparser data).
- No fair-play tiebreaker.
- No new runtime dependencies (stdlib `csv` only; core stays zero-dep).

## Open questions

None blocking. Exact K-by-tournament table and assertion thresholds are finalised
empirically during implementation and documented in code.
