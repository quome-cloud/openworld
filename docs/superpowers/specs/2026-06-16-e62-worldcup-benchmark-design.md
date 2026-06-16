# E62 — Benchmark the World Cup forecaster against other models

**Date:** 2026-06-16
**Status:** Design — approved approach, pending spec review
**Experiment number:** e62 (E61 is the latest)
**Branch:** bundled onto `jenia/e61-worldcup-backtest` (reuses E61's engine, which isn't on `main` yet)

## Purpose

E61 showed our Elo→Poisson world model has modest-but-real skill on four past World
Cups (decisive hit-rate ~70%, Brier skill +13%, KO advancement ~71%, all four
champions in our pre-tournament top 6). But a backtest in a vacuum is hard to
interpret. **E62 puts the model on the same footing as other forecasters** — a suite
of statistical baselines we control plus FiveThirtyEight's published SPI forecasts —
and scores everyone with the same proper, ordinal metric (RPS). The headline result
is *where our model lands relative to the field*, reported honestly even if it loses
to 538.

## Scope

- **Match-level W/D/L** forecasting only (that is where 538's data and the standard
  literature metrics live). Tournament-level calibration stays in E61.
- Four cups: **2010, 2014, 2018, 2022** for our models and the statistical baselines
  (256 matches). **2018 & 2022 only** for the 538 head-to-head (128 matches — 538
  did not forecast 2010/2014).
- Bundled onto the E61 branch; reuses `examples/worldcup_history.py`.

## Data

Vendor 538's archived World Cup match files into `datasets/fivethirtyeight/`:

| File | Source | Rows | Use |
|---|---|---|---|
| `wc_2018.csv` | 538 via Wayback (snapshot `20250306125411`) | 64 matches | 538 per-match W/D/L probs + actual scores |
| `wc_2022.csv` | 538 via Wayback (snapshot `20250306125414`) | 64 matches | same |
| `README.md` | — | — | provenance: original `projects.fivethirtyeight.com/soccer-api/international/{year}/wc_matches.csv`, Wayback URLs, license note |

538 columns used: `date, team1, team2, prob1, prob2, probtie, score1, score2`
where `prob1`=P(team1 win), `prob2`=P(team2 win), `probtie`=P(draw); `team1` is the
home/first team. 538's `spi1/spi2` are **pre-match** ratings that update through the
tournament, so 538 is a **walk-forward** forecaster (this motivates our walk-forward
variant). Team names are reconciled to `results.csv` spellings via a small map and
**verified**: each cup's 64 538 rows must align 1:1 (by date + unordered team pair)
to the 64 real fixtures — assert on mismatch.

Fetch is already proven reachable (Wayback returns HTTP 200, 64-match CSVs). The
implementation re-fetches from the Wayback `id_` raw URLs and vendors the files; if
the fetch fails it falls back to a clear error asking for a manual drop.

## Metric: RPS (plus Brier, log-loss, hit-rate)

**Ranked Probability Score** is the standard proper score for ordinal W/D/L football
forecasts (Constantinou–Fenton). Categories are ordered by home-team margin:
`[away-win, draw, home-win]`. For forecast probabilities `p = [p_away, p_draw, p_home]`
and one-hot outcome `o`,

```
RPS = (1/(r-1)) * Σ_{i=1}^{r-1} ( Σ_{j=1}^{i} (p_j - o_j) )^2      # r = 3
```

Lower is better; a perfect forecast scores 0; RPS penalises distance (calling
away-win when home-win occurred is worse than calling a draw). We also keep Brier
(unordered), log-loss, and argmax hit-rate for continuity with E61. All metrics are
reported per model, **pooled and per cup**, on the relevant match set.

A small `rps(probs, actual)` helper (stdlib) is added; `score_matches(predictions,
actuals)` aggregates a model's per-match probability list into {rps, brier, logloss,
hit_rate, decisive_hit_rate, n}.

## Model suite

Each model is a function `model(cup, ...) -> {match_key: {"W","D","L"}}` over a
cup's real matches, where probabilities are from the **home (team1) perspective**
and `match_key` identifies the fixture. All are leakage-free (train only on data
strictly before each match/cup).

1. **uniform** — (⅓,⅓,⅓) for every match. The floor; a model below this is worthless.
2. **Elo-logistic (Davidson)** — from the same pre-tournament Elo (E61 engine),
   map the Elo difference to W/D/L via the Davidson draw model:
   `P(home) ∝ 10^(d/400)`, `P(away) ∝ 10^(-d/400)`, `P(draw) ∝ ν·10^(0)` scaled so a
   single draw-spread parameter ν is fit by MLE on pre-cup internationals. The
   probabilistic cousin of "chalk".
3. **ours — Elo→Poisson FROZEN** — the E61 model: `score_group_matches`-style W/D/L
   from `_wdl_probs` on frozen pre-tournament Elo, generalised to all matches
   (group + knockout) for benchmark coverage.
4. **ours — Elo→Poisson WALK-FORWARD** — start from frozen pre-cup Elo; process the
   cup's matches in date order, predict each from Elo *as of just before it*, then
   update Elo with the actual result (via `EloEngine.update_match`, K=World-Cup).
   This absorbs in-tournament information exactly as 538 does → the fair anchor.
5. **Poisson team-strength (Maher)** — independent-Poisson attack/defense model:
   `log λ_home = μ + atk_home − def_away + γ` (home effect γ),
   `log λ_away = μ + atk_away − def_home`; parameters fit by MLE (scipy.optimize) on
   internationals in the **4 years before each cup's freeze date** (leakage-free,
   keeps strength current). Teams with too few matches in-window fall back to a
   league-average strength. W/D/L from the resulting independent-Poisson score grid.
6. **538 SPI** — read `prob1/probtie/prob2` directly; 2018 & 2022 only.

**Chalk** (E61's deterministic higher-Elo bracket) is reported as a **hit-rate
reference only**, not RPS-scored — a degenerate all-or-nothing distribution would be
unfairly punished by a proper score, and Elo-logistic is its calibrated stand-in.

## Fairness framing

The headline head-to-head is **walk-forward-ours vs 538 vs Elo-logistic vs Maher on
2018+2022** (all see comparable information). **Frozen-ours** is reported alongside as
the genuine pre-tournament forecast, with an explicit note that 538/walk-forward have
an in-tournament information advantage over it. We do not hide which models had more
information.

## Architecture

- **`examples/worldcup_benchmark.py`** (new) — the suite: `rps`, `score_matches`,
  the six model functions, the 538 loader + name reconciliation, and a
  `run_benchmark(cups)` that returns the full results structure. **May use
  numpy/scipy** (experiment-grade, like e12/e50). Imports `worldcup_history` for the
  Elo engine, cups, and `_wdl_probs`. `worldcup_history.py` itself **stays
  stdlib-only** (unchanged except possibly exposing a tiny helper if needed).
- **`experiments/e62_worldcup_benchmark.py`** (new) — thin driver: fetch/verify 538
  data presence, run the benchmark, `save_results("e62_worldcup_benchmark", payload)`
  **before** asserts, self-check, print the ranking table.
- **`tests/test_e62_worldcup_benchmark.py`** (new) — RPS correctness (perfect=0,
  ordering sensitivity, ≤ uniform for a confident-correct forecast), 538 alignment
  (64 matches/cup map 1:1), each model emits normalised probs over the right match
  count, leakage (Maher/Elo-logistic fit uses only pre-cup data), determinism.
- **Paper integration** — `fig_worldcup_benchmark` (RPS-by-model bar chart, 538 bar
  shown for 2018/22), `table_worldcup_benchmark` (model × {RPS, Brier, hit-rate}),
  macros (e.g. `\BenchOursRPS`, `\BenchFTERPS`, `\BenchMaherRPS`, `\BenchUniformRPS`),
  `\NumExperiments` bump. Letters-only macro names.

## Results JSON shape

```
{model_set, cups, n_matches_all, n_matches_538,
 per_model: {<name>: {pooled:{rps,brier,logloss,hit_rate,n},
                      per_cup:{"2010":{...},...}}},
 head_to_head_538: {matches:128, per_model:{walk_forward:{rps,..}, frozen:{..},
                    elo_logistic:{..}, maher:{..}, five_thirty_eight:{..}, uniform:{..}}},
 ranking: [(model, pooled_rps), ... sorted]}
```

## Self-checks (after `save_results`)

- `rps` of a perfect forecast == 0; `rps([⅓,⅓,⅓], decisive) == 5/18 ≈ 0.2778` and
  `rps([⅓,⅓,⅓], draw) == 1/9 ≈ 0.1111` (uniform RPS depends on the outcome — it is
  *not* a flat 0.25). Pooled uniform RPS lands ~0.24 given ~21% draws.
- Every probabilistic model's pooled RPS < the pooled uniform RPS (~0.24) on
  all-4-cups — i.e. every model beats the floor. (If Maher or Elo-logistic doesn't,
  that's a real finding — report, don't suppress.)
- On the 128-match 538 set: report ours-walk-forward RPS vs 538 RPS. **No assertion
  that we beat 538** (we may not); assert only that both are < uniform and that the
  538 column has exactly 128 matches.
- Determinism: identical results across two runs (fixed seeds for any sampling;
  Maher MLE is deterministic given data).

## Honest reporting

The writeup states plainly where our model ranks: expected order is roughly
538 ≈ Maher ≲ ours-walk-forward < Elo-logistic < ours-frozen < uniform, but the
actual order is whatever the data says. If our model loses to 538 and/or Maher, we
say so — "competitive with standard statistical baselines, behind the market-grade
538 forecast" is a true and respectable result. We foreground RPS, and we never
compare a frozen model to a walk-forward one without flagging the information gap.

## Non-goals (YAGNI)

- No bookmaker closing-odds baseline (deferred — clean WC odds data is the hard part;
  separate follow-up).
- No negative-binomial / Dixon-Coles ρ-correction (Maher independent-Poisson is the
  canonical baseline; the ρ refinement is a second-order tweak).
- No tournament-level (title-odds) benchmark — 538 publishes those but our match-level
  comparison is the cleaner, more defensible head-to-head; E61 already covers our own
  tournament calibration.
- No new core dependencies; numpy/scipy used only in the benchmark/experiment layer,
  never in `worldcup_history.py` or the core library.

## Open questions

None blocking. Maher fit window (4 years) and the team-name reconciliation map are
finalised empirically during implementation and documented in code.
