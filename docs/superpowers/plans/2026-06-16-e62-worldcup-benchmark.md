# E62 — World Cup Model Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Benchmark the Elo→Poisson World Cup forecaster against a suite of baselines (uniform, Elo-logistic/Davidson, Maher Poisson team-strength) and FiveThirtyEight SPI, scored match-by-match with RPS/Brier on 2010–2022 (538 head-to-head on 2018/2022), including a walk-forward variant of our model for a fair comparison.

**Architecture:** A new `examples/worldcup_benchmark.py` holds the RPS/Brier scoring, the six model functions, a self-contained results-CSV reader (cup matches + pre-cup training window), and the 538 loader; it may use numpy/scipy (experiment-grade) and imports `worldcup_history.py` (which stays stdlib-only and unchanged). A thin `experiments/e62_worldcup_benchmark.py` runs the benchmark, saves results before asserts, and prints the ranking. Everything is leakage-free: fitted models and ratings use only data strictly before each match/cup.

**Tech Stack:** Python stdlib + numpy 2.3 + scipy 1.17 (benchmark layer only); reuses `worldcup_history.EloEngine`, `_wdl_probs`, `_cup_freeze_date`, `load_cup`, `RESULTS_CSV`, `_k_for`; `experiments/common.save_results`; matplotlib in `make_paper_assets.py`.

**Branch:** `jenia/e61-worldcup-backtest` (bundled; reuses unmerged E61 engine). One PR to `main`.

---

## Reusable API (from examples/worldcup_history.py — already implemented & tested)

- `RESULTS_CSV` (Path), `HOME_ADVANTAGE=100.0`, `_k_for(tournament)->float`, `_gd_multiplier(margin)`.
- `EloEngine.from_results(RESULTS_CSV, until=None, base=1500.0)`; `eng.rating(team)`; `eng.update_match(home, away, hg, ag, neutral, k)`; `eng.ratings_asof(date)->{team:rating}` (teams unseen before `date` are absent; default-fill via `.get(team, base)`).
- `_cup_freeze_date(year)->str` (2010-06-11/2014-06-12/2018-06-14/2022-11-20).
- `load_cup(year)->Cup`; `cup.groups`, `cup.host`, `cup.group_result(a,b)->(home,hg,ag)|None`, `cup.knockout_matches()->[{home,away,hg,ag,winner,date}]`.
- `_wdl_probs(home, away, elo, host, base, sims, rng)->{"W","D","L"}` (Elo→Poisson, home perspective).
- `_eff(team, elo, host, base)`, `_sample(...)`.

`worldcup_history.py` is NOT modified by this plan.

## File Structure

- **Create** `datasets/fivethirtyeight/{wc_2018.csv,wc_2022.csv,README.md}` — vendored 538 data.
- **Create** `examples/worldcup_benchmark.py` — RPS/Brier, results reader, 6 models, 538 loader, `run_benchmark`.
- **Create** `experiments/e62_worldcup_benchmark.py` — driver: run, `save_results`, assert, print ranking.
- **Create** `tests/test_e62_worldcup_benchmark.py` — RPS, models, 538 alignment, leakage, determinism.
- **Modify** `scripts/make_paper_assets.py` — `EXPERIMENTS` entry, `fig_/table_worldcup_benchmark`, `main()` calls, macros, `\NumExperiments` bump.
- **Output (generated)** `experiments/results/e62_worldcup_benchmark.json`, `paper/figs/e62_worldcup_benchmark.pdf`, `paper/tables/worldcup_benchmark.tex`.

`match_key` convention (used everywhere): `(date, home, away)` from `results.csv`, where `home`=listed home team, all probabilities are from the **home perspective** with keys `"W"`(home win)/`"D"`/`"L"`(away win).

---

## Task 1: Vendor the 538 data

**Files:**
- Create: `datasets/fivethirtyeight/{wc_2018.csv,wc_2022.csv,README.md}`

- [ ] **Step 1: Fetch the two archived 538 WC files from the Wayback Machine**

```bash
mkdir -p datasets/fivethirtyeight
curl -sSL --max-time 90 -o datasets/fivethirtyeight/wc_2018.csv \
  "http://web.archive.org/web/20250306125411id_/https://projects.fivethirtyeight.com/soccer-api/international/2018/wc_matches.csv"
curl -sSL --max-time 90 -o datasets/fivethirtyeight/wc_2022.csv \
  "http://web.archive.org/web/20250306125414id_/https://projects.fivethirtyeight.com/soccer-api/international/2022/wc_matches.csv"
```

- [ ] **Step 2: Verify both files are real 64-match CSVs with the expected header**

Run:
```bash
for y in 2018 2022; do
  echo "== $y =="; head -1 datasets/fivethirtyeight/wc_$y.csv
  python3 -c "import csv; rows=list(csv.DictReader(open('datasets/fivethirtyeight/wc_$y.csv'))); print(len(rows),'rows'); assert len(rows)==64; r=rows[0]; assert all(k in r for k in ['date','team1','team2','prob1','prob2','probtie','score1','score2']); print('ok')"
done
```
Expected: header line, `64 rows`, `ok` for both. If a file is HTML (Wayback miss) or row count ≠ 64, STOP and report BLOCKED — do not fabricate data.

- [ ] **Step 3: Write the provenance README**

Create `datasets/fivethirtyeight/README.md`:
```markdown
# FiveThirtyEight World Cup match forecasts (vendored for E62)

538's per-match SPI win/draw/loss probabilities + actual scores for the 2018 and
2022 World Cups. Used by `examples/worldcup_benchmark.py` as the external benchmark.

| File | Matches | Original source | Retrieved via |
|---|---|---|---|
| `wc_2018.csv` | 64 | projects.fivethirtyeight.com/soccer-api/international/2018/wc_matches.csv | Wayback snapshot 20250306125411 |
| `wc_2022.csv` | 64 | projects.fivethirtyeight.com/soccer-api/international/2022/wc_matches.csv | Wayback snapshot 20250306125414 |

538 shut down in 2023 and the live endpoints now redirect to ABC News; these are the
Internet Archive copies. Columns used: date, team1 (home), team2 (away), prob1
(P team1 win), prob2 (P team2 win), probtie (P draw), score1, score2. 538's SPI
updates during the tournament, so these are walk-forward forecasts. CC-licensed per
538's data repo (github.com/fivethirtyeight/data).
```

- [ ] **Step 4: Commit**

```bash
git add datasets/fivethirtyeight/
git commit -m "data: vendor 538 World Cup 2018/2022 match forecasts (Wayback) for E62

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: RPS + scoring helpers

**Files:**
- Create: `examples/worldcup_benchmark.py`
- Test: `tests/test_e62_worldcup_benchmark.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_e62_worldcup_benchmark.py`:
```python
"""Tests for the World Cup model benchmark (examples/worldcup_benchmark.py)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import worldcup_benchmark as wb  # noqa: E402


def test_rps_perfect_is_zero():
    assert wb.rps({"W": 1.0, "D": 0.0, "L": 0.0}, "W") == 0.0
    assert wb.rps({"W": 0.0, "D": 0.0, "L": 1.0}, "L") == 0.0


def test_rps_uniform_values():
    u = {"W": 1 / 3, "D": 1 / 3, "L": 1 / 3}
    assert abs(wb.rps(u, "W") - 5 / 18) < 1e-9   # decisive
    assert abs(wb.rps(u, "L") - 5 / 18) < 1e-9   # decisive
    assert abs(wb.rps(u, "D") - 1 / 9) < 1e-9    # draw


def test_rps_ordering_sensitivity():
    # Predicting the adjacent outcome (draw) beats predicting the far one (away win)
    # when home actually won.
    near = {"W": 0.0, "D": 1.0, "L": 0.0}
    far = {"W": 0.0, "D": 0.0, "L": 1.0}
    assert wb.rps(near, "W") < wb.rps(far, "W")


def test_score_matches_aggregates():
    preds = {("d", "A", "B"): {"W": 0.7, "D": 0.2, "L": 0.1}}
    actuals = {("d", "A", "B"): "W"}
    s = wb.score_matches(preds, actuals)
    assert s["n"] == 1
    assert 0.0 <= s["rps"] <= 1.0
    assert 0.0 <= s["brier"] <= 2.0
    assert s["hit_rate"] == 1.0
```

- [ ] **Step 2: Run, verify fail**

Run: `python3 -m pytest tests/test_e62_worldcup_benchmark.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'worldcup_benchmark'`.

- [ ] **Step 3: Implement the module header + scoring**

Create `examples/worldcup_benchmark.py`:
```python
"""World Cup model benchmark (E62): RPS/Brier head-to-head of the OpenWorld
Elo->Poisson forecaster vs statistical baselines and FiveThirtyEight SPI.

Scores match-level W/D/L forecasts (home perspective) on 2010/2014/2018/2022 with
the Ranked Probability Score (the proper ordinal metric for football), plus Brier,
log-loss and hit-rate. 538 is compared on 2018 & 2022 only. Leakage-free: every
fitted model / rating uses only data strictly before each match or cup.

May use numpy/scipy (experiment-grade). worldcup_history.py stays stdlib-only.

    python experiments/e62_worldcup_benchmark.py
"""

from __future__ import annotations

import csv
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(__file__))
import worldcup_history as wh  # noqa: E402

FTE_DIR = Path(__file__).resolve().parents[1] / "datasets" / "fivethirtyeight"
CUPS = [2010, 2014, 2018, 2022]
FTE_CUPS = [2018, 2022]
_ORDER = ["L", "D", "W"]   # ordinal: away-win < draw < home-win (by home margin)


def rps(probs: Dict[str, float], actual: str) -> float:
    """Ranked Probability Score for an ordinal W/D/L forecast (home perspective).

    Categories ordered [L, D, W] by home margin. 0 = perfect, larger = worse.
    """
    cum_p = cum_o = 0.0
    total = 0.0
    for cat in _ORDER[:-1]:          # r-1 = 2 cumulative terms
        cum_p += probs[cat]
        cum_o += 1.0 if cat == actual else 0.0
        total += (cum_p - cum_o) ** 2
    return total / (len(_ORDER) - 1)


def _brier(probs: Dict[str, float], actual: str) -> float:
    return sum((probs[c] - (1.0 if c == actual else 0.0)) ** 2 for c in "WDL")


def score_matches(predictions: Dict[tuple, Dict[str, float]],
                  actuals: Dict[tuple, str]) -> dict:
    """Aggregate one model's per-match probabilities vs actual outcomes.

    Only match_keys present in BOTH dicts are scored. Returns rps/brier/logloss/
    hit_rate/decisive_hit_rate/n (means over the scored matches).
    """
    keys = [k for k in predictions if k in actuals]
    n = len(keys)
    if n == 0:
        return {"n": 0, "rps": None, "brier": None, "logloss": None,
                "hit_rate": None, "decisive_hit_rate": None}
    rps_s = brier_s = ll_s = hits = 0.0
    dec_hits = dec_n = 0
    for k in keys:
        p, a = predictions[k], actuals[k]
        rps_s += rps(p, a)
        brier_s += _brier(p, a)
        ll_s += -math.log(max(p[a], 1e-12))
        fav = max(p, key=p.get)
        hits += fav == a
        if a != "D":
            dec_n += 1
            dec_hits += (p["W"] > p["L"]) if a == "W" else (p["L"] > p["W"])
    return {"n": n, "rps": rps_s / n, "brier": brier_s / n, "logloss": ll_s / n,
            "hit_rate": hits / n,
            "decisive_hit_rate": dec_hits / dec_n if dec_n else None}
```

- [ ] **Step 4: Run tests, verify pass**

Run: `python3 -m pytest tests/test_e62_worldcup_benchmark.py -v`
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add examples/worldcup_benchmark.py tests/test_e62_worldcup_benchmark.py
git commit -m "feat(e62): RPS + Brier scoring helpers for the model benchmark

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Results-CSV reader (cup matches + pre-cup training window + actuals)

**Files:**
- Modify: `examples/worldcup_benchmark.py`
- Test: `tests/test_e62_worldcup_benchmark.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_e62_worldcup_benchmark.py`:
```python
def test_cup_matches_are_64_in_date_order():
    m = wb.cup_matches(2014)
    assert len(m) == 64
    dates = [r["date"] for r in m]
    assert dates == sorted(dates)
    r = m[0]
    assert set(r) >= {"date", "home", "away", "hg", "ag", "neutral"}


def test_actual_outcomes_home_perspective():
    out = wb.actual_outcomes(2014)
    assert len(out) == 64
    # Brazil opened 2014 by beating Croatia 3-1 at home -> home win "W".
    key = next(k for k in out if k[1] == "Brazil" and k[2] == "Croatia")
    assert out[key] == "W"


def test_training_matches_are_leakage_free():
    cutoff = wh._cup_freeze_date(2014)
    tr = wb.training_matches(2014, years=4)
    assert tr, "expected pre-cup internationals"
    assert all(r["date"] < cutoff for r in tr)
    assert all(r["date"] >= "2010-06-12" for r in tr)  # 4-year window
```

- [ ] **Step 2: Run, verify fail**

Run: `python3 -m pytest tests/test_e62_worldcup_benchmark.py -k "cup_matches or actual_outcomes or training" -v`
Expected: FAIL (missing `cup_matches`/`actual_outcomes`/`training_matches`).

- [ ] **Step 3: Implement the reader**

Add to `examples/worldcup_benchmark.py`:
```python
def _read_results() -> List[dict]:
    rows = []
    with open(wh.RESULTS_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                hg, ag = int(r["home_score"]), int(r["away_score"])
            except (ValueError, KeyError):
                continue
            rows.append({"date": r["date"], "home": r["home_team"],
                         "away": r["away_team"], "hg": hg, "ag": ag,
                         "tournament": r["tournament"],
                         "neutral": str(r["neutral"]).strip().upper() == "TRUE"})
    rows.sort(key=lambda r: r["date"])
    return rows


def cup_matches(year: int) -> List[dict]:
    """The cup's 64 real matches (date-sorted), home perspective from results.csv."""
    return [r for r in _read_results()
            if r["tournament"] == "FIFA World Cup" and r["date"][:4] == str(year)]


def actual_outcomes(year: int) -> Dict[tuple, str]:
    """match_key (date, home, away) -> actual W/D/L from the home perspective."""
    out = {}
    for r in cup_matches(year):
        out[(r["date"], r["home"], r["away"])] = (
            "W" if r["hg"] > r["ag"] else ("D" if r["hg"] == r["ag"] else "L"))
    return out


def training_matches(year: int, years: int = 4) -> List[dict]:
    """All internationals strictly before the cup's freeze date, within `years`."""
    cutoff = wh._cup_freeze_date(year)
    lo = f"{year - years:04d}{cutoff[4:]}"
    return [r for r in _read_results() if lo <= r["date"] < cutoff]
```

- [ ] **Step 4: Run tests, verify pass**

Run: `python3 -m pytest tests/test_e62_worldcup_benchmark.py -k "cup_matches or actual_outcomes or training" -v`
Expected: 3 PASS. (If `test_actual_outcomes_home_perspective` fails because 2014's opener is stored away-perspective, inspect — results.csv lists Brazil as home for the opener, so "W" is correct; a failure indicates a reader bug.)

- [ ] **Step 5: Commit**

```bash
git add examples/worldcup_benchmark.py tests/test_e62_worldcup_benchmark.py
git commit -m "feat(e62): results-CSV reader (cup matches, actuals, training window)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Trivial + our-model predictions (uniform, frozen, walk-forward)

**Files:**
- Modify: `examples/worldcup_benchmark.py`
- Test: `tests/test_e62_worldcup_benchmark.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_e62_worldcup_benchmark.py`:
```python
def test_uniform_predictions():
    eng = wh.EloEngine.from_results(wh.RESULTS_CSV)
    p = wb.predict_uniform(2014, eng)
    assert len(p) == 64
    v = next(iter(p.values()))
    assert v == {"W": 1 / 3, "D": 1 / 3, "L": 1 / 3}


def test_ours_frozen_predictions_normalised():
    eng = wh.EloEngine.from_results(wh.RESULTS_CSV)
    p = wb.predict_ours_frozen(2014, eng, sims=2000)
    assert len(p) == 64
    for v in p.values():
        assert abs(sum(v.values()) - 1.0) < 1e-9


def test_walk_forward_beats_uniform_on_2014():
    eng = wh.EloEngine.from_results(wh.RESULTS_CSV)
    preds = wb.predict_ours_walk_forward(2014, eng, sims=3000)
    actuals = wb.actual_outcomes(2014)
    s = wb.score_matches(preds, actuals)
    u = wb.score_matches(wb.predict_uniform(2014, eng), actuals)
    assert len(preds) == 64
    assert s["rps"] < u["rps"]   # real skill beats the floor
```

- [ ] **Step 2: Run, verify fail**

Run: `python3 -m pytest tests/test_e62_worldcup_benchmark.py -k "uniform or frozen or walk_forward" -v`
Expected: FAIL (missing predictors).

- [ ] **Step 3: Implement**

Add to `examples/worldcup_benchmark.py`:
```python
import random as _random


def _host_of(year: int) -> str:
    return wh.load_cup(year).host


def predict_uniform(year: int, eng: "wh.EloEngine") -> Dict[tuple, Dict[str, float]]:
    third = 1 / 3
    return {(r["date"], r["home"], r["away"]): {"W": third, "D": third, "L": third}
            for r in cup_matches(year)}


def predict_ours_frozen(year: int, eng: "wh.EloEngine", sims: int = 20000,
                        seed: int = 2026) -> Dict[tuple, Dict[str, float]]:
    """Elo->Poisson W/D/L from FROZEN pre-tournament Elo (the E61 model)."""
    elo = eng.ratings_asof(wh._cup_freeze_date(year))
    host = _host_of(year)
    rng = _random.Random(seed)
    out = {}
    for r in cup_matches(year):
        out[(r["date"], r["home"], r["away"])] = wh._wdl_probs(
            r["home"], r["away"], elo, host, 1500.0, sims, rng)
    return out


def predict_ours_walk_forward(year: int, eng: "wh.EloEngine", sims: int = 20000,
                              seed: int = 2026) -> Dict[tuple, Dict[str, float]]:
    """Predict each match from Elo as of just before it, then update Elo with the
    real result (K = World Cup) — absorbs in-tournament info like 538 does."""
    elo = dict(eng.ratings_asof(wh._cup_freeze_date(year)))
    host = _host_of(year)
    base = 1500.0
    wc_k = wh._k_for("FIFA World Cup")
    rng = _random.Random(seed)
    out = {}
    for r in cup_matches(year):              # date-sorted
        h, a, hg, ag = r["home"], r["away"], r["hg"], r["ag"]
        out[(r["date"], h, a)] = wh._wdl_probs(h, a, elo, host, base, sims, rng)
        # update a local Elo copy (World Football Elo, same formula as the engine)
        ha = 0.0 if r["neutral"] else wh.HOME_ADVANTAGE
        rh, ra = elo.get(h, base), elo.get(a, base)
        we = 1.0 / (1.0 + 10 ** (-((rh + ha) - ra) / 400.0))
        w = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
        delta = wc_k * wh._gd_multiplier(hg - ag) * (w - we)
        elo[h] = rh + delta
        elo[a] = ra - delta
    return out
```

- [ ] **Step 4: Run tests, verify pass**

Run: `python3 -m pytest tests/test_e62_worldcup_benchmark.py -k "uniform or frozen or walk_forward" -v`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add examples/worldcup_benchmark.py tests/test_e62_worldcup_benchmark.py
git commit -m "feat(e62): uniform + ours-frozen + ours-walk-forward predictors

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Elo-logistic (Davidson) model

**Files:**
- Modify: `examples/worldcup_benchmark.py`
- Test: `tests/test_e62_worldcup_benchmark.py`

The Davidson model gives ordinal W/D/L from an Elo gap with one fitted draw parameter
ν: with `f = 10^(d/400)` and `g = 10^(-d/400)` (d = home Elo − away Elo, home incl.
host bump), `P(home) = f/(f+g+ν√(fg))`, `P(draw) = ν√(fg)/(f+g+ν√(fg))`,
`P(away) = g/(f+g+ν√(fg))`. ν ≥ 0 is fit by max-likelihood on pre-cup internationals.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_e62_worldcup_benchmark.py`:
```python
def test_davidson_probs_normalise_and_favour_stronger():
    p = wb._davidson_probs(elo_diff=200.0, nu=1.0)
    assert abs(sum(p.values()) - 1.0) < 1e-9
    assert p["W"] > p["L"]                      # home stronger by 200 Elo
    eq = wb._davidson_probs(elo_diff=0.0, nu=1.0)
    assert abs(eq["W"] - eq["L"]) < 1e-9        # symmetric when even


def test_fit_davidson_nu_positive_and_predicts():
    eng = wh.EloEngine.from_results(wh.RESULTS_CSV)
    nu = wb.fit_davidson_nu(2014, eng)
    assert nu > 0.0
    preds = wb.predict_elo_logistic(2014, eng)
    actuals = wb.actual_outcomes(2014)
    s = wb.score_matches(preds, actuals)
    u = wb.score_matches(wb.predict_uniform(2014, eng), actuals)
    assert len(preds) == 64
    assert s["rps"] < u["rps"]
```

- [ ] **Step 2: Run, verify fail**

Run: `python3 -m pytest tests/test_e62_worldcup_benchmark.py -k "davidson or elo_logistic" -v`
Expected: FAIL (missing functions).

- [ ] **Step 3: Implement**

Add to `examples/worldcup_benchmark.py`:
```python
import numpy as np
from scipy.optimize import minimize_scalar


def _davidson_probs(elo_diff: float, nu: float) -> Dict[str, float]:
    f = 10 ** (elo_diff / 400.0)
    g = 10 ** (-elo_diff / 400.0)
    draw = max(nu, 0.0) * math.sqrt(f * g)
    z = f + g + draw
    return {"W": f / z, "D": draw / z, "L": g / z}


def fit_davidson_nu(year: int, eng: "wh.EloEngine") -> float:
    """MLE of the Davidson draw parameter on pre-cup internationals (leakage-free).

    Uses each training match's pre-match Elo gap from a fresh engine frozen at the
    cup's freeze date (ratings_asof), so no post-cup info leaks.
    """
    elo = eng.ratings_asof(wh._cup_freeze_date(year))
    tr = training_matches(year, years=4)
    gaps, outs = [], []
    for r in tr:
        ha = 0.0 if r["neutral"] else wh.HOME_ADVANTAGE
        d = (elo.get(r["home"], 1500.0) + ha) - elo.get(r["away"], 1500.0)
        gaps.append(d)
        outs.append("W" if r["hg"] > r["ag"] else ("D" if r["hg"] == r["ag"] else "L"))

    def neg_ll(log_nu: float) -> float:
        nu = math.exp(log_nu)
        s = 0.0
        for d, o in zip(gaps, outs):
            s += -math.log(max(_davidson_probs(d, nu)[o], 1e-12))
        return s

    res = minimize_scalar(neg_ll, bounds=(-6.0, 3.0), method="bounded")
    return math.exp(res.x)


def predict_elo_logistic(year: int, eng: "wh.EloEngine") -> Dict[tuple, Dict[str, float]]:
    elo = eng.ratings_asof(wh._cup_freeze_date(year))
    host = _host_of(year)
    nu = fit_davidson_nu(year, eng)
    out = {}
    for r in cup_matches(year):
        d = wh._eff(r["home"], elo, host, 1500.0) - wh._eff(r["away"], elo, host, 1500.0)
        out[(r["date"], r["home"], r["away"])] = _davidson_probs(d, nu)
    return out
```

- [ ] **Step 4: Run tests, verify pass**

Run: `python3 -m pytest tests/test_e62_worldcup_benchmark.py -k "davidson or elo_logistic" -v`
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add examples/worldcup_benchmark.py tests/test_e62_worldcup_benchmark.py
git commit -m "feat(e62): Elo-logistic (Davidson) baseline with fitted draw parameter

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Maher Poisson team-strength model

**Files:**
- Modify: `examples/worldcup_benchmark.py`
- Test: `tests/test_e62_worldcup_benchmark.py`

Independent-Poisson attack/defense model fit by MLE on the 4-year pre-cup window:
`log λ_home = μ + atk[home] − def[away] + γ`, `log λ_away = μ + atk[away] − def[home]`.
Identifiability: fix `mean(atk)=mean(def)=0` via a soft sum-to-zero penalty. W/D/L
from the independent-Poisson score grid (0..MAXG goals each side).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_e62_worldcup_benchmark.py`:
```python
def test_poisson_grid_probs_normalise():
    p = wb._poisson_wdl(1.6, 0.9, max_goals=10)
    assert abs(sum(p.values()) - 1.0) < 1e-6
    assert p["W"] > p["L"]            # home expects more goals


def test_maher_fit_and_predict_beats_uniform():
    eng = wh.EloEngine.from_results(wh.RESULTS_CSV)
    model = wb.fit_maher(2014)
    assert "atk" in model and "dee" in model and "home" in model and "mu" in model
    preds = wb.predict_maher(2014, model)
    actuals = wb.actual_outcomes(2014)
    s = wb.score_matches(preds, actuals)
    u = wb.score_matches(wb.predict_uniform(2014, eng), actuals)
    assert len(preds) == 64
    assert s["rps"] < u["rps"]
```

- [ ] **Step 2: Run, verify fail**

Run: `python3 -m pytest tests/test_e62_worldcup_benchmark.py -k "poisson_grid or maher" -v`
Expected: FAIL (missing functions).

- [ ] **Step 3: Implement**

Add to `examples/worldcup_benchmark.py`:
```python
def _poisson_wdl(lam_home: float, lam_away: float, max_goals: int = 10) -> Dict[str, float]:
    """W/D/L from two independent Poissons over a 0..max_goals score grid."""
    def pmf(lam, k):
        return math.exp(-lam) * lam ** k / math.factorial(k)
    ph = [pmf(lam_home, k) for k in range(max_goals + 1)]
    pa = [pmf(lam_away, k) for k in range(max_goals + 1)]
    w = d = l = 0.0
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            p = ph[i] * pa[j]
            if i > j:
                w += p
            elif i == j:
                d += p
            else:
                l += p
    z = w + d + l
    return {"W": w / z, "D": d / z, "L": l / z}


def fit_maher(year: int, years: int = 4) -> dict:
    """MLE of a Maher independent-Poisson attack/defense model on the pre-cup window.

    Returns {teams, atk, dee (defense), home (γ), mu}. Leakage-free: training_matches
    only includes internationals before the cup's freeze date.
    """
    tr = training_matches(year, years=years)
    teams = sorted({r["home"] for r in tr} | {r["away"] for r in tr})
    idx = {t: i for i, t in enumerate(teams)}
    nt = len(teams)
    hi = np.array([idx[r["home"]] for r in tr])
    ai = np.array([idx[r["away"]] for r in tr])
    hg = np.array([r["hg"] for r in tr], dtype=float)
    ag = np.array([r["ag"] for r in tr], dtype=float)

    # params: [mu, gamma, atk(nt), def(nt)]
    def unpack(x):
        return x[0], x[1], x[2:2 + nt], x[2 + nt:2 + 2 * nt]

    def neg_ll(x):
        mu, gamma, atk, dee = unpack(x)
        lh = np.exp(mu + gamma + atk[hi] - dee[ai])
        la = np.exp(mu + atk[ai] - dee[hi])
        ll = np.sum(hg * np.log(lh) - lh) + np.sum(ag * np.log(la) - la)
        pen = 1e3 * (atk.mean() ** 2 + dee.mean() ** 2)   # soft sum-to-zero
        return -ll + pen

    from scipy.optimize import minimize
    x0 = np.concatenate([[0.0, 0.2], np.zeros(nt), np.zeros(nt)])
    res = minimize(neg_ll, x0, method="L-BFGS-B")
    mu, gamma, atk, dee = unpack(res.x)
    return {"teams": teams, "idx": idx, "mu": float(mu), "home": float(gamma),
            "atk": atk, "dee": dee}


def predict_maher(year: int, model: dict) -> Dict[tuple, Dict[str, float]]:
    idx, atk, dee = model["idx"], model["atk"], model["dee"]
    mu, gamma = model["mu"], model["home"]
    mean_atk = float(atk.mean()) if len(atk) else 0.0
    mean_dee = float(dee.mean()) if len(dee) else 0.0

    def a_of(t):
        return atk[idx[t]] if t in idx else mean_atk

    def d_of(t):
        return dee[idx[t]] if t in idx else mean_dee

    out = {}
    for r in cup_matches(year):
        h, a = r["home"], r["away"]
        lam_h = math.exp(mu + gamma + a_of(h) - d_of(a))
        lam_a = math.exp(mu + a_of(a) - d_of(h))
        out[(r["date"], h, a)] = _poisson_wdl(lam_h, lam_a)
    return out
```

- [ ] **Step 4: Run tests, verify pass**

Run: `python3 -m pytest tests/test_e62_worldcup_benchmark.py -k "poisson_grid or maher" -v`
Expected: 2 PASS. (Fitting on ~3-4k matches × ~150 teams may take a few seconds; acceptable. If L-BFGS-B is slow/non-converging, that's a real tuning issue — report, don't fake the result.)

- [ ] **Step 5: Commit**

```bash
git add examples/worldcup_benchmark.py tests/test_e62_worldcup_benchmark.py
git commit -m "feat(e62): Maher independent-Poisson team-strength baseline (scipy MLE)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: 538 loader with name reconciliation + alignment verification

**Files:**
- Modify: `examples/worldcup_benchmark.py`
- Test: `tests/test_e62_worldcup_benchmark.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_e62_worldcup_benchmark.py`:
```python
@pytest.mark.parametrize("year", [2018, 2022])
def test_fte_aligns_to_real_fixtures(year):
    preds = wb.predict_fte(year)
    actuals = wb.actual_outcomes(year)
    assert len(preds) == 64
    # every 538 match_key must be a real fixture (1:1 alignment after name mapping)
    assert all(k in actuals for k in preds), set(preds) - set(actuals)
    for v in preds.values():
        assert abs(sum(v.values()) - 1.0) < 1e-6


def test_fte_probs_oriented_home_perspective():
    # 2022 opener Qatar(home) vs Ecuador: 538 favoured Ecuador, so L > W from
    # Qatar's home perspective.
    preds = wb.predict_fte(2022)
    key = next(k for k in preds if k[1] == "Qatar" and k[2] == "Ecuador")
    assert preds[key]["L"] > preds[key]["W"]
```

- [ ] **Step 2: Run, verify fail**

Run: `python3 -m pytest tests/test_e62_worldcup_benchmark.py -k "fte" -v`
Expected: FAIL (missing `predict_fte`).

- [ ] **Step 3: Implement**

Add to `examples/worldcup_benchmark.py`:
```python
# 538 team name -> results.csv spelling, where they differ. Extended during impl
# until every cup's 64 538 rows align 1:1 to the real fixtures (the test is the gate).
FTE_NAME = {
    "IR Iran": "Iran",
    "Korea Republic": "South Korea",
    "USA": "United States",
    "China PR": "China",
}


def _fte_name(n: str) -> str:
    return FTE_NAME.get(n, n)


def predict_fte(year: int) -> Dict[tuple, Dict[str, float]]:
    """Load 538's per-match W/D/L probs, oriented to our (home=results.csv) keys.

    538 rows give team1/team2 + prob1/probtie/prob2. We match each to the real
    fixture by date + unordered team pair, and orient to the real home team.
    """
    real = {frozenset((r["home"], r["away"])): (r["date"], r["home"], r["away"])
            for r in cup_matches(year)}
    out = {}
    path = FTE_DIR / f"wc_{year}.csv"
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            t1, t2 = _fte_name(row["team1"]), _fte_name(row["team2"])
            p1, ptie, p2 = float(row["prob1"]), float(row["probtie"]), float(row["prob2"])
            key = real.get(frozenset((t1, t2)))
            if key is None:
                raise ValueError(f"538 {year}: no real fixture for {t1} vs {t2} "
                                 f"(extend FTE_NAME)")
            _date, home, _away = key
            if home == t1:                       # 538 team1 == real home
                probs = {"W": p1, "D": ptie, "L": p2}
            else:                                # flip orientation
                probs = {"W": p2, "D": ptie, "L": p1}
            z = sum(probs.values())
            out[key] = {k: v / z for k, v in probs.items()}
    return out
```

- [ ] **Step 4: Run tests, verify pass**

Run: `python3 -m pytest tests/test_e62_worldcup_benchmark.py -k "fte" -v`
Expected: 3 PASS. If `test_fte_aligns_to_real_fixtures` raises "no real fixture for X vs Y (extend FTE_NAME)", add the missing 538→results.csv name mapping to `FTE_NAME` and re-run. Do NOT weaken the 1:1 alignment assertion.

- [ ] **Step 5: Commit**

```bash
git add examples/worldcup_benchmark.py tests/test_e62_worldcup_benchmark.py
git commit -m "feat(e62): 538 SPI loader with name reconciliation + fixture alignment

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: run_benchmark + experiment driver

**Files:**
- Modify: `examples/worldcup_benchmark.py`
- Create: `experiments/e62_worldcup_benchmark.py`
- Test: `tests/test_e62_worldcup_benchmark.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_e62_worldcup_benchmark.py`:
```python
def test_run_benchmark_structure_and_floor():
    res = wb.run_benchmark(sims=2000)
    assert set(res["per_model"]) >= {"uniform", "elo_logistic", "ours_frozen",
                                     "ours_walk_forward", "maher"}
    # every probabilistic model beats the uniform floor on pooled RPS
    u = res["per_model"]["uniform"]["pooled"]["rps"]
    for name in ("elo_logistic", "ours_frozen", "ours_walk_forward", "maher"):
        assert res["per_model"][name]["pooled"]["rps"] < u, name
    # 538 head-to-head present on 128 matches
    assert res["head_to_head_538"]["n"] == 128
    assert "five_thirty_eight" in res["head_to_head_538"]["per_model"]
```

- [ ] **Step 2: Run, verify fail**

Run: `python3 -m pytest tests/test_e62_worldcup_benchmark.py::test_run_benchmark_structure_and_floor -v`
Expected: FAIL (missing `run_benchmark`).

- [ ] **Step 3: Implement `run_benchmark`**

Add to `examples/worldcup_benchmark.py`:
```python
def _pooled_and_per_cup(predict_fn, cups) -> dict:
    """predict_fn(year) -> {match_key: probs}. Returns pooled + per-cup scores."""
    all_preds, all_acts = {}, {}
    per_cup = {}
    for y in cups:
        preds = predict_fn(y)
        acts = actual_outcomes(y)
        per_cup[str(y)] = score_matches(preds, acts)
        all_preds.update(preds)
        all_acts.update(acts)
    return {"pooled": score_matches(all_preds, all_acts), "per_cup": per_cup}


def run_benchmark(sims: int = 20000, seed: int = 2026) -> dict:
    eng = wh.EloEngine.from_results(wh.RESULTS_CSV)
    maher_models = {y: fit_maher(y) for y in CUPS}
    model_fns = {
        "uniform": lambda y: predict_uniform(y, eng),
        "elo_logistic": lambda y: predict_elo_logistic(y, eng),
        "ours_frozen": lambda y: predict_ours_frozen(y, eng, sims=sims, seed=seed),
        "ours_walk_forward": lambda y: predict_ours_walk_forward(y, eng, sims=sims, seed=seed),
        "maher": lambda y: predict_maher(y, maher_models[y]),
    }
    per_model = {name: _pooled_and_per_cup(fn, CUPS) for name, fn in model_fns.items()}

    # 538 head-to-head on 2018+2022 only: re-score every model on the shared matches.
    h2h_preds = {name: {} for name in list(model_fns) + ["five_thirty_eight"]}
    h2h_acts = {}
    for y in FTE_CUPS:
        acts = actual_outcomes(y)
        h2h_acts.update(acts)
        for name, fn in model_fns.items():
            h2h_preds[name].update(fn(y))
        h2h_preds["five_thirty_eight"].update(predict_fte(y))
    h2h = {name: score_matches(h2h_preds[name], h2h_acts) for name in h2h_preds}

    ranking = sorted(((n, m["pooled"]["rps"]) for n, m in per_model.items()),
                     key=lambda kv: kv[1])
    return {"cups": CUPS, "fte_cups": FTE_CUPS, "sims": sims, "seed": seed,
            "per_model": per_model,
            "head_to_head_538": {"n": sum(len(actual_outcomes(y)) for y in FTE_CUPS),
                                 "per_model": h2h},
            "ranking": ranking}
```

- [ ] **Step 4: Run the test, verify pass**

Run: `python3 -m pytest tests/test_e62_worldcup_benchmark.py::test_run_benchmark_structure_and_floor -v`
Expected: PASS. (If a model does NOT beat uniform, investigate before weakening — a fitted model below uniform is a real finding worth reporting, but usually signals a bug in that model.)

- [ ] **Step 5: Write the experiment driver**

Create `experiments/e62_worldcup_benchmark.py`:
```python
"""E62 - Benchmark the World Cup forecaster against other models.

Scores six match-level W/D/L forecasters (uniform, Elo-logistic/Davidson, our
Elo->Poisson frozen, our walk-forward, Maher Poisson team-strength, and 538 SPI)
with RPS/Brier on 2010-2022 (538 head-to-head on 2018/2022). The honest question is
where our model ranks relative to the field. save_results() precedes the asserts.

    python experiments/e62_worldcup_benchmark.py
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import worldcup_benchmark as wb  # noqa: E402
from common import save_results  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    res = wb.run_benchmark(sims=args.sims, seed=args.seed)
    save_results("e62_worldcup_benchmark", res)   # BEFORE asserts

    pm = res["per_model"]
    u = pm["uniform"]["pooled"]["rps"]
    for name in ("elo_logistic", "ours_frozen", "ours_walk_forward", "maher"):
        assert pm[name]["pooled"]["rps"] < u, (name, pm[name]["pooled"]["rps"], u)
    assert res["head_to_head_538"]["n"] == 128
    h2h = res["head_to_head_538"]["per_model"]
    assert h2h["five_thirty_eight"]["rps"] < u  # 538 also beats the floor
    assert h2h["ours_walk_forward"]["n"] == 128

    print("[E62] pooled RPS ranking (lower is better):")
    for name, r in res["ranking"]:
        print(f"  {name:<20} {r:.4f}")
    print("\n538 head-to-head (2018+2022, 128 matches):")
    for name in ("five_thirty_eight", "ours_walk_forward", "ours_frozen",
                 "maher", "elo_logistic", "uniform"):
        s = h2h[name]
        print(f"  {name:<20} RPS {s['rps']:.4f}  Brier {s['brier']:.3f}  "
              f"hit {s['hit_rate']*100:.0f}%")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run the driver from repo root (small then full)**

Run: `python3 experiments/e62_worldcup_benchmark.py --sims 3000`
Expected: `[saved]` line, a pooled-RPS ranking, and the 538 head-to-head block, no AssertionError.
Then full: `python3 experiments/e62_worldcup_benchmark.py`
Expected: same at default sims. Note runtime (Maher fits 4× + sampling; expect tens of seconds).

- [ ] **Step 7: Commit (module + driver + results JSON)**

```bash
git add examples/worldcup_benchmark.py experiments/e62_worldcup_benchmark.py experiments/results/e62_worldcup_benchmark.json tests/test_e62_worldcup_benchmark.py
git commit -m "feat(e62): run_benchmark aggregation + experiment driver + results

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: Paper integration

**Files:**
- Modify: `scripts/make_paper_assets.py`

- [ ] **Step 1: Add to `EXPERIMENTS`**

Append `"e62_worldcup_benchmark"` to the `EXPERIMENTS` list (after `"e61_worldcup_backtest",`).

- [ ] **Step 2: Add fig + table functions**

After `def table_worldcup_backtest` (search for it), add:
```python
def fig_worldcup_benchmark(e62):
    h2h = e62["head_to_head_538"]["per_model"]
    order = ["five_thirty_eight", "ours_walk_forward", "maher", "elo_logistic",
             "ours_frozen", "uniform"]
    labels = {"five_thirty_eight": "538", "ours_walk_forward": "ours (WF)",
              "maher": "Maher", "elo_logistic": "Elo-logistic",
              "ours_frozen": "ours (frozen)", "uniform": "uniform"}
    names = [n for n in order if n in h2h]
    rps = [h2h[n]["rps"] for n in names]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    colors = [TEAL if n.startswith("ours") else (BLUE if n == "five_thirty_eight" else SLATE)
              for n in names]
    ax.bar([labels[n] for n in names], rps, color=colors)
    ax.set_ylabel("RPS (lower = better)")
    ax.set_title("E62: match-level RPS, 538 head-to-head (2018+2022)")
    for i, v in enumerate(rps):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout(); fig.savefig(FIGS / "e62_worldcup_benchmark.pdf"); plt.close(fig)


def table_worldcup_benchmark(e62):
    pm = e62["per_model"]
    order = ["uniform", "elo_logistic", "maher", "ours_frozen", "ours_walk_forward"]
    label = {"uniform": "Uniform", "elo_logistic": "Elo-logistic", "maher": "Maher Poisson",
             "ours_frozen": "Ours (frozen)", "ours_walk_forward": "Ours (walk-fwd)"}
    lines = [r"\begin{tabular}{lrrr}", r"\toprule",
             r"Model (all 4 cups) & RPS & Brier & Hit \\", r"\midrule"]
    for n in order:
        s = pm[n]["pooled"]
        lines.append(f"{label[n]} & {s['rps']:.3f} & {s['brier']:.3f} & "
                     f"{s['hit_rate']*100:.0f}\\% \\\\")
    h = e62["head_to_head_538"]["per_model"]
    lines += [r"\midrule",
              r"\multicolumn{4}{l}{\emph{538 head-to-head (2018+2022, 128 matches)}} \\",
              f"538 & {h['five_thirty_eight']['rps']:.3f} & "
              f"{h['five_thirty_eight']['brier']:.3f} & "
              f"{h['five_thirty_eight']['hit_rate']*100:.0f}\\% \\\\",
              f"Ours (walk-fwd) & {h['ours_walk_forward']['rps']:.3f} & "
              f"{h['ours_walk_forward']['brier']:.3f} & "
              f"{h['ours_walk_forward']['hit_rate']*100:.0f}\\% \\\\",
              r"\bottomrule", r"\end{tabular}"]
    (TABLES / "worldcup_benchmark.tex").write_text("\n".join(lines))
```

- [ ] **Step 3: Register calls + macros**

In `main()`, after `table_worldcup_backtest(...)`:
```python
    fig_worldcup_benchmark(data["e62_worldcup_benchmark"])
    table_worldcup_benchmark(data["e62_worldcup_benchmark"])
```
In the macros block (the function with parameter `d`, near `macro("NumExperiments"`), add before the list closes (letters-only names; note `d` not `data`):
```python
        macro("BenchOursWFRPS", f"{d['e62_worldcup_benchmark']['head_to_head_538']['per_model']['ours_walk_forward']['rps']:.3f}"),
        macro("BenchFTERPS", f"{d['e62_worldcup_benchmark']['head_to_head_538']['per_model']['five_thirty_eight']['rps']:.3f}"),
        macro("BenchMaherRPS", f"{d['e62_worldcup_benchmark']['per_model']['maher']['pooled']['rps']:.3f}"),
        macro("BenchOursFrozenRPS", f"{d['e62_worldcup_benchmark']['per_model']['ours_frozen']['pooled']['rps']:.3f}"),
        macro("BenchUniformRPS", f"{d['e62_worldcup_benchmark']['per_model']['uniform']['pooled']['rps']:.3f}"),
```
Then bump `\NumExperiments` by 1 (e.g. "60" -> "61"); report before/after.

- [ ] **Step 4: Regenerate + compile**

Run: `python3 scripts/make_paper_assets.py`
Expected: completes, writes `paper/figs/e62_worldcup_benchmark.pdf`, `paper/tables/worldcup_benchmark.tex`, updated `paper/numbers.tex`. If it crashes on a DIFFERENT experiment's missing JSON (pre-existing), report it and instead verify ours in isolation:
```bash
python3 -c "import sys,json; sys.path.insert(0,'scripts'); import make_paper_assets as m; e=json.load(open('experiments/results/e62_worldcup_benchmark.json')); m.fig_worldcup_benchmark(e); m.table_worldcup_benchmark(e); print(open('paper/tables/worldcup_benchmark.tex').read())"
```
Then: `cd paper && tectonic main.tex 2>&1 | tail -20` — expect a clean compile, no "Missing \begin{document}" (digit in a macro name) and no undefined refs. Do NOT edit main.tex prose.

- [ ] **Step 5: Commit**

```bash
git add scripts/make_paper_assets.py paper/numbers.tex paper/figs/e62_worldcup_benchmark.pdf paper/tables/worldcup_benchmark.tex
git commit -m "paper: integrate E62 model benchmark (fig, table, macros)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```
(If make_paper_assets was blocked by a pre-existing missing JSON, commit only the script + the e62 pdf/tex and report that numbers.tex/tectonic were blocked by a pre-existing condition.)

---

## Task 10: Final verification + honest writeup

**Files:** none (verification); possibly `experiments/e62_worldcup_benchmark.py` docstring

- [ ] **Step 1: Run the full E62 + E61 + forecaster suites**

Run: `python3 -m pytest tests/test_e62_worldcup_benchmark.py tests/test_e61_worldcup_backtest.py tests/test_worldcup2026.py -q`
Expected: all pass.

- [ ] **Step 2: Run the benchmark end-to-end at full sims**

Run: `python3 experiments/e62_worldcup_benchmark.py`
Expected: ranking + 538 head-to-head printed, no AssertionError.

- [ ] **Step 3: Revert unrelated collateral from make_paper_assets**

`make_paper_assets.py` rebuilds ALL figures and `main.pdf`; the experiment re-run rewrites the JSON timestamp. Keep only E62-intended changes:
```bash
git checkout -- experiments/results/e62_worldcup_benchmark.json 2>/dev/null || true
git checkout -- paper/figs/*.png paper/main.pdf 2>/dev/null || true
git status --short
```
Expected: clean working tree (only the intended committed E62 files; `.DS_Store`/`worldcup2026_bracket.svg` remain untracked). If the committed results JSON differs from a re-run only by `timestamp`, the committed one stands.

- [ ] **Step 4: Honest writeup pass**

Open the E62 results JSON and confirm the driver docstring + (if you add prose later) any claims match the actual ranking. State plainly where our model lands — e.g. "ours-walk-forward RPS X vs 538 Y vs Maher Z": if we trail 538 and/or Maher, say so. Do not tune to a desired ranking. If `ours_frozen` ranks worse than `ours_walk_forward` (expected — less information), note that as the information-gap effect, not a defect.

---

## Self-Review (completed during planning)

- **Spec coverage:** 538 data vendoring (T1) ✓; RPS+Brier metric (T2) ✓; results reader/leakage window (T3) ✓; uniform + frozen + walk-forward (T4) ✓; Elo-logistic/Davidson (T5) ✓; Maher Poisson (T6) ✓; 538 loader + alignment + orientation (T7) ✓; run_benchmark + driver + 538 head-to-head + ranking (T8) ✓; paper fig/table/macros/NumExperiments (T9) ✓; final verification + honest writeup + collateral cleanup (T10) ✓. Chalk-hit-rate-only: intentionally NOT re-scored here (lives in E61); not a gap.
- **Placeholder scan:** no TBD/TODO; every code step has full code. `FTE_NAME` is seeded with likely mappings and the test gates completeness (extend-until-aligned is explicit, not a placeholder).
- **Type consistency:** `match_key = (date, home, away)` used uniformly; model fns all return `{key: {"W","D","L"}}`; `score_matches` consumes that + `actual_outcomes` (`{key: "W"|"D"|"L"}`); `fit_maher` returns keys `teams/idx/mu/home/atk/dee` consumed verbatim by `predict_maher`; `run_benchmark` keys (`per_model`,`head_to_head_538`,`ranking`) match the driver and paper functions.
- **Known follow-through:** numpy/scipy imported inside `worldcup_benchmark.py` only (T5/T6); `worldcup_history.py` untouched.
