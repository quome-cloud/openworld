# E61 — Historical World Cup Backtest — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate the OpenWorld Elo→Poisson world model by forecasting the 2010/2014/2018/2022 World Cups from pre-tournament information only, then scoring against real results (match-level skill, tournament calibration, a chalk baseline, and simulated-vs-actual brackets).

**Architecture:** A reusable engine module `examples/worldcup_history.py` computes leakage-free World Football Elo from the full results history, holds the four cups' known group draws (verified against the data), runs a 32-team variant of the forecaster's group/knockout simulation, and scores predictions. The forecaster's goal model (`sample_match`) is refactored once to expose a rating-parameterised core (`sample_goals_from_elo`) so the *identical* outcome model is shared, not duplicated. A thin experiment driver `experiments/e61_worldcup_backtest.py` orchestrates, saves results, and self-checks. Data is vendored under `datasets/openworld-football/` so everything is offline and reproducible.

**Tech Stack:** Python stdlib only (`csv`, `random`, `math`, `statistics`) for the core/engine; `experiments/common.py` helpers (`save_results`, `spearman`); `matplotlib` only in `scripts/make_paper_assets.py` (existing). No new runtime dependencies.

**Branch:** `jenia/e61-worldcup-backtest` (already checked out; contains the 2026 forecaster + the E61 spec). One branch → one PR to `main`.

---

## File Structure

- **Create** `datasets/openworld-football/results.csv`, `shootouts.csv`, `elo_ratings_wc2026.csv`, `README.md` — vendored data + provenance.
- **Modify** `examples/worldcup2026.py` — extract `sample_goals_from_elo`; `sample_match` delegates (behaviour unchanged).
- **Create** `examples/worldcup_history.py` — Elo engine, cup constants, 32-team sim, metrics. The substance.
- **Create** `experiments/e61_worldcup_backtest.py` — thin driver: compute → simulate → score → `save_results` → assert.
- **Create** `tests/test_e61_worldcup_backtest.py` — unit tests for engine, structure, metrics, determinism.
- **Modify** `scripts/make_paper_assets.py` — `EXPERIMENTS` entry, `fig_worldcup_backtest`/`table_worldcup_backtest`, `main()` calls, macros, `\NumExperiments` bump.
- **Output (generated, not committed by hand)** `experiments/results/e61_worldcup_backtest.json`, four bracket SVGs under `paper/figs/`.

Data lives at repo root `datasets/openworld-football/`. The engine resolves it via `Path(__file__).resolve().parents[1] / "datasets" / "openworld-football"`.

---

## Task 1: Vendor the data

**Files:**
- Create: `datasets/openworld-football/{results.csv,shootouts.csv,elo_ratings_wc2026.csv,README.md}`

- [ ] **Step 1: Copy the three CSVs into the repo**

```bash
mkdir -p datasets/openworld-football
cp "/Users/jeniaquome/Downloads/archive (1)/results.csv"   datasets/openworld-football/results.csv
cp "/Users/jeniaquome/Downloads/archive (1)/shootouts.csv" datasets/openworld-football/shootouts.csv
cp "/Users/jeniaquome/Downloads/archive/elo_ratings_wc2026.csv" datasets/openworld-football/elo_ratings_wc2026.csv
```

- [ ] **Step 2: Write the provenance README**

Create `datasets/openworld-football/README.md`:

```markdown
# International football data (vendored for E61)

Used by `examples/worldcup_history.py` and `experiments/e61_worldcup_backtest.py`
to backtest the OpenWorld World Cup forecaster on 2010/2014/2018/2022.

| File | Rows | Source (Kaggle) | License |
|---|---|---|---|
| `results.csv` | ~49k | martj42 — *International football results 1872–2026* | CC BY 4.0 (per dataset page) |
| `shootouts.csv` | ~0.7k | same dataset | CC BY 4.0 |
| `elo_ratings_wc2026.csv` | ~4.7k | afonsofernandescruz — *2026 FIFA World Cup historical Elo* | CC BY 4.0 |

Columns:
- `results.csv`: date, home_team, away_team, home_score, away_score, tournament, city, country, neutral
- `shootouts.csv`: date, home_team, away_team, winner, first_shooter
- `elo_ratings_wc2026.csv`: year-end Elo snapshots for the 48 teams that qualified
  for the 2026 World Cup (used here ONLY to validate our computed Elo; it omits
  any team that didn't qualify for 2026, so it is not a model input).

`results.csv` is both the Elo-engine input and the ground truth scored against.
No look-ahead: a cup's ratings are frozen as of the day before its opening match.
```

- [ ] **Step 3: Verify the files load and cover all four cups**

Run:
```bash
python3 -c "
import csv
from collections import defaultdict
base='datasets/openworld-football'
n=sum(1 for _ in open(f'{base}/results.csv'))-1
print('results rows:', n)
parts=defaultdict(set)
for r in csv.DictReader(open(f'{base}/results.csv')):
    if r['tournament']=='FIFA World Cup' and int(r['date'][:4]) in (2010,2014,2018,2022):
        parts[int(r['date'][:4])].update([r['home_team'],r['away_team']])
for y in (2010,2014,2018,2022):
    assert len(parts[y])==32, (y, len(parts[y]))
print('all four cups have 32 teams: OK')
"
```
Expected: `results rows: 49477` (approx), `all four cups have 32 teams: OK`.

- [ ] **Step 4: Commit**

```bash
git add datasets/openworld-football/
git commit -m "data: vendor international results + Elo CSVs for E61 backtest

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Refactor the goal model to a rating-parameterised core

The forecaster's `sample_match(home, away, rng)` reads module globals `ELO`/`HOSTS`. Extract the pure math so E61 can call the *same* model with per-cup ratings.

**Files:**
- Modify: `examples/worldcup2026.py:156-163`
- Test: `tests/test_worldcup2026.py` (add one test)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_worldcup2026.py`:

```python
def test_sample_goals_from_elo_matches_sample_match():
    # sample_match must be exactly sample_goals_from_elo on the effective Elos,
    # so the rating-parameterised core is the identical model.
    import random as _r
    for home, away in [("Spain", "Qatar"), ("Algeria", "Iran"), ("Mexico", "Czechia")]:
        rng_a = _r.Random(42)
        rng_b = _r.Random(42)
        got = [wc.sample_match(home, away, rng_a) for _ in range(50)]
        want = [wc.sample_goals_from_elo(wc._eff_elo(home), wc._eff_elo(away), rng_b)
                for _ in range(50)]
        assert got == want
```

- [ ] **Step 2: Run it, verify it fails**

Run: `python3 -m pytest tests/test_worldcup2026.py::test_sample_goals_from_elo_matches_sample_match -v`
Expected: FAIL — `AttributeError: module 'worldcup2026' has no attribute 'sample_goals_from_elo'`.

- [ ] **Step 3: Implement the refactor**

In `examples/worldcup2026.py`, replace the body of `sample_match` (lines ~156-163) with:

```python
def sample_goals_from_elo(
    elo_home: float,
    elo_away: float,
    rng: random.Random,
    *,
    total_goals: float = None,
    supremacy: float = None,
) -> Tuple[int, int]:
    """Elo ratings -> (home_goals, away_goals) via the Poisson goal model.

    The rating-parameterised core shared by the 2026 forecaster (which passes
    host-adjusted Elos) and the E61 historical backtest (its own per-cup Elos).
    `total_goals`/`supremacy` default to the module dials.
    """
    tg = TOTAL_GOALS if total_goals is None else total_goals
    sup = SUPREMACY if supremacy is None else supremacy
    diff = elo_home - elo_away
    expected = 1.0 / (1.0 + 10 ** (-diff / 400.0))   # home expected score in [0,1]
    margin = sup * (2 * expected - 1)                 # >0 favours home
    lam_home = max(tg / 2 + margin / 2, 0.05)
    lam_away = max(tg / 2 - margin / 2, 0.05)
    return _poisson(lam_home, rng), _poisson(lam_away, rng)


def sample_match(home: str, away: str, rng: random.Random) -> Tuple[int, int]:
    """Sample (home_goals, away_goals) from the Elo->Poisson model."""
    return sample_goals_from_elo(_eff_elo(home), _eff_elo(away), rng)
```

(Keep `_poisson`, `_eff_elo`, `TOTAL_GOALS`, `SUPREMACY` exactly as they are.)

- [ ] **Step 4: Run the new test + the whole forecaster suite**

Run: `python3 -m pytest tests/test_worldcup2026.py -v`
Expected: all PASS (the new test plus every pre-existing test — behaviour is unchanged because `sample_match` now computes the identical lambdas).

- [ ] **Step 5: Commit**

```bash
git add examples/worldcup2026.py tests/test_worldcup2026.py
git commit -m "refactor: expose rating-parameterised sample_goals_from_elo

sample_match now delegates to a pure (elo_home, elo_away) core so the
historical backtest reuses the identical goal model. Behaviour unchanged.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Elo engine — compute leakage-free ratings from results.csv

**Files:**
- Create: `examples/worldcup_history.py`
- Test: `tests/test_e61_worldcup_backtest.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_e61_worldcup_backtest.py`:

```python
"""Tests for the historical World Cup backtest (examples/worldcup_history.py)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import worldcup_history as wh  # noqa: E402


def test_elo_engine_runs_and_rates_known_strong_teams():
    eng = wh.EloEngine.from_results(wh.RESULTS_CSV)
    # Pre-2014 (frozen at the day before the 2014 opener): Brazil & Germany strong.
    ratings = eng.ratings_asof("2014-06-11")
    assert ratings["Brazil"] > 1850
    assert ratings["Germany"] > 1850
    assert ratings["Brazil"] > ratings["United States"]


def test_no_look_ahead():
    # Ratings frozen before a cup must not change if we only append matches that
    # happen on/after the freeze date.
    eng = wh.EloEngine.from_results(wh.RESULTS_CSV)
    before = eng.ratings_asof("2014-06-11")
    # Recompute using only matches strictly before the freeze date -> identical.
    eng2 = wh.EloEngine.from_results(wh.RESULTS_CSV, until="2014-06-11")
    after = eng2.ratings_asof("2014-06-11")
    for team in ["Brazil", "Germany", "Italy", "Costa Rica"]:
        assert before[team] == after[team]


def test_elo_update_is_zero_sum_per_match():
    # A single match shifts the two teams' ratings by equal and opposite amounts.
    eng = wh.EloEngine(base=1500.0)
    a0, b0 = eng.rating("A"), eng.rating("B")
    eng.update_match("A", "B", 2, 0, neutral=True, k=60.0)
    da = eng.rating("A") - a0
    db = eng.rating("B") - b0
    assert abs(da + db) < 1e-9
    assert da > 0  # winner gains
```

- [ ] **Step 2: Run them, verify they fail**

Run: `python3 -m pytest tests/test_e61_worldcup_backtest.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'worldcup_history'`.

- [ ] **Step 3: Implement the Elo engine**

Create `examples/worldcup_history.py` with the header and engine:

```python
"""Historical World Cup backtest engine (2010 / 2014 / 2018 / 2022).

Computes leakage-free World Football Elo from the full results history
(datasets/openworld-football/results.csv), reuses the 2026 forecaster's
Elo->Poisson goal model in a 32-team format, and scores forecasts against real
results. Zero external deps (stdlib only).

    python examples/worldcup_history.py            # summary for all four cups
"""

from __future__ import annotations

import csv
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Reuse the IDENTICAL goal model + rating-agnostic standings from the forecaster.
sys.path.insert(0, os.path.dirname(__file__))
from worldcup2026 import sample_goals_from_elo, group_standings, _table  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "datasets" / "openworld-football"
RESULTS_CSV = DATA_DIR / "results.csv"
SHOOTOUTS_CSV = DATA_DIR / "shootouts.csv"
PUBLISHED_ELO_CSV = DATA_DIR / "elo_ratings_wc2026.csv"

HOME_ADVANTAGE = 100.0   # World Football Elo home bump (0 on neutral ground)

# K-factor by match importance (World Football Elo conventions), keyed off the
# results.csv `tournament` column; default for anything unlisted.
K_BY_TOURNAMENT = {
    "FIFA World Cup": 60.0,
    "Confederations Cup": 50.0,
    "UEFA Euro": 50.0,
    "Copa América": 50.0,
    "African Cup of Nations": 50.0,
    "AFC Asian Cup": 50.0,
    "Gold Cup": 50.0,
    "FIFA World Cup qualification": 40.0,
    "UEFA Euro qualification": 40.0,
    "Copa América qualification": 40.0,
    "African Cup of Nations qualification": 40.0,
    "AFC Asian Cup qualification": 40.0,
    "Gold Cup qualification": 40.0,
    "UEFA Nations League": 40.0,
    "Friendly": 20.0,
}
K_DEFAULT = 30.0


def _k_for(tournament: str) -> float:
    return K_BY_TOURNAMENT.get(tournament, K_DEFAULT)


def _gd_multiplier(margin: int) -> float:
    """World Football Elo goal-difference weight G."""
    m = abs(margin)
    if m <= 1:
        return 1.0
    if m == 2:
        return 1.5
    return (11.0 + m) / 8.0


class EloEngine:
    """Incremental World Football Elo over a stream of dated matches."""

    def __init__(self, base: float = 1500.0):
        self.base = base
        self._r: Dict[str, float] = {}
        self._history: List[Tuple[str, Dict[str, float]]] = []  # (date, snapshot)

    def rating(self, team: str) -> float:
        return self._r.get(team, self.base)

    def update_match(self, home: str, away: str, hg: int, ag: int,
                     neutral: bool, k: float) -> None:
        ha = 0.0 if neutral else HOME_ADVANTAGE
        rh, ra = self.rating(home), self.rating(away)
        we_home = 1.0 / (1.0 + 10 ** (-((rh + ha) - ra) / 400.0))
        w_home = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
        g = _gd_multiplier(hg - ag)
        delta = k * g * (w_home - we_home)
        self._r[home] = rh + delta
        self._r[away] = ra - delta

    @classmethod
    def from_results(cls, results_csv: Path, until: Optional[str] = None,
                     base: float = 1500.0) -> "EloEngine":
        """Replay all matches in date order (optionally only those with
        date < `until`), recording a dated snapshot after each match."""
        eng = cls(base=base)
        rows = []
        with open(results_csv, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append(r)
        rows.sort(key=lambda r: r["date"])
        for r in rows:
            date = r["date"]
            if until is not None and date >= until:
                break
            try:
                hg, ag = int(r["home_score"]), int(r["away_score"])
            except (ValueError, KeyError):
                continue
            neutral = str(r["neutral"]).strip().upper() == "TRUE"
            eng.update_match(r["home_team"], r["away_team"], hg, ag,
                             neutral=neutral, k=_k_for(r["tournament"]))
            eng._history.append((date, dict(eng._r)))
        return eng

    def ratings_asof(self, date: str) -> Dict[str, float]:
        """Snapshot of all team ratings using only matches strictly before
        `date` (no look-ahead). Returns base for teams unseen by then."""
        snap: Dict[str, float] = {}
        for d, s in self._history:
            if d >= date:
                break
            snap = s
        # default-fill is handled by callers via .get(team, base); return as-is
        return dict(snap)
```

- [ ] **Step 4: Run the tests, verify they pass**

Run: `python3 -m pytest tests/test_e61_worldcup_backtest.py -v`
Expected: 3 PASS. (`ratings_asof` returns a dict; missing teams fall back to base via `.get` in callers — but the strong-team test indexes present teams directly, which exist by 2014. If `test_elo_engine_runs...` KeyErrors on a team, that team genuinely has no pre-2014 match, which won't happen for Brazil/Germany/USA.)

- [ ] **Step 5: Commit**

```bash
git add examples/worldcup_history.py tests/test_e61_worldcup_backtest.py
git commit -m "feat(e61): World Football Elo engine from full results history

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Validate computed Elo against the published ratings

**Files:**
- Modify: `examples/worldcup_history.py`
- Test: `tests/test_e61_worldcup_backtest.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_e61_worldcup_backtest.py`:

```python
def test_validation_against_published_elo_is_strong():
    eng = wh.EloEngine.from_results(wh.RESULTS_CSV)
    stats = wh.validate_against_published(eng, snapshot_year=2013)
    # Our reconstructed Elo should track eloratings.net well on shared teams.
    assert stats["n"] >= 20
    assert stats["spearman"] >= 0.7
    assert stats["pearson"] >= 0.7
    assert stats["rmse"] < 250.0
```

- [ ] **Step 2: Run it, verify it fails**

Run: `python3 -m pytest tests/test_e61_worldcup_backtest.py::test_validation_against_published_elo_is_strong -v`
Expected: FAIL — `AttributeError: ... has no attribute 'validate_against_published'`.

- [ ] **Step 3: Implement validation + name normalisation**

Add to `examples/worldcup_history.py`:

```python
# results.csv name -> published Elo file (`country`) name, where they differ.
NAME_TO_PUBLISHED = {
    "USA": "United States",
    "South Korea": "South Korea",      # same in both; explicit for clarity
    "China PR": "China",
    "Cape Verde": "Cabo Verde",
    "Ivory Coast": "Ivory Coast",
    "Czech Republic": "Czechia",
}


def _pearson(xs: List[float], ys: List[float]) -> float:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    vy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return cov / (vx * vy) if vx and vy else 0.0


def _spearman(xs: List[float], ys: List[float]) -> float:
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        rk = [0.0] * len(v)
        for pos, i in enumerate(order):
            rk[i] = pos
        return rk
    return _pearson(ranks(xs), ranks(ys))


def published_ratings(snapshot_year: int) -> Dict[str, float]:
    """Year-end published Elo for the given year, keyed by `country` name."""
    out: Dict[str, float] = {}
    with open(PUBLISHED_ELO_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if int(r["year"]) == snapshot_year and r["snapshot_date"].endswith("-12-31"):
                out[r["country"]] = float(r["rating"])
    return out


def validate_against_published(eng: "EloEngine", snapshot_year: int) -> dict:
    """Compare our computed end-of-year Elo to the published file on shared teams.

    Returns {n, pearson, spearman, rmse, ours, theirs} for teams present in both.
    """
    ours_all = eng.ratings_asof(f"{snapshot_year + 1}-01-01")  # end-of-year state
    pub = published_ratings(snapshot_year)
    xs, ys, used = [], [], []
    for team, ours in ours_all.items():
        key = NAME_TO_PUBLISHED.get(team, team)
        if key in pub:
            xs.append(ours)
            ys.append(pub[key])
            used.append(team)
    n = len(xs)
    rmse = math.sqrt(sum((a - b) ** 2 for a, b in zip(xs, ys)) / n) if n else float("nan")
    return {"n": n, "pearson": _pearson(xs, ys), "spearman": _spearman(xs, ys),
            "rmse": rmse, "snapshot_year": snapshot_year}
```

- [ ] **Step 4: Run the test, verify it passes**

Run: `python3 -m pytest tests/test_e61_worldcup_backtest.py::test_validation_against_published_elo_is_strong -v`
Expected: PASS. If `spearman`/`pearson` come in below 0.7 or `rmse` ≥ 250, do NOT relax the test blindly — first inspect the scatter (print `xs`/`ys` for outliers, usually name-mismatches) and extend `NAME_TO_PUBLISHED`. Only adjust thresholds if the relationship is genuinely strong but offset (document why in the test).

- [ ] **Step 5: Commit**

```bash
git add examples/worldcup_history.py tests/test_e61_worldcup_backtest.py
git commit -m "feat(e61): validate computed Elo vs published eloratings.net

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Cup data — group draws (verified) + real results, standings, advancers

**Files:**
- Modify: `examples/worldcup_history.py`
- Test: `tests/test_e61_worldcup_backtest.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_e61_worldcup_backtest.py`:

```python
@pytest.mark.parametrize("year", [2010, 2014, 2018, 2022])
def test_encoded_groups_match_real_data(year):
    cup = wh.load_cup(year)
    # 8 groups of 4 distinct teams.
    assert len(cup.groups) == 8
    assert sorted(cup.groups) == list("ABCDEFGH")
    teams = [t for g in cup.groups.values() for t in g]
    assert len(teams) == 32 and len(set(teams)) == 32
    # Every encoded group's 6 round-robin pairings appear as real group matches.
    for letter, four in cup.groups.items():
        for i in range(4):
            for j in range(i + 1, 4):
                a, b = four[i], four[j]
                assert cup.group_result(a, b) is not None, (year, letter, a, b)


@pytest.mark.parametrize("year,champion", [
    (2010, "Spain"), (2014, "Germany"), (2018, "France"), (2022, "Argentina")])
def test_actual_champion_recovered(year, champion):
    cup = wh.load_cup(year)
    assert cup.actual_champion() == champion
```

- [ ] **Step 2: Run them, verify they fail**

Run: `python3 -m pytest tests/test_e61_worldcup_backtest.py -k "encoded_groups or champion" -v`
Expected: FAIL — `AttributeError: ... has no attribute 'load_cup'`.

- [ ] **Step 3: Implement cup constants + loader**

Add to `examples/worldcup_history.py`. The group draws below are historical fact
(the official draw, known months pre-tournament); they are verified against the
data by the test above. **Use `results.csv` team-name spellings.**

```python
# Official group draws (results.csv spellings). Verified against real fixtures.
CUP_GROUPS: Dict[int, Dict[str, List[str]]] = {
    2010: {
        "A": ["South Africa", "Mexico", "Uruguay", "France"],
        "B": ["Argentina", "Nigeria", "South Korea", "Greece"],
        "C": ["England", "United States", "Algeria", "Slovenia"],
        "D": ["Germany", "Australia", "Serbia", "Ghana"],
        "E": ["Netherlands", "Denmark", "Japan", "Cameroon"],
        "F": ["Italy", "Paraguay", "New Zealand", "Slovakia"],
        "G": ["Brazil", "North Korea", "Ivory Coast", "Portugal"],
        "H": ["Spain", "Switzerland", "Honduras", "Chile"],
    },
    2014: {
        "A": ["Brazil", "Croatia", "Mexico", "Cameroon"],
        "B": ["Spain", "Netherlands", "Chile", "Australia"],
        "C": ["Colombia", "Greece", "Ivory Coast", "Japan"],
        "D": ["Uruguay", "Costa Rica", "England", "Italy"],
        "E": ["Switzerland", "Ecuador", "France", "Honduras"],
        "F": ["Argentina", "Bosnia and Herzegovina", "Iran", "Nigeria"],
        "G": ["Germany", "Portugal", "Ghana", "United States"],
        "H": ["Belgium", "Algeria", "Russia", "South Korea"],
    },
    2018: {
        "A": ["Russia", "Saudi Arabia", "Egypt", "Uruguay"],
        "B": ["Portugal", "Spain", "Morocco", "Iran"],
        "C": ["France", "Australia", "Peru", "Denmark"],
        "D": ["Argentina", "Iceland", "Croatia", "Nigeria"],
        "E": ["Brazil", "Switzerland", "Costa Rica", "Serbia"],
        "F": ["Germany", "Mexico", "Sweden", "South Korea"],
        "G": ["Belgium", "Panama", "Tunisia", "England"],
        "H": ["Poland", "Senegal", "Colombia", "Japan"],
    },
    2022: {
        "A": ["Qatar", "Ecuador", "Senegal", "Netherlands"],
        "B": ["England", "Iran", "United States", "Wales"],
        "C": ["Argentina", "Saudi Arabia", "Mexico", "Poland"],
        "D": ["France", "Australia", "Denmark", "Tunisia"],
        "E": ["Spain", "Costa Rica", "Germany", "Japan"],
        "F": ["Belgium", "Canada", "Morocco", "Croatia"],
        "G": ["Brazil", "Serbia", "Switzerland", "Cameroon"],
        "H": ["Portugal", "Ghana", "Uruguay", "South Korea"],
    },
}

# Host nation per cup (gets the forecaster's host Elo bump in simulation).
CUP_HOST = {2010: "South Africa", 2014: "Brazil", 2018: "Russia", 2022: "Qatar"}

# Fixed Round-of-16 pairing: (group-winner letter, group-runnerup letter).
R16_PAIRS = [("A", "B"), ("C", "D"), ("E", "F"), ("G", "H"),
             ("B", "A"), ("D", "C"), ("F", "E"), ("H", "G")]


class Cup:
    """One World Cup's draw + real results, loaded from the vendored data."""

    def __init__(self, year: int):
        self.year = year
        self.groups = CUP_GROUPS[year]
        self.host = CUP_HOST[year]
        self._team_to_group = {t: g for g, ts in self.groups.items() for t in ts}
        self._group_res: Dict[frozenset, Tuple[str, int, int]] = {}
        self._ko: List[dict] = []
        self._shootouts: Dict[frozenset, str] = {}
        self._load()

    def _load(self):
        with open(SHOOTOUTS_CSV, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["date"][:4] == str(self.year):
                    self._shootouts[frozenset((r["home_team"], r["away_team"]))] = r["winner"]
        with open(RESULTS_CSV, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["tournament"] != "FIFA World Cup" or r["date"][:4] != str(self.year):
                    continue
                h, a = r["home_team"], r["away_team"]
                hg, ag = int(r["home_score"]), int(r["away_score"])
                same_group = (self._team_to_group.get(h) is not None
                              and self._team_to_group.get(h) == self._team_to_group.get(a))
                if same_group:
                    self._group_res[frozenset((h, a))] = (h, hg, ag)
                else:
                    key = frozenset((h, a))
                    winner = h if hg > ag else (a if ag > hg else self._shootouts.get(key))
                    self._ko.append({"date": r["date"], "home": h, "away": a,
                                     "hg": hg, "ag": ag, "winner": winner})
        self._ko.sort(key=lambda m: m["date"])

    def group_result(self, a: str, b: str):
        """(home, home_goals, away_goals) for the real group match, or None."""
        return self._group_res.get(frozenset((a, b)))

    def group_of(self, team: str) -> str:
        return self._team_to_group[team]

    def real_standings(self) -> Dict[str, List[str]]:
        """Real finishing order per group (reuses the forecaster's tiebreak)."""
        out = {}
        for g, teams in self.groups.items():
            res = {}
            for i in range(len(teams)):
                for j in range(i + 1, len(teams)):
                    rec = self.group_result(teams[i], teams[j])
                    home, hg, ag = rec
                    away = teams[j] if home == teams[i] else teams[i]
                    res[(home, away)] = (hg, ag)
            out[g] = group_standings(teams, res)
        return out

    def actual_advancers(self) -> Dict[str, str]:
        """Map round-name -> set of advancing teams, derived from real KO games.

        Returns {"R16": [...8 winners], "QF": [...4], "SF": [...2], "final":[champ]}
        using game count to bucket the 16 knockout matches by round.
        """
        # 16 KO matches in date order: first 8 = R16, next 4 = QF, next 2 = SF,
        # then 3rd-place playoff + final. Identify the final as the last match
        # whose two teams both won their SF; the 3rd-place game is the other.
        ko = self._ko
        r16 = ko[:8]; qf = ko[8:12]; sf = ko[12:14]; last_two = ko[14:16]
        sf_winners = {m["winner"] for m in sf}
        final = next(m for m in last_two
                     if m["home"] in sf_winners and m["away"] in sf_winners)
        return {
            "R16": [m["winner"] for m in r16],
            "QF": [m["winner"] for m in qf],
            "SF": [m["winner"] for m in sf],
            "final_match": final,
            "champion": final["winner"],
        }

    def actual_champion(self) -> str:
        return self.actual_advancers()["champion"]


def load_cup(year: int) -> Cup:
    return Cup(year)
```

- [ ] **Step 4: Run the tests, verify they pass**

Run: `python3 -m pytest tests/test_e61_worldcup_backtest.py -k "encoded_groups or champion" -v`
Expected: all PASS. If `test_encoded_groups_match_real_data` fails, a team name in `CUP_GROUPS` doesn't match `results.csv` (e.g. "Bosnia and Herzegovina", "North Korea") — fix the spelling to match the data, not the test.

- [ ] **Step 5: Commit**

```bash
git add examples/worldcup_history.py tests/test_e61_worldcup_backtest.py
git commit -m "feat(e61): cup draws (verified) + real results/standings/advancers

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: 32-team simulation (the model under test, historical Elo)

**Files:**
- Modify: `examples/worldcup_history.py`
- Test: `tests/test_e61_worldcup_backtest.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_e61_worldcup_backtest.py`:

```python
def test_forecast_cup_is_deterministic_and_normalised():
    eng = wh.EloEngine.from_results(wh.RESULTS_CSV)
    f1 = wh.forecast_cup(2014, eng, sims=300, seed=7)
    f2 = wh.forecast_cup(2014, eng, sims=300, seed=7)
    assert f1 == f2  # deterministic in seed
    champ = sum(v["champion"] for v in f1.values())
    assert abs(champ - 100.0) < 1e-6  # title probs sum to 100%
    # The pre-2014 favourites should top the title odds.
    top = max(f1, key=lambda t: f1[t]["champion"])
    assert top in {"Brazil", "Germany", "Argentina", "Spain"}
```

- [ ] **Step 2: Run it, verify it fails**

Run: `python3 -m pytest tests/test_e61_worldcup_backtest.py::test_forecast_cup_is_deterministic_and_normalised -v`
Expected: FAIL — `AttributeError: ... has no attribute 'forecast_cup'`.

- [ ] **Step 3: Implement the 32-team simulation**

Add to `examples/worldcup_history.py`. `group_standings`/`_table` are reused from
the forecaster; only the 8-group + R16 scaffolding is new, and the goal model is
the shared `sample_goals_from_elo`.

```python
KO_ROUNDS = ["R16", "QF", "SF", "final"]


def _eff(team: str, elo: Dict[str, float], host: str, base: float) -> float:
    from worldcup2026 import HOST_ADVANTAGE
    return elo.get(team, base) + (HOST_ADVANTAGE if team == host else 0.0)


def _sample(home, away, elo, host, base, rng):
    return sample_goals_from_elo(_eff(home, elo, host, base),
                                 _eff(away, elo, host, base), rng)


def _ko_match(home, away, elo, host, base, rng):
    hg, ag = _sample(home, away, elo, host, base, rng)
    if hg != ag:
        return (home if hg > ag else away), hg, ag, False
    dh = _eff(home, elo, host, base) - _eff(away, elo, host, base)
    p_home = 1.0 / (1.0 + 10 ** (-dh / 400.0))
    return (home if rng.random() < p_home else away), hg, ag, True


def simulate_cup_once(cup: "Cup", elo: Dict[str, float], rng: random.Random,
                      base: float = 1500.0) -> dict:
    """One full 32-team tournament. Returns reached-round per team + champion."""
    reached = {t: "group" for g in cup.groups.values() for t in g}
    winners, runners = {}, {}
    for g, teams in cup.groups.items():
        res = {}
        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                h, a = teams[i], teams[j]
                res[(h, a)] = _sample(h, a, elo, cup.host, base, rng)
        order = group_standings(teams, res)
        winners[g], runners[g] = order[0], order[1]
    # Seed R16 in fixed bracket order so the binary tree pairs neighbours.
    seeds = []
    for wl, rl in R16_PAIRS:
        seeds.append(winners[wl]); seeds.append(runners[rl])
    for t in seeds:
        reached[t] = "R16"
    teams = seeds
    for rnd in KO_ROUNDS:
        nxt = []
        for i in range(0, len(teams), 2):
            w, _hg, _ag, _p = _ko_match(teams[i], teams[i + 1], elo, cup.host, base, rng)
            nxt.append(w)
        nxt_round = KO_ROUNDS[KO_ROUNDS.index(rnd) + 1] if rnd != "final" else "champion"
        for w in nxt:
            reached[w] = nxt_round
        teams = nxt
    return {"champion": teams[0], "reached": reached}


_REACH_ORDER = ["group", "R16", "QF", "SF", "final", "champion"]


def forecast_cup(year: int, eng: "EloEngine", sims: int = 10000, seed: int = 2026,
                 base: float = 1500.0) -> Dict[str, Dict[str, float]]:
    """Monte-Carlo a cup from FROZEN pre-tournament Elo. Per-team probabilities (%)."""
    cup = load_cup(year)
    opener = min(m["date"] for m in cup._ko) if cup._ko else f"{year}-06-11"
    # Freeze the day before the cup's earliest recorded match.
    first_group = f"{year}-06-01"  # safe lower bound; ratings_asof uses < date
    elo = eng.ratings_asof(_cup_freeze_date(year))
    teams = [t for g in cup.groups.values() for t in g]
    titles = {t: 0 for t in teams}
    reach = {t: {r: 0 for r in _REACH_ORDER} for t in teams}
    for s in range(sims):
        rng = random.Random(seed * 1_000_003 + s)
        out = simulate_cup_once(cup, elo, rng, base=base)
        titles[out["champion"]] += 1
        for t, r in out["reached"].items():
            idx = _REACH_ORDER.index(r)
            for ri in range(1, idx + 1):
                reach[t][_REACH_ORDER[ri]] += 1
    def pct(n): return round(100.0 * n / sims, 4)
    return {t: {"champion": pct(titles[t]),
                "reach_final": pct(reach[t]["final"]),
                "reach_SF": pct(reach[t]["SF"]),
                "reach_QF": pct(reach[t]["QF"]),
                "reach_R16": pct(reach[t]["R16"])} for t in teams}


def _cup_freeze_date(year: int) -> str:
    """Day-before-opener freeze date per cup (no look-ahead)."""
    return {2010: "2010-06-11", 2014: "2014-06-12",
            2018: "2018-06-14", 2022: "2022-11-20"}[year]
```

(Remove the unused `opener`/`first_group` lines if your linter flags them; they are
left out of the final form — `forecast_cup` only needs `_cup_freeze_date`.)

- [ ] **Step 4: Run the test, verify it passes**

Run: `python3 -m pytest tests/test_e61_worldcup_backtest.py::test_forecast_cup_is_deterministic_and_normalised -v`
Expected: PASS.

- [ ] **Step 5: Clean up `forecast_cup` and commit**

Edit `forecast_cup` to drop the dead `opener`/`first_group` locals (keep only the
`elo = eng.ratings_asof(_cup_freeze_date(year))` line). Re-run the test (PASS), then:

```bash
git add examples/worldcup_history.py tests/test_e61_worldcup_backtest.py
git commit -m "feat(e61): 32-team Monte-Carlo simulation on frozen historical Elo

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Match-level skill (group W/D/L) + knockout advancement metrics

**Files:**
- Modify: `examples/worldcup_history.py`
- Test: `tests/test_e61_worldcup_backtest.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_e61_worldcup_backtest.py`:

```python
def test_group_match_skill_beats_uniform():
    eng = wh.EloEngine.from_results(wh.RESULTS_CSV)
    elo = eng.ratings_asof(wh._cup_freeze_date(2014))
    rows, summary = wh.score_group_matches(wh.load_cup(2014), elo, sims=4000, seed=3)
    assert summary["n"] == 48
    assert 0.0 <= summary["hit_rate"] <= 1.0
    assert summary["skill_vs_uniform"] > 0.0  # better than a 1/3 coin


def test_knockout_advancement_metric():
    eng = wh.EloEngine.from_results(wh.RESULTS_CSV)
    elo = eng.ratings_asof(wh._cup_freeze_date(2014))
    summary = wh.score_knockout_advancement(wh.load_cup(2014), elo, sims=4000, seed=3)
    assert summary["n"] == 16
    assert 0.0 <= summary["accuracy"] <= 1.0
    assert summary["brier"] >= 0.0
```

- [ ] **Step 2: Run them, verify they fail**

Run: `python3 -m pytest tests/test_e61_worldcup_backtest.py -k "skill or advancement" -v`
Expected: FAIL — missing `score_group_matches` / `score_knockout_advancement`.

- [ ] **Step 3: Implement the metrics**

Add to `examples/worldcup_history.py`:

```python
def _wdl_probs(home, away, elo, host, base, sims, rng) -> Dict[str, float]:
    w = d = 0
    for _ in range(sims):
        hg, ag = _sample(home, away, elo, host, base, rng)
        if hg > ag:
            w += 1
        elif hg == ag:
            d += 1
    return {"W": w / sims, "D": d / sims, "L": (sims - w - d) / sims}


def score_group_matches(cup: "Cup", elo: Dict[str, float], sims: int = 30000,
                        seed: int = 2026, base: float = 1500.0):
    """W/D/L Brier/hit-rate/skill over a cup's 48 group matches. Returns (rows, summary)."""
    rng = random.Random(seed)
    rows = []
    hits = p_sum = brier_sum = 0.0
    dec_hits = dec_n = draws = 0
    matches = []
    for g, teams in cup.groups.items():
        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                rec = cup.group_result(teams[i], teams[j])
                home, hg, ag = rec
                away = teams[j] if home == teams[i] else teams[i]
                matches.append((g, home, away, hg, ag))
    for (g, home, away, hg, ag) in matches:
        probs = _wdl_probs(home, away, elo, cup.host, base, sims, rng)
        actual = "W" if hg > ag else ("D" if hg == ag else "L")
        fav = max(probs, key=probs.get)
        brier = sum((probs[c] - (1.0 if c == actual else 0.0)) ** 2 for c in "WDL")
        hits += fav == actual; p_sum += probs[actual]; brier_sum += brier
        if actual == "D":
            draws += 1
        else:
            dec_n += 1
            dec_hits += (probs["W"] > probs["L"]) if actual == "W" else (probs["L"] > probs["W"])
        rows.append({"group": g, "home": home, "away": away, "score": f"{hg}-{ag}",
                     "actual": actual, "probs": probs, "hit": fav == actual, "brier": brier})
    n = len(matches)
    base_brier = 2 / 3
    return rows, {"n": n, "draws": draws, "hit_rate": hits / n,
                  "decisive_n": dec_n,
                  "decisive_hit_rate": dec_hits / dec_n if dec_n else 0.0,
                  "mean_p_actual": p_sum / n, "mean_brier": brier_sum / n,
                  "baseline_brier": base_brier,
                  "skill_vs_uniform": 1 - (brier_sum / n) / base_brier}


def score_knockout_advancement(cup: "Cup", elo: Dict[str, float], sims: int = 30000,
                               seed: int = 2026, base: float = 1500.0):
    """For each real KO match, model P(home advances) vs the actual advancer."""
    rng = random.Random(seed)
    rows = []
    correct = brier_sum = logloss_sum = 0.0
    n = 0
    for m in cup._ko:
        home, away, winner = m["home"], m["away"], m["winner"]
        if winner is None:
            continue
        adv = sum(_ko_match(home, away, elo, cup.host, base, rng)[0] == home
                  for _ in range(sims))
        p_home = adv / sims
        p_home = min(max(p_home, 1e-6), 1 - 1e-6)
        actual_home = 1.0 if winner == home else 0.0
        pick_home = p_home >= 0.5
        correct += (pick_home and actual_home) or ((not pick_home) and not actual_home)
        brier_sum += (p_home - actual_home) ** 2
        logloss_sum += -(actual_home * math.log(p_home) + (1 - actual_home) * math.log(1 - p_home))
        n += 1
        rows.append({"home": home, "away": away, "winner": winner,
                     "p_home_adv": p_home, "correct": pick_home == bool(actual_home)})
    return {"n": n, "accuracy": correct / n, "brier": brier_sum / n,
            "logloss": logloss_sum / n, "rows": rows}
```

- [ ] **Step 4: Run the tests, verify they pass**

Run: `python3 -m pytest tests/test_e61_worldcup_backtest.py -k "skill or advancement" -v`
Expected: both PASS. (If `skill_vs_uniform` is ≤ 0 for a single cup it's a real signal — but 2014 should be positive. Investigate before weakening the assertion.)

- [ ] **Step 5: Commit**

```bash
git add examples/worldcup_history.py tests/test_e61_worldcup_backtest.py
git commit -m "feat(e61): group W/D/L skill + knockout advancement metrics

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Tournament calibration + chalk baseline

**Files:**
- Modify: `examples/worldcup_history.py`
- Test: `tests/test_e61_worldcup_backtest.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_e61_worldcup_backtest.py`:

```python
def test_tournament_calibration_fields():
    eng = wh.EloEngine.from_results(wh.RESULTS_CSV)
    f = wh.forecast_cup(2014, eng, sims=800, seed=5)
    cal = wh.tournament_calibration(wh.load_cup(2014), f)
    assert cal["champion"] == "Germany"
    assert 1 <= cal["champion_rank"] <= 32
    assert cal["champion_prob"] >= 0.0
    assert cal["champion_logloss"] >= 0.0


def test_chalk_baseline_picks_higher_elo():
    eng = wh.EloEngine.from_results(wh.RESULTS_CSV)
    elo = eng.ratings_asof(wh._cup_freeze_date(2014))
    base = wh.chalk_baseline(wh.load_cup(2014), elo)
    assert 0.0 <= base["group_hit_rate"] <= 1.0
    assert base["champion"] in [t for g in wh.load_cup(2014).groups.values() for t in g]
```

- [ ] **Step 2: Run them, verify they fail**

Run: `python3 -m pytest tests/test_e61_worldcup_backtest.py -k "calibration or chalk" -v`
Expected: FAIL — missing `tournament_calibration` / `chalk_baseline`.

- [ ] **Step 3: Implement calibration + baseline**

Add to `examples/worldcup_history.py`:

```python
def tournament_calibration(cup: "Cup", forecast: Dict[str, Dict[str, float]]) -> dict:
    """Where did the actual champion / finalists / semifinalists rank in our odds?"""
    adv = cup.actual_advancers()
    champ = adv["champion"]
    ranked = sorted(forecast, key=lambda t: forecast[t]["champion"], reverse=True)
    champ_rank = ranked.index(champ) + 1
    p_champ = forecast[champ]["champion"] / 100.0
    finalists = {adv["final_match"]["home"], adv["final_match"]["away"]}
    semis = set(adv["SF"]) | finalists  # SF winners are finalists; SF list = winners
    # Semifinalists = the 4 teams that reached SF = QF winners.
    semifinalists = set(adv["QF"])
    return {
        "champion": champ,
        "champion_rank": champ_rank,
        "champion_prob": p_champ,
        "champion_logloss": -math.log(max(p_champ, 1e-6)),
        "finalists": sorted(finalists),
        "mean_finalist_reach_final_prob":
            sum(forecast[t]["reach_final"] for t in finalists) / 200.0,
        "semifinalists": sorted(semifinalists),
        "mean_semifinalist_reach_SF_prob":
            sum(forecast[t]["reach_SF"] for t in semifinalists) / (100.0 * len(semifinalists)),
    }


def reach_round_calibration(cups_forecasts, key="reach_QF", actual_round="QF",
                            n_buckets=5) -> List[dict]:
    """Pool teams across cups; bucket by predicted prob; compare to observed freq.

    `cups_forecasts` is a list of (Cup, forecast). Returns per-bucket
    {lo, hi, n, predicted, observed}.
    """
    pts = []  # (predicted_prob_in_[0,1], hit 0/1)
    for cup, fc in cups_forecasts:
        adv = cup.actual_advancers()
        reached = set(adv["R16"]) | set(adv["QF"]) | set(adv["SF"]) | {adv["champion"]}
        # "reached QF" = team is in QF winners OR went further; build per round:
        round_sets = {
            "R16": set(t for g in cup.groups for t in []),  # placeholder; filled below
        }
        reached_qf = set(adv["QF"]) | set(adv["SF"]) | {adv["final_match"]["home"],
                                                        adv["final_match"]["away"]}
        reached_map = {
            "R16": set(adv["R16"]) | reached_qf,
            "QF": reached_qf,
            "SF": set(adv["SF"]) | {adv["final_match"]["home"], adv["final_match"]["away"]},
        }
        target = reached_map[actual_round]
        for t in fc:
            pts.append((fc[t][key] / 100.0, 1.0 if t in target else 0.0))
    out = []
    for b in range(n_buckets):
        lo, hi = b / n_buckets, (b + 1) / n_buckets
        sel = [(p, h) for (p, h) in pts if (lo <= p < hi or (b == n_buckets - 1 and p == 1.0))]
        if not sel:
            out.append({"lo": lo, "hi": hi, "n": 0, "predicted": None, "observed": None})
            continue
        out.append({"lo": lo, "hi": hi, "n": len(sel),
                    "predicted": sum(p for p, _ in sel) / len(sel),
                    "observed": sum(h for _, h in sel) / len(sel)})
    return out


def chalk_baseline(cup: "Cup", elo: Dict[str, float], base: float = 1500.0) -> dict:
    """Deterministic 'higher pre-tournament Elo always wins' bracket."""
    def stronger(a, b):
        ea, eb = _eff(a, elo, cup.host, base), _eff(b, elo, cup.host, base)
        return a if (ea, a) >= (eb, b) else b  # name breaks exact ties
    # Group hit-rate: predict higher-Elo as the W/D/L call (never predicts draw).
    hits = n = 0
    for g, teams in cup.groups.items():
        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                rec = cup.group_result(teams[i], teams[j])
                home, hg, ag = rec
                away = teams[j] if home == teams[i] else teams[i]
                pick = stronger(home, away)
                actual = home if hg > ag else (away if ag > hg else None)
                hits += (pick == actual)
                n += 1
    # Chalk bracket from real group standings (so it sees the same qualifiers).
    standings = cup.real_standings()
    winners = {g: standings[g][0] for g in cup.groups}
    runners = {g: standings[g][1] for g in cup.groups}
    teams = []
    for wl, rl in R16_PAIRS:
        teams.append(winners[wl]); teams.append(runners[rl])
    for _rnd in KO_ROUNDS:
        teams = [stronger(teams[i], teams[i + 1]) for i in range(0, len(teams), 2)]
    return {"group_hit_rate": hits / n, "champion": teams[0]}
```

- [ ] **Step 4: Run the tests, verify they pass**

Run: `python3 -m pytest tests/test_e61_worldcup_backtest.py -k "calibration or chalk" -v`
Expected: both PASS.

- [ ] **Step 5: Commit**

```bash
git add examples/worldcup_history.py tests/test_e61_worldcup_backtest.py
git commit -m "feat(e61): tournament calibration + chalk baseline

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: Experiment driver — assemble, save, self-check

**Files:**
- Create: `experiments/e61_worldcup_backtest.py`

- [ ] **Step 1: Write the driver**

Create `experiments/e61_worldcup_backtest.py`:

```python
"""E61 - Historical World Cup backtest (2010 / 2014 / 2018 / 2022).

Forecasts four past World Cups from PRE-tournament info only, then scores against
reality. Computes leakage-free World Football Elo from the full results history
(validated against published eloratings.net), reuses the 2026 forecaster's
Elo->Poisson goal model in a 32-team format, and reports: match-level skill
(group W/D/L), knockout advancement, tournament calibration, and a chalk baseline.

Deterministic & offline. save_results() is called BEFORE the asserts.

    python experiments/e61_worldcup_backtest.py            # default sims
    python experiments/e61_worldcup_backtest.py --sims 5000
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import worldcup_history as wh  # noqa: E402
from common import save_results  # noqa: E402

CUPS = [2010, 2014, 2018, 2022]
CHAMPIONS = {2010: "Spain", 2014: "Germany", 2018: "France", 2022: "Argentina"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=10000)
    ap.add_argument("--match-sims", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    eng = wh.EloEngine.from_results(wh.RESULTS_CSV)

    validation = {y: wh.validate_against_published(eng, y)
                  for y in (2009, 2013, 2017, 2021)}

    per_cup = {}
    cups_forecasts = []
    pooled_group_brier = pooled_group_n = 0.0
    pooled_group_hits = 0.0
    pooled_ko_correct = pooled_ko_n = 0.0
    for y in CUPS:
        cup = wh.load_cup(y)
        elo = eng.ratings_asof(wh._cup_freeze_date(y))
        forecast = wh.forecast_cup(y, eng, sims=args.sims, seed=args.seed)
        cups_forecasts.append((cup, forecast))
        _rows, gsum = wh.score_group_matches(cup, elo, sims=args.match_sims, seed=args.seed)
        ko = wh.score_knockout_advancement(cup, elo, sims=args.match_sims, seed=args.seed)
        cal = wh.tournament_calibration(cup, forecast)
        chalk = wh.chalk_baseline(cup, elo)
        per_cup[str(y)] = {"group": gsum, "knockout": ko, "calibration": cal,
                           "chalk": chalk, "host": cup.host}
        pooled_group_brier += gsum["mean_brier"] * gsum["n"]
        pooled_group_hits += gsum["hit_rate"] * gsum["n"]
        pooled_group_n += gsum["n"]
        pooled_ko_correct += ko["accuracy"] * ko["n"]
        pooled_ko_n += ko["n"]

    base_brier = 2 / 3
    pooled = {
        "group_n": int(pooled_group_n),
        "group_hit_rate": pooled_group_hits / pooled_group_n,
        "group_mean_brier": pooled_group_brier / pooled_group_n,
        "group_skill_vs_uniform": 1 - (pooled_group_brier / pooled_group_n) / base_brier,
        "knockout_n": int(pooled_ko_n),
        "knockout_accuracy": pooled_ko_correct / pooled_ko_n,
        "mean_champion_logloss":
            sum(per_cup[str(y)]["calibration"]["champion_logloss"] for y in CUPS) / len(CUPS),
        "mean_chalk_group_hit_rate":
            sum(per_cup[str(y)]["chalk"]["group_hit_rate"] for y in CUPS) / len(CUPS),
    }
    calib_qf = wh.reach_round_calibration(cups_forecasts, key="reach_QF", actual_round="QF")

    payload = {"model": "elo_poisson_32team", "sims": args.sims,
               "match_sims": args.match_sims, "seed": args.seed,
               "cups": CUPS, "validation": validation,
               "per_cup": per_cup, "pooled": pooled,
               "reach_qf_calibration": calib_qf}
    save_results("e61_worldcup_backtest", payload)   # BEFORE asserts

    # --- self-checks (after save) ---
    for y in (2013, 2017):  # representative snapshots
        assert validation[y]["spearman"] >= 0.7, (y, validation[y])
    assert pooled["group_skill_vs_uniform"] > 0.0, pooled
    assert pooled["group_hit_rate"] > 1 / 3, pooled  # beats a 3-way coin
    # The model should, on average, beat chalk's group calls OR match it closely.
    assert pooled["knockout_accuracy"] >= 0.5, pooled
    for y in CUPS:  # actual champion should land in the upper half of our odds
        assert per_cup[str(y)]["calibration"]["champion_rank"] <= 16, (y, per_cup[str(y)])

    print(f"[E61] pooled group hit-rate {pooled['group_hit_rate']*100:.1f}% "
          f"(skill {pooled['group_skill_vs_uniform']*100:+.0f}%), "
          f"KO accuracy {pooled['knockout_accuracy']*100:.0f}%, "
          f"mean champ log-loss {pooled['mean_champion_logloss']:.2f}")
    for y in CUPS:
        c = per_cup[str(y)]["calibration"]
        print(f"  {y}: champ {c['champion']} ranked #{c['champion_rank']} "
              f"(p={c['champion_prob']*100:.1f}%)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the experiment with small sims to verify it completes + asserts pass**

Run: `cd experiments && python3 e61_worldcup_backtest.py --sims 1000 --match-sims 3000`
Expected: prints `[saved] .../e61_worldcup_backtest.json`, then the pooled summary
and four per-cup champion ranks, with no AssertionError. (If a champion-rank
assertion fires, that's a real finding for a chaotic cup — relax `<= 16` to a
documented value only after confirming the forecast is otherwise well-calibrated;
prefer reporting it honestly in the writeup.)

- [ ] **Step 3: Run the full experiment to write real numbers**

Run: `cd experiments && python3 e61_worldcup_backtest.py`
Expected: same, with default sims (10000 / 20000). Note runtime; if too slow,
20000 match-sims × 256 matches is the cost — acceptable on a laptop (~1-2 min).

- [ ] **Step 4: Commit (script + results JSON)**

```bash
git add experiments/e61_worldcup_backtest.py experiments/results/e61_worldcup_backtest.json
git commit -m "feat(e61): experiment driver + results JSON

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 10: Bracket vs reality — per-cup SVG

**Files:**
- Modify: `examples/worldcup_history.py`
- Test: `tests/test_e61_worldcup_backtest.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_e61_worldcup_backtest.py`:

```python
def test_bracket_svg_is_self_contained():
    eng = wh.EloEngine.from_results(wh.RESULTS_CSV)
    f = wh.forecast_cup(2014, eng, sims=400, seed=2)
    svg = wh.render_cup_svg(wh.load_cup(2014), f)
    assert svg.startswith("<svg") and svg.rstrip().endswith("</svg>")
    assert "http://www.w3.org/2000/svg" in svg
    assert "Germany" in svg          # actual champion appears
    assert "<image" not in svg and "xlink:href=\"http" not in svg  # no fetched resources
```

- [ ] **Step 2: Run it, verify it fails**

Run: `python3 -m pytest tests/test_e61_worldcup_backtest.py::test_bracket_svg_is_self_contained -v`
Expected: FAIL — missing `render_cup_svg`.

- [ ] **Step 3: Implement a compact 32-team SVG (atlas palette)**

Add to `examples/worldcup_history.py`. This renders two columns of text — our
modal forecast (most-probable champion + each team's title odds for the top 8)
beside the actual final four — in the forecaster's palette. Keep it self-contained.

```python
_BG, _ACCENT, _OCHRE, _TEAL, _INK = "#fcfbf8", "#1d4ed8", "#b45309", "#0f766e", "#1f2937"


def _esc(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def render_cup_svg(cup: "Cup", forecast: Dict[str, Dict[str, float]]) -> str:
    """Self-contained SVG: our title-odds top-8 beside the cup's actual final four."""
    adv = cup.actual_advancers()
    ranked = sorted(forecast, key=lambda t: forecast[t]["champion"], reverse=True)[:8]
    champ = adv["champion"]
    finalists = [adv["final_match"]["home"], adv["final_match"]["away"]]
    runner_up = [t for t in finalists if t != champ][0]
    W, H = 720, 460
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
           f'viewBox="0 0 {W} {H}" font-family="ui-sans-serif,system-ui,sans-serif">']
    out.append(f'<rect width="{W}" height="{H}" fill="{_BG}"/>')
    out.append(f'<text x="24" y="40" font-size="22" font-weight="700" fill="{_INK}">'
               f'World Cup {cup.year} — forecast vs reality</text>')
    out.append(f'<text x="24" y="64" font-size="13" fill="{_TEAL}">'
               f'host {_esc(cup.host)} · model: Elo→Poisson, pre-tournament ratings</text>')
    # Left: our title odds
    out.append(f'<text x="24" y="104" font-size="15" font-weight="700" '
               f'fill="{_ACCENT}">Our title odds (top 8)</text>')
    y = 132
    for t in ranked:
        p = forecast[t]["champion"]
        mark = "  ← actual winner" if t == champ else ""
        col = _OCHRE if t == champ else _INK
        out.append(f'<text x="24" y="{y}" font-size="14" fill="{col}">'
                   f'{_esc(t)} — {p:.1f}%{mark}</text>')
        y += 26
    # Right: actual final four
    out.append(f'<text x="400" y="104" font-size="15" font-weight="700" '
               f'fill="{_ACCENT}">What actually happened</text>')
    rows = [("Champion", champ), ("Runner-up", runner_up)]
    rows += [("Semi-finalist", t) for t in adv["QF"] if t not in finalists]
    y = 132
    for label, t in rows:
        odds = forecast.get(t, {}).get("champion", 0.0)
        out.append(f'<text x="400" y="{y}" font-size="14" fill="{_INK}">'
                   f'{label}: {_esc(t)} (we gave {odds:.1f}% to win)</text>')
        y += 26
    out.append('</svg>')
    return "\n".join(out)
```

- [ ] **Step 4: Run the test, verify it passes**

Run: `python3 -m pytest tests/test_e61_worldcup_backtest.py::test_bracket_svg_is_self_contained -v`
Expected: PASS.

- [ ] **Step 5: Wire SVG output into the driver and regenerate**

In `experiments/e61_worldcup_backtest.py`, after `save_results(...)`, add:

```python
    from pathlib import Path
    figdir = Path(__file__).resolve().parents[1] / "paper" / "figs"
    figdir.mkdir(parents=True, exist_ok=True)
    for (cup, forecast) in cups_forecasts:
        svg = wh.render_cup_svg(cup, forecast)
        (figdir / f"e61_bracket_{cup.year}.svg").write_text(svg, encoding="utf-8")
```

Run: `cd experiments && python3 e61_worldcup_backtest.py --sims 1000 --match-sims 3000`
Expected: four `paper/figs/e61_bracket_20XX.svg` files written, no errors.

- [ ] **Step 6: Commit**

```bash
git add examples/worldcup_history.py tests/test_e61_worldcup_backtest.py experiments/e61_worldcup_backtest.py paper/figs/e61_bracket_*.svg
git commit -m "feat(e61): per-cup forecast-vs-reality SVG

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 11: Paper integration

**Files:**
- Modify: `scripts/make_paper_assets.py`

- [ ] **Step 1: Add the experiment to `EXPERIMENTS`**

In `scripts/make_paper_assets.py`, append `"e61_worldcup_backtest"` to the
`EXPERIMENTS` list (after `"e60_io_boundary",` at line ~40).

- [ ] **Step 2: Add a figure + table function**

After `fig_io_boundary`/`table_io_boundary` (search for `def table_io_boundary`),
add:

```python
def fig_worldcup_backtest(e61):
    cups = e61["cups"]
    hit = [e61["per_cup"][str(y)]["group"]["hit_rate"] * 100 for y in cups]
    ko = [e61["per_cup"][str(y)]["knockout"]["accuracy"] * 100 for y in cups]
    chalk = [e61["per_cup"][str(y)]["chalk"]["group_hit_rate"] * 100 for y in cups]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    x = range(len(cups))
    ax.plot(x, hit, "-o", color=BLUE, label="model group hit-rate")
    ax.plot(x, ko, "-s", color=TEAL, label="model KO advancement")
    ax.plot(x, chalk, "--", color=SLATE, label="chalk group hit-rate")
    ax.axhline(100 / 3, color="#999", lw=0.8, ls=":")
    ax.set_xticks(list(x)); ax.set_xticklabels([str(c) for c in cups])
    ax.set_ylabel("accuracy (%)"); ax.set_ylim(0, 100)
    ax.legend(fontsize=8); ax.set_title("E61: forecast accuracy by cup")
    fig.tight_layout(); fig.savefig(FIGS / "e61_worldcup_backtest.pdf"); plt.close(fig)


def table_worldcup_backtest(e61):
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Cup & Group hit & KO adv. & Champ rank & Champ \% \\", r"\midrule"]
    for y in e61["cups"]:
        pc = e61["per_cup"][str(y)]
        cal = pc["calibration"]
        lines.append(
            f"{y} & {pc['group']['hit_rate']*100:.0f}\\% & "
            f"{pc['knockout']['accuracy']*100:.0f}\\% & "
            f"\\#{cal['champion_rank']} & {cal['champion_prob']*100:.1f}\\% \\\\")
    p = e61["pooled"]
    lines += [r"\midrule",
              f"Pooled & {p['group_hit_rate']*100:.0f}\\% & "
              f"{p['knockout_accuracy']*100:.0f}\\% & -- & -- \\\\",
              r"\bottomrule", r"\end{tabular}"]
    (TABLES / "worldcup_backtest.tex").write_text("\n".join(lines))
```

- [ ] **Step 3: Register calls in `main()` and add macros**

In `main()` (search `def main`), after the `table_io_boundary(...)` call add:

```python
    fig_worldcup_backtest(data["e61_worldcup_backtest"])
    table_worldcup_backtest(data["e61_worldcup_backtest"])
```

In the macros `lines = [...]` block (search `macro("NumExperiments"`), add before
the closing `]` (use letters-only macro names):

```python
        macro("WCGroupHitRate", pct(data["e61_worldcup_backtest"]["pooled"]["group_hit_rate"])),
        macro("WCKnockoutAcc", pct(data["e61_worldcup_backtest"]["pooled"]["knockout_accuracy"])),
        macro("WCGroupSkill", pct(data["e61_worldcup_backtest"]["pooled"]["group_skill_vs_uniform"])),
        macro("WCChampLogLoss", f"{data['e61_worldcup_backtest']['pooled']['mean_champion_logloss']:.2f}"),
        macro("WCChalkHitRate", pct(data["e61_worldcup_backtest"]["pooled"]["mean_chalk_group_hit_rate"])),
        macro("WCEloSpearman", f"{data['e61_worldcup_backtest']['validation']['2013']['spearman']:.2f}"),
```

(Note: JSON keys are strings, so `validation['2013']` — confirm the driver writes
year keys as strings; `validate_against_published` is stored under int keys in the
dict but `save_results` JSON-encodes them as strings. Access with `["2013"]`.)

And bump `\NumExperiments`: change `macro("NumExperiments", "59")` to the next
integer (`"60"`), reflecting the added experiment count.

- [ ] **Step 4: Regenerate assets and verify no crash**

Run: `python3 scripts/make_paper_assets.py`
Expected: completes without KeyError/exception; prints its normal output and
writes `paper/figs/e61_worldcup_backtest.pdf`, `paper/tables/worldcup_backtest.tex`,
and an updated `paper/numbers.tex`.

- [ ] **Step 5: Compile the paper to check for undefined refs**

Run: `cd paper && tectonic main.tex 2>&1 | tail -20`
Expected: compiles (a PDF is produced). If `main.tex` doesn't `\input` the new
table/fig, that's fine — the assets exist for when prose references them; do NOT
edit `main.tex` prose in this task unless an undefined-control-sequence error from
an *existing* reference appears. Verify no "Missing \begin{document}" (would mean a
digit slipped into a macro name).

- [ ] **Step 6: Commit**

```bash
git add scripts/make_paper_assets.py paper/numbers.tex paper/figs/e61_worldcup_backtest.pdf paper/tables/worldcup_backtest.tex
git commit -m "paper: integrate E61 World Cup backtest (fig, table, macros)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 12: Final verification

**Files:** none (verification only)

- [ ] **Step 1: Run the full E61 test file**

Run: `python3 -m pytest tests/test_e61_worldcup_backtest.py -v`
Expected: all tests PASS.

- [ ] **Step 2: Run the forecaster tests (confirm the refactor didn't regress)**

Run: `python3 -m pytest tests/test_worldcup2026.py -v`
Expected: all PASS.

- [ ] **Step 3: Run the experiment end-to-end at full sims**

Run: `cd experiments && python3 e61_worldcup_backtest.py`
Expected: `[saved] ...e61_worldcup_backtest.json`, pooled summary, four per-cup
lines, no AssertionError.

- [ ] **Step 4: Regenerate paper assets once more from the full-sims JSON**

Run: `python3 scripts/make_paper_assets.py && cd paper && tectonic main.tex 2>&1 | tail -5`
Expected: clean regeneration + compile.

- [ ] **Step 5: Review the diff, then commit any regenerated assets**

Run: `git status && git add -A experiments/results paper/figs paper/tables paper/numbers.tex && git commit -m "chore(e61): regenerate assets at full sims

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"` (skip if nothing changed)

- [ ] **Step 6: Honest writeup pass**

Open `examples/worldcup_history.py` and `experiments/e61_worldcup_backtest.py`
docstrings; confirm they state where the model shines (favourites, calibration)
and where it misses (Brazil 1–7 2014, Italy group exits, 2022 Argentina from a
non-top Elo). If any pooled metric came out weak/flat, say so in the docstring —
do not tune to a target.

---

## Self-Review (completed during planning)

- **Spec coverage:** Elo engine (T3) ✓; validation (T4) ✓; unchanged model + 32-team
  format (T2, T6) ✓; group reconstruction→verified draws (T5) ✓; match-level group
  W/D/L + knockout advancement (T7) ✓; tournament calibration + reach-round (T8) ✓;
  chalk baseline (T8) ✓; bracket-vs-reality SVG (T10) ✓; vendored data (T1) ✓;
  experiment + save-before-assert (T9) ✓; tests (T3–T10) ✓; paper integration (T11) ✓.
- **Placeholder scan:** `reach_round_calibration` had a leftover placeholder line
  (`round_sets`/`reached`) — left in the code block intentionally minimal; the
  implementer should delete the unused `round_sets`/`reached` locals (only
  `reached_map`/`target` are used). Flagged here so it isn't shipped.
- **Type consistency:** `forecast` dicts use keys `champion/reach_final/reach_SF/
  reach_QF/reach_R16` everywhere; `_cup_freeze_date` used in T6/T7/T8/T9; `Cup`
  API (`group_result`, `real_standings`, `actual_advancers`, `actual_champion`,
  `groups`, `host`, `_ko`) consistent across tasks.
- **Known cleanup for the implementer:** (a) drop dead `opener`/`first_group` in
  `forecast_cup` (T6 step 5); (b) drop unused `round_sets`/`reached` in
  `reach_round_calibration` (T8); (c) verify `NAME_TO_PUBLISHED` covers any
  validation outliers (T4 step 4).
