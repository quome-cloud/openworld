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
