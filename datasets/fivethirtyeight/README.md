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
