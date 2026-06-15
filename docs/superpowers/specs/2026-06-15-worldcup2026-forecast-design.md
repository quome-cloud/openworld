# World Cup 2026 forecast — design

**Goal.** A deterministic, reproducible Monte-Carlo forecaster for the FIFA World
Cup 2026 (Canada/Mexico/USA), built as an OpenWorld world model. Output: each
team's probability of winning the title and of reaching each round, plus a
most-likely bracket. "Start fresh" — no real results are applied; every match is
predicted.

## Decisions (locked with the user)

- **Goal:** forecast / probabilities.
- **Outcome source:** Elo ratings from eloratings.net → a coded goal model.
  Predict all matches (real results may be injected later via `play_match`).
- **Scope:** full tournament — 48 teams, 12 groups, top-2 + 8 best thirds → a
  Round of 32, then R16 → QF → SF → Final.

## Data (captured in the module, editable)

- **`ELO`** — all 48 teams → numeric Elo (eloratings.net 2026 World Cup data,
  June 2026). E.g. Spain 2129, Argentina 2115, France 2063, England 2024.
- **`GROUPS`** — A–L, each 4 teams, from the Dec 5 2025 final draw. The five
  former "playoff winner" slots are resolved: A+Czechia, B+Bosnia, D+Turkey,
  F+Sweden, I+Iraq; K's third slot = DR Congo.
- **`R32`** — the 16 official Round-of-32 pairings (ESPN), in slot notation:
  winners of A,B,D,E,G,I,K,L face a best-third; C/F/H/J winners face runners-up;
  the rest are runner-up vs runner-up.

## Architecture

One source of truth for the rules, used by both the Monte-Carlo driver and the
served OpenWorld `World`:

1. **Outcome model** (`sample_match`, seeded `random.Random`): Elo → expected
   score `e = 1/(1+10^(-Δ/400))` → goal supremacy → two Poisson goal draws
   (mean total ≈ 2.7, supremacy scale ≈ 1.9 — both tunable). Gives realistic
   draws in groups; knockouts resolve ties by an Elo-leaning penalty coin-flip.
   A modest **host advantage** (USA/Mexico/Canada, default +60 Elo) is applied
   and exposed as a tunable dial.
2. **Rules engine** (plain functions): `group_standings` (3/1/0 pts; tiebreak
   points → GD → GF → goals; deterministic), `rank_thirds` (rank all 12 thirds,
   take best 8), `build_r32` (assign the 8 qualifying thirds to the 8 third-slots
   respecting each slot's allowed-group set), `play_knockout` (single-elim tree).
3. **`simulate_tournament(rng)`** — runs one full tournament, returns champion +
   per-team furthest round reached.
4. **`forecast(n, seed)`** — runs N independent seeded tournaments, aggregates to
   champion% / reach-final% / reach-SF% / reach-QF% / reach-R16% / win-group%.
5. **OpenWorld `World`** (`build_world()`) — `FunctionTransition` over symbolic
   state (`groups`, `elo`, `results`, `phase`, `standings`, `bracket`,
   `champion`); actions `play_match` (apply a real or sampled result) and
   `simulate_rest` (play out remaining matches from a seed). Serialises to a spec
   and serves at `/view` for interactive stepping. One rules implementation backs
   both paths — no duplication.

## Honest approximations (documented in code)

- **Third-place → slot assignment** uses a deterministic greedy fill over the
  allowed-group sets rather than the literal FIFA 495-row combination table.
  Second-order effect on champion probabilities; the slot table is editable.
- **R16+ bracket adjacency** pairs the listed R32 matches as a fixed binary tree;
  the exact official adjacency is a single editable list.
- Both are flagged in the source so the boundary is visible, per project norms.

## Testing

- **Determinism:** same seed → identical champion and bracket.
- **Invariants:** exactly 24 group qualifiers + 8 thirds = 32 into R32; round
  sizes 32→16→8→4→2→1; champion is always exactly one real team.
- **Probability sanity:** all champion probabilities sum to ~100%; every team in
  [0,100]; higher-Elo team has ≥ win prob in a head-to-head (model monotonicity).
- **Rules:** win=3pts, standings sort order, third-place ranking, host advantage
  raises a host's advance probability vs the no-advantage baseline.
- **E2E:** `forecast(2000)` returns a complete, normalized table over all 48
  teams; top of the table is a strong side (sanity, not asserted to a fixed team).

## Non-goals (YAGNI)

- No live data scraping at runtime (Elo/groups are captured constants).
- No LLM in the loop — dynamics are hand-written verified code, fully offline.
- No betting-market calibration; this is a strength-based structural forecast.
