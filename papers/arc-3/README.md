# OpenWorld on ARC-AGI-3

A standalone paper: port OpenWorld's **verified code world model** recipe to the **ARC-AGI-3**
interactive-reasoning benchmark, where frontier agents score near zero. The wager: each game is a
*world* whose deterministic dynamics OpenWorld can **synthesize as code, verify exactly, and plan
through** — exact where learned next-frame models compound error.

- `main.tex` — the manuscript (symlinks share `../assets/` like the other papers).
- `RECIPE.md` — the experimental recipe (explore → synthesize-and-verify → plan) + the experiment
  plan (E86 fidelity, E87 planning/level-completion, E88 cross-game / cross-generation transfer).
- Experiments live in `experiments/e86_arc3.py` (+ E87/E88 as built); the ARC-AGI-3 games are the
  official public set via the `arc-agi` toolkit (local play, gym-like).

Status: integration + determinism precondition + baselines established (PoC); E86 synthesis
harness in place; quantitative runs in progress.
