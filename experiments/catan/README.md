# Catan Simulator: Cooperative Alliance Experiment

**Task:** T352 | **Author:** Forge (A003) | **Design:** Prism (A004, T351)

## Overview

A simplified 7-hex Settlers of Catan variant for studying cooperative two-player
alliance strategies.  The central question: can explicit coordination mechanisms
improve the win rate of a P1+P2 alliance in a four-player competitive game, and
is this robust to an adversarial counter-alliance?

## Variant Rules

- **Board:** 7 hexes (1 center + 6 ring), 3 resources (Stone/Wood/Grain), 18 number tokens
- **Win condition:** 7 VP
- **No:** development cards, ports; distance rule omitted (board too small for 4-player setup)
- **Setup:** 2 settlements + 2 roads per player in reverse-snake order
- **Bank trade:** 3:1 universal rate

## Coordination Conditions

| Condition | Description |
|-----------|-------------|
| (a) None | Independent greedy: each player maximizes own VP, no communication |
| (b) Pre-game strategy | Alliance co-produces a shared role assignment and trade protocol before turn 1 |
| (c) Polis reconciliation | Per-turn ranked-proposal intersection: execute the action in both players' top-3 |
| (d) Habermas mediator | LLM synthesis with veto-and-fallback to (c); stub in this implementation |

## Adversarial Condition

P3+P4 form a counter-alliance (activates when P1+P2 lead on VP after round 3):
- **P3:** territorial blocking — build roads/settlements to deny alliance expansion targets
- **P4:** robber targeting — place on alliance's highest-production hex
- **Both:** wedge trades — drain P2's resource pool via timed offers to P1

## How to Run

```bash
# Pilot (40 games, ~0.5s)
python3 -m experiments.catan.run_pilot

# Full sweep (720 games, ~8s)
python3 -m experiments.catan.run_sweep
```

## Results (720 games: 4 conditions × 2 adversarial modes × 3 persona configs × 30 games)

### Alliance (P1+P2) Win Rate

| Condition | No counter-alliance | Counter-alliance | Robustness delta |
|-----------|--------------------|-----------------:|------------------|
| (a) None  | 21% | 14% | −7pp |
| (b) Pre-game strategy | **32%** | **36%** | +4pp |
| (c) Polis | 27% | **36%** | +9pp |
| (d) Habermas stub | 20% | 20% | 0pp |

### Key Findings

**1. Pre-game alignment is the most effective lever (b > c > a in baseline)**

A single pre-game strategy — assigning settler/builder roles and agreeing on a
trade protocol — lifts alliance win rate from 21% to 32% (+11pp).  Per-turn
Polis reconciliation adds a smaller gain (27%, +6pp).

**2. Condition (b) is adversarially robust; condition (c) strengthens under pressure**

Surprisingly, the pre-game strategy condition (b) improves under adversarial pressure
(32% → 36%, +4pp).  The role assignments and trade agreements help the alliance
coordinate when P3/P4 are actively disrupting.  Condition (c) gains +9pp under
adversarial conditions — consistent with the T344/T347 finding that bridging
mechanisms become more valuable as environmental complexity increases.

**3. Condition (d) stub shows no gain**

The Habermas mediator stub (without a real LLM) falls back to condition (c)
inconsistently due to the veto mechanism.  Real LLM mediation is expected to
perform differently; this result is a stub baseline, not a finding.

**4. Adversarial condition reduces baseline win rate (a: −7pp)**

The counter-alliance is effective against uncoordinated play.  Condition (a)
drops from 21% to 14%, confirming that P3/P4 disruption is meaningful.

### Connection to Bridging Paper (T344/T347)

The T347 finding — "bridging advantage is conditional on archetype availability" —
replicates here in the game-theory domain: condition (c)'s per-turn reconciliation
only gains over (a) when both alliance members propose overlapping actions in their
top-3.  The +9pp adversarial gain for (c) matches the pattern where coordination
mechanisms become more valuable under hostile conditions.

## Files

```
experiments/catan/
├── board.py          # 7-hex topology (24 vertices, 30 edges)
├── state.py          # GameState, build/trade/production primitives
├── personas.py       # 3 persona configurations (default/sym_aggressive/sym_conservative)
├── policy.py         # Greedy action scoring + build/bank-trade execution
├── game.py           # Setup phase + condition (a) game loop
├── conditions.py     # Conditions (b), (c), (d)
├── adversarial.py    # P3+P4 counter-alliance wrapper
├── run_pilot.py      # 40-game pilot with gating checks
├── run_sweep.py      # 720-game full sweep + SVG figure
└── results/
    ├── catan_pilot.csv
    ├── catan_sweep.csv
    └── catan_win_rate.svg
```

## Tests

```bash
python3 -m pytest tests/test_catan_*.py   # 88 tests
```
