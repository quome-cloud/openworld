# World 1: US Two-Party Political Simulator — Bridging Baselines

Experiment testing bridging-ranked algorithms against majority vote in a stylised US political
policy simulator. Implemented on top of `openworld` as part of T344 (D.O.H. researchy team).

## Setup

**Policy space**: 8 issues × {−2,−1,0,+1,+2} = 5^8 = 390,625 candidate bundles.

Issues: `healthcare`, `climate`, `fiscal`, `civil_rights`, `foreign_policy`,
`criminal_justice`, `education`, `immigration`.

**Personas**: 300 agents with bimodal ideology (40% left + 40% right + 20% centrist),
Dirichlet(α=0.5) issue weights, SBM network structure (6 communities), quadratic welfare loss.

**Candidate slate**: K=7 bundles per trial — 5 archetypes (all-issues at −2,−1,0,+1,+2)
plus 2 random bundles drawn per trial seed.

## Conditions

| Condition | Algorithm | Description |
|-----------|-----------|-------------|
| Z | Random | Uniform draw from slate (floor) |
| A | Majority vote | Each persona votes for their welfare-maximising bundle; plurality wins |
| C_CN | Community Notes | L2-regularised matrix factorisation r̂=μ+i_u+i_n+f_u·f_n; rank by i_n |
| C_PP | Polarity-product | Geometric mean of per-community endorsement rates across SBM communities |
| D | Oracle | Precomputed argmax of global welfare G(b) = Σ_p welfare(p, b) + spillovers |

**C_CN implementation**: production Community Notes spec (Wojcik et al. 2022, arXiv:2210.15723),
λ_i=0.15, λ_f=0.03, d=1, 200 gradient-descent iterations.

## Spillover Configurations

Spillovers add a fixed global bonus when a bundle matches a specific sub-pattern.

**centrist**: positive-sum bundles at or near the policy centre
```
healthcare=0, fiscal=0          → +0.05
climate=−1, foreign_policy=0   → +0.03
criminal_justice=−1, education=0 → +0.04
```

**off_axis**: positive-sum bundles at *non-centrist* intersections (stress test for centrism-laundering)
```
climate=−2, fiscal=+1           → +0.05
civil_rights=+1, criminal_justice=−1 → +0.03
education=+2, healthcare=−1     → +0.04
```

## Results

**N=20 personas, 50 trials per (condition, spillover_config) cell.**

### Gap fraction by condition

`gap_fraction = (G_achieved − G_random) / (G_oracle − G_random)`

where G_random is the mean welfare over all 390,625 bundles (population average,
not a single Z draw).

| Condition | centrist (med) | off_axis (med) |
|-----------|---------------|----------------|
| Z | 0.199 | 0.244 |
| A (majority) | 0.306 | 0.495 |
| C_CN | **0.793** | **0.904** |
| C_PP | **0.793** | **0.904** |
| D (oracle) | 1.000 | 1.000 |

See [`results/gap_fraction_boxplot.svg`](results/gap_fraction_boxplot.svg).

### Key findings

**1. Bridging captures 2–3× more of the achievable gap than majority vote.**
At centrist spillovers: C_CN/C_PP gap=0.793 vs A gap=0.306 (+2.6×).
At off_axis spillovers: C_CN/C_PP gap=0.904 vs A gap=0.495 (+1.8×).

**2. Centrism-laundering null rejected.**
Prism's null hypothesis: "bridging advantage is an artifact of centrist spillover placement
— bridging algorithms just launder centrism, not cross-cluster signal."
Result: bridging advantage *strengthens* on off_axis spillovers (0.793→0.904 for C).
Bridging finds real cross-cluster agreement, not a centrist bias.

**3. C_CN ≈ C_PP at K=7 slate size.**
Full Community Notes matrix factorisation (gradient descent, L2 regularisation, latent factors)
is empirically indistinguishable from a 50-line polarity-product bridge score.
*Suggests: latent factorisation complexity is overkill for small-slate aggregation decisions.*
The geometric-mean multi-cluster insight is doing the work; the full CN model adds nothing at K=7.

**4. Majority vote is deterministic given fixed personas.**
With the 5 archetype bundles always present, A always selects the same archetype regardless
of which 2 random bundles are added per trial (IQR=0 across 50 trials). Bridging algorithms
are similarly stable — they lock onto the spillover-positive bundle via cross-cluster signal.

**5. Bridging advantage is conditional on archetype availability (slate-composition sensitivity).**
Sensitivity check (`run_cycle5b.py`): 50 trials with K=7 fully-random slates (no archetypes).

| Condition | centrist (med) | off_axis (med) |
|-----------|---------------|----------------|
| A (majority) | 0.222 | 0.330 |
| C_CN | 0.228 | 0.345 |
| C_PP | 0.226 | 0.325 |

With purely random slates, C_CN ≈ A (delta ≤ 0.016). The large bridging advantage from
finding 4 requires that the centrist archetype bundle — which the spillover configuration
rewards — is explicitly present in the slate. Without it, majority vote and bridging
algorithms are equivalent: neither can reliably find a spillover-positive bundle it hasn't
been shown.

*Implication: bridging algorithms require structured candidate slates with broad coverage
to outperform majority vote. Slate design is a load-bearing design choice, not a neutral
setup parameter.*

## Reproducing the results

```bash
# From the openworld repo root:
python -m experiments.bridging.run_cycle5   # archetype slates (main results)
python -m experiments.bridging.run_cycle5b  # fully-random slates (sensitivity check)

# Oracle PayoffTables are cached in experiments/bridging/.cache/ (gitignored).
# First run for a new (n_personas, spillover_config) pair takes 2–5 min.
# Subsequent runs are instant (pickle cache).
```

## File structure

```
experiments/bridging/
├── personas.py          # Persona generator: bimodal ideology, SBM network, welfare
├── policy.py            # PolicyBundle, PayoffTable, oracle precomputation
├── simulation.py        # Conditions Z/A/D, trial runner, CSV output
├── conditions_c.py      # Conditions C_CN (Community Notes) + C_PP (polarity-product)
├── run_cycle5.py        # Experiment runner (Cycle 5: 50 trials × 5 cond × 2 config, archetype slates)
├── run_cycle5b.py       # Sensitivity check (Cycle 5b: fully-random slates, no archetypes)
├── results/
│   ├── cycle5_results.csv      # 500 trial rows (archetype slates)
│   ├── gap_fraction_boxplot.svg
│   ├── cycle5b_results.csv     # 500 trial rows (fully-random slates)
│   └── cycle5b_summary.txt     # C vs A comparison, slate-sensitivity verdict
└── .cache/              # Oracle PayoffTable pickle cache (gitignored)
```
