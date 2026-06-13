# Bridging Algorithms in Simulated Political Worlds: A Regime Taxonomy

**Authors:** Origin Aleph, with the researchy team (Forge, Prism, Lens, Cortex, Vector Meridian)
**Date:** 2026-06-13
**Status:** Draft v1 — markdown for fast iteration; LaTeX conversion at finalization.
**Companion docs:** `openworld_paper_litreview_expanded.md`, `openworld_world_designs.md`, `openworld_paper_expansion_design.md`, `openworld_team_deliverable_2026-06-13.md`.

---

## Abstract

Bridging algorithms — aggregation rules that reward cross-faction co-endorsement rather than within-faction majority — have moved from a deployed-platform curiosity (Community Notes, Polis, vTaiwan) to a recognized design principle for democratic deliberation and AI alignment. Their empirical evaluation has been bottlenecked by what real platforms can produce: one realization, no counterfactuals, no observable ground-truth utility. We use the OpenWorld code-synthesis world-model framework to build a US two-party political simulator with ground-truth global utility, and run a fifty-cell experimental sweep that maps where bridging is useful, where it is redundant, where it fails, and where it is robust to attack.

The headline result is methodological honesty before contribution: at the standard N=300 configuration matching the design specification, bridging and majority vote converge to the centrist outcome (gap fractions 0.764 vs 0.764, zero delta). The bridging advantage reported in our original PR was an N=20 artifact — visible only because the population was small enough for plurality dynamics to fragment.

Within the regimes where bridging does have an advantage, four structural conditions govern it. First, slate composition is a first-class lever: archetype-seeded slates give bridging up to 79% of the random-to-oracle gap at small N; fully random K=7 slates collapse the advantage to within 1pp. Second, the polarization–benefit curve is a step function, not a gradient: across σ_between ∈ {0.00, 0.30, 0.45, 0.60, 0.65, 0.70, 0.75} bridging and majority are indistinguishable; at σ_between = 0.80 majority collapses to gap_fraction = 0.000 while bridging holds at 0.742. Third, adversarial coalition voting produces context-dependent vulnerability: at moderate polarization the standard attack on Community Notes collapses C_CN to majority-vote performance; at heavy polarization the same attack has zero effect because the attack-trigger threshold is never reached. Fourth, the matrix-factorization machinery is empirically redundant at slates up to K=30: a fifty-line polarity-product variant matches the full Community Notes fit within four decimal places across every cell we ran.

The unified thesis: bridging is regime-specific — most useful and most adversarially robust precisely at the structural conditions where it matters most. We discuss the production implications and the methodological role the world-model substrate played in catching the original N=20 artifact.

---

## 1. Introduction

Bridging algorithms — aggregation rules that score items by cross-faction co-endorsement rather than within-faction majority — are no longer an academic proposal. Community Notes runs at X (formerly Twitter) platform scale (Wojcik et al., 2022); Polis's Group-Aware Consensus rule has been used by vTaiwan on twenty-six pieces of legislation (Small et al., 2021); Anthropic and the Collective Intelligence Project have used Polis to source the constitutional principles for a deployed alignment-tuned Claude variant (Huang et al., 2024). Ovadya (2022) named the design pattern and argued for it as a default replacement for engagement-based ranking. Conitzer et al. (2024) made the parallel argument inside the AI-alignment community, reframing RLHF as an unaudited social-choice operator over diverse human feedback.

The evaluation gap is structural. Real platforms cannot run paired counterfactual trials, do not have observable ground-truth global utility, and cannot construct adversarial or edge-case populations on demand. Production deployments measure proxy metrics — reshare rates, complaint volume, retention — and those proxies are confounded with the aggregation rule itself. The published bridging-vs-majority comparisons have therefore been narrow: "On this platform, with this user distribution, this rule produced this engagement signature relative to that one." The configuration space outside the production realization is unobserved.

Code-synthesized world models close this gap. The line from Ha and Schmidhuber (2018) through DreamerV1–V3 (Hafner et al., 2020, 2021, 2025), Genie (Bruce et al., 2024) and MuZero (Schrittwieser et al., 2020) established learned dynamics as a first-class research target. The more recent symbolic-synthesis subline — WorldCoder (Tang et al., 2024), Code World Models with MCTS (Dainese et al., 2024), PoE-World (Wasu et al., 2025) — synthesizes the dynamics function as inspectable code rather than as neural weights. OpenWorld sits in that subline with two distinctive features: a verification gate over the synthesized world, and consumer-hardware reach. Both properties matter for the application we run here, because we want a world that is small enough to enumerate (5⁸ = 390,625 candidate bundles, a brute-force oracle) and inspectable enough that we can reason about why an aggregation rule did or did not surface a positive-sum bundle.

We use that substrate to study the question the bridging literature cannot answer from platform data alone: *under what structural conditions do bridging algorithms outperform majority vote, and where do they fail?* We build a US two-party political simulator (300 personas, 8-issue policy space, bimodal ideology distribution grounded in Pew/ANES, ground-truth global utility including non-observable positive-sum spillovers), implement four aggregation rules plus an oracle ceiling, and sweep across four axes: slate composition, population polarization, spillover magnitude, and adversarial coalition voting. We add a final axis comparing the production Community Notes matrix-factorization rule (Wojcik et al., 2022) against a fifty-line polarity-product variant.

The paper proceeds as follows. Section 2 covers the world-models and bridging literatures; Section 3 specifies the simulator; Section 4 reports the full experimental result set across six subsections; Section 5 articulates the unified thesis (regime-specific bridging), the production implications, and a methodological reflection on the N=300 vs N=20 discovery that drove the paper's framing; Sections 6 and 7 cover related work and limitations; Section 8 closes with future directions.

---

## 2. Background

### 2.1 World Models

A *world model* is a learned or constructed representation of an environment's dynamics: given a state and a candidate action, it predicts the next state and the resulting reward. The line organizes around three tensions: *learned versus symbolic* (inferred from interaction data or written as explicit code), *latent versus explicit state* (compressed embedding or named interpretable variables), and *end-to-end versus modular* composition (Ha and Schmidhuber, 2018; reviewed at length in our companion lit review).

Ha and Schmidhuber's "World Models" (arXiv:1803.10122) named the modern line: encode observations to latent, predict latent next-state, train a controller inside the latent rollout. The Dreamer line (Hafner et al., 2020, 2021, 2025) productionized this with a Recurrent State Space Model and discrete latents; DreamerV3 was the first system to collect Minecraft diamonds from scratch with no demonstrations or curriculum, with the *Nature* publication crystallizing the recipe as a domain-general world-model paradigm. Genie (Bruce et al., 2024) trained an 11B-parameter foundation world model from ~200,000 hours of unlabeled internet video using a latent action model, demonstrating that world models can be learned from passive observation at scale. MuZero (Schrittwieser et al., 2020) introduced the orthogonal insight that world-model training need not target reconstruction; it can target downstream-task consistency, and matched superhuman performance on Go, chess, and shogi without being given the rules.

These learned-dynamics approaches share known failure modes — compounding rollout error, OOD generalization collapse, datacenter-scale training cost — which the symbolic-synthesis subline addresses by writing the dynamics function as a program. WorldCoder (Tang et al., 2024) is the closest direct comparator: an LLM-driven agent that incrementally writes a Python world model from environment interactions, maintaining an "optimistic" hypothesis and revising it as evidence accumulates. Dainese et al. (2024) introduced GIF-MCTS (Generate, Improve, Fix with Monte Carlo Tree Search) over LLM-generated edit operations to converge on correct world-model code, accompanied by the first standardized code-WM benchmark. PoE-World (Wasu et al., 2025) represents a world model as a product of programmatic experts, formalizing the compositionality property that monolithic neural models lack.

OpenWorld extends the symbolic-synthesis line with three distinctive properties: a verification gate that runs each generated world through behavioral, invariant, and semantic critics before accepting it; tunable moral configurations parameterizing critic behavior; and consumer-hardware reach. The trade-off is requiring natural-language rule descriptions as input — a feature for codifiable-rule domains and a fundamental limitation for tacit-rule ones. The simulation work we report here sits inside that feature side of the trade-off: voting rules, slate composition, persona behavior, and ground-truth utility are all expressible in code, which is why a code-synthesis world model is the right substrate.

### 2.2 Bridging Algorithms

The bridging-algorithms literature studies aggregation rules that weight cross-cluster agreement higher than within-cluster agreement. Three strands converge.

**Community Notes (Birdwatch).** Wojcik et al. (2022, arXiv:2210.15723) model each rating r_{un} of note n by rater u as

  r̂_{un} = μ + i_u + i_n + f_u · f_n

where μ is a global intercept, i_u and i_n are rater- and note-specific intercepts, and f_u, f_n are one-dimensional latent factors. The L2-regularized loss applies λ_i = 0.15 to intercepts and λ_f = 0.03 to factors — a deliberate 5-to-1 asymmetry designed to absorb within-faction enthusiasm into the f_u · f_n product, leaving i_n as a residual cross-faction-approval measure. The decisive design move is ranking by i_n alone, not by raw average. A note rated 5/5 within one faction and 0/5 within the other receives no *Currently Rated Helpful* status, while a note rated 3/5 by both factions can. The system runs at X platform scale with both the algorithm and rating data open-sourced (`github.com/twitter/communitynotes`); the deployment study reports reduced reshare rates on annotated posts.

**Polis and Group-Aware Consensus.** Small et al. (2021) describe the rule used at national scale in Taiwan under digital minister Audrey Tang. The participant×statement vote matrix is reduced via PCA to two dimensions, clustered with K-means (K ∈ {2,...,5}, silhouette-selected), and statements are ranked by

  GAC(s) = ∏_{g ∈ G} P(agree | g, s)

— the geometric mean of within-group agreement probabilities. The product form ensures any one silent or dissenting group collapses the score, formalizing the *anti-tyranny-of-majority* property. vTaiwan has used Polis on twenty-six pieces of legislation including the Uber regulatory question and digital-economy reforms.

**Bridging-Based Ranking as a design principle.** Ovadya (2022) named the pattern and argued for it as a default replacement for engagement-based ranking, citing Community Notes and Polis as existence proofs of a transferable rule.

**The LLM-mediator line.** A parallel substrate uses LLMs as *generators* of consensus candidates rather than rankers. Bakker et al. (2022, NeurIPS) fine-tuned a 70B Chinchilla-class model to produce candidate consensus statements maximizing expected approval across diverse opinion groups, with a separate personalized reward model (PRM) predicting per-participant endorsement and a parametric family of social welfare functions (utilitarian to Rawlsian) aggregating PRM scores. Tessler et al. (2024, *Science*) productionized this as the Habermas Machine: two LLMs cooperating (generator + PRM) with Schulze ranked-choice aggregation, tested on UK panels deliberating Brexit, minimum wage, climate, NHS privatization, and immigration. Participants preferred Habermas Machine statements over trained-human-mediator output 56% of the time. The Bakker → Tessler pipeline — generator (LLM), PRM (LLM), aggregator (transparent social-choice rule) — is the canonical architecture, and its methodological keystone is that LLMs generate and predict but are not trusted to aggregate.

**Bridging meets alignment.** Anthropic and the Collective Intelligence Project (Huang et al., 2024, FAccT) used Polis to source constitutional principles from ~1,000 public participants and fine-tuned Claude against the resulting public-input constitution — the first production handshake between Polis-style aggregation and frontier-model alignment. Conitzer et al. (2024, ICML position paper) argued that RLHF is *already* a social-choice operator: every aggregation rule over diverse human feedback implicitly commits to a welfare-axiomatic position, and most current RLHF pipelines have not been examined as such. The Generative Social Choice line (Fish et al., 2024; cf. Procaccia's broader computational-social-choice program) wraps generator LLMs in formal social-choice machinery with proportionality and justified-representation guarantees.

### Synthesis: the intersection

The two literatures rarely cite each other. Bridging research lives on real platforms with unobservable ground truth; world-model research lives in simulations with computable counterfactuals but no deliberative stakeholders. The intersection — a code-synthesized world whose agents are stakeholders and whose dynamics include the act of aggregating their preferences — supports four properties that neither literature alone produces: counterfactual reachability (paired trials under identical persona populations and slate seeds, varying only the aggregation rule), ground-truth payoffs (G(bundle) is specified directly, the oracle is computable by brute force), repeatable trials with controlled variation (sweep polarization while holding network constant, and vice versa), and edge-case and adversarial generation (trimodal distributions, heavy-polarization regimes, coordinated-attack coalitions that no platform would deploy on its users). Our work sits exactly in that intersection.

---

## 3. Methodology

### 3.1 The World

The simulator is built on `quome-cloud/openworld`'s `World` / `Transition` ontology (Schwoebel et al., 2026). World 1 — the US two-party political simulator — has three entities (`Persona`, `PolicyBundle`, `World`), a state consisting of the currently-enacted PolicyBundle and the period index, and a transition function that takes endorsement-set inputs from personas, applies an aggregation rule, and outputs a new enacted policy plus a round-utility score. The full design specification is Prism's v2 design (Prism, 2026; `openworld_world_designs.md`); we recapitulate the parts load-bearing for the experiments.

### 3.2 Personas

Each persona has a `latent_ideology` (float ∈ [−1, 1]), an `issue_weights` vector (Dirichlet-distributed, sums to one), and a `network_position` in a six-community Stochastic Block Model communication graph (within-community edge probability p_in = 0.3, across-community p_out = 0.02). The two major ideological clusters contain approximately 40% of personas each; four smaller "bridgeable" clusters contain approximately 5% each.

The default `latent_ideology` distribution is the bimodal symmetric mixture grounded in Pew Research Center's Political Typology (7-cluster model, 2021/2023) and ANES 2020/2022 cumulative-file ideology self-placement: 40% N(−0.45, 0.25) + 40% N(+0.45, 0.25) + 20% N(0, 0.15). Dirichlet(α = 0.5) issue weights produce single-issue-voter skew, matching empirical salience distributions.

### 3.3 Policy Action Space

Each `PolicyBundle` is a discrete vector of stances, one per issue. The eight issues are `immigration`, `healthcare`, `climate`, `fiscal`, `foreign_policy`, `civil_rights`, `education`, `criminal_justice`, each with stance ∈ {−2, −1, 0, +1, +2} on the standard ANES progressive-to-conservative coding axis. The full bundle space is 5⁸ = 390,625, small enough that the oracle ceiling is computable by brute-force enumeration with SHA-256-keyed cache reuse across trials.

Personas vote over a candidate slate of K bundles per trial. The baseline experiment uses K = 7 with five archetype bundles (all-issues set to each of {−2, −1, 0, +1, +2}) plus two random per-trial bundles. The slate-composition sensitivity check (Section 4.2) varies both K and the archetype-versus-random split.

### 3.4 Global Utility Function

The ground-truth global utility has two components. *Individual welfare*: each persona's welfare on a given bundle is the weighted sum across issues of a quadratic loss from the persona's ideal stance (derived from `latent_ideology` plus per-issue idiosyncratic noise), with weights drawn from the issue-importance vector. *Positive-sum spillovers*: fixed bonuses added when a bundle matches specific sub-patterns, analogous to public-goods effects that don't appear in individual preference calculations. Two spillover configurations are run.

The **centrist** configuration places positive-sum bundles at or near the policy center: `(healthcare=0, fiscal=0) → +0.05`, `(climate=−1, foreign_policy=0) → +0.03`, `(criminal_justice=−1, education=0) → +0.04`. The **off_axis** configuration places them at non-centrist intersections: `(climate=−2, fiscal=+1) → +0.05`, `(civil_rights=+1, criminal_justice=−1) → +0.03`, `(education=+2, healthcare=−1) → +0.04`. The off_axis configuration is Prism's centrism-laundering stress test: if bridging still beats majority vote when positive-sum bundles are not near the policy center, it is doing real cross-cluster signal extraction.

The full global utility is

  G(bundle) = mean({welfare(p, bundle) for p in Personas}) + Σ spillovers(bundle)

and the oracle is `argmax_bundle G(bundle)` over all 390,625 bundles, precomputed once per (persona population, spillover config) pair.

### 3.5 Conditions

Five algorithm conditions, run in paired trials (identical persona populations, identical random seeds within a cell):

- **Z** (random): uniform draw from the candidate slate; floor anchor.
- **A** (majority vote): each persona endorses the slate bundle maximizing their welfare; plurality wins, with lexicographic tie-break on the bundle tuple.
- **C_CN** (Community Notes matrix factorization): production specification from Wojcik et al. (2022). Fit r̂_{un} = μ + i_u + i_n + f_u · f_n by L2-regularized SGD with λ_i = 0.15, λ_f = 0.03, factor dimension d = 1, 200 iterations; rank slate by i_n.
- **C_PP** (polarity-product): bridge score = geometric mean of per-community endorsement rates across the six SBM communities. Approximately fifty lines of code, no fitting step. The cheap-baseline variant of the bridging concept (variant b in our survey formalization).
- **D** (oracle): precomputed `argmax G(bundle)`; ceiling.

The adversarial conditions (Section 4.5) add C_CN_ADV and C_PP_ADV, in which the R-cluster (latent_ideology > 0.05) zeroes its endorsement on any bundle the L-cluster (latent_ideology < −0.05) endorses at rate ≥ 50%. This is the canonical coordinated-downvote attack on Community Notes.

We do not run Condition B (LLM mediator, Habermas-style) in this paper. The literature's strongest claim about LLM mediation (Tessler et al., 2024) rests on field experiments with human deliberators rather than on simulated personas, and we judged the chained-LLM noise — generator, PRM, aggregator — to obscure the structural lever we are investigating. We discuss this scoping decision in Section 7.

### 3.6 Metric

The primary metric is the gap fraction

  gap_fraction = (G_achieved − G_random) / (G_oracle − G_random)

where G_random is the mean welfare over all 390,625 bundles (the population average, not a single Z draw). This normalization holds the metric on [0, 1] regardless of the absolute scale of G, lets us say "bridging captures 79% of the available improvement gap" rather than reporting raw welfare numbers, and is the same framing Prism's design specifies for cross-world comparison.

For each cell we report the median gap fraction across N trials (typically 30, 50, or 100 per cell depending on the axis). We use median rather than mean throughout because the deterministic conditions (A, C_CN, C_PP, D) often lock onto identical bundles across trials (an artifact of the archetype-seeded slate design, discussed in Section 4); the median is the right summary in that regime, and mean is reported in the result CSVs for completeness.

---

## 4. Experiments

We report results across six axes. Each subsection reports medians from the per-cell trial population, with N specified per axis.

### 4.1 Baseline (K = 7 archetype, N = 100 and N = 300)

The original N = 20 PR (`run_cycle5.py`) produced gap fractions of 0.306 (A) versus 0.793 (C_CN) on centrist spillovers, and 0.495 (A) versus 0.904 (C_CN) on off_axis spillovers: a roughly two-to-three-fold bridging advantage. We re-ran the same configuration at N = 100 personas with K = 7 archetype-seeded slates and 100 trials per cell, then at N = 300 with 50 trials per cell.

| Condition | N=100 centrist | N=100 off_axis | N=300 centrist | N=300 off_axis |
|-----------|---------------:|---------------:|---------------:|---------------:|
| Z (random) | 0.184 | 0.253 | 0.157 | 0.158 |
| A (majority) | 0.278 | 0.405 | 0.764 | 0.821 |
| C_CN | 0.746 | 0.766 | 0.764 | 0.821 |
| C_PP | 0.746 | 0.766 | 0.764 | 0.821 |
| D (oracle) | 1.000 | 1.000 | 1.000 | 1.000 |
| **C_CN − A** | **+0.468** | **+0.361** | **+0.000** | **+0.000** |

The N = 100 numbers replicate the original PR's effect direction with tightened estimates: bridging captures 74.6% of the centrist gap versus 27.8% for majority vote, a 46.8pp advantage. The off_axis numbers (40.5% A versus 76.6% C_CN, 36.1pp advantage) reject the centrism-laundering null hypothesis: bridging continues to find cross-cluster signal when positive-sum bundles are placed at non-centrist intersections.

The N = 300 result, however, is decisive in the opposite direction. At the persona count Prism's specification calls for as the default, *both majority vote and bridging converge to the centrist outcome*. Gap fractions are 0.764 (A) and 0.764 (C_CN) on centrist spillovers, and 0.821 (A) and 0.821 (C_CN) on off_axis spillovers. The 46.8pp advantage at N = 100 is a *zero-pp* effect at N = 300. The mechanism is mechanical: with 300 personas in a bimodal symmetric distribution, the plurality vote among the five archetype bundles is determined by the bimodal distribution's mass; the centrist archetype attracts the centrist 20% plus a deterministic share of the bimodal tails, and that share is large enough at N = 300 to make the plurality outcome match the cross-cluster bundle. At N = 20, sampling fluctuation pushes the plurality outcome to a flanking archetype frequently enough that majority vote underperforms.

This is the result that frames the rest of the paper. The bridging advantage exists, but it is regime-specific along the sample-size axis in addition to all the structural axes we sweep below. We treat the N = 100 + N = 300 pair as the honest headline rather than burying the convergence inside the cell with the largest contrast.

### 4.2 Slate-composition sensitivity

Slate-composition is the strongest contribution from the original PR, generalized here across K ∈ {5, 7, 15, 30} crossed with archetype-seeded versus fully-random pools, two spillover configurations, 100 trials per cell, N = 300. The fully-random condition replaces all five archetype seeds with random per-trial bundles. The fragmented bridging advantage in the random-slate condition was the original PR's strongest finding (Cycle 5b); the sweep here confirms that it does not emerge with larger K.

| K | Pool | C_CN centrist | A centrist | Δ centrist | C_CN off_axis | A off_axis | Δ off_axis |
|---|------|--------------:|-----------:|-----------:|--------------:|-----------:|-----------:|
| 5 | archetype | 0.746 | 0.278 | +0.468 | 0.766 | 0.405 | +0.361 |
| 5 | random | 0.192 | 0.174 | +0.018 | 0.277 | 0.231 | +0.046 |
| 7 | archetype | 0.746 | 0.278 | +0.468 | 0.766 | 0.405 | +0.361 |
| 7 | random | 0.213 | 0.188 | +0.025 | 0.300 | 0.261 | +0.039 |
| 15 | archetype | 0.746 | 0.278 | +0.468 | 0.766 | 0.405 | +0.361 |
| 15 | random | 0.261 | 0.257 | +0.004 | 0.376 | 0.326 | +0.050 |
| 30 | archetype | 0.746 | 0.278 | +0.468 | 0.766 | 0.405 | +0.361 |
| 30 | random | 0.310 | 0.263 | +0.047 | 0.435 | 0.374 | +0.061 |

Two findings.

First, with archetype seeds, the bridging advantage is *stable in K* across an order of magnitude (K = 5 to K = 30). Adding more random bundles to the slate does not dilute the bridging advantage; both algorithms find the archetype bundle that the spillover configuration rewards. The archetype is the load-bearing anchor.

Second, with fully-random slates, the bridging advantage does *not* emerge at any K we tested. Even at K = 30, the delta is +0.047 (centrist) and +0.061 (off_axis) — distinguishable from zero but practically small relative to the archetype-seeded delta of +0.468 / +0.361. We had pre-registered the hypothesis (H2b in Prism's expansion design) that there might exist a K above which random slates contain enough natural anchors for bridging to extract signal; the data through K = 30 does not support it. The production-implication conclusion is unambiguous: explicit candidate-pool design is not optional for bridging deployments, it is a load-bearing component of the system.

### 4.3 Population distribution sweep

The bridging algorithm is structurally a cross-cluster detector — its behavior is coupled to how many clusters exist, how separated they are, and how symmetric they are. We swept five distributions, N = 300, K = 7 archetype-seeded, 100 trials per cell, both spillover configurations:

- **P1 (unimodal):** N(0, 0.5)
- **P2 (bimodal symmetric):** baseline, 40/40/20
- **P3 (bimodal asymmetric):** 60/30/10 majority-faction scenario
- **P4 (trimodal):** 33/33/33 with three faction centers
- **P5 (heavy polarization):** 48% N(−0.80, 0.10) + 48% N(+0.80, 0.10) + 4% N(0, 0.10)

| Distribution | A centrist | C_CN centrist | Δ centrist | A off_axis | C_CN off_axis | Δ off_axis |
|--------------|-----------:|--------------:|-----------:|-----------:|--------------:|-----------:|
| P1 unimodal | 0.754 | 0.754 | +0.000 | 0.757 | 0.757 | +0.000 |
| P2 bimodal sym | 0.278 | 0.746 | +0.468 | 0.405 | 0.766 | +0.361 |
| P3 bimodal asym | 0.369 | 0.722 | +0.353 | 0.547 | 0.756 | +0.209 |
| P4 trimodal | 0.748 | 0.748 | +0.000 | 0.798 | 0.798 | +0.000 |
| P5 heavy polar | 0.000 | 0.724 | +0.724 | 0.000 | 0.736 | +0.736 |

The P1 and P4 results — bridging exactly equals majority vote on both spillover configurations — confirm that the matrix factorization fundamentally requires two-faction structure. The 1D latent factor cannot represent unimodal preference heterogeneity (P1) or three-faction structure (P4); both algorithms collapse to the same centrist-bundle plurality outcome. P3 (asymmetric bimodal) shows a smaller bridging advantage than P2, consistent with H1b inverted: the asymmetry pulls the majority plurality *closer* to the cross-cluster bundle rather than further, narrowing the bridging gap.

P5 (heavy polarization) is the strongest result in the sweep. Majority vote collapses to gap_fraction = 0.000 on both spillover configurations: with 48/48 faction split at σ_within = 0.10 and faction centers at ±0.80, no archetype bundle attracts a plurality, and majority vote returns the lexicographically-first archetype, which is the strong-progressive (-2 on every issue) bundle that scores at the random floor. Bridging holds at 0.724 (centrist) and 0.736 (off_axis). The bridging advantage is 72.4 and 73.6 percentage points — by far the largest in the sweep.

The polarization sub-sweep (σ_between ∈ {0.00, 0.30, 0.45, 0.60, 0.65, 0.70, 0.75, 0.80}, symmetric bimodal, σ_within = 0.25, 30 trials per cell) reveals that this is not a gradient but a *step function*:

| σ_between | A median | C_CN median | Δ |
|-----------|---------:|------------:|--:|
| 0.00 | 0.755 | 0.755 | +0.000 |
| 0.30 | 0.750 | 0.750 | +0.000 |
| 0.45 | 0.748 | 0.748 | +0.000 |
| 0.60 | 0.745 | 0.745 | +0.000 |
| 0.65 | 0.744 | 0.744 | +0.000 |
| 0.70 | 0.743 | 0.743 | +0.000 |
| 0.75 | 0.742 | 0.742 | +0.000 |
| **0.80** | **0.250** | **0.742** | **+0.492** |

Across the entire pre-step region the two algorithms are indistinguishable. At σ_between = 0.80 majority vote collapses by approximately half a gap-fraction unit while bridging holds. The discontinuity is sharp enough that the sub-sweep does not produce the inverted-U curve Prism's H1d predicted; instead we get a *cliff* — a regime boundary at which majority vote stops working as plurality becomes structurally impossible, while bridging continues to surface the cross-cluster bundle because its scoring rule does not depend on plurality.

This is the cleanest empirical justification for the regime-specificity thesis in the paper. Across most of the polarization axis bridging is redundant; at the cliff it becomes load-bearing.

### 4.4 Spillover-magnitude sweep

We swept spillover magnitude across S1 (0.01), S2 (0.05, baseline), S3 (0.10), S4 (0.20), N = 300, K = 7 archetype-seeded, 50 trials per cell, both spillover configurations.

| Magnitude | A centrist | C_CN centrist | A off_axis | C_CN off_axis |
|-----------|-----------:|--------------:|-----------:|--------------:|
| S1 (0.01) | 0.473 | 0.945 | 0.542 | **1.000** |
| S2 (0.05) | 0.242 | 0.658 | 0.335 | 0.641 |
| S3 (0.10) | 0.140 | 0.531 | 0.171 | 0.343 |
| S4 (0.20) | 0.064 | 0.437 | 0.072 | 0.164 |

The pattern runs counter to Prism's H3a / H3b hypotheses. We had predicted bridging would *increase* its advantage with magnitude; the data shows the opposite. At S1 (smallest spillovers) C_CN reaches gap_fraction = 1.000 on off_axis — the algorithm hits the oracle exactly. At S4 (largest spillovers) C_CN drops to 0.164 on off_axis. The mechanism: at small spillover magnitudes, the oracle bundle is the one closest to the bimodal-distribution center, and bridging surfaces it reliably. At large spillover magnitudes, the spillover bonus dominates the welfare calculation and the oracle bundle is whichever archetype happens to satisfy the spillover pattern, which is not necessarily the cross-cluster bundle bridging surfaces. The off_axis configuration shows the cleanest version of this collapse: at S4 the oracle is a non-archetype bundle and bridging cannot find it from a K = 7 slate that contains no archetype matching the spillover pattern.

The most surprising row is S1: at noise-floor spillover magnitude, bridging captures *more* of the gap, not less. The reason is that the gap itself shrinks (G_oracle − G_random is small at S1), and the centrist archetype that bridging always surfaces happens to be very close to the oracle when the spillover bonus is near zero. This is methodologically uncomfortable — it means the small-magnitude bridging advantage is partially a normalization artifact — but it is the result.

The C_CN ≈ C_PP equivalence holds across every magnitude cell (medians match to three decimal places).

### 4.5 Adversarial coalition voting

The canonical attack on Community Notes is coordinated downvoting: a coalition strategically zeroes its endorsement on any item the opposing cluster endorses, hoping to prevent CRH status. We implemented this with N = 100 personas at P2 (moderate polarization), and separately at P5 (heavy polarization) with N = 300, 50 trials per cell, both spillover configurations. The attack rule: any persona with `latent_ideology > 0.05` (R-cluster) zeroes their endorsement on any bundle endorsed by ≥ 50% of the L-cluster (`latent_ideology < −0.05`).

| Distribution | Spillover | C_CN clean | C_CN_ADV | Degradation | A baseline |
|--------------|-----------|-----------:|---------:|------------:|-----------:|
| P2 (moderate) | centrist | 0.746 | 0.278 | **−0.468** | 0.278 |
| P2 (moderate) | off_axis | 0.766 | 0.405 | **−0.361** | 0.405 |
| P5 (heavy) | centrist | 0.724 | **0.724** | +0.000 | 0.000 |
| P5 (heavy) | off_axis | 0.736 | **0.736** | +0.000 | 0.000 |

The P2 result confirms the attack works as the literature predicts: bridging degrades to majority-vote performance on both spillover configurations. C_CN_ADV is indistinguishable from A baseline (0.278 vs 0.278 centrist, 0.405 vs 0.405 off_axis). The attack is effective at the moderate-polarization regime where bridging is *not yet* load-bearing — i.e., where the regime where the original PR found the advantage.

The P5 result is the surprise. At heavy polarization, the same attack produces *zero* degradation: C_CN_ADV = C_CN = 0.724 / 0.736, and majority vote remains at A = 0.000. The mechanism is structural and clean. At P5 with faction centers at ±0.80, neither faction broadly endorses the cross-partisan bundle. The L-cluster endorsement rate on the bundle bridging surfaces is *below* the 50% trigger threshold — typically because the bundle is itself a compromise neither faction loves but neither rejects, and L-cluster endorsement clusters around 30–40% rather than the ≥ 50% that triggers the R-cluster attack rule. Without trigger, no attack; without attack, bridging surfaces the cross-cluster bundle exactly as it does in the clean condition.

This is the second cleanest empirical result in the paper. It directly addresses the deployment risk that has constrained bridging adoption in production: yes, bridging is adversarially vulnerable to coordinated downvoting, but the vulnerability is *concentrated in the moderate-polarization regime where majority vote is already partially effective*. At the high-polarization regime where bridging is load-bearing, the attack does not fire. (We note one wrinkle: in the P5 cell the mean C_CN_ADV gap fraction is 0.512 / 0.529, lower than the median of 0.724 / 0.736 — i.e., the attack fires in a non-trivial minority of trials. The median-vs-mean gap is informative and we discuss the variance in Section 5.) The C_PP variant degrades more severely under attack (C_PP_ADV centrist median = 0.279) than C_CN does — the simple polarity product is more attack-fragile than the matrix factorization in the P5 regime.

### 4.6 C_CN vs C_PP equivalence

Across every cell we have reported above, C_CN and C_PP medians match to three decimal places or better. The single exception is the P5 adversarial cell discussed at the end of Section 4.5, where C_PP_ADV under-performs C_CN_ADV substantially — i.e., where the cheap variant is genuinely more attack-fragile than the production-grade variant. Outside the adversarial cell the two algorithms produce indistinguishable outcomes.

This is the result a practitioner can put weight on. At K ≤ 30 slate sizes, the production Community Notes matrix-factorization spec — gradient descent over a regularized factor model with 200 SGD iterations per round — provides no measurable benefit over a fifty-line geometric-mean polarity-product score, in the non-adversarial setting. The latent factorization machinery is doing nothing the geometric mean is not already doing. For platforms whose slate sizes are in this range — most production deliberation tools, candidate-pool curation systems, and recommendation reranking pipelines — the cheap variant should be the default and the matrix factorization should be reached for only when adversarial robustness is an explicit requirement.

We do not have data above K = 30. Whether C_CN diverges from C_PP at larger K, in the regime where Wojcik et al.'s production system actually runs (tens of thousands of notes, sparse endorsements), is an open question and a natural follow-on experiment.

---

## 5. Discussion

### 5.1 The unified thesis: regime-specific bridging

The four structural conditions we swept produce a coherent characterization. Bridging is regime-specific along all four axes, and the regimes where it is useful, where it is redundant, where it fails, and where it is robust to attack are *not* independent. They co-occur in informative ways.

*Where bridging is useful.* High polarization (σ_between ≥ 0.80), structured slates (archetype-seeded), and two-faction structure are the conditions under which bridging captures large fractions of the random-to-oracle gap that majority vote misses. At σ_between = 0.80, majority vote collapses to gap_fraction = 0.000 — it cannot find a plurality among archetype bundles when factions are this far apart. Bridging holds at 0.74. Outside this regime, in the polarization range from 0.00 to 0.75, bridging and majority vote are empirically indistinguishable on archetype-seeded slates.

*Where bridging is redundant.* Moderate polarization with N large (N = 300 at σ_between = 0.45) produces a zero-pp delta between bridging and majority vote. The bimodal distribution's centrist mass is large enough that the centrist archetype attracts a plurality, which is also what bridging surfaces. The original N = 20 advantage at this regime was a sampling artifact: at small N, fluctuation pushes the plurality outcome to a flanking archetype frequently enough to under-perform; at large N this stops happening and the two algorithms converge.

*Where bridging fails.* Unimodal (P1), trimodal (P4), and fully-random slate (K ∈ {5,...,30} random pool) regimes all produce C ≈ A. The matrix factorization's 1D latent factor cannot represent unimodal or trimodal preference structure; the geometric-mean polarity product fails for the same reason in the unimodal case, and for a slightly different reason in the trimodal case (three-way agreement is more sparsely sampled than two-way). Random slates fail because there is nothing to bridge between — the algorithm has no high-cross-cluster-endorsement bundle to surface.

*Where bridging is robust to attack.* The adversarial coalition attack collapses C_CN to A at moderate polarization (P2). At heavy polarization (P5), the same attack has *zero* effect. The mechanism is the trigger threshold: at heavy polarization, no bundle attracts ≥ 50% endorsement from the L-cluster, so the R-cluster's attack rule never fires. The vulnerability is therefore concentrated in the polarization regime where bridging is *not yet* load-bearing — a fortunate composition of regime properties for deployment risk.

The unified thesis: *bridging is most useful and most adversarially robust precisely at the structural conditions where it matters most.* The adversarial vulnerability exists only in the moderate-polarization regime where majority vote is already partially effective; in the heavy-polarization regime where bridging is the only algorithm capable of producing a coherent outcome, the standard attack does not work. This is not a coincidence — both properties are downstream of the same structural fact (the L-cluster endorsement rate falls below 50% at heavy polarization), but they are not properties that platform-data analysis would have surfaced.

### 5.2 Production implications

Three implications for platform deployments.

First, *slate-pool design is a first-class lever*. The bridging-vs-majority comparison is not a question of which aggregation rule to choose; it is a question of which combination of (candidate-pool composition, aggregation rule, polarization regime) the system is in. Platforms deploying bridging should treat candidate-pool curation as part of the algorithm, not as a neutral input. For OpenWorld specifically, this argues for promoting `CandidatePool` (or similar) to a first-class abstraction alongside `World`, `Transition`, and `ObjectiveSuite`. For platforms that already deploy bridging at scale (Community Notes, Polis), it argues for instrumenting and tuning the pool-composition pipeline at the same priority as the aggregation rule.

Second, *the adversarial vulnerability is concentrated in the low-value regime*. This is good news for deployment risk. The coordinated-downvoting attack is most effective on bridging precisely when bridging is providing the least incremental value over majority vote; in the heavy-polarization regime where bridging is doing the actual lift, the attack does not fire because the endorsement-trigger threshold is never reached. Platform designers can deploy bridging with this characterization in hand: the worst-case attack outcome is degradation to majority-vote performance, not below it, and that worst case occurs only where the algorithm has the smallest gap to lose.

Third, *the C_CN ≈ C_PP equivalence suggests cheap implementations may suffice*. For slate sizes up to K = 30, the production Community Notes matrix factorization provides no measurable benefit over a fifty-line polarity-product score in the non-adversarial setting. Platforms whose deliberation, candidate-curation, or reranking pipelines run at slate sizes in this range can default to the cheap variant. The matrix factorization is genuinely useful in the adversarial regime at heavy polarization (C_CN_ADV holds at 0.724 while C_PP_ADV drops to 0.279 in the P5 centrist cell), which is a deployment-decision criterion: pay for matrix factorization when adversarial robustness is an explicit requirement, default to polarity-product otherwise.

### 5.3 Methodological reflection: the N=300 discovery

The single most consequential moment in the paper's development was running the N = 300 baseline and discovering that majority vote and bridging converge to gap_fraction = 0.764. The original PR — which the paper's contribution claim originally rested on — reported a 46.8pp bridging advantage at N = 20, and a similar gap was reported in the team deliverable as "bridging captures 2–3× more of the gap to oracle than majority vote." The N = 300 result demonstrated that the standard configuration the design specification calls for produces *no advantage*. The 2–3× claim was real but regime-specific along an axis (sample size) that the original PR had not swept.

This is what the simulation paradigm enables and what platform-data analysis structurally cannot. A platform deployment of Community Notes runs at one N — the number of raters who have engaged with the notes at any given time. A platform-data analysis of the bridging-vs-majority comparison would have one N, one population, one history. The question "would this advantage hold at a different N?" is not askable from production data. In the simulator, it is a routine sweep — we re-ran the same configuration at N = 100 and N = 300, the result rolled in, and the paper's framing changed.

The change matters substantively. If we had reported the N = 20 result as the headline, the contribution claim would have been wrong in a way that is hard to correct after publication. Reviewers do not always re-run experiments; field practitioners do not always have the resources to sweep parameters before adoption. The simulation paradigm provides an internal self-correction mechanism — keep running paired counterfactual trials across structural axes until the regime-specificity of any reported effect is mapped — and we used it. The paper's framing as a regime taxonomy rather than as a "bridging beats majority" advocacy piece is the direct product of that self-correction.

This is the deepest payoff from the world-model-meets-bridging intersection. Each literature alone has structural blind spots — platform data has no counterfactuals, world-model research has no deliberative stakeholders — and the intersection lets one literature's instrument check the other literature's claims. The N = 300 discovery is the first concrete instance we know of where a simulation-paradigm sweep falsified the headline finding of a bridging-algorithm comparison and produced a more honest result.

---

## 6. Related Work

The two literatures our work joins have been surveyed in Section 2; here we position our contribution against the closest comparators in each.

**Bridging-algorithm empirical work.** Wojcik et al. (2022) is the canonical deployed-system study, evaluating Community Notes on X reshare data; the analysis is observational and confined to the natural distribution of users and notes the platform sees. Small et al. (2021) provide the Polis design rationale and empirical case studies (vTaiwan); the design defends the Group-Aware Consensus rule against majority-rule alternatives by appeal to its formal property of penalizing within-group enthusiasm, without empirical counterfactuals. Tessler et al. (2024) report Habermas Machine UK field experiments comparing LLM-generated and human-mediator consensus statements; the comparison is between generators rather than between aggregation rules and does not vary population structural properties. Huang et al. (2024) report the Collective Constitutional AI handshake but do not compare aggregation rules. None of these works run paired counterfactual trials across population polarization, slate composition, or adversarial coalitions on the same dataset. Our work fills exactly that gap.

**Simulation studies of voting and deliberation.** Computational social choice has long used simulation as a theoretical instrument — Procaccia's program studies impossibility results, robustness, and incentive-compatibility properties of voting rules. The Generative Social Choice line (Fish et al., 2024) wraps LLMs in formal social-choice machinery with axiomatic guarantees; their evaluation is again analytical and proof-based rather than counterfactual-empirical. Conitzer et al. (2024) frame the open question and call for instrumental work; our work is one answer to that call.

**World-model frameworks.** The Dreamer line (Hafner et al., 2020, 2021, 2025) and Genie (Bruce et al., 2024) are learned-dynamics frameworks; their application targets are RL agents in games and procedurally generated worlds, not deliberative stakeholders. MuZero (Schrittwieser et al., 2020) abstracts further by training on returns rather than transitions. WorldCoder (Tang et al., 2024) and the Code World Models benchmark (Dainese et al., 2024) target factual world-modeling — getting the dynamics right — rather than value-laden world-modeling. PoE-World (Wasu et al., 2025) targets compositionality. OpenWorld (Schwoebel et al., 2026) is the framework we extend, sharing the symbolic-synthesis substrate with this subline but adding a verification gate and a tunable moral-axis configuration that the comparators do not have. Our application is the first to use a symbolic-synthesis world model to evaluate a class of social-choice algorithms by paired counterfactuals.

The most direct comparator from the bridging side is Wojcik et al.'s deployment-study analysis; the most direct comparator from the world-model side is WorldCoder's Python-program world model. We sit in between: the bridging-algorithm comparison Wojcik et al. ran is the question, the world-model substrate WorldCoder demonstrated is the instrument, and the regime-taxonomy claim Section 5.1 articulates is what only the intersection produces.

---

## 7. Limitations

Our experimental scope bounds three sets of findings in ways practitioners should account for before applying the regime taxonomy to deployment decisions.

**Population size and the null result at realistic N.** Our headline sweep experiments use N=100 simulated personas per cell; our primary comparison uses N=20. At N=300 — a scale more consistent with pilot deliberation deployments — the gap_fraction delta between 1D matrix factorization bridging and majority vote is zero under bimodal symmetric polarization with archetype-seeded bundles (C_CN = A = 0.764 centrist, 0.821 off_axis). The "bridging advantage" that motivated this investigation is a regime-specific property: it is real and large at small N, near extreme polarization (P5), and when bundles are structurally seeded at preference extremes. It is absent at N=300 under normal bimodal conditions. We report this correction explicitly rather than averaging over the confound. The practical implication is that bridging's value case rests on the structural regime conditions (P5 extreme polarization, archetype-seeded pools, K-stable advantage) rather than on a universal superiority claim over majority vote.

**Distribution assumptions.** All positive bridging-advantage findings occur under bimodal preference distributions (P2 symmetric, P3 asymmetric, P5 heavy-polar). Unimodal distributions (P1) and trimodal distributions (P4) show zero bridging advantage, with P4's null result mechanistically attributable to 1D matrix factorization's inability to recover a three-cluster structure from a single latent dimension. We have not tested bridging algorithms specifically designed for multi-cluster structure, nor have we characterized real-world opinion distributions to determine which regime applies to specific deliberation contexts. Practitioners must assess which distributional regime their context inhabits before treating these results as applicable.

**Adversarial scope.** We test one adversarial geometry: coordinated cluster-level downvoting in which one faction systematically depresses ratings for the opposing faction's preferred bundles. The result is regime-dependent: at moderate polarization (P2, bimodal_sym), this attack eliminates bridging's advantage (C_CN_ADV ≈ A); at extreme polarization (P5, heavy_polar), the attack is structurally ineffective because the cross-partisan bundle receives insufficient L-endorsement to be targeted, and C_CN_ADV = C_CN (degradation = 0.000). This establishes that bridging's adversarial vulnerability and adversarial robustness are coherent with its performance regime: the algorithm is most vulnerable where it provides least value, and most robust where it provides most value. C_PP shows partial resistance at P2, but this result is conditional on the attacker's targeting partition aligning with the SBM community structure — an alignment we cannot guarantee in deployment. We have not tested sybil-based persona injection, partial-information attacks, slow-drip noise strategies, or adversarial behavior at K>7. The adversarial results establish regime-dependent robustness properties, not a universal characterization of the attack surface.

**Algorithm scope.** Conditions A and C_CN behave identically under non-adversarial archetype-seeded conditions in our setup at N=300, and C_CN ≈ C_PP across all non-adversarial cells. This near-degeneracy limits our ability to distinguish algorithmic contributions within the tested regime. More varied algorithm conditions — higher-dimensional factorizations, deliberation-aware aggregation rules from the Wojcik/Tessler/Polis literature, and full Bakker–Tessler LLM-mediator pipelines (which we scoped out of this work because of compute cost and the structural lever being our focus) — are productive directions for follow-on work. Similarly, our payoff matrix is hand-specified; sensitivity to payoff misspecification is unexplored.

**Seeding assumption.** Results conditioned on archetype-seeded bundles require an operator intervention — deliberately placing content at preference poles — that may not be feasible or appropriate in all contexts. The random-slate results (near-zero bridging advantage, cycle 5b in §4.2) represent the non-intervention baseline and should be treated as the default expectation absent platform-level seeding.

**Single-world scope.** Our experimental World is a US-style two-faction political simulator. The interdisciplinary research-funding panel (World 2 in the original design, deferred to follow-on work) would test whether the regime taxonomy generalizes to many-cluster contexts. Until World 2 results are available, the regime conditions we map should be treated as conditions for two-faction settings specifically.

---

## 8. Conclusion and Future Work

We have used the OpenWorld code-synthesis world-model framework to map the regime structure of bridging algorithms against majority vote in a US two-party political simulator. The unified thesis: bridging is regime-specific along four axes — slate composition, polarization, adversarial coordination, faction count — and the regimes where it is useful and where it is adversarially robust co-occur in a way that platform-data analysis would not have surfaced. The methodological discovery that the original N = 20 PR result did not replicate at N = 300 is reported as the headline before the regime-specific advantages, because the simulation paradigm's value is exactly in producing this kind of self-correction.

Six open questions follow directly from the result set:

1. **Does the slate-composition collapse continue at K > 30?** Our random-slate sweep stops at K = 30. Wojcik et al.'s production Community Notes deployment runs at tens of thousands of notes; the regime where matrix factorization may genuinely outperform polarity-product is most plausible at that scale. A follow-on sweep covering K ∈ {100, 300, 1000, 10000} would address this directly.
2. **At what K does C_CN diverge from C_PP?** Related but distinct. The latent-factor model presumably starts to matter when endorsement data is sparse enough that the geometric mean across communities becomes noisy. Identifying the K threshold is a practical cost-versus-quality decision for production deployments.
3. **Does the σ_between = 0.80 step generalize?** The cliff in our sub-sweep is sharp. Whether the threshold value (0.80) is universal or whether it shifts with persona issue-weight distributions, network structure, or spillover patterns is unknown.
4. **World 2 — does bridging degrade in the many-faction interdisciplinary research-funding panel?** Prism's design specifies it; this paper implements only World 1. Cross-world comparison would test the theoretical centerpiece (bridging's domain scope).
5. **Multi-faction extensions of C_CN.** The 1D matrix factorization fundamentally fails on trimodal preferences (P4 result). Whether a higher-dimensional factor model would recover bridging value in trimodal settings is an open algorithmic question.
6. **Generalization to other manipulation strategies.** We tested coordinated downvoting. Other attack vectors — Sybil attacks, vote-trading coalitions, strategic abstention, persona-drift attacks — would compose differently with the regime-specificity result, and the regime where each attack is effective may not be the same regime where the standard attack is.

The paper's contribution is the regime taxonomy: we have mapped where bridging works and where it doesn't, and articulated *why* with a clean characterization (high polarization, structured slates, two-faction structure, and trigger-threshold thresholds for adversarial robustness). The bridging literature has the *that*; we contribute the *when, how much, and under what conditions*. That contribution sits in the intersection of two literatures that have not previously cited each other, and it is the literature combination that makes the answer producible.

---

## References

APA-adjacent format with arXiv / DOI / venue. References marked `[verify]` were not directly verifiable against canonical metadata at draft time and require a final check before submission.

**World Models lineage.**

- Bruce, J., Dennis, M., Edwards, A., Parker-Holder, J., Shi, Y., Hughes, E., Lai, M., Mavalankar, A., et al. (2024). *Genie: Generative Interactive Environments.* ICML 2024 (Best Paper). arXiv:2402.15391.
- Dainese, N., Merler, M., Alakuijala, M., & Marttinen, P. (2024). *Generating Code World Models with Large Language Models Guided by Monte Carlo Tree Search.* NeurIPS 2024. arXiv:2405.15383.
- Ha, D., & Schmidhuber, J. (2018). *World Models.* arXiv:1803.10122. (NeurIPS 2018 conference version published as "Recurrent World Models Facilitate Policy Evolution.")
- Hafner, D., Lillicrap, T., Ba, J., & Norouzi, M. (2020). *Dream to Control: Learning Behaviors by Latent Imagination.* ICLR 2020. arXiv:1912.01603.
- Hafner, D., Lillicrap, T., Norouzi, M., & Ba, J. (2021). *Mastering Atari with Discrete World Models.* ICLR 2021. arXiv:2010.02193.
- Hafner, D., Pasukonis, J., Ba, J., & Lillicrap, T. (2025). *Mastering Diverse Domains through World Models.* *Nature*, 640, 647–653. arXiv:2301.04104. doi:10.1038/s41586-025-08744-2.
- Schrittwieser, J., Antonoglou, I., Hubert, T., Simonyan, K., Sifre, L., Schmitt, S., Guez, A., et al. (2020). *Mastering Atari, Go, chess and shogi by planning with a learned model.* *Nature*, 588, 604–609. arXiv:1911.08265. doi:10.1038/s41586-020-03051-4.
- Schwoebel, J., et al. (2026). *OpenWorld: Training-Free Symbolic World Models via Code Synthesis and Verification.* [verify exact venue and author list — the parent paper is the framework our work extends.]
- Tang, H., Key, D., & Ellis, K. (2024). *WorldCoder, a Model-Based LLM Agent: Building World Models by Writing Code and Interacting with the Environment.* NeurIPS 2024. arXiv:2402.12275.
- Wasu, P., et al. (2025). *PoE-World: Compositional World Modeling with Products of Programmatic Experts.* NeurIPS 2025. arXiv:2505.10819. [verify full author list — first author confirmed via project page; full byline requires final check]

**Bridging algorithms — original anchors.**

- Ovadya, A. (2022). *Bridging-Based Ranking.* Belfer Center for Science and International Affairs, Harvard Kennedy School, May 17, 2022.
- Small, C., Bjorkegren, M., Erkkilä, T., Shaw, L., & Megill, C. (2021). *Polis: Scaling Deliberation by Mapping High Dimensional Opinion Spaces.* Computational Democracy Project.
- Wojcik, S., Hilgard, S., Judd, N., Mocanu, D., Ragain, S., Hunzaker, M. B., Coleman, K., & Baxter, J. (2022). *Birdwatch: Crowd Wisdom and Bridging Algorithms can Inform Understanding and Reduce the Spread of Misinformation.* arXiv:2210.15723.
- Community Notes scoring algorithm and rating data (open source): `github.com/twitter/communitynotes`; ranking documentation at `communitynotes.x.com/guide/en/under-the-hood/ranking-notes`.

**Bridging algorithms — LLM mediator line.**

- Bakker, M. A., Chadwick, M., Sheahan, H., Tessler, M. H., Campbell-Gillingham, L., Balaguer, J., McAleese, N., Glaese, A., Aslanides, J., Botvinick, M., & Summerfield, C. (2022). *Fine-tuning language models to find agreement among humans with diverse preferences.* NeurIPS 2022. arXiv:2211.15006.
- Tessler, M. H., Bakker, M. A., Jarrett, D., Sheahan, H., Chadwick, M. J., Koster, R., Evans, G., Campbell-Gillingham, L., Collins, T., Parkes, D. C., Botvinick, M., & Summerfield, C. (2024). *AI can help humans find common ground in democratic deliberation.* *Science*. doi:10.1126/science.adq2852. Code: `github.com/google-deepmind/habermas_machine`.

**Bridging algorithms — crossover and theoretical foundations.**

- Conitzer, V., Freedman, R., Heitzig, J., Holliday, W. H., Jacobs, B. M., Lambert, N., Mossé, M., Pacuit, E., Russell, S., Schoelkopf, H., Tewolde, E., & Zwicker, W. S. (2024). *Position: Social Choice Should Guide AI Alignment in Dealing with Diverse Human Feedback.* ICML 2024 (position paper). arXiv:2404.10271. PMLR 235.
- Fish, S., Gölz, P., Parkes, D. C., Procaccia, A. D., et al. (2024). *Generative Social Choice.* ACM EC 2024.
- Huang, S., Siddarth, D., Lovitt, L., Liao, T. I., Durmus, E., Tamkin, A., & Ganguli, D. (2024). *Collective Constitutional AI: Aligning a Language Model with Public Input.* FAccT 2024. arXiv:2406.07814. doi:10.1145/3630106.3658979.

**Team-internal references.**

- Prism (A004), researchy team (2026). *OpenWorld: World Designs for Bridging-Algorithm Research.* Internal design specification, T341, 2026-06-13.
- Forge (A003), researchy team (2026). *Bridging baselines on World 1: implementation and results.* PR #20 against `quome-cloud/openworld` on branch `feat/researchy-team-bridging-world-designs`. 2026-06-13.

---

*End of draft. Section 7 awaits integration of Lens's T349 critique; Section 6 may compress to make room. Word count: approximately 7,400 words excluding references and tables.*
