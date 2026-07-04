# Peer Review: Hybrid Perceptual World Models: World-Time Compute for Interactive Reasoning (ARC-AGI-3)

**Reviewer:** Fable (claude-fable-5)
**Date:** 2026-07-04
**Venue:** Internal Review — botXiv / OpenWorld Project
**Materials reviewed:** `papers/arc-3/main.tex` (1461 lines, branch `arc3-beats-sota`), `papers/arc-3/arc3_numbers.tex`

---

## Summary

The paper ports the OpenWorld verified-code world-model recipe to ARC-AGI-3 and argues, primarily through an extensive battery of negative results, that the benchmark's bottleneck is not dynamics modeling but *goal discovery*: wins are "goal-as-procedure" (an ordered protocol) rather than "goal-as-state," so a family of principled automated goal-inference methods (atomic objectives, frontier-LLM hypotheses, Bayesian sub-worlds, engine reconstruction, a formal solving cycle, spectral exploration) solves zero hard games even on fidelity-1.0 models. A live coding agent (Claude Code) that writes, reasons the win condition for, and replay-verifies its own world model crosses these walls, yielding 16/25 games completed source-free (158/183 levels, 89.4 per-game RHAE) — claimed to beat the prior SOTA baseline1 (15/25, 145 levels, 58.12 RHAE) on games, levels, and efficiency — and 24/25 source-assisted as an upper bound. The paper reports unusually strong reproducibility machinery (deterministic replay verification, one-command re-verification, per-run leakage audits, OpenWorld round-trips) and consolidates its limitations candidly. However, the headline SOTA claim rests on an N=1-per-game, offline, unbounded-reset protocol, a one-game margin with no statistical treatment, a self-described "fresh" result from a sweep that was "still finishing," and an efficiency comparison whose protocol comparability to baseline1 is not established.

## Strengths

- **The goal-as-procedure diagnosis is a genuine conceptual contribution.** Locating the failure of automated ARC-AGI-3 solvers at goal inference rather than dynamics modeling, and supporting it with the fidelity-1.0 control (methods fail *even on a perfect model*, §5, App. B), is the kind of mechanism-isolating experiment design the field needs. The forming-vs-verifiable world-model asymmetry (Fig. `world_model_forming`, lines 485–497) is a crisp, memorable articulation.

- **Negative results as first-class citizens.** The strategy landscape (§6, Fig. heatmap) reporting a dozen strategies including six-plus principled failures (E102–E104, E127, E130, E135) is rare and valuable. The E127 shared-prior-bias diagnostic (two frontier models agreeing with each other more than with reality, lines 845–851) is a publishable observation in its own right.

- **Reproducibility machinery is well above field norm.** Deterministic replay verification of every banked solve, a one-command re-verification script (`bench/verify_arc3_solves.sh`), a single asset-regeneration pipeline (`build_arc3_fullgame.py → e120_rhae.py → e121_openworld_roundtrip.py → make_arc3_assets.py`, lines 613–634), and scoring with the engine's *own* `EnvironmentScoreCalculator` rather than a re-implementation are all commendable.

- **The random-play baseline is honest and self-deflating in the right way.** Reporting that 11/25 first levels are reachable by random play under the offline protocol, that one agent "solve" (lp85) is no better than random, and preferring "16/25 above random" over the inflated ≥1-level union (§8, lines 533–548) meaningfully raises the paper's credibility.

- **The two-protocol design (source-free vs. source-assisted) with an audited, quantified gap** is a real methodological advance over baseline1's asserted leakage audit, and the framing of the 8-game gap as "the measured cost of discovery" (lines 344–351) is clean. Catching their own agent reading source via telemetry is exactly the right kind of audit.

- **The limitations section (§9) is unusually complete** — offline/unbounded-resets, N=1, training contamination, uncontrolled cost, and the fresh-result caveat are all stated plainly rather than buried.

- **Model portability check.** The Claude-vs-Codex source-assisted head-to-head with identical per-game levels on 24/25 games (App. A, lines 657–666) is a useful robustness result, and the paper correctly flags the source-free Claude-vs-Codex comparison as confounded (lines 709–716).

## Weaknesses and Concerns

Ranked by severity.

1. **(Critical) The headline SOTA claim is statistically and procedurally fragile.** The margin over baseline1 is one game (16 vs. 15) out of 25, with N=1 per game and no variance estimate (Limitation 3, lines 565–568). No statistical treatment is offered; a two-proportion or paired-by-game comparison at this margin is nowhere near significant. Worse, Limitation 5 (lines 576–586) discloses that an *earlier run of the same recipe lost to baseline1*, and the winning number comes from a re-run after fixing "harness/budget confounds," with the sweep "still finishing, so numbers may be slightly revised." Re-running a stochastic agent until it clears a competitor by one game, then headlining the result in the abstract and title-adjacent claims ("beats SOTA on all three axes," lines 182, 237–239), is a garden-of-forking-paths risk the paper does not confront. The confound-fixing is asymmetric by construction: the authors' arm was optimized (crash fixed, budget uncapped, max reasoning, retry-resilient) while baseline1's number is taken as-reported. At minimum the paper needs (a) multiple independent seeds/sessions per game with a success-rate distribution, (b) the final (not in-flight) sweep, and (c) a pre-registered or at least clearly stated stopping rule.

2. **(Critical) The efficiency comparison to baseline1 is not established as like-for-like, yet it is called "the decisive axis."** The 89.4 vs. 58.12 per-game RHAE comparison (§A.1, lines 769–799) scores the authors' *executed, precomputed, replay-verified* plans — after offline search with unbounded resets — against human baselines, while Limitation 1 (lines 553–559) concedes "an explicit live RHAE score would be lower" and calls action economy "our weak axis." The paper cannot simultaneously headline a ~54% efficiency win in the abstract (line 183) and concede in the limitations that its live efficiency would be lower. Whether baseline1's 58.12 includes in-run exploration actions (which would make the comparison strongly biased in the authors' favor) is never resolved. The per-level source-free score of 113.3 — *above* the human cap of 100 on replayed plans — is exactly the signature of this protocol mismatch. The honest caveat at lines 801–807 belongs in the abstract sentence making the claim, and the "confirming live run" (acknowledged as a next step) is arguably a prerequisite for the SOTA-efficiency claim, not a follow-up.

3. **(Major) Aggregation asymmetry in the headline source-free number.** Lines 709–716 disclose that Claude's source-free figure is a *cumulative archive* — "multiple methods over many best-keeper rounds" — and correctly refuse to compare it to a single Codex sweep (60 levels) because "the gap conflates capability with pipeline maturity." But the same logic applies to the headline comparison: baseline1's 15/25 appears to be a single-system result, while the 16/25 aggregates the best outcome per game across many methods and rounds. If the headline number is a best-of-union over methods and attempts, the abstract must say so, and the fair comparison is either single-sweep-vs-single-sweep or an explicitly matched-compute protocol.

4. **(Major) "Modeling is largely solved" (§4, lines 386–403) overstates the evidence.** What is shown: (i) replay-determinism is 1.00, so dynamics are code-expressible *in principle*; (ii) Claude's synthesized `predict()` reaches 49% mean held-out exact-match with only 3/21 near-perfect models, and fidelity 0 on the densest game (cn04, line 913). Meanwhile E87 (lines 944–952) shows planning is *fidelity-gated* — "partial models do not yet support reliable multi-step control." So for ~18/21 games, automated synthesis produces models through which one cannot reliably plan. "Modeling is solvable by a capable reasoning agent per-game" is supported; "modeling is largely solved" is not, on the paper's own numbers. Note also that "100% determinism" and 49% fidelity are both *single-step* quantities (lines 402, 914–916); the abstract should carry that qualifier.

5. **(Major) The goal-as-procedure/goal-as-state dichotomy is not engaged with its own literature and is formally leaky.** Any "ordered procedure" objective is expressible as a state-based objective over a history- or automaton-augmented state space — this is the entire premise of reward machines (Icarte et al., ICML 2018), LTL-conditioned RL, and options/temporally-extended goals. The paper's negative results show that goals scored over the *raw single frame* fail; they do not show that goal-as-state fails in general, because none of the tested methods search over product-automaton or memory-augmented goal spaces. The diagnosis may well survive this test (the hypothesis space is enormous), but as written the central claim ("a goal scored over a single state cannot rank an ordered procedure," lines 173–174) conflates a representational choice with an impossibility, and the related-work engagement is limited to ARC-AGI-3 systems — no reward machines, no LTL goals, no IRL/goal-inference literature, and no code-world-model prior art (WorldCoder, theory-based RL/EMPA, AutumnSynth), all directly relevant.

6. **(Major) Training-data contamination undercuts the "source-free" framing more than the paper allows.** Limitation 4 (lines 569–575) is candid that the public games may sit in Claude's weights, and the intro itself notes a memorized game is "exactly the 'memorized pattern' the benchmark seeks to exclude" (lines 213–215). But the abstract and headline nonetheless brand 16/25 as the "benchmark-intended" source-free result. The strongest supportable claim is "no runtime source access (audited)" — the paper says this once, in the limitations, and should say it wherever the source-free number is headlined. The proposed contamination control (a model with cutoff predating game release) is the right direction; even a cheap proxy — e.g., prompting the model for game code/win conditions by game ID before play — would materially strengthen the claim.

7. **(Moderate) The winning method is under-characterized.** The live coding agent (§7, App. G) is "Claude Code, one agentic session per game, unbounded offline budget." There is no ablation of *what the agent does that matters* (does it actually use the synthesized `predict()` model? how often does it fall back to determinism-exploiting sweeps? what is the distribution of session lengths/actions?), no failure analysis on the 9 source-free-unsolved games, and no compute matching against the automated baselines it beats. "Reasoning crosses the wall" is asserted from outcomes; the paper concedes some solves are sweeps (lp85, ft09, lines 512–515) but does not quantify the deliberate/sweep split beyond anecdote. Given that this agent *is* the paper's positive result, it receives strikingly less experimental scrutiny than the methods that fail.

8. **(Moderate) Internal inconsistencies in per-game accounting.** The App. E per-game table (lines 1310–1342) lists "our best" as level 1 for most games (e.g., ka59 = 1, tu93 = 1), directly contradicting the full-game results (§2: 16 games *completed* source-free; App. F replay figures showing ar25 8/8, re86 8/8, sb26 8/8, tu93 presumably 9/9, etc.). The table appears to reflect a stale ≥1-level snapshot of the automated-strategy era, not the current full-game results. Similarly, the "procedural wall" count drifts: nine walls in §8 (line 545), eleven games listed in §7 (lines 507–508) under "solves 15/25," twelve in the Fig. heatmap caption (adds bp35, line 479). And s5i5 is both "a wall" (heatmap caption) and cheap-tier-solved (router table). These must be reconciled before the numbers can be trusted end-to-end — which is otherwise the paper's strongest suit.

## Detailed Technical Questions

1. **Stopping rule and run selection.** How many source-free campaign attempts (full or partial) were made in total, and how was the reported one selected? Was the decision to stop re-running made before or after observing 16 > 15? Please report all attempts, or re-run K independent sessions per game and report per-game success rates with intervals.

2. **baseline1 protocol comparability.** Does baseline1's 15/25 (and 58.12 RHAE) arise under offline unbounded resets like yours, or a live/limited-reset protocol? Does its RHAE include exploration actions? Table 1 marks baseline1 "pre-patch" (line 305) — what patch, and does a post-patch number exist? Without answers, none of the three "axes" are established as commensurable.

3. **Leakage audit mechanism.** Precisely what does the per-run audit monitor — file-system access, process tree, network, prompt contents? What happened in the disclosed incident where "we catch our own agent reading source" (line 273)? Was that run discarded and the game re-attempted, and how many runs were discarded overall?

4. **Contamination probe.** Before play, can you query the agent's model for each game ID's mechanics/win condition (and for the game source itself) and report recall? A model that can recite `dc22.py` from its weights makes "source-free" a misnomer for that game regardless of runtime audits.

5. **Goal-as-state over augmented states.** Did any tested goal-discovery method search over history-augmented or automaton-structured goal spaces (e.g., reward machines, sequences of sub-goal predicates with memory)? E124's "ordered list of subgoal predicates" (lines 1219–1222) is exactly this and is only a design — doesn't its existence concede that the dichotomy is about hypothesis-space choice rather than a fundamental limit of state-based scoring?

6. **Agent mechanism ablation.** For the 15 agent solves: in how many did the synthesized `predict()` world model causally contribute to the solution (vs. direct environment search)? An ablation removing the world-model step from the agent harness would test whether "verified world models" — the paper's framework — or raw agentic search is doing the work. This is the single most important missing experiment: the paper's title credits hybrid world models, but the evidence could equally credit Claude Code plus determinism.

7. **RHAE accounting.** For the source-free RHAE of 89.4: are reset-and-replay actions, failed exploration episodes, and discovery actions counted in `a`, or only the final executed plan? If only the final plan, what is the total action count including discovery, and what would RHAE be under that accounting?

8. **Fidelity–solving disconnect.** Claude's mean synthesis fidelity is 49% with 3/21 near-perfect, yet the agent completes 16–24 games. What fidelity do the *agent's own* per-game world models reach on held-out transitions? If they are near-perfect where E86's one-shot/agentic synthesis is not, the capability claim in §4 needs restating around the agent, not the synthesizer.

9. **Stale table and wall counts.** Please regenerate App. E's per-game table from `arc3_fullgame.json` and give a single, definitional list of "procedural walls" (with the criterion that makes a game a wall), reconciling the 9/11/12 discrepancy and s5i5's dual classification.

10. **Unused Go-Explore results.** `arc3_numbers.tex` defines `\ArcGoExplore*` macros (13 games, 300,116 steps, gain 1) that appear nowhere in the manuscript. Was a Go-Explore experiment run and cut? Given Go-Explore is the canonical hard-exploration baseline, its result (positive or negative) belongs in the strategy landscape.

## Minor Comments

1. **Experiment-ID collisions:** "E120" is both the RHAE scorer (`e120_rhae.py`, line 619) and the committee-of-experts method (§ selfrebuild, line 1152); "E121" is both the OpenWorld round-trip (line 623) and the rule-change detector (line 1162). Renumber — as written, the reproducibility section and the methods section refer to different experiments by the same name.

2. **The abstract is ~450 words and argues with itself** ("read honestly," "not as the headline but..."). Venue abstracts need the claim, the number, and the protocol in ~150 words; move the epistemic negotiation to §1 and §9. Similarly, in-progress designs (E119, E124, E131) are framed as sections coequal with results; mark them as a "Directions" appendix or cut — they dilute an already very long paper.

3. **"World-time compute"** appears in the title but is defined only in passing (lines 1040–1043) as inference-time search that constructs the model during play. Either define it crisply in §1 and use it consistently, or drop it from the title; currently it does no work.

4. **The security reading (lines 676–689)** upgrades "solves 64×64 grid games given source" to "concrete cyber-offensive capability" — a real codebase lacks a deterministic replay oracle, a level counter, and a 7-action interface. Hedge this to a hypothesis or support it with a non-game demonstration.

5. **The critique of baseline1** (sole-authored, no ablations, asserted audit, lines 262–275) is fair but boomerangs: this paper also has no ablations of its winning method, N=1, and an uncontrolled cost estimate (~$7 from subscription proration, lines 587–592). Recommend tempering the rhetoric to match the shared limitations. Also, the AI-use declaration (lines 1440–1448) discloses Claude drafted the manuscript reporting Claude's results; a sentence on how author review guarded against favorable-framing bias would be well placed.

## Overall Assessment

**Recommendation:** Major Revision
**Confidence:** 4

The diagnostic core — goal discovery, not dynamics modeling, is the ARC-AGI-3 bottleneck, established via fidelity-1.0 controls and a disciplined battery of negative results — is novel, well-instrumented, and worth publishing, and the reproducibility engineering is exemplary. But the headline SOTA claim currently outruns its evidence on four independent grounds (one-game margin at N=1 after a re-run that reversed a loss; unverified protocol comparability on the efficiency axis; union-vs-single-system aggregation asymmetry; unmeasured training contamination), and the paper's own limitations section concedes most of this while the abstract does not. A revision that (a) reports multi-seed success rates, (b) resolves baseline1 protocol comparability or reframes the comparison as indicative, (c) ablates the world-model contribution inside the winning agent, and (d) fixes the internal accounting inconsistencies would make this a strong paper whose claims match its considerable substance.

## Scores

| Dimension | Score (1-5) |
|-----------|-------------|
| Novelty | 4 |
| Technical Quality | 2 |
| Clarity | 3 |
| Significance | 4 |
| **Overall** | **3** |

*Scoring notes: Novelty rewards the goal-as-procedure diagnosis and the audited two-protocol design; Technical Quality is held down by N=1, the post-hoc re-run, comparability gaps, and the missing agent ablation rather than by any error in what was measured; Significance assumes the diagnosis survives the augmented-goal-space test (Q5) — if it does, this reframes how the community should attack interactive-reasoning benchmarks.*
