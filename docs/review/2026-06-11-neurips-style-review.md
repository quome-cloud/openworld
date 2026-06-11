# Peer Review — NeurIPS style

**Paper:** OpenWorld: Training-Free Symbolic World Models with Verified Code
Dynamics, Tunable Moral Configurations, and Agents-as-a-Judge

**Recommendation:** Borderline reject (4/10) in current form; the system and
direction are strong, but the empirical claims outrun the statistical support.
A revision that addresses W1–W6 below could plausibly reach accept.

---

## Summary

The paper presents a zero-dependency Python framework operationalizing the
Code World Model paradigm on local LLMs: declared symbolic worlds, an LLM
plan–generate–verify relay producing executable transition code, dial-weighted
"open" objectives, a configuration tuner, and an LLM judge for behavior
selection. Ten experiments on Qwen2.5 7B/3B/1.5B compare verified synthesized
dynamics against per-step LLM next-state prediction, ablate verification
gates, evaluate program repair with judge-based candidate selection, and
demonstrate dial-swept Pareto frontiers.

## Strengths

- **S1.** Clear, falsifiable central claim (compounding error eliminated by
  construction) with a clean ground-truth-instrumented methodology.
- **S2.** Genuine end-to-end reproducibility: one script regenerates every
  number, figure, and table; raw JSON committed.
- **S3.** Honest reporting culture: pilot saturations and post-hoc protocol
  changes are disclosed rather than hidden; judge accuracy is audited, not
  assumed.
- **S4.** The coding world is a smart choice of flagship: dynamics are
  *literally* test execution, so the simulator is exact by construction.

## Weaknesses

- **W1 — Sample sizes are far too small for the strength of the claims.**
  E1 rests on **n=3** rollouts on **one** world. The abstract's flagship
  sentence ("bit-exact over every rollout") is supported by three action
  scripts. E7's correlation uses 12 points with no significance test. Nothing
  in the paper carries a p-value or a paired test.

- **W2 — No actually-*learned* baseline.** The "learned-style" baseline is an
  LLM predicting next states zero-shot. Whatever its merits as a structural
  proxy, it is not a learned dynamics model: it has never seen a transition
  from the environment. The learned-vs-symbolic framing demands at least one
  model *trained on environment transitions* (even a small MLP or k-NN
  regressor trained on oracle rollouts), with a sample-efficiency curve
  (accuracy vs. number of training transitions). Without it, the central
  comparison is symbolic-vs-prompted, not symbolic-vs-learned.

- **W3 — The judge experiment conflates two effects.** E6 changes *two*
  variables relative to E5: (i) best-of-3 high-temperature sampling
  (diversity) and (ii) judge selection. The observed lift could come entirely
  from (i). Required controls, on shared candidate sets: **random-of-3**
  (isolates diversity), **oracle-of-3** (ceiling), and a paired statistical
  test (McNemar) of judge vs. random. LLM judges also have documented
  position bias; no order-randomization audit is reported.

- **W4 — Single world for the headline fidelity results.** E1/E10 use only
  the sprint world. The synthesized-code claim should replicate across all
  three instrumented worlds with the same protocol.

- **W5 — Benchmark size.** Ten repair tasks with n=10 Wilson intervals
  spanning ~35 points cannot support "lifts solve rates from 70% to 90%."
  The benchmark needs to grow (even to 20 tasks) and/or evaluations need
  multiple seeds per task, with intervals reported in the abstract.

- **W6 — Judge-alignment robustness.** E7 uses a single rubric phrasing and
  12 deterministic episodes. Report at minimum: a second rubric paraphrase
  (wording sensitivity), and a permutation p-value for the Spearman
  correlation.

- **W7 (minor).** The 47,087× speed ratio is an artifact of comparing native
  code to network-served LLM inference; fine to report, but should be framed
  per-device. The "bit-exact" terminology should be confined to worlds where
  the accepted program was verified perfect on probes; E2 shows 3B-accepted
  code is *not* generally rule-perfect.

## Questions for the authors

1. How much of the E6 lift survives against random-of-3? (W3)
2. What does a trained MLP need — how many transitions — to match the
   synthesized program's probe accuracy in-distribution, and does it transfer
   at 10×? (W2)
3. Does first-divergence replicate on orchard and triage? (W4)
4. Is the judge order-consistent when candidates are reversed? (W3)

## Requested experiments (mapped to revision plan)

| Review item | Experiment |
|---|---|
| W1, W4 | **E11**: fidelity with n=8 scripts × 3 worlds (24 rollouts), CIs |
| W2 | **E12**: MLP dynamics trained on K∈{100, 1k, 10k} oracle transitions; probe/rollout/OOD comparison and sample-efficiency curve |
| W3 | **E13**: paired first/random/judge/oracle selection on shared candidates; McNemar; position-bias audit |
| W5 | benchmark expanded to 20 tasks; E5/E6 rerun |
| W6 | **E15**: second rubric + permutation test |
| W1 | paired tests and CIs added throughout the Results section |

---

*Review produced as part of the project's own red-team process; the revision
below addresses each item with new committed experiments.*
