# Peer Review — NeurIPS style, Round 3

**Paper:** OpenWorld: Training-Free Symbolic World Models with Verified Code
Dynamics, Tunable Moral Configurations, and Agents-as-a-Judge (revision 2)

**Recommendation:** Accept conditional on scope honesty (7/10). Validity and
generality concerns from rounds 1–2 are resolved; what remains is the gap
between what is measured (simulator fidelity on small deterministic worlds)
and what world models are *for* (planning in rich, uncertain environments).
The paper should either demonstrate the next ring of that gap or shrink its
framing to match.

---

## Assessment of revision 2

The cross-family replication (Gemma-2-9B rule-perfect on all compiles) and
the scale ladder with a strengthened learned baseline close R1 and R4
convincingly. The pooled judge analysis (p=0.078, reported as estimated) and
the filter-vs-repair decomposition are model examples of honest mechanism
reporting. The repository's two-defeats-of-the-timeout story in the
limitations is the kind of operational candor reviewers rarely get.

## Remaining weaknesses

- **Q1 — World complexity is never varied.** The three instrumented worlds
  have 4–7 state fields and 2–4 rules with simple arithmetic. The synthesis
  claim is therefore supported only at toy scale, and nothing in the paper
  indicates whether quality degrades gracefully or collapses as declared
  rules multiply and interact. Required: a parametric complexity stress
  test --- synthesize worlds of increasing rule count/interaction (e.g., 4,
  8, 12, 16 coupled rules) under a fixed protocol and plot probe accuracy
  versus complexity. Where is the cliff?

- **Q2 — Determinism is an undisclosed scope restriction.** The sandbox
  forbids randomness and the generator prompt demands deterministic code, so
  stochastic environments --- most of the world --- are out of scope *by
  construction*, and the paper never says so. Either state the restriction
  prominently, or better: extend the paradigm (seeded RNG threaded through
  state preserves replayability and verification) and test whether a local
  model can synthesize dynamics whose *distributions* match declared
  probabilities (total-variation distance against a stochastic oracle over
  thousands of seeded transitions).

- **Q3 — Fidelity is measured; planning utility is not.** World models exist
  to improve decisions. The paper never runs a planner. Demonstrate the
  payoff chain: lookahead planning through the synthesized simulator should
  beat a reactive policy on task score (executed in the ground-truth
  environment), and planning through the LLM next-state engine should be
  both slower and worse --- connecting the fidelity and throughput tables to
  the quantity that matters.

- **Q4 — No oracle-free error signal.** Probe accuracy requires the
  hand-written oracle the practitioner does not have. The 3B results (0.87)
  show accepted-but-wrong dynamics occur; users need a diagnostic that does
  not presume ground truth. Evaluate self-consistency: independently
  synthesized programs disagree somewhere --- does cross-program disagreement
  on probe states *predict* ground-truth error (precision/recall)? The stored
  artifacts from E2/E16 make this nearly free.

- **Q5 (minor) — Judge cost accounting.** Judge-of-3 spends ~5 model calls
  per round versus 1 for the baseline. Report the comparison cost-normalized
  so the marginal-significance lift is priced.

- **Q6 (nit) — State the quantization/runtime dependence** of all
  LLM-dependent numbers (Ollama version, Q4_K_M quantization).

## Requested experiments

| Item | Experiment |
|---|---|
| Q1 | **E20**: parametric rule-complexity ladder, accuracy vs. R |
| Q2 | **E21**: seeded-RNG support + stochastic-world synthesis with distributional verification |
| Q3 | **E22**: lookahead planning through code/LLM/none, executed in the oracle |
| Q4 | **E23**: cross-program disagreement as an oracle-free error detector |
| Q5, Q6 | text: cost-normalized judge table; runtime/quantization statement |

---

*Round-3 review produced as part of the project's red-team process; revision 3
addresses each item below.*
