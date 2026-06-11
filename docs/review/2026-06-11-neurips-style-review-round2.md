# Peer Review — NeurIPS style, Round 2

**Paper:** OpenWorld: Training-Free Symbolic World Models with Verified Code
Dynamics, Tunable Moral Configurations, and Agents-as-a-Judge (revision 1)

**Recommendation:** Weak accept (6/10). The revision is responsive and the
honesty of the statistical reporting is now a strength. The remaining
weaknesses are about generality and power rather than validity. A second
revision addressing R1–R4 would move this to accept.

---

## Assessment of the first revision

The authors did what was asked, and reported results that *weakened* their
own narrative where the data said so (judge margin non-significant at n=60;
rubric-paraphrase correlation marginal; pass@1 cost of high-temperature
sampling). E12's memorization finding (1-NN: 8/8 on-policy rollouts, 30%
probe accuracy) is a genuinely useful methodological observation. E11's
24/24-vs-0/24 with CIs now adequately supports the central claim *for this
model family*.

## Remaining weaknesses

- **R1 — Single model family (carried over, now the top issue).** Every
  generator, judge, critic, and proposer is Qwen2.5. The synthesis-reliability
  claim ("local models write verified, near-rule-perfect dynamics") is a claim
  about local LLMs, supported by one vendor's models. Replicate E2-style
  synthesis (verified acceptance + ground-truth probe accuracy + wall-clock
  cost) with at least two other model families at comparable scale (e.g.,
  Llama-3.1-8B, Gemma-2-9B). If cross-family results hold, the headline
  generalizes; if they don't, that's essential information.

- **R2 — The judge comparison remains underpowered.** The authors honestly
  report McNemar p=0.289 at 60 rounds and "a larger benchmark is needed."
  Then run one: the protocol is cheap and paired. Triple the proposer seeds
  (120+ pooled rounds) and report the pooled discordant counts. Either the
  margin sharpens into significance or the effect estimate shrinks --- both
  outcomes are informative and publishable.

- **R3 — The verification ablation conflates filtering with repair.** The
  "full gate" condition both *rejects* bad candidates and *feeds back*
  verifier errors for regeneration (up to 4 iterations). These are different
  mechanisms with different costs. Ablate the loop: full gate with
  max_iters=1 (filter only — note this can reject all attempts) versus
  max_iters=4 (filter + repair), paired seeds.

- **R4 — OOD is a single axis at a single magnitude.** 10× state scaling is
  one shift. Show a scale ladder (1×/10×/100×) — for the symbolic engine this
  is nearly free, and for the trained baselines it tests whether degradation
  is graceful or catastrophic. Additionally, the E12 MLP is plausibly
  under-tuned (full-batch, fixed lr, absolute-state target). Offer a stronger
  variant (e.g., delta-state target, larger capacity, longer training) so the
  sample-efficiency claim is against a defensible learned baseline, not a
  strawman.

- **R5 (minor) — Benchmark contamination.** The 20 defect archetypes are
  classic; the repair agent has plausibly seen all of them in pretraining.
  This does not undermine the world-model claims (the environment is exact
  regardless), but agent solve rates should not be read as debugging-skill
  measurements. State this.

- **R6 (nit) — Experiment numbering skips E14**; either renumber or note the
  gap. Synthesis wall-clock cost is asserted ("a few seconds") but not
  reported as data; record it in the cross-family replication.

## Requested experiments

| Item | Experiment |
|---|---|
| R1 | **E16**: cross-family synthesis (Llama-3.1-8B, Gemma-2-9B) × 3 worlds × 3 seeds, with acceptance, probe accuracy, iterations, and wall-time |
| R2 | **E17**: extend E13 to 6 proposer seeds (120 pooled rounds), pooled McNemar + bias audit |
| R3 | **E18**: repair-loop ablation, max_iters 1 vs 4 under the full gate |
| R4 | **E19**: scale ladder 1×/10×/100× for code/MLP/1-NN/LLM engines; stronger delta-MLP baseline |
| R5, R6 | text changes (contamination caveat; numbering note; report synthesis cost from E16) |

---

*Round-2 review produced as part of the project's red-team process; revision 2
addresses each item below.*
