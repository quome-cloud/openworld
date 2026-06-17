# Reachable-OOD audit + corrected Pillar-1 narrative (2026-06-16)

## TL;DR

The paper's OOD differentiation rests **pervasively** on a ×10 (and 10×/100×) magnitude-scaling
probe. E66 shows that probe is **off-manifold** — it scales every state field by 10, producing states
the worlds never visit (e.g. sprint's invariant `shipped + backlog = 12` is violated). On a **held-out
reachable region** instead, the "learned models collapse to zero OOD" claim is **only true for
memorizers**; parametric learners partially generalize (Koopman 0.42–0.84). Verified code's exactness is
unaffected. The blanket "statistical learning collapses to zero OOD" claim is **overstated as written**
and needs scoping before it ships.

This is NOT a fabrication problem — the ×10 numbers are real. It's a **construct-validity** problem:
×10 measures "can't handle impossible states," which overstates the generalization gap.

## Corrected Pillar-1 paragraph (drop-in replacement for the OOD claim)

> Across the reachable state space, no learned dynamics model is exact: 1-nearest-neighbour and tabular
> memorizers reproduce trained states (≈0.99) but fail to generalize to held-out reachable states
> (0.00–0.17), while parametric learners partially recover the dynamics — a degree-2 Koopman lift attains
> 0.84/0.57 interpolation/extrapolation accuracy on sprint and 0.58/0.42 on triage — yet none reach
> exactness and all degrade with distance from the trained region. Verified synthesized code, by
> contrast, is exact and constant (1.0) across the entire reachable manifold at zero training cost. A
> naive ×10-magnitude OOD probe drives every learned model to 0.00, but those scaled states lie off the
> reachable manifold (they violate the worlds' own invariants), so that figure measures extrapolation to
> impossible states rather than failure to generalize the dynamics; we therefore report OOD on a held-out
> region of the reachable state space (Table~\ref{tab:reachable-ood}).

## Stale spots to fix in `paper/` (×10-based OOD claims)

`numbers.tex`:
- `\MLPOODTenK` = `0%` — the headline "MLP collapses to 0% OOD" macro. On reachable OOD, MLP = 0.12–0.61,
  not 0. Either rescope to "×10 magnitude OOD" or replace with reachable-OOD numbers.
- `\CodeOODRate` = `100%`, `\LLMOODRate` = `40%` — fine for the ×10 probe, but the surrounding prose
  treats 100-vs-0 as the generalization story; reframe as reachable-OOD where the gap is exact-vs-approximate.

`main.tex` (each asserts learned-model OOD collapse on the ×10/10×/100× probe):
- L159–162  "MLP … compound error and collapse out of distribution"
- L300–304  "compounding-error failure mode of learned world models is eliminated"
- L315–317  "10× out-of-distribution probes at \CodeOODRate"
- L631      "ten 10× out-of-distribution probes"
- L685–696  E19 ladder "collapses identically out of distribution … 1×/10×/100×" (ladder.tex, wired at L2393)
- L716–726  E37/E38 induction "learned baselines fit magnitudes and collapse to zero OOD on every replicate"
- L743–744  "the learned baselines collapse out of distribution"
- L757–773  "scale-invariant (in-dist = OOD)"

## Scope caveats (don't overclaim the correction either)

- E66 covered **sprint + triage**, n=5 seeds, run locally. **Orchard not re-tested** (nested-dict state).
- E66 tested the **classical learned baselines** (1-NN/tabular/linear/Koopman/MLP). The **induction
  experiments (E37/E38)** compare LLM rule-induction vs MLP/1-NN — a different setup E66 did not rerun.
  Their "collapse to zero OOD" claims use the **same ×10 probe**, so they warrant the **same reachable-OOD
  recheck** before being trusted. Recommend rerunning E12/E37/E38 with the E66 probe.
- Koopman (deg-2 lift) is **not currently a baseline in the paper**. Its on-manifold generalization
  (0.42–0.84) is the strongest reason the "statistical learning collapses OOD" framing is too strong —
  consider adding it as a baseline so the comparison is against a competent learner, not just MLP/1-NN.

## INDUCTION RECHECK (E37b, 2026-06-16) — splits the OOD claim: interpolation vs extrapolation

Reran the E37 induction pipeline (qwen2.5:7b code-induction + E12 MLP/1-NN) on the reachable
train-region split (sprint, K=[100,1000]×2 reps). Aggregate (in-region / interp-OOD / extrap-OOD /
in-dist-orig / ×10-old):

| model | in | interp | extrap | in-dist | ×10 |
|---|---|---|---|---|---|
| code (induced) | 0.72 | 0.73 | 0.51 | 0.45 | 0.45 |
| MLP | 0.80 | **0.80** | 0.07 | 0.33 | 0.00 |
| 1-NN | 0.79 | 0.29 | 0.00 | 0.15 | 0.00 |

**The OOD claim splits in two and they land differently:**
1. **Interpolation → "collapse to zero OOD" is an artifact.** MLP = 0.0 on ×10 but **0.80 on reachable
   interpolation.** Learned models generalize fine to unseen on-manifold states. Soften the claim.
2. **Extrapolation → the claim HOLDS.** On the genuinely-unseen reachable region (shipped past median),
   MLP collapses (0.07), 1-NN fully (0.0); induced code degrades gracefully (0.51), still beating both.
   Code is scale-invariant (in-dist 0.45 = ×10 0.45) — the paper's structural point survives.

**Honest fix = precision, not retraction:** the differentiator is **extrapolation**, not "OOD" broadly.
Relabel the ×10 probe "extreme extrapolation," add an interpolation column, and the claim becomes "learned
models fail to *extrapolate*; code extrapolates by construction" — accurate and defensible.

⚠️ **Confound (don't trust the induced-code magnitudes here):** the LLM induction was DEGRADED this run —
train-reproduction only 0.55–0.81, not the ≈1.0 the paper's setup achieves (restricting training to the
held-out region thinned the induction examples). So the induced-code numbers are a **lower bound**. The
MLP/1-NN OOD numbers are solid; the induced-code numbers need a clean rerun holding induction quality
fixed (sample induction examples from the full manifold but score on a held-out reachable-extrapolation
region, or gate on repro=1.0) BEFORE any induced-code number ships. Result: `results/e37b_reachable_induction.json`.

## CLEAN INDUCTION RUN (E37c, 2026-06-16) — extrapolation claim supported; 7b can't hit repro=1.0

Reran with branch-covering examples + a repro=1.0 acceptance gate (8 attempts × 5 seeds = 40/replicate),
sprint, K=1000×3. **The gate was never reached: 0/3** (best repro 0.93/0.82/0.95). qwen2.5:7b cannot
induce a fully-correct sprint program from held-out-region data (residual miss = the post-ship
`bugs += debt//4` compound rule). Aggregate (in / interp / extrap / in-dist / ×10):

| model | in | interp | extrap | in-dist | ×10 |
|---|---|---|---|---|---|
| code (induced) | 0.95 | 0.90 | **0.70** | 0.47 | 0.47 |
| MLP | 0.83 | 0.74 | 0.10 | 0.47 | 0.00 |
| 1-NN | 0.97 | 0.25 | 0.005 | 0.23 | 0.00 |

1. **Extrapolation claim strongly supported.** Even imperfect induced code extrapolates 0.70 vs MLP 0.10,
   1-NN 0.005. Clear repro→extrap correlation across replicates (0.93→0.625, 0.82→0.625, 0.95→0.858),
   trending to ≈1.0 at repro=1.0. The e37b 0.51 lower bound is now 0.70 and rising with induction quality.
2. **Interpolation artifact reconfirmed** (MLP ×10 0.0 / interp 0.74).
3. **Code scale-invariant** (in-dist 0.47 = ×10 0.47 exactly) — holds even for imperfect induction.

⚠️ **New caveat for the paper:** E37 says induction is "accepted only if it reproduces the observed
transitions" (implies repro=1.0), but qwen2.5:7b reached that **0/3** on held-out-region data (tops ~0.95).
So induction success is **model/data-dependent**; the paper likely used full-manifold induction data or a
stronger model. **Next step to nail the clean repro=1.0 point: rerun the inducer with qwen2.5:14b**
(installed locally). Result: `results/e37c_clean_induction.json`.

## Honest net

The verified-code value survives intact: **exact + zero-data + flat across the reachable manifold**.
What does NOT survive is "learned world models collapse to zero OOD" as a blanket claim. The corrected,
two-part version: learned models **interpolate fine within the reachable manifold but fail to extrapolate
to unseen regions**; verified/induced code extrapolates by construction. So the real advantage is
**exact-and-extrapolating vs approximate-and-interpolation-bound** — a real, publishable result, just not
the 100-vs-0 cliff the ×10 probe implied. The ×10 probe should be relabeled extreme-extrapolation and
paired with an interpolation column throughout (numbers.tex `\MLPOODTenK` + the ~8 main.tex spots).
