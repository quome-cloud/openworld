# Note to Jim — OOD methodology finding (2026-06-16)

Hey Jim — while going over the world-model results I hit a methodology issue in the OOD evaluation that
affects a central claim, so flagging it early with a fix in hand. Short version: **the thesis survives,
but the "learned world models collapse to zero OOD" framing is too strong as written and a reviewer who
re-runs with on-manifold OOD states would catch it.** Here's the issue, the corrected result, and what I'd
change.

## The issue

The OOD probe scales every state field ×10 (`SPRINT_PROBES_SCALED`, and the 10×/100× ladder). Those
scaled states are **off the reachable manifold** — e.g. sprint's `shipped + backlog = 12` invariant is
violated (shipped=70, backlog=50). A learned model scoring 0% on states the world can never produce
measures "can't handle impossible inputs," not "can't generalize the dynamics." So the ×10 = 0% result
overstates the gap.

## The corrected result (held-out *reachable* region instead of ×10)

I built an on-manifold OOD probe — hold out a region of the reachable state space, train the learned
baselines on the rest, test on the held-out region (`experiments/e65_reachable_ood.py`, and the induction
version `e37b`/`e37c`). The picture splits cleanly in two:

- **Interpolation (unseen but on-manifold states): the collapse claim is an artifact.** MLP scores 0% on
  ×10 but **~0.74–0.80** on reachable interpolation. Learned models generalize fine within the manifold.
- **Extrapolation (genuinely unseen reachable region): the claim holds.** MLP collapses (~0.07), 1-NN
  fully (~0.00); induced/verified code extrapolates (induced code ~0.70 and rising with induction quality,
  trending to ~1.0; verified code exact by construction). Code stays scale-invariant (in-dist = ×10).

So the real differentiator is **extrapolation**, not "OOD" broadly. Memorizers (1-NN/tabular) fail on both;
parametric learners interpolate but don't extrapolate; code extrapolates by construction.

## What I'd change in the paper

1. **Split the OOD claim into interpolation vs extrapolation.** Soften "collapse to zero OOD" (false for
   interpolation), keep the extrapolation result (true and strong). Relabel the ×10 probe "extreme
   extrapolation" and add an interpolation column.
2. **Lead the verified-code correctness claim on MiniGrid** — it validates against the real Farama
   `minigrid` env (bit-exact, 600/600), which is genuinely independent. The sprint/triage fidelity number
   for verified code is exact *by construction* (the model wraps the same oracle it's scored against), so
   it's a reference, not a head-to-head win.
3. Specific spots flagged with line numbers in `paper/REACHABLE-OOD-AUDIT-2026-06-16.md`: the
   `\MLPOODTenK` macro in `numbers.tex` + ~8 prose spots in `main.tex` (incl. the E19 ladder and the
   E37/E38 induction "collapse to zero OOD" lines).

## Honest caveats

- The reachable rerun covered **sprint/triage** with the classical baselines (n=5). Orchard and the E38
  scale ladder weren't re-run — same probe should be applied there before those claims ship.
- One open item (now resolved into a finding): **neither qwen2.5:7b nor 14b** could induce a fully-correct
  program (repro=1.0) from the held-out-region data (0/3 each) — the post-ship `bugs += debt//4` compound
  rule resists induction from this distribution at both sizes. So induction success here is
  **data-distribution-limited, not just model-size-limited**. The induced-code extrapolation number is
  therefore a *trend*, not a single point: imperfect induced code already extrapolates ~0.66–0.70 (vs MLP
  ~0.10, 1-NN ~0.005), and the repro→extrap correlation is clean and monotone (14b: 0.86→0.50, 0.94→0.67,
  0.94→0.80; 7b: 0.95→0.86), extrapolating to ≈1.0 at repro=1.0 — which the verified-code reference
  realizes exactly (1.0 by construction). Data: `e37c` (7b), `e37d_clean_induction_14b` (14b).

Nothing's pushed — this is all on a local branch (`gpu-bench-reachable-ood`), additive (your `main.tex`
and `numbers.tex` are untouched). Happy to push it as a branch + PR, or walk through it live, whichever you
prefer. Full detail + the proposed paragraph rewrite are in the audit doc.
