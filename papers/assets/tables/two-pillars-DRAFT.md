# Two pillars — combined table draft (2026-06-16)

Lead with Pillar 1 (the win); Pillar 2 is the supporting negative.

## Pillar 1 — Verified world MODEL vs learned world models (e63, sprint/triage, 5 seeds)

Fidelity (probe in-dist / **probe OOD** / rollout) and sprint control return:

| Method | probe in | **OOD** | rollout | sprint control ↑ | steps/s |
|---|---|---|---|---|---|
| **Verified code (CWM)** | 1.00 | **1.00** | 1.00 | **7.5** | 2.25M |
| 1-NN | 1.00 | **0.00** | 0.95 | 5.7 | 6.5k |
| tabular | 1.00 | **0.00** | 0.95 | 5.7 | 1.89M |
| koopman | 0.96 | **0.00** | 0.88 | 5.6 | 82k |
| linear | 0.88 | **0.00** | 0.38 | −7.4 | 145k |
| MLP | 0.83 | **0.00** | 0.38 | −2.6 | 69k |
| LLM-as-WM (E10/11/22) | — | — | **0.00** | 0.00 | — |

Baselines: reactive 5.0, random 3.8. Trained MLP needs k≈5000 to escape negative return (e61).
**Every learned method collapses to 0.0 OOD; the verified world holds 1.0 with zero training data.**

Precondition (real, unconditional): verified world reproduces `MiniGrid-DoorKey-6x6-v0` bit-for-bit —
600/600 steps, 0 mismatches (A100-validated).

## Pillar 2 — Verified-trace distillation gives no robust lift (supporting negative)

Δ = distilled − base, n=10 heldout, two teachers:

| student | 14b: single-shot Δ | 14b: in-world Δ | 70b: single-shot Δ | 70b: in-world Δ |
|---|---|---|---|---|
| 1.5b | +1 (ns) | −2 | 0 | — |
| 3b | — | +1 | 0 | — |
| 7b | −1 | −1 | +1 | **0** (discordant 0/0) |

All |Δ| ≤ 2 at n=10 = noise. **Two independent teachers (14b near-size, 70b large-gap), same null.**
Separable positive sub-finding: in-world feedback-usage emerges with scale (1.5b mean-att 1.0 / can't
use feedback → 3b/7b can).

## Framing

"Worlds help" holds for the verified-world-MODEL sense (Pillar 1), NOT the distill-into-small-policy
sense (Pillar 2). Two meanings of "world." The project's claim is the first; the second is the footnote.

## DO NOT overclaim
- Pillar 1 head-to-head is vs **classical** baselines only. DreamerV3/MuZero/V-JEPA/Genie/Sora are
  compared **on properties**, not run — that head-to-head is still a sprint (`bench/README.md`).
- Pillar 1 numbers: sprint/triage, n=5, local macOS. Directional. Only the MiniGrid fidelity is
  unconditional.
