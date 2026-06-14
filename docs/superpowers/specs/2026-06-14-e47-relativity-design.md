# E47 - Relativity as a verified world model: reference frames and atomic clocks

**Date:** 2026-06-14
**Status:** approved (design); pending spec review

## Goal

Demonstrate the framework on physics whose defining feature is *reference frames
and changing references*: special + general relativity. A verified symbolic world
model encodes the exact relativistic dynamics (Lorentz time dilation, velocity
addition, gravitational potential); agents in different frames carry atomic
clocks and observe each other; and the model is validated against REAL
atomic-clock measurements (GPS, Hafele-Keating). The contrast is the paper's:
the symbolic model is exact and transfers to the relativistic regime (near c,
real orbits) where a learned model approximates and a Newtonian model is simply
wrong.

## The world

State: lab-frame time `t` plus, per agent, `{position, velocity, tau}` where
`tau` is the agent's proper time (its atomic clock), and a designated observer
frame. The transition advances each clock by the time-dilated amount
`d(tau) = dt / gamma`, with `gamma = 1/sqrt(1 - (v/c)^2)`. A **changing
reference** is an action that changes an agent's velocity (acceleration /
turnaround). **Agent observation** is one agent perceiving another's clock
through the Lorentz transform (a perception boundary). General relativity adds a
gravitational rate factor `1 + Phi/c^2` from the local potential `Phi = -GM/r`.

## Claims and experiments

1. **Time-dilation fidelity + OOD near c.** The symbolic model gives the clock
   rate `1/gamma` exactly at any velocity. A learned regressor trained on
   LOW-velocity clock pairs (`v <= 0.3c`) fits in-distribution but errs sharply
   OOD as `v -> c` (the nonlinearity of gamma); a Newtonian baseline predicts
   rate 1 (no dilation) and its error grows with v. Metric: clock-rate error vs
   v, in-distribution vs OOD.
2. **Reciprocity & relativistic velocity addition.** Each inertial frame sees the
   other's clock as slow (symmetric). Composing boosts must yield
   `(v1 + v2)/(1 + v1 v2 / c^2) <= c`. Symbolic is exact; the Galilean baseline
   `v1 + v2` exceeds c (unphysical) for large inputs. Metric: composed speed vs
   c across input pairs.
3. **Twin paradox (the changing-reference centerpiece, via agent worldlines).**
   A traveling agent accelerates out, turns around, and returns; its clock
   integrates `dt/gamma` along its worldline and reads less than the stay-at-home
   agent on reunion. The asymmetry is entirely the frame change at turnaround. A
   frame-naive (Galilean) model predicts equal aging - wrong. Metric: predicted
   age difference vs the exact worldline integral; reported per leg.
4. **Real atomic-clock validation (SR + GR).**
   - **GPS** (computed exactly from first principles): orbital-velocity SR slows
     the satellite clock (~ -7 us/day), the weaker gravity at altitude speeds it
     up (GR, ~ +45 us/day), net ~ +38 us/day - the documented correction.
   - **Hafele-Keating (1971)**: with documented flight-averaged parameters
     (altitude, ground speed, latitude, duration) and Earth's rotation, the model
     reproduces the eastward (~ -40 ns) / westward (~ +275 ns) shifts within the
     measured error bars; the Newtonian model predicts 0 (off by hundreds of ns).
     The east/west asymmetry exists ONLY because of relativity in the rotating
     frame.

## Baselines

- **symbolic (ours)** - the verified relativistic transition (SR + GR).
- **newtonian** - no dilation (rate 1, Galilean velocity addition): the
  "classical simulator" that GPS/HK prove wrong.
- **learned** - a numpy regressor fit on low-velocity clock data (claim 1 only).

## Framework integration

The relativistic dynamics are a verified `Transition` over the world state;
agents are state entries carrying clocks; observation between frames is a Lorentz
transform (perception boundary); changing references are velocity-change actions
(ties to E41 changing rules, E42 frame-hopping agents). Newtonian/learned are
alternative transitions for the contrast.

## Self-checks (asserts)

- symbolic clock rate equals `1/gamma` to machine precision across v; learned OOD
  error >> symbolic; newtonian error grows with v.
- relativistic velocity addition `<= c` for all tested pairs; Galilean exceeds c.
- twin-paradox age difference matches the worldline integral; > 0 (traveler
  younger); newtonian difference == 0.
- GPS net rate within ~1 us/day of +38 us/day; SR and GR terms each correct sign
  and magnitude.
- Hafele-Keating: eastward < 0 < westward (correct asymmetry/signs); each within
  the published observed error bars; newtonian == 0.

## Deliverables

- `experiments/e47_relativity.py` (+ `results/e47_relativity.json`),
  deterministic/offline/self-checking (pure physics, no Ollama).
- Figure (time-dilation curve symbolic vs learned vs newtonian with OOD region;
  velocity-addition saturating at c; twin-paradox worldline; real-clock
  validation bars GPS + HK predicted/observed) + table; paper subsection
  (relativity / reference frames); `\NumExperiments` 45 -> 46.
- PR (stacked on e45-next-token-final).

## Honest boundaries

- These are the standard textbook SR/GR weak-field formulas, not a numerical GR
  solver; valid in the regimes used (low gravity, sub-c speeds), which is exactly
  where atomic clocks operate.
- The GPS prediction is computed from first principles and matches the documented
  ~38 us/day; the Hafele-Keating prediction uses documented flight-AVERAGED
  parameters, so it reproduces the published values within error rather than to
  the last nanosecond (the original used detailed flight logs) - stated plainly.
- "Atomic clock" here means an ideal proper-time clock; we model the relativistic
  rate, not clock hardware noise.
