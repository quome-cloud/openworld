# Strategy-lift proposals — E140 arms compared, ideas for improving Opus / Sonnet

Based on head-to-head of `arc3_fullgame_sourcefree.json` (Opus @ max) vs `arc3_fullgame_sourcefree_codex.json` (Codex @ xhigh), plus the E148 strategy fingerprints. Fable comparison hooks are in the script for when its archive lands.

## What the E140 head-to-head shows

**Opus 16/25 full · Codex 7/25 full · Fable ~11/25 running (per operator).**

- **Codex adds zero unique wins.** Every game Codex fully solves, Opus also fully solves. So Codex isn't a diversification argument on its own; it's a cost argument. Whether Fable adds *unique* wins is the key open question — worth calling out explicitly in the paper as the falsifiable multi-arm hypothesis.
- **7 games solved by both** (ar25, dc22, ft09, ka59, lp85, su15, tu93). Cheap; likely to stay cheap on Sonnet.
- **9 games are Opus-only** (cd82, cn04, g50t, m0r0, re86, s5i5, sb26, sc25, tr87). The reasoning gap between Opus @ max and Codex @ xhigh is exactly here. Try Sonnet on these first — if Sonnet gets any of them, capability is not the discriminator, iteration is.
- **6 games are shared walls** where every arm gets stuck within ±1 level: `lf52 (6/10)`, `ls20 (5/7)`, `r11l (4/6)`, `sp80 (4/6)`, `tn36 (2/7)`, `vc33 (4/7)`. **These are the target set for goal-predicate synthesis experiments** (E102/103/104 negatives were on exactly this class). If Fable moves the needle here, it's a big deal; if not, it confirms the goal-as-procedure wall is the same wall for all three arms.

## Six concrete experiments to lift Opus (and to check Sonnet)

Each is scoped to fit in ~1 cycle at existing infra. Ordered by expected ROI.

### Proposal L1 — Sonnet arm at E140 budget (highest-value, cheapest)

Run Sonnet-4.6 @ max under `run_e140_backoff.sh` with the same 4-hour cap. Report `arc3_fullgame_sourcefree_sonnet.json` in the same schema so the head-to-head extends automatically.

Predicted outcomes:
- If Sonnet lands ≥13/25, capability isn't the wall for 80% of games; iteration budget and reasoning-effort mode are. Cheaper deploy path.
- If Sonnet lands ~7/25 (Codex-tier), the 9 Opus-only games are genuinely capability-gated. Sharper story for the paper.
- If Sonnet lands >16/25, unlikely but publishable.

Cost: ~4 hours wall per game × 25 = one weekend on a single agent slot.

### Proposal L2 — Cross-arm CEGIS on the shared walls (high leverage)

The 6 shared-wall games (lf52, ls20, r11l, sp80, tn36, vc33) are where all arms fail at the same level. Try:

1. Have Opus and Codex each write a synthesized `predict(frame, action)` for the last-cleared level of the walled game.
2. Roll each forward until they disagree.
3. Use the disagreement frame as a targeted probe in the real env.
4. Feed the ground-truth transition back to both models as an extra training example.
5. Re-synthesize.

This is essentially E127's differential CEGIS applied across models rather than within one model. Predicted: for at least 2/6 shared walls, disagreement discovers the mechanic the individual arms missed.

Cost: ~1 day. Reuses E127 infrastructure.

### Proposal L3 — "Fable signature" transfer to Opus (memory-heavy prompting)

E148 shows Fable is memory-heavy (memory rate 13.0 vs Opus 9.2) and mechanic-heavy (4.4 vs 3.1). Try:

- Boost Opus's memory strategy usage by adding an explicit "notes on what worked before / what didn't" scratchpad at every action, forced by the TASK.md prompt.
- Boost mechanic-inference by adding an explicit "hypothesize the local mechanic" step before each new zone.

Predicted: +1 to +3 games. Cheap to test — TASK.md tweak + rerun on the 9 shared-wall + partial-solve games.

Cost: <1 day. Zero-code.

### Proposal L4 — Fable-vs-Opus disagreement ensemble on shared walls

Once Fable's archive lands and confirms which walls it *did* beat (if any):

- On each Fable-only-wins game, extract Fable's solved.json actions.
- Replay in a fresh Opus session with the actions as inline evidence in TASK.md.
- Ask Opus: "explain WHY these actions solve it".
- Then feed Opus similar-shaped unsolved games with the extracted principle as a prior.

Tests whether the specific *win mechanic* transfers even if Opus wouldn't have found it alone. If it transfers, that argues for a "curriculum" strategy: run Fable first on hard games, use its wins to bootstrap Opus on the walls.

Cost: 2–3 days. Publishable as its own ablation.

### Proposal L5 — E140 second-round with reset

E140's `e140_uncapped.rounds = 2`. Look at the run logs: did the second-round help on the games Round 1 failed, or did it just repeat the failure mode? If round-2 helps, add round-3 selectively on unsolved games. If it doesn't help, prune to round-1 and reallocate budget to another arm.

Quick to check from the traces; adjust runners accordingly.

Cost: 4 hours analysis, no new runs.

### Proposal L6 — "Frozen tools" ablation (kills a reviewer attack)

Currently the agent has full Bash + Read + Write + Edit. Try:
- Remove Edit — force the agent to overwrite whole files. Does completion drop?
- Remove Read on transcripts — force the agent to rely on its own memory across sessions. Does it drop?

Predicts which tools are load-bearing for the win rate. Not a lift itself, but a paper-ready ablation that closes the "your agent is just a scaffolding win" attack.

Cost: 1 day. Runs on the same 25 games.

## Consolidated recommendation

For the paper's next revision:
1. **Do L1 first.** Sonnet at E140 budget is the single most informative missing datapoint.
2. **Do L2 second.** Cross-arm CEGIS on shared walls is the biggest research-content lift.
3. **Fold L3 (memory-heavy prompt) into a small ablation table.** Even a null result is a useful paper artefact.
4. **Do L4 once Fable's archive stabilises.** This is the paper's "curriculum" story if it works.

For the deployment pipeline (referencing operator's mention of qusast):
- Wire the agent-generated Python through qusast's accept/block layer BEFORE the sandbox executor. That closes the paper's implicit trust assumption ("we run whatever code the agent writes"). A one-line addition to the agent harness + a policy file.

## Note on Sonnet

Sonnet is Anthropic's mid-tier reasoning model — cheaper than Opus, better than Haiku. If Sonnet + E140 budget matches or beats Codex-xhigh, that's a very shippable result: same benchmark performance at ~1/3 the compute. Worth running before the paper freezes.

## Falsifiable predictions worth committing to before the runs

- **P1.** Sonnet + E140 will land between Codex-xhigh (7) and Opus-max (16), probably 11–13.
- **P2.** Fable will add at least 1 unique win vs Opus (otherwise the paper's 3-arm framing is a distraction).
- **P3.** Cross-arm CEGIS will crack ≥1 of the 6 shared walls (lf52 is the likeliest candidate — its long chain suggests procedure-not-state, so disagreement should localise the missing predicate).

If any of these are wrong, the paper's story tightens (or shifts) around the actual result. Either way it's cheap to run.
