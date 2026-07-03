# Paper feedback — ARC-3 manuscript, arc3-runner-fix @ HEAD

Reviewer-style notes on `papers/arc-3/main.tex` at the current tip. Goal: preserve the honest,
diagnostic character of the paper while tightening claims a hostile reviewer would attack.

## What is genuinely strong

1. **The core diagnosis (goal-as-procedure) is a real contribution.** Three principled attacks
   (E102 atomic goals, E103 LLM hypotheses, E104 Bayesian sub-world) all fail on fidelity-1.0
   models. That's a clean negative result and it explains a puzzle the field has been dancing
   around. Keep this front-and-centre.
2. **Reproducibility discipline is exceptional** for an ARC-benchmark paper — every number
   regenerates from `scripts/make_arc3_assets.py`, action traces are replay-verified with one
   command, and E121 round-trips every solve through the OpenWorld framework. Reviewers will
   notice and it materially strengthens the paper.
3. **Honest caveats section** — offline unbounded resets, 11/25 free at ≥1 level, 14/25
   distinctive above-random. Very few benchmark papers admit their tutorial-level artefact.
   Keep this even if the SOTA number stays in the abstract.
4. **The strategy figures (E148)** are a rare piece of introspection into WHY the solves work —
   which is exactly what most benchmark-cracking papers omit.

## Where reviewers will push (in expected-severity order)

### 1. "Modeling is largely solved" is overstated

- Abstract claims "near-perfect (≥90%) models" but only 3/21 real-dynamics games qualify.
- Average fidelity is 49% (Claude); 12% (qwen-32B). Neither reads as "solved".
- On dense games (cn04, 174 changed cells/step) Claude is at 0%.

**Suggested fix.** Rephrase the abstract from "modeling is largely solved" to something like
"modeling is *tractable* — Claude synthesizes near-perfect (≥90%) code-dynamics on 3/21
real-dynamics games and materially beats random-copy on 19/21; goal-discovery, not modeling
capacity, is the wall." Costs one sentence, buys back the reviewer's trust for the whole
diagnosis section.

### 2. Goal-as-procedure is diagnosed but not taxonomised

The E102–E104 story is that these methods fail *on perfect models* — which is the whole point.
But the paper doesn't answer the natural next question: **which wins are procedures, which are
states, and can we predict the split?** Right now a reviewer will ask "are these all failure
modes of the same shape, or three different shapes lumped together?".

**Suggested fix (cheap, big win).** Add a 3–5 row table: for each failed game, one column each for
{state-configurable?, ordered-sequence?, timed-window?, hidden-state?, precise-click?}. Ternary marks
(✓ / partial / ✗). Immediately shows which failure modes E102–E104 are actually diagnosing.
This is the taxonomy the paper is one 20-line table away from having. `SOLVING_LOG.md` already
contains the raw material.

### 3. RHAE 95.8 is measured offline; the headline reads live

The paper correctly flags this in the methods section, but the abstract and §9 both lead with
the number without the caveat inline. A reviewer who spot-checks the leaderboard will find a
smaller number and cry foul.

**Suggested fix.** Two options.
- **Cheapest:** parenthesise every headline mention: "RHAE 95.8 (offline replay-verified;
  live-protocol validation ongoing)".
- **Better:** report BOTH offline and any live-protocol number you have, even if small. If none
  yet, add a stub sentence: "A single live-protocol run on `ar25` yielded RHAE=X (see §9.3);
  full sweep pending."

### 4. Cost comparison ($7 vs $350) is unmetered and uncontrolled

Different engines, different harness, unmetered tokens. Paper says "estimate, not controlled"
but the "50× cheaper" claim is used in the abstract-adjacent framing. Baseline1 authors WILL
push back.

**Suggested fix.** Move the cost comparison out of any abstract-facing summary and into an
"Economics" subsection with three columns:
1. tokens per game (measured);
2. USD estimate at posted rates (calculated);
3. wall-clock hours (measured).
Report each cell for both approaches with a "same task, different engine" disclaimer. Same
number, less exposure.

### 5. Router / cascade is a deployment strategy, not a contribution

§8 describes the cheap-then-agent cascade as a system feature, but there's no ablation showing
the cascade beats "just run the agent on everything". This is the section a reviewer will
recommend cutting.

**Suggested fix.** Either (a) run the ablation and report the delta (probably marginal —
agent-only reaches same 23/25, cascade is just cheaper), or (b) *cut §8 entirely* and put the
one useful sentence ("we deploy try-cheap-then-escalate so 12/25 games skip the LLM entirely")
into a footnote in §9. Losing §8 saves ~1 page.

### 6. Multi-perception consensus (s5i5) — surprise result under-explained

Multi-perception voting solves s5i5, which E102–E104 could not solve on a fidelity-1.0 model.
The paper notes this but doesn't cash in on the surprise — this is arguably the single most
interesting empirical finding in the whole paper. It suggests that when perception itself is
underdetermined, the goal-discovery wall was really a perception wall in disguise.

**Suggested fix.** Add a paragraph in §7 (or promote to §5.5): "One game, s5i5, was solved by
multi-perception consensus after all three principled goal methods failed on a perfect-fidelity
model. Post-hoc, the click-modality perceptor discovered a target sprite the directional
perceptor was masking. This suggests that some fraction of E102–E104's failures were
perception-limited, not goal-limited." That single paragraph reframes multi-perception from
"cheap tier" to "principled diagnostic".

### 7. Fidelity numbers per-model vary wildly in structure not just magnitude

The paper reports Claude 49% / qwen-32B 12% as if they're differences in degree. But
`e148_strategy_space` shows they use qualitatively different STRATEGIES — Fable is memory +
verify; opus is perceive + verify; codex is stripped-down. This is a story about *how* not
just *how well*.

**Suggested fix.** Cite the E148 radar figure explicitly in §3 with one sentence: "Beyond raw
fidelity, arms differ in *how* they solve: Fable is memory-heavy, opus is perception-heavy,
codex is minimal (see E148 radar)." Then in §8b, note that this is why the cascade helps:
routing to the right arm is orthogonal to just adding compute.

## Small polish items

- **Abstract order.** The current opening dives into the SOTA number; consider leading with the
  diagnosis ("frontier LLMs score near-zero because…") because that's the paper's real
  contribution. SOTA follows.
- **"Fable" name convention.** The paper mixes `claude-fable-5` and `Fable` and (in some
  captions) "F". Pick one and stick with it.
- **Determinism = 1.00.** This is a beautiful, load-bearing fact. Currently mentioned in
  passing. Consider a callout box or a bolded sentence: "Every one of the 25 games is
  deterministic under replay (Determinism = 1.00)." One line, huge weight.
- **The recipe figure (Fig. 5).** Dense but powerful. Consider adding an overlay that shows
  which recipe steps E102 / E103 / E104 attack (dashed arrows), so the negative results are
  literally visible in the architecture diagram.
- **`SOLVING_LOG.md` is gold.** Currently a supplement. Consider promoting 2–3 quotes into
  §5 as inline captions on E103 failures. E.g. one of the s5i5 hypotheses that failed. Turns a
  claim into a scene.

## What NOT to change

- The above-random honest metric. Keep it. Reviewers will respect it.
- The negative results (E102–E104). Do NOT hide them or reframe them as positive.
- The cost/protocol caveats. Move them, don't remove them.

## Priority ranking (if resource-constrained)

1. **Do #1 (rephrase modeling-solved) + #2 (add taxonomy table).** Both are cheap, both directly
   address the biggest reviewer attack surface.
2. **Do #3 (offline caveat) + #4 (economics section).** Same idea: quarantine risky claims.
3. **Do #6 (s5i5 promotion) + #7 (E148 radar cite).** These are gain, not defence — they make
   the paper more interesting.
4. **Cut §8 (cascade) if you need page count back.** The paper is stronger without it.

## Timing

If Jim wants a fast tightening pass, #1 through #4 above are ~2 hours of writing. #6 and #7
are ~1 hour. Cutting §8 is 30 min of paragraph reflow. Total < 4 hours for a materially
stronger paper.
