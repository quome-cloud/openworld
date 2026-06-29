"""Expert STRATEGY LENSES for the ARC-AGI-3 agent panel (the Bayesian-experts router tier).

Idea (MSA, arXiv 2507.12547): when normal agent retries plateau on a game, the single agent has tunnelled
on ONE framing of the mechanic (e.g. ka59's L7 'topology-sealed' proof). A panel of differently-primed
experts synthesises genuinely different hypotheses, covering parts of hypothesis space a single framing
never visits. Selection stays ENV-GROUNDED: whichever expert's plan actually raises g.levels wins
(replay-verified); if none beat the wall, nothing is banked (abstain). Cost-neutral: on a stuck game the
panel REPLACES the repeated identical prompt with rotating lenses -- same attempt budget, diverse strategy.

SOURCE-FREE: these are general puzzle-solving FRAMINGS, not game answers. The agent still must discover the
rule by acting. No lens names a game mechanic or a solution. Keep it that way (the runs are audited).
"""

EXPERTS = [
    ("topology",
     "TOPOLOGY / REACHABILITY: hypothesise the win is about REACHING or CONNECTING regions. Map which "
     "cells are reachable; find sealed pockets; test whether an action OPENS a barrier or whether an "
     "object must be ROUTED through a one-way conduit. If a goal looks unreachable, hunt for a non-obvious "
     "gate -- a cell whose toggle changes a wall ELSEWHERE."),
    ("temporal",
     "TEMPORAL / PHASE: hypothesise the rule depends on TIME or ORDER, not static state. Probe whether "
     "barriers change after K steps, whether a sub-sequence must be done in a specific ORDER, or whether "
     "waiting / cycling an action mutates the board. Re-test every 'static wall' across many steps before "
     "trusting it."),
    ("force",
     "FORCE / PUSH TRANSMISSION: hypothesise objects PUSH one another. Test whether moving into an object "
     "transmits force through a chain, whether locking one object changes how another moves, and whether a "
     "piece can only be delivered by an ADJACENT pusher. Try pushing chains you previously assumed inert."),
    ("color_algebra",
     "COLOR ALGEBRA / PARITY: hypothesise the win is a COLORING or parity rule -- toggling a cell flips its "
     "neighbours, or you must reach a target pattern (think lights-out / GF(2)). Track per-colour counts "
     "and parities; test whether one action flips a NEIGHBOURHOOD rather than a single cell."),
    ("symmetry",
     "SYMMETRY / CIPHER: hypothesise a SUBSTITUTION or SYMMETRY rule -- mirror, rotation, or a per-cell "
     "mapping between two regions. Compare regions under dihedral transforms; test whether one region must "
     "be made the mirror, rotation, or recolouring of another."),
    ("perception_reframe",
     "PERCEPTION REFRAME: distrust your own perception. What are you MASKING or ignoring -- a status bar, a "
     "counter, a faint or rare-coloured cell -- that might actually BE the signal? Re-include masked cells; "
     "treat the 'noise' as possibly encoding the rule; re-derive the goal without your prior assumptions."),
    ("counting",
     "COUNTING / BUDGET: hypothesise the win depends on COUNTS or a move BUDGET -- collect N of something, "
     "hold a counter at a value, or finish within a step budget. Track every number on the board and every "
     "counter you can find; test threshold effects at specific values."),
]

_BY_NAME = {n: t for n, t in EXPERTS}


def names():
    return [n for n, _ in EXPERTS]


def lens(key):
    """Return the strategy-lens text for an expert, selected by name or by a rotating integer index."""
    if isinstance(key, str) and key in _BY_NAME:
        name, text = key, _BY_NAME[key]
    else:
        try:
            i = int(key) % len(EXPERTS)
        except (TypeError, ValueError):
            i = 0
        name, text = EXPERTS[i]
    return name, text


# Shared with every panel attempt: operationalises surprise-driven regime resets (E121). A stuck agent
# often fails because it is forcing a STALE world model onto a level whose rules already changed.
REGIME_RESET = (
    "WATCH FOR RULE CHANGES (the levels are compositional -- mechanics get added). Two signals that the "
    "rules just changed: (a) a board RELOAD -- a large, sudden change across most of the board (often a "
    "level-up); (b) SURPRISE -- your predict() starts mis-predicting transitions it used to get right. When "
    "either fires, do NOT carry your old dynamics forward: treat it as a NEW regime, re-explore from "
    "scratch, and rebuild predict()/the goal for it. If you are STUCK and your model keeps mis-predicting, "
    "that is the signal your model is wrong for the current regime -- rebuild it rather than forcing the old "
    "rules (a wall is often a stale model, not an impossible level).")


def task_addendum(key):
    """The block appended to TASK.md when this game is routed to the expert panel."""
    name, text = lens(key)
    return (f"\n\n--- STRATEGY LENS (expert: {name}) ---\n"
            f"Earlier attempts on this game PLATEAUED -- they likely tunnelled on one framing. For THIS "
            f"attempt, lead with a DIFFERENT working hypothesis:\n  {text}\nStill discover everything by "
            f"acting and replay-verify; this lens only changes which hypothesis you test FIRST, never what "
            f"counts as a solve (only raising g.levels does). If this lens clearly does not fit after honest "
            f"probing, say so and fall back to open exploration.\n\n"
            f"--- WHEN THE RULES CHANGE (surprise-driven regime reset) ---\n  {REGIME_RESET}\n")


if __name__ == "__main__":
    import sys
    # CLI: print the addendum for a given expert key (used by the shell runner).
    print(task_addendum(sys.argv[1] if len(sys.argv) > 1 else 0))
