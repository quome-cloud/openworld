# Submission proposal — Agentworld special issue (Antikythera / MIT Press)

Form: <https://link.antikythera.org/agentworldform> (review window opened June 25, 2026).
Format: full paper draft (`main.pdf`) plus the abstract below. Lead author / contact: James
Schwoebel (Quome, Inc.). The abstract below is the one in `main.tex` with the generated macros
expanded; keep them in sync by regenerating, not by editing here.

## Title

**A World You Did Not Author.** Modeling Is Cheap, Intent Is Expensive: What a Solved Benchmark
Says to the Anthropology of Agents

## Abstract (≤ 500 words)

Agentworld asks what happens when minds that did not grow up in a world are dropped into it in
large numbers. We report on a controlled miniature of exactly that situation. ARC-AGI-3 is an
interactive benchmark that places an agent in an unfamiliar grid game with no instructions, no
language, and no stated goal; the agent must learn the rules by acting and then reach an end it
was never told. Humans clear every game. Frontier agents scored under one percent at launch. Our
agent, built on the OpenWorld framework, completes all 25/25 public games under an audited
*source-free* protocol, never reading a game's code: it acts, writes down what it learns as a
small verified program, plans inside that program, and replays the plan against the real engine.

The lesson we carry to Agentworld is not the score. It is *where the difficulty sat*. Learning the
*dynamics* of a world that did not author you turned out to be cheap: an agent that writes its
model as code and checks it against held-out transitions recovers the rules exactly. Learning the
*intent* of that world was the wall. Four principled ways of inferring the goal without a
reasoning agent solved zero hard games, several while holding a perfect model of the dynamics. The
reason is structural. An ARC-AGI-3 win is not a state you can score from a screenshot; it is an
ordered *procedure*, a protocol that must be followed in sequence. No score over a single frame
can rank a procedure, so goal-conditioned planning never finds it, while undirected play
occasionally stumbles into it. We call this diagnosis *goal-as-procedure*.

We argue that this gap, between the brute facts of a world (what happens when you press a button)
and its institutional facts (what counts as having done the right thing), is the same gap a social
anthropology of agents has to explain. Searle's constitutive rules, Wittgenstein's rule-following,
Garfinkel's breaching experiments, and Tomasello's shared intentionality all locate the difficulty
of entering a human world in the second kind of fact, not the first. Our miniature confirms the
location experimentally and gives it a cost: modeling is cheap, intent is expensive, and only an
agent that *reasons* about what the world is for gets through.

The essay draws six correspondences with Agentworld's questions: open-world entry, norms as
procedures, agents as decomposable assemblages, the phenotypic diversity of non-human solvers, the
centaur workflow that actually produced the result, and legibility as the price of admission. Two
design components accompany the argument. The *audit protocol* shows what an agent's claim to have
learned honestly can and cannot certify (no runtime access to the world's source, audited over 844
transcripts; no guarantee about prior exposure). The *atlas* is a set of serveable, inspectable
world models, one per game, in which the model of a world is itself a world, and worlds nest. We
close with what the miniature cannot say and with five propositions for builders and
anthropologists of centaur societies.

## Design, visual, and web-based components

- **The atlas**: one self-contained SVG world-model card per solved game
  (`papers/arc-3/maps/*.svg`), viewable in any browser; gallery figure in the essay.
- **The live view**: `openworld serve <specs> --allow-code --open` serves every solved world as an
  interactive, steppable state graph with the perceive → world → emit boundary drawn.
- **The audit**: source-free harness, per-transcript audit script, and the audit summary,
  reproducible from the open-source repository.
- **Replay trajectories**: the banked, replay-verified winning action sequence per game, rendered
  as frame strips.
- **Diagrams**: world anatomy and goal-as-procedure, in the framework's shared design language.

## Still needed for the form

Author biographies and CVs; a portfolio work sample (suggest the ARC-3 technical report and the
atlas); a trimmed author list if the essay should carry fewer names than the technical report.
