# Peer Review — Round 4: A Moral Philosopher's Reading

**Paper:** OpenWorld: Training-Free Symbolic World Models with Verified Code
Dynamics, Tunable Moral Configurations, and Agents-as-a-Judge (revision 3)

**Reviewer stance:** I take no issue with the empirical machinery — the
engineering reviews have done their work. My concern is the moral machinery,
which the paper treats as solved by making it *tunable*. Tunability is a
genuine advance over frozen weights, but the paper's "moral configuration"
embeds substantive ethical commitments it never argues for, and its framing
("the specification stays open because it stays code") obscures that the
*space* of expressible moralities is itself closed in a particular way.

**Recommendation:** The alignment claims need either philosophical narrowing
or structural broadening (5/10 on those claims; no objection to the rest).

---

## The central objection: the dial is crypto-utilitarianism

Everything the framework can express about value reduces to
$\sum_i w_i \cdot o_i(s, a, s')$ — a weighted sum of scalar outcome scores.
That is not a neutral container for ethics; it is *act consequentialism with
a linear social welfare function*. The morality "dial" does not let an
operator choose among ethical theories; it lets them choose among points in
one theory's parameter space. Three consequences:

- **P1 — Incommensurability is asserted, then ignored.** The paper cites
  Berlin's value pluralism (via the specification-trap literature) as
  motivation, but a weighted sum is precisely the device Berlin's argument
  rules out: it presumes a common scalar currency between, say, a patient's
  deterioration and a budget line. If the paper takes pluralism seriously,
  the framework must support aggregation structures that do *not* reduce to
  a single sum — at minimum lexical (priority) orderings and maximin.

- **P2 — No deontic structure.** Objectives evaluate outcomes; nothing in
  the framework can say an action is *impermissible regardless of outcomes*
  (Nozick's side constraints; the clinician's "never abandon a treatable
  critical patient"). Under a pure weighting scheme there will exist dial
  settings at which the optimizer commits acts most operators would call
  forbidden — and the framework cannot even express the prohibition. Show
  this empirically, then fix it: implement constraints as vetoes (filters on
  the action set), not penalties (which are just more utilitarianism), and
  measure the welfare price of deontology.

- **P3 — Distribution-blindness.** A weighted sum is indifferent between
  giving 2 utility to the best-off and 2 to the worst-off. Rawls's
  difference principle and Sen's capability critique both demand
  aggregators sensitive to *who* receives value. The commons world already
  tracks per-agent outcomes; compare the operating points selected by the
  utilitarian sum, maximin, and a lexical priority rule on the same world,
  and report what each sacrifices.

## Second objection: one theory at a time is the wrong epistemic posture

- **P4 — Moral uncertainty.** Operators do not *know* the right theory;
  MacAskill-style moral uncertainty argues for decision procedures that
  hedge across theories rather than maximize within one. A "moral
  parliament" — delegates for utilitarian, Rawlsian, and deontological
  positions voting over actions — is implementable in this framework in an
  afternoon. The interesting empirical question: does the parliament avoid
  every theory's worst-case (the hedging property), and what does it pay
  for that insurance on each theory's own metric?

- **P5 — Pluralism should be measurable in the judge, too.** E7/E15 graded
  trajectories under one welfare-flavored rubric and a paraphrase. If value
  pluralism is real, philosophically distinct rubrics (utilitarian,
  deontological, care-ethical) should induce *different orderings* over the
  same trajectories — rubric framing is a value choice, not a wording
  choice. Measure the rank correlations between rubric-induced orderings.

- **P6 — Procedural legitimacy (text, not experiment).** A dial set by one
  operator answers "what does the operator want," not "what is justifiable
  to those affected" (Scanlon). The tuner's "solving manifold" gestures at
  this — the set of configurations meeting every stakeholder's minimum is a
  crude contractualist surface — but the paper should stop implying that
  open specification *resolves* the specification trap. It relocates it:
  from frozen weights to an operator's hand on a dial. Say so.

## Requested changes

| Item | Response requested |
|---|---|
| P1, P3 | **E24**: same world, three aggregators (sum / maximin / lexical); report total vs worst-off vs gap at each rule's chosen operating point |
| P2 | **E25**: framework support for deontological constraints as action vetoes; violation counts across the dial sweep with and without constraints; the welfare cost of the constraint |
| P4 | **E26**: moral parliament vs three pure-theory agents; hedging property measured on every theory's own metric |
| P5 | **E27**: judge rubrics for three ethical traditions over identical trajectories; rank correlations between orderings |
| P6 | Reframe: "open specification" manages the specification trap procedurally, it does not resolve it; add Rawls/Nozick/Berlin/MacAskill/Scanlon to the bibliography and the discussion |

---

*Round-4 review produced as part of the project's red-team process; revision 4
addresses each item below.*
