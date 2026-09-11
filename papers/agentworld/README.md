# A World You Did Not Author

An essay for the **Agentworld** special issue of *Antikythera: Journal for the Philosophy of
Planetary Computation* (MIT Press, Fall 2026; call at <https://agentworld.antikythera.org/>).
Subtitle: *Modeling Is Cheap, Intent Is Expensive: What a Solved Benchmark Says to the
Anthropology of Agents.*

It reframes the ARC-AGI-3 technical report (`../arc-3/`) around its transferable diagnosis,
**goal-as-procedure**: an agent dropped into a world it did not author learns the world's
*dynamics* cheaply (verified code, exact-matched against held-out transitions) but hits a wall at the
world's *intent*, because a win is an ordered procedure that no score over a single state can rank.
The essay argues this is the same gap (brute facts vs. institutional facts; Searle, Wittgenstein,
Garfinkel, Dennett, Tomasello) a social anthropology of agents has to explain, draws six
correspondences with Agentworld's research questions, and offers two design components the call
invites: the **audit protocol** (what "source-free" can and cannot certify) and the **atlas** of
serveable world models.

- `main.tex` — the essay. Same house template and author block as `../arc-3/`.
- `PROPOSAL.md` — the ≤500-word abstract/proposal and the component list, ready for the
  submission form.
- Symlinks share `../assets/` (`figs/`, `tables/`, `numbers.tex`, `refs.bib`, `sections/`) like the
  other papers, plus `arc3_numbers.tex -> ../arc-3/arc3_numbers.tex`. **Every number is a macro**
  from the asset pipeline (`scripts/make_paper_assets.py`, `scripts/make_arc3_assets.py`); never
  hand-edit numbers here.
- Figures are reused from the technical report (`arc3_anatomy`, `arc3_goal_as_procedure`,
  `arc3_source_matrix`, `arc3_maps_gallery`); no new assets are generated.

Build: `tectonic main.tex` from this directory.

Status: full draft (~6k words, 14 pp.). Author list copied from the ARC-3 report; trim before
submission. Not yet submitted.
