# LaTeX paper template

A reusable LaTeX template for the papers in this repository (Task Force for AI Agents
in Healthcare papers), derived from the QFIRE paper. It reproduces the house style — title
page with the **Quome logo** (`figs/quome.png`; `taskforce_logo.png` is also bundled if you
prefer it), gray abstract panel with keywords + code/data + correspondence,
bold-underlined figure/table captions, framed figures, and the policy-card / `listings`
infrastructure — plus a generic section skeleton and the back-matter declarations
(including a **Declaration of generative AI use**) that journals expect.

## Files

| File | Purpose |
|---|---|
| `main.tex` | The template. Fill in every `% TODO:` marker. |
| `numbers.tex` | Headline result macros, `\input` by `main.tex` (override the `\providecommand` defaults). Generate these from your benchmark output. |
| `refs.bib` | Starter bibliography (carried over from QFIRE — trim to what you cite; keep `haarf2026`). |
| `Makefile` | `make` builds `main.pdf` (prefers `tectonic`, falls back to `pdflatex`+`bibtex`). |
| `figs/` | Figures. The `quome.png` (title-page logo) and `taskforce_logo.png` are included. |
| `tables/` | Generated/`\input`-able tables. |

## Quick start

```bash
cp -R papers/_template papers/<NNN-your-paper>   # e.g. papers/016-new-thing
cd papers/<NNN-your-paper>
make            # builds main.pdf with placeholder boxes where figures will go
```

Then:

1. Fill in the title, authors, affiliations, abstract, and keywords on the title page.
2. Work through each `% TODO:` in the body; delete sections you don't need (e.g. the
   declarative-config subsection if your system has no DSL).
3. Replace each `\todofig{...}` placeholder box with `\includegraphics{figs/...}`.
4. Add your citations to `refs.bib`.
5. Edit the **Declaration of generative AI use** and the other back-matter
   declarations to reflect what you actually did.

## Conventions (match the rest of the program)

- Report rate metrics with 95% Wilson confidence intervals; pre-register hypotheses
  and seeds.
- Keep one figure that captures the whole paper (the `\label{fig:hero}` "paper in one
  figure").
- Map each contribution to the relevant HAARF sub-controls in the Discussion.
- All patient data synthetic; no real PHI.
- Every table/figure should regenerate from a single build command for reproducibility.

## The `\todofig` helper

`\todofig{<height>}{<description>}` draws a labeled placeholder box so the document
compiles before figures exist. Replace each with a real `\includegraphics` (which is
auto-framed) when the figure is ready.
