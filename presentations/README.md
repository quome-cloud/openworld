# Conference presentations

Four talks built from the OpenWorld papers, each available as **Beamer (PDF)** and a
**self-contained HTML deck** — one shared theme, generated from a single content spec so the
two formats stay identical.

| Deck | Talk | Source paper |
|------|------|--------------|
| `world_computing/` | **An Introduction to World Computing** | [`papers/framework`](../papers/framework) |
| `world_time_computing/` | **World-Time Computing: A Next Frontier for Simulated Learning** | [`papers/world-time-compute`](../papers/world-time-compute) |
| `solving_arc_agi_3/` | **Solving ARC-AGI-3 with Code World Models** | [`papers/arc-3`](../papers/arc-3) |
| `world_models_overview/` | **World Models: An Overview and Applications** (combined) | all three |

Each `<deck>/` folder is self-contained: `<deck>.html`, `<deck>.tex`, and a local `figs/` copied
from `papers/assets/figs/`. Every paper figure and the relevant appendix figures are included.

## Present

**HTML** (no dependencies, works offline on any machine — the whole point of the vendored
`slides.js`): open `<deck>/<deck>.html` in a browser.

- Arrow keys / Space / PageUp-Down — navigate · `f` — fullscreen · `Home`/`End` — jump
- Swipe on touch; the URL hash (`#12`) deep-links a slide; a progress bar + counter show position
- `Cmd/Ctrl-P` → "Save as PDF" prints every slide (the `@media print` rule un-stacks them)

**Beamer PDF**: build with `tectonic` (no system LaTeX needed):

```bash
cd presentations/world_computing && tectonic world_computing.tex
```

## Build / regenerate

One generator turns each content spec into both formats:

```bash
python presentations/build_decks.py                 # all four decks
python presentations/build_decks.py solving_arc_agi_3   # just one
```

- Content lives in `decks/<name>.py` — `TITLE`, `SUBTITLE`, `AUTHOR`, `VENUE`, and a `slides` list.
- Slide types: `section` · `bullets` · `figure` (image + optional bullets) · `twocol` · `statement`.
- Figures are referenced as `figs/NAME.png` (as in the papers) and copied from `papers/assets/figs/`
  into each deck's own `figs/`, so a deck folder is portable on its own.
- The shared **OpenWorld theme** (blue/teal/ochre depth ramp, warm paper, nested-worlds mark) is
  defined once in `build_decks.py` and applied to both Beamer and HTML.

The compiled `*.pdf` are build artifacts (git-ignored); rebuild them with `tectonic` as above.

## Add a deck

Drop a `decks/<name>.py` defining the four strings and `slides = [...]`, then run the generator.
Numbers and claims should track the papers (the specs were extracted from them directly).
