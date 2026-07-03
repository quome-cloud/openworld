# arXiv Submission Packaging

This note is for agents preparing future OpenWorld paper submissions. It records the workflow that
worked for the framework paper and the failure modes to avoid.

## Goal

Submit a TeX source package that arXiv can compile with `pdflatex`, even though local paper builds
usually use `tectonic`. arXiv's file handling and engine choice can differ from local builds, so the
upload package should be self-contained, conservative, and easy to inspect.

## Checklist

1. Start from the paper directory, for example `papers/framework/`.
2. Use source files, not the generated `main.pdf`.
3. Include:
   - `main.tex`
   - `main.bbl`
   - `refs.bib`
   - `numbers.tex`
   - all referenced table `.tex` files
   - all referenced section `.tex` files
   - all referenced figure files
4. Exclude build artifacts:
   - `main.pdf`
   - `main.aux`
   - `main.out`
   - `main.log`
   - `main.blg`
   - `.DS_Store`
5. Prefer PNG/JPG/PDF figures for `pdflatex`; do not rely on arXiv to convert SVG/EPS.
6. Avoid transparent PNGs for logos. Flatten them onto white before packaging.
7. Avoid fragile subdirectories if arXiv appears to lose folder contents. Use a flat archive and rewrite
   `main.tex` references from `figs/foo.png`, `tables/foo.tex`, and `sections/foo.tex` to `foo.png`,
   `foo.tex`, and `foo.tex`.
8. Include `main.bbl`. arXiv may run BibTeX, but a checked-in/generated `.bbl` makes references more
   deterministic.
9. Before upload, unpack the archive and verify the paths in `main.tex` exactly match files in the archive.
10. On arXiv, delete old uploaded files before uploading a replacement archive. Mixed old/new files caused
    repeated false missing-file failures.

## Engine Notes

The local nice-looking PDFs may be built by `tectonic`/XeTeX-style tooling, while arXiv may compile with
`pdflatex`. This can change fonts, line breaks, page count, and first-page flow. For `pdflatex` packages,
add these near the top of `main.tex` in the upload copy:

```tex
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{microtype}
\ifdefined\DeclareUnicodeCharacter
  \DeclareUnicodeCharacter{2014}{---}
  \DeclareUnicodeCharacter{2192}{\ensuremath{\to}}
\fi
```

Also replace raw Unicode glyphs that pdflatex fonts may miss:

- `—` -> `---`
- `→` -> `\ensuremath{\to}` or `$\to$`

## Figure Handling

For the framework paper, these were the critical fixes:

- The OpenWorld title banner had to be rasterized from the good PDF asset:

```bash
pdftoppm -png -r 300 -singlefile papers/assets/figs/title_band.pdf /tmp/title_band_good
```

- The Quome logo PNG had transparency that rendered badly under arXiv's `pdflatex`; flatten it:

```bash
magick quome.png -background white -alpha remove -alpha off quome_flat.png
```

If ImageMagick cannot rasterize a PDF because Ghostscript is missing, use `pdftoppm` for PDF-to-PNG.

## Flat Archive Template

This is the robust fallback when arXiv loses `figs/`, `tables/`, or `sections/` directories.

```bash
ROOT=/Users/jim/Desktop/openworld
PAPER=framework
STAGE=/private/tmp/openworld_${PAPER}_arxiv_flat

rm -rf "$STAGE"
mkdir -p "$STAGE"

cp "$ROOT/papers/$PAPER/main.tex" "$STAGE/main.tex"
cp "$ROOT/papers/assets/numbers.tex" "$STAGE/numbers.tex"
cp "$ROOT/papers/assets/refs.bib" "$STAGE/refs.bib"
cp "$ROOT/papers/$PAPER/main.bbl" "$STAGE/main.bbl"  # generate first if missing
cp "$ROOT/papers/assets/tables/"*.tex "$STAGE/"
cp "$ROOT/papers/assets/sections/"*.tex "$STAGE/"

# Copy only figures used by the paper, or all PNG/PDF figures if size is acceptable.
cp "$ROOT/papers/assets/figs/"*.png "$STAGE/"

# Rewrite include/input paths for a flat archive.
perl -0pi -e 's/\{figs\//\{/g; s/\{tables\//\{/g; s/\{sections\//\{/g' "$STAGE/"*.tex

# Optional pdflatex font compatibility for upload copy.
perl -0pi -e 's/\\usepackage\[utf8\]\{inputenc\}/\\usepackage[utf8]{inputenc}\n\\usepackage[T1]{fontenc}\n\\usepackage{lmodern}\n\\usepackage{microtype}/' "$STAGE/main.tex"

cd "$STAGE"
tectonic --keep-intermediates main.tex
tar -czf "$ROOT/papers/${PAPER}_arxiv_upload.tar.gz" \
  --exclude='main.pdf' --exclude='main.aux' --exclude='main.out' --exclude='main.log' \
  --exclude='*.blg' --exclude='.DS_Store' .
```

After packaging:

```bash
tar -tzf "$ROOT/papers/${PAPER}_arxiv_upload.tar.gz" | sort | less
tar -xOzf "$ROOT/papers/${PAPER}_arxiv_upload.tar.gz" ./main.tex | rg 'includegraphics|input|bibliography'
```

Make sure every referenced file exists at the same archive path.

## arXiv Metadata

Use the manuscript source for title/authors/abstract. The arXiv abstract may be a plain-text shortened
version of the PDF abstract, but it should not change claims, numbers, scope, or limitations.

Leave these blank unless known:

- Report number
- Journal reference
- External DOI
- ACM class
- MSC class

For comments, use a concise factual summary such as:

```text
64 pages, 35 figures, 25 tables. Code and data available at https://github.com/quome-cloud/openworld
```

## Debugging arXiv Logs

Search the log for `Error:` and `not found`. If the same missing file remains after uploading a new
archive, arXiv is probably compiling stale files. Delete all uploaded files in the arXiv file list and
upload one archive only.

Citation warnings often appear before BibTeX or after a failed compile. If `main.bbl` is present and read
near the end of the log, fix fatal missing-file errors first.
