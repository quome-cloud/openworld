# Build the paper. Prefers tectonic (self-contained, no shell-escape needed);
# falls back to pdflatex + bibtex. Output: main.pdf
.PHONY: paper clean

paper:
	@if command -v tectonic >/dev/null 2>&1; then \
		echo ">> building with tectonic"; \
		tectonic main.tex; \
	else \
		echo ">> tectonic not found; building with pdflatex + bibtex"; \
		pdflatex -interaction=nonstopmode main.tex && \
		bibtex main && \
		pdflatex -interaction=nonstopmode main.tex && \
		pdflatex -interaction=nonstopmode main.tex; \
	fi

clean:
	rm -f main.aux main.bbl main.blg main.log main.out main.toc main.fls main.fdb_latexmk
