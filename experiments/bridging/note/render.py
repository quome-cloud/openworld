"""Render llm_validation_note.md → llm_validation_note.pdf via markdown-it + headless Chrome.

Style mirrors experiments/catan/note/note.pdf (Liberation Serif body, 12pt, ~1.5 inch
margins, typographic numerals in tables).

Usage:
    python3 experiments/bridging/note/render.py
"""

from __future__ import annotations

import pathlib
import subprocess
import tempfile

from markdown_it import MarkdownIt

HERE = pathlib.Path(__file__).parent
MD_PATH = HERE / "llm_validation_note.md"
PDF_PATH = HERE / "llm_validation_note.pdf"

TITLE = "LLM-Persona Validation Note (Bridging Paper Addendum)"

CSS = """
@page { size: Letter; margin: 1in 1.1in; }
html { font-family: 'Liberation Serif', 'Times New Roman', Times, serif; }
body {
  font-size: 11.5pt;
  line-height: 1.45;
  color: #111;
  max-width: 720px;
  margin: 0 auto;
}
h1 {
  font-size: 19pt;
  line-height: 1.2;
  margin-top: 0.4em;
  margin-bottom: 0.6em;
  font-weight: bold;
}
h2 {
  font-size: 14pt;
  margin-top: 1.4em;
  margin-bottom: 0.5em;
  font-weight: bold;
  border-bottom: 1px solid #888;
  padding-bottom: 0.2em;
}
h3 {
  font-size: 12pt;
  margin-top: 1.1em;
  margin-bottom: 0.4em;
  font-weight: bold;
}
p { margin: 0.55em 0; text-align: justify; hyphens: auto; }
ul, ol { margin: 0.4em 0 0.6em 1.6em; padding: 0; }
li { margin: 0.25em 0; }
hr {
  border: 0;
  border-top: 1px solid #888;
  margin: 1em 0;
}
code {
  font-family: 'Liberation Mono', 'Courier New', Courier, monospace;
  font-size: 0.92em;
  background: #f3f3f3;
  padding: 0 0.2em;
  border-radius: 2px;
}
pre {
  font-family: 'Liberation Mono', 'Courier New', Courier, monospace;
  background: #f3f3f3;
  padding: 0.5em 0.7em;
  border-radius: 3px;
  overflow-x: auto;
  font-size: 0.92em;
  line-height: 1.35;
}
table {
  border-collapse: collapse;
  margin: 0.8em auto;
  font-size: 10.5pt;
}
th, td {
  border: 1px solid #999;
  padding: 0.25em 0.65em;
  text-align: left;
}
th {
  background: #eee;
  font-weight: bold;
}
em { font-style: italic; }
strong { font-weight: bold; }
blockquote {
  border-left: 3px solid #888;
  padding-left: 0.9em;
  color: #333;
  margin: 0.6em 0;
}
"""


def md_to_html_body(md_text: str) -> str:
    md = MarkdownIt("commonmark", {"breaks": False, "html": True}).enable("table")
    return md.render(md_text)


def wrap_html(body: str, title: str) -> str:
    return (
        "<!doctype html>\n"
        "<html lang=\"en\">\n"
        "<head>\n"
        "<meta charset=\"utf-8\">\n"
        f"<title>{title}</title>\n"
        f"<style>{CSS}</style>\n"
        "</head>\n"
        "<body>\n"
        f"{body}\n"
        "</body>\n"
        "</html>\n"
    )


def render_pdf(html: str, pdf_path: pathlib.Path) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_html = pathlib.Path(tmpdir) / "note.html"
        tmp_html.write_text(html, encoding="utf-8")
        cmd = [
            "/opt/google/chrome/chrome",
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--no-pdf-header-footer",
            "--virtual-time-budget=10000",
            f"--print-to-pdf={pdf_path}",
            f"file://{tmp_html}",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            raise RuntimeError(
                f"Chrome render failed (rc={result.returncode})\n"
                f"stdout: {result.stdout}\nstderr: {result.stderr}"
            )


def main() -> None:
    md_text = MD_PATH.read_text(encoding="utf-8")
    html_body = md_to_html_body(md_text)
    html = wrap_html(html_body, TITLE)
    render_pdf(html, PDF_PATH)
    print(f"Wrote {PDF_PATH} ({PDF_PATH.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
