"""Single-source conference-deck generator: one slide spec -> Beamer (.tex) AND self-contained
HTML (.html, driven by a vendored zero-dependency slides.js). Shared OpenWorld theme so all four
talks look identical.

    python presentations/build_decks.py           # build every deck in presentations/decks/
    python presentations/build_decks.py world_computing   # just one

Each deck module in presentations/decks/<name>.py defines:
    TITLE, SUBTITLE, AUTHOR, VENUE  (strings)
    slides = [ {"type": ...}, ... ]   # see the slide types below

Slide types: section | bullets | figure | twocol | statement  (fields documented in the README).
Figures are given as "figs/NAME.png" (as in the papers); they are copied from papers/assets/figs/
into each deck's own figs/ so the folder is self-contained and portable.
"""
import os, sys, shutil, importlib.util, html, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PRES = ROOT / "presentations"
DECKS = PRES / "decks"
FIGSRC = ROOT / "papers" / "assets" / "figs"

# ---- OpenWorld palette (blue / teal / ochre depth ramp on warm paper) --------------------
C = dict(paper="FBFAF6", ink="16242E", deep="0B2E4F", blue="1E6FB0",
         teal="0F8C8C", ochre="D98A2B", muted="5B6B78", line="E4DFD3")

def subst(template, colors):
    """Replace @name@ color tokens; leaves literal % (LaTeX comments, CSS units) untouched."""
    for k, v in colors.items():
        template = template.replace(f"@{k}@", v)
    return template

# =========================================================================================
# BEAMER
# =========================================================================================
BEAMER_PREAMBLE = r"""\documentclass[aspectratio=169,11pt]{beamer}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{graphicx}
\usepackage{xcolor}
\usepackage{tikz}
\usetikzlibrary{calc}
\graphicspath{{figs/}}
\usefonttheme{professionalfonts}
\setbeamertemplate{navigation symbols}{}
\definecolor{owpaper}{HTML}{@paper@}
\definecolor{owink}{HTML}{@ink@}
\definecolor{owdeep}{HTML}{@deep@}
\definecolor{owblue}{HTML}{@blue@}
\definecolor{owteal}{HTML}{@teal@}
\definecolor{owochre}{HTML}{@ochre@}
\definecolor{owmuted}{HTML}{@muted@}
\setbeamercolor{background canvas}{bg=owpaper}
\setbeamercolor{normal text}{fg=owink}
\setbeamercolor{frametitle}{fg=owdeep}
\setbeamercolor{itemize item}{fg=owteal}
\setbeamercolor{itemize subitem}{fg=owochre}
\setbeamercolor{structure}{fg=owblue}
\setbeamerfont{frametitle}{series=\bfseries,size=\Large}
\setbeamertemplate{itemize item}{\small\raise0.5pt\hbox{\textbullet}}
\setbeamertemplate{frametitle}{%
  \vskip6pt\hbox{\color{owochre}\rule[2pt]{16pt}{2.2pt}}\hskip4pt
  \insertframetitle\par\vskip-2pt}
% nested-worlds mark
\newcommand{\owmark}[1][0.5]{\begin{tikzpicture}[scale=#1]
  \draw[owdeep,line width=1.1pt,rounded corners=1pt] (0,0) rectangle (1,1);
  \draw[owblue,line width=1.1pt,rounded corners=1pt] (0.22,0.22) rectangle (0.78,0.78);
  \fill[owteal,rounded corners=0.5pt] (0.40,0.40) rectangle (0.60,0.60);
\end{tikzpicture}}
\setbeamertemplate{footline}{\hbox{}\hfill{\color{owmuted}\tiny\insertframenumber}\hskip8pt\vskip4pt}
"""

BEAMER_TITLE = r"""\begin{frame}[plain]
\centering\vfill
{\owmark[1.4]}\par\vskip14pt
{\color{owdeep}\bfseries\LARGE %(title)s\par}\vskip8pt
{\color{owblue}\large %(subtitle)s\par}\vskip18pt
{\color{owink}\normalsize %(author)s\par}\vskip3pt
{\color{owmuted}\small %(venue)s\par}
\vfill
\end{frame}
"""

def _btext(s):
    return (s.replace("\\", r"\textbackslash{}").replace("&", r"\&").replace("%", r"\%")
             .replace("_", r"\_").replace("#", r"\#").replace("$", r"\$")
             .replace("~", r"\textasciitilde{}").replace("^", r"\textasciicircum{}"))

def beamer_slide(s):
    t = s.get("type")
    if t == "section":
        return (r"""\begin{frame}[plain]\centering\vfill
{\color{owochre}\rule{40pt}{2.5pt}}\par\vskip10pt
{\color{owdeep}\bfseries\Large %s\par}\vfill\end{frame}""" % _btext(s["title"]))
    title = _btext(s.get("title", ""))
    if t == "statement":
        return (r"""\begin{frame}[plain]\centering\vfill
{\color{owdeep}\bfseries\LARGE %s\par}\vfill\end{frame}""" % _btext(s["text"]))
    def bullets(bs):
        if not bs: return ""
        items = "\n".join(r"  \item %s" % _btext(b) for b in bs)
        return "\\begin{itemize}\n%s\n\\end{itemize}" % items
    if t == "bullets":
        return (r"""\begin{frame}{%s}
%s
\end{frame}""" % (title, bullets(s.get("bullets", []))))
    if t == "figure":
        cap = s.get("caption", "")
        capl = (r"\vskip3pt{\color{owmuted}\footnotesize %s\par}" % _btext(cap)) if cap else ""
        has_b = bool(s.get("bullets"))
        bl = ("\\vskip3pt{\\footnotesize " + bullets(s.get("bullets", [])) + "}") if has_b else ""
        h = "0.48" if has_b else "0.74"   # leave room for caption + bullets when present
        return (r"""\begin{frame}{%s}
\centering
\includegraphics[width=\linewidth,height=%s\textheight,keepaspectratio]{%s}%s
\flushleft %s
\end{frame}""" % (title, h, Path(s["image"]).name, capl, bl))
    if t == "twocol":
        return (r"""\begin{frame}{%s}
\begin{columns}[T]
\begin{column}{0.52\textwidth}%s\end{column}
\begin{column}{0.46\textwidth}\centering
\includegraphics[width=\linewidth,height=0.72\textheight,keepaspectratio]{%s}\end{column}
\end{columns}
\end{frame}""" % (title, bullets(s.get("bullets", [])), Path(s["image"]).name))
    return ""

def build_beamer(deck):
    body = "\n".join(beamer_slide(s) for s in deck["slides"])
    meta = {k: _btext(deck[k]) for k in ("title", "subtitle", "author", "venue")}
    return (subst(BEAMER_PREAMBLE, C) + "\n\\begin{document}\n"
            + BEAMER_TITLE % meta + "\n" + body + "\n\\end{document}\n")

# =========================================================================================
# HTML  (self-contained: inline CSS + vendored slides.js)
# =========================================================================================
HTML_CSS = """
:root{--paper:#@paper@;--ink:#@ink@;--deep:#@deep@;--blue:#@blue@;--teal:#@teal@;--ochre:#@ochre@;--muted:#@muted@;--line:#@line@;}
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%%;background:#0a1620;color:var(--ink);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}
#deck{position:fixed;inset:0}
/* every slide: title anchored at top, body fills the rest, generous bottom safe-area for the chrome */
.slide{position:absolute;inset:0;display:none;flex-direction:column;
  padding:6vh 7vw 8vh;background:var(--paper);opacity:0;transition:opacity .25s ease}
.slide.on{display:flex;opacity:1}
.slide h2{color:var(--deep);font-size:3.7vh;line-height:1.15;font-weight:800;flex:0 0 auto;margin-bottom:2.2vh}
.slide h2::before{content:"";display:inline-block;width:32px;height:5px;background:var(--ochre);
  border-radius:3px;margin-right:14px;vertical-align:middle}
/* body: fills remaining height and vertically centres its content, never overflows */
.body{flex:1 1 auto;min-height:0;display:flex;flex-direction:column;justify-content:center;gap:1.8vh}
ul{list-style:none;max-width:74ch}
li{position:relative;padding-left:28px;margin:1.7vh 0;font-size:3.05vh;line-height:1.34;color:var(--ink)}
li::before{content:"";position:absolute;left:0;top:.55em;width:11px;height:11px;border-radius:3px;background:var(--teal)}
/* figure fits into whatever height remains after title/caption/bullets -- so nothing spills */
.fig{display:flex;align-items:center;justify-content:center;width:100%%;min-height:0}
.fig img{max-width:84vw;object-fit:contain;border-radius:8px;box-shadow:0 5px 26px rgba(11,46,79,.12)}
.cap{flex:0 0 auto;color:var(--muted);font-size:2.1vh;text-align:center;line-height:1.3}
.figbody{align-items:center}
.figbody .fig img{max-height:64vh}
.figbody.hasbul .fig img{max-height:44vh}
.figbody ul{flex:0 0 auto;max-width:82ch;width:100%%}
.figbody li{font-size:2.45vh;margin:1vh 0}
.two{display:flex;gap:4vw;flex:1 1 auto;align-items:center;min-height:0;width:100%%}
.two .col{flex:1;min-width:0}
.two .col.img{display:flex;align-items:center;justify-content:center}
.two .col.img img{max-width:42vw;max-height:70vh;object-fit:contain;border-radius:8px;box-shadow:0 5px 26px rgba(11,46,79,.12)}
.two li{font-size:2.7vh}
/* title / section / statement: full-slide centred */
.title,.section,.statement{align-items:center;justify-content:center;text-align:center;padding-bottom:6vh}
.title .t{color:var(--deep);font-size:5.6vh;font-weight:800;line-height:1.1;max-width:22ch;margin:2.4vh 0 1.6vh}
.title .s{color:var(--blue);font-size:3.1vh;max-width:34ch;line-height:1.3}
.title .a{color:var(--ink);font-size:2.5vh;margin-top:3vh}
.title .v{color:var(--muted);font-size:2vh;margin-top:.6vh}
.section .st{color:var(--deep);font-size:5vh;font-weight:800;max-width:24ch;line-height:1.15}
.section .rule,.title .rule{width:54px;height:6px;background:var(--ochre);border-radius:3px}
.statement .big{color:var(--deep);font-size:4.6vh;font-weight:800;line-height:1.24;max-width:26ch}
/* nested-worlds mark */
.mark{display:inline-block;position:relative}
.mark i{position:absolute;border-radius:3px}
/* chrome (kept clear of content by the slide's bottom padding) */
#bar{position:fixed;left:0;bottom:0;height:5px;background:var(--teal);z-index:9;transition:width .25s}
#num{position:fixed;right:16px;bottom:12px;color:var(--muted);font-size:1.8vh;z-index:9;font-variant-numeric:tabular-nums}
#brand{position:fixed;left:16px;bottom:9px;z-index:9;display:flex;align-items:center;gap:8px;color:var(--muted);font-size:1.7vh;opacity:.85}
@media print{.slide{display:flex!important;opacity:1!important;position:relative;page-break-after:always;height:100vh}}
"""

def _mark_html(scale=1.0):
    u = 74 * scale
    return (f'<span class="mark" style="width:{u}px;height:{u}px">'
            f'<i style="inset:0;border:2.4px solid var(--deep)"></i>'
            f'<i style="inset:{u*0.22:.0f}px;border:2.4px solid var(--blue)"></i>'
            f'<i style="inset:{u*0.40:.0f}px;background:var(--teal)"></i></span>')

HTML_JS = """
(function(){
  var slides=[].slice.call(document.querySelectorAll('.slide'));
  var bar=document.getElementById('bar'), num=document.getElementById('num');
  var i=Math.max(0,Math.min(slides.length-1,parseInt(location.hash.slice(1))||0));
  function show(n){i=Math.max(0,Math.min(slides.length-1,n));
    slides.forEach(function(s,k){s.classList.toggle('on',k===i)});
    bar.style.width=((i+1)/slides.length*100)+'%';
    num.textContent=(i+1)+' / '+slides.length; location.hash=i;}
  function next(){show(i+1)} function prev(){show(i-1)}
  document.addEventListener('keydown',function(e){
    if(['ArrowRight','ArrowDown','PageDown',' '].indexOf(e.key)>=0){next();e.preventDefault();}
    else if(['ArrowLeft','ArrowUp','PageUp'].indexOf(e.key)>=0){prev();e.preventDefault();}
    else if(e.key==='Home'){show(0);} else if(e.key==='End'){show(slides.length-1);}
    else if(e.key==='f'){var d=document.documentElement; (document.fullscreenElement?document.exitFullscreen():d.requestFullscreen&&d.requestFullscreen());}
  });
  var sx=null;
  document.addEventListener('touchstart',function(e){sx=e.touches[0].clientX});
  document.addEventListener('touchend',function(e){if(sx===null)return;var dx=e.changedTouches[0].clientX-sx;
    if(Math.abs(dx)>50){dx<0?next():prev();}sx=null;});
  window.addEventListener('hashchange',function(){var n=parseInt(location.hash.slice(1));if(!isNaN(n)&&n!==i)show(n);});
  show(i);
})();
"""

def _h(s):
    return html.escape(str(s), quote=True)

def html_slide(s):
    t = s.get("type")
    if t == "section":
        return f'<section class="slide section"><div class="rule"></div><div class="st">{_h(s["title"])}</div></section>'
    if t == "statement":
        return f'<section class="slide statement"><div class="big">{_h(s["text"])}</div></section>'
    title = f'<h2>{_h(s.get("title",""))}</h2>'
    def ul(bs):
        if not bs: return ""
        return "<ul>" + "".join(f"<li>{_h(b)}</li>" for b in bs) + "</ul>"
    if t == "bullets":
        return f'<section class="slide">{title}<div class="body">{ul(s.get("bullets",[]))}</div></section>'
    if t == "figure":
        cap = f'<div class="cap">{_h(s["caption"])}</div>' if s.get("caption") else ""
        bl = ul(s.get("bullets", []))
        img = f'<div class="fig"><img src="figs/{Path(s["image"]).name}" alt="{_h(s.get("caption") or s.get("title",""))}"></div>'
        hasb = " hasbul" if s.get("bullets") else ""
        return f'<section class="slide">{title}<div class="body figbody{hasb}">{img}{cap}{bl}</div></section>'
    if t == "twocol":
        return (f'<section class="slide">{title}<div class="body"><div class="two">'
                f'<div class="col">{ul(s.get("bullets",[]))}</div>'
                f'<div class="col img"><img src="figs/{Path(s["image"]).name}" alt="{_h(s.get("title",""))}"></div>'
                f'</div></div></section>')
    return ""

def build_html(deck):
    title_slide = (f'<section class="slide title on">{_mark_html(1.3)}'
                   f'<div class="rule" style="margin-top:2vh"></div>'
                   f'<div class="t">{_h(deck["title"])}</div>'
                   f'<div class="s">{_h(deck["subtitle"])}</div>'
                   f'<div class="a">{_h(deck["author"])}</div>'
                   f'<div class="v">{_h(deck["venue"])}</div></section>')
    body = title_slide + "".join(html_slide(s) for s in deck["slides"])
    css = subst(HTML_CSS, C)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_h(deck["title"])}</title><style>{css}</style></head>
<body><div id="deck">{body}</div>
<div id="bar"></div><div id="num"></div>
<div id="brand">{_mark_html(0.28)}<span>OpenWorld</span></div>
<script>{HTML_JS}</script></body></html>"""

# =========================================================================================
def load_deck(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return dict(title=mod.TITLE, subtitle=mod.SUBTITLE, author=getattr(mod, "AUTHOR", ""),
                venue=getattr(mod, "VENUE", ""), slides=mod.slides, slug=path.stem)

def copy_figs(deck, outdir):
    figs = outdir / "figs"; figs.mkdir(parents=True, exist_ok=True)
    missing = []
    for s in deck["slides"]:
        img = s.get("image")
        if not img: continue
        name = Path(img).name
        src = FIGSRC / name
        if src.exists():
            shutil.copy(src, figs / name)
        else:
            missing.append(name)
    return missing

def build(path):
    deck = load_deck(path)
    outdir = PRES / deck["slug"]; outdir.mkdir(parents=True, exist_ok=True)
    missing = copy_figs(deck, outdir)
    (outdir / f"{deck['slug']}.tex").write_text(build_beamer(deck))
    (outdir / f"{deck['slug']}.html").write_text(build_html(deck))
    n = len(deck["slides"]) + 1
    print(f"  {deck['slug']}: {n} slides -> {deck['slug']}.tex + .html"
          + (f"  [MISSING figs: {missing}]" if missing else ""))

def main():
    which = sys.argv[1:]
    mods = sorted(DECKS.glob("*.py"))
    mods = [m for m in mods if not which or m.stem in which]
    print("building decks:")
    for m in mods:
        try: build(m)
        except Exception as e: print(f"  {m.stem}: ERROR {e}")

if __name__ == "__main__":
    main()
