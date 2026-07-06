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

# ---- on-brand monoline icons (24x24, stroke=currentColor) for cards / flow / stats ----------
ICONS = {
    "eye":     '<path d="M2 12s3.6-6.5 10-6.5 10 6.5 10 6.5-3.6 6.5-10 6.5S2 12 2 12z"/><circle cx="12" cy="12" r="3"/>',
    "compass": '<circle cx="12" cy="12" r="9"/><path d="M15.6 8.4l-2.3 4.9-4.9 2.3 2.3-4.9z"/>',
    "code":    '<path d="M9 7l-5 5 5 5"/><path d="M15 7l5 5-5 5"/>',
    "target":  '<circle cx="12" cy="12" r="8.5"/><circle cx="12" cy="12" r="4"/><circle cx="12" cy="12" r="1" fill="currentColor" stroke="none"/>',
    "branch":  '<circle cx="6" cy="18" r="2.4"/><circle cx="6" cy="6" r="2.4"/><circle cx="18" cy="6" r="2.4"/><path d="M6 8.4v7.2"/><path d="M18 8.4v1.6a4 4 0 01-4 4H6"/>',
    "check":   '<circle cx="12" cy="12" r="9"/><path d="M8 12.5l2.6 2.6L16 9.5"/>',
    "shield":  '<path d="M12 3l7.5 3v5c0 5-3.7 8.7-7.5 10-3.8-1.3-7.5-5-7.5-10V6z"/><path d="M8.5 12l2.3 2.3L15.5 10"/>',
    "bolt":    '<path d="M13 2L4 13h6l-1 9 9-11h-6z"/>',
    "bars":    '<path d="M5 20V11M12 20V4M19 20v-6"/><path d="M4 20h16"/>',
    "nested":  '<rect x="3.5" y="3.5" width="17" height="17" rx="2"/><rect x="8" y="8" width="8" height="8" rx="1.5"/>',
    "grid":    '<rect x="3.5" y="3.5" width="17" height="17" rx="2"/><path d="M3.5 9.2h17M3.5 14.8h17M9.2 3.5v17M14.8 3.5v17"/>',
    "gear":    '<circle cx="12" cy="12" r="3.2"/><path d="M12 3v3M12 18v3M3 12h3M18 12h3M5.6 5.6l2.1 2.1M16.3 16.3l2.1 2.1M5.6 18.4l2.1-2.1M16.3 7.7l2.1-2.1"/>',
    "stack":   '<path d="M12 3l9 4.5-9 4.5-9-4.5z"/><path d="M3 12l9 4.5 9-4.5"/><path d="M3 16.5L12 21l9-4.5"/>',
    "gauge":   '<path d="M4 17a8 8 0 1116 0"/><path d="M12 17l3.4-4.4"/><circle cx="12" cy="17" r="1" fill="currentColor" stroke="none"/>',
    "lock":    '<rect x="5" y="10.5" width="14" height="9.5" rx="2"/><path d="M8 10.5V7a4 4 0 018 0v3.5"/>',
    "flask":   '<path d="M9 3h6M10 3v5.5l-4.6 8.1A2 2 0 007.2 20h9.6a2 2 0 001.8-3.4L14 8.5V3"/>',
    "route":   '<circle cx="5" cy="19" r="1.8"/><circle cx="19" cy="5" r="1.8"/><path d="M5 17c0-6 8-6 8-12"/>',
    "scan":    '<path d="M4 8V5a1 1 0 011-1h3M16 4h3a1 1 0 011 1v3M20 16v3a1 1 0 01-1 1h-3M8 20H5a1 1 0 01-1-1v-3"/><path d="M4 12h16"/>',
    "cube":    '<path d="M12 3l8 4.5v9L12 21l-8-4.5v-9z"/><path d="M12 12l8-4.5M12 12v9M12 12L4 7.5"/>',
    "atom":    '<circle cx="12" cy="12" r="2"/><ellipse cx="12" cy="12" rx="9" ry="4"/><ellipse cx="12" cy="12" rx="9" ry="4" transform="rotate(60 12 12)"/><ellipse cx="12" cy="12" rx="9" ry="4" transform="rotate(120 12 12)"/>',
}

def _svg(name, cls="ic"):
    p = ICONS.get(name)
    if not p:
        return ""
    return (f'<svg class="{cls}" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            f'stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round">{p}</svg>')

# ---- animated concept diagrams (self-contained SVG/CSS; animations play when the slide is shown) ----
DIAGRAMS = {
    # verified code stays exact; a per-step LLM drifts -- the two curves draw themselves
    "compounding": '''<svg viewBox="0 0 860 430" class="dgm" preserveAspectRatio="xMidYMid meet">
  <line class="axis" x1="72" y1="368" x2="820" y2="368"/><line class="axis" x1="72" y1="42" x2="72" y2="368"/>
  <text class="axl" x="446" y="410">rollout depth &#8594;</text>
  <text class="axl" x="30" y="205" transform="rotate(-90 30 205)">state error &#8594;</text>
  <path class="err" d="M72 250 L820 250 L820 74 C 450 158, 260 246, 72 250 Z"/>
  <path class="code draw" pathLength="1" d="M72 250 L820 250"/>
  <path class="llm draw" pathLength="1" d="M72 250 C 260 246, 450 158, 820 74"/>
  <circle class="dot code-dot" cx="820" cy="250" r="7"/><circle class="dot llm-dot" cx="820" cy="74" r="7"/>
  <text class="lbl-code" x="430" y="238">verified code &#8212; exact at every depth</text>
  <text class="lbl-llm" x="470" y="66">per-step LLM &#8212; drifts by ~step 2.3</text>
</svg>''',
    # the win is an ordered procedure that lights up A->B->C->WIN; state-scoring fills a gauge and stalls
    "procedure": '''<div class="dgmp">
  <div class="lane">
    <span class="plabel">Goal-as-procedure &#8212; the win</span>
    <div class="pnodes"><span class="pnode n1">A</span><span class="parrow a1">&#8594;</span>
      <span class="pnode n2">B</span><span class="parrow a2">&#8594;</span>
      <span class="pnode n3">C</span><span class="parrow a3">&#8594;</span>
      <span class="pwin">WIN &#10003;</span></div>
    <div class="pcap">Reason the ordered sequence &#8212; the only path through</div>
  </div>
  <div class="lane">
    <span class="plabel muted">Goal-as-state &#8212; fails</span>
    <div class="gauge"><div class="gfill"></div><span class="gx">&#10007; stalls, never wins</span></div>
    <div class="pcap">Score one screen: climbs to a high value, but cannot rank the sequence</div>
  </div>
</div>''',
    # a monolith cracks past ~8 rules; composed verified parts stay high -- two bars grow in
    "cliff": '''<div class="dgmc">
  <div class="cbar-wrap"><div class="cbar cbar-mono"><span class="cval">0.31</span></div>
    <div class="clbl">Monolithic<br><small>one 16-rule synthesis &#8212; cracks</small></div></div>
  <div class="cbar-wrap"><div class="cbar cbar-comp"><span class="cval">0.92</span></div>
    <div class="clbl">Composition<br><small>four verified 4-rule parts + bridges</small></div></div>
</div>''',
    # what a world model IS: (state, action) -> World -> next state, agent is separate
    "worldmodel": '''<div class="dgmm">
  <div class="mm-row">
    <div class="mm-in"><span class="mm-chip mm-s">state <b>s</b></span><span class="mm-chip mm-a">action <b>a</b></span></div>
    <span class="mm-ar a1">&#8594;</span>
    <div class="mm-world"><span class="mm-mark"></span><span class="mm-wl">World</span><small>verified transition</small></div>
    <span class="mm-ar a2">&#8594;</span>
    <span class="mm-chip mm-sp">next state <b>s&#8242;</b></span>
  </div>
  <div class="mm-cap">One World is both the environment and the planning model &#8212; the agent is a separate policy</div>
</div>''',
    # many verified worlds feed one skill, which generalises to a held-out world
    "worldtime": '''<div class="dgmw ag">
  <div class="wt-src"><span class="wt-w w1"></span><span class="wt-w w2"></span><span class="wt-w w3"></span>
    <span class="wt-w w4"></span><span class="wt-w w5"></span><span class="wt-w w6"></span>
    <div class="wt-cap">many verified worlds</div></div>
  <div class="wt-flow"><span class="wt-arrow"></span></div>
  <div class="wt-hub"><span class="wt-core"></span><div class="wt-cap">one distilled skill</div></div>
  <div class="wt-flow"><span class="wt-arrow a2"></span></div>
  <div class="wt-out"><span class="wt-new"></span><div class="wt-cap">a new, held-out world</div></div>
</div>''',
}

# =========================================================================================
# BEAMER
# =========================================================================================
BEAMER_PREAMBLE = r"""\documentclass[aspectratio=169,11pt]{beamer}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{graphicx}
\usepackage{xcolor}
\usepackage{tikz}
\usetikzlibrary{calc,positioning,arrows.meta}
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

def _beamer_atlas(parts, current=None):
    node, lbl = [], []
    for i, p in enumerate(parts):
        sty = "stn" if current is None else ("std" if i < current else "stc" if i == current else "stx")
        x = i * 3.4
        node.append(r"\node[%s] (s%d) at (%.1f,0) {%d};" % (sty, i, x, i + 1))
        lbl.append(r"\node[stlbl] at (%.1f,-1.15) {%s};" % (x, _btext(p)))
    lines = "".join(r"\draw[stln] (s%d)--(s%d);" % (i - 1, i) for i in range(1, len(parts)))
    return (r"\resizebox{0.95\textwidth}{!}{\begin{tikzpicture}["
            r"stn/.style={circle,draw=owteal,line width=1pt,minimum size=9mm,font=\bfseries,inner sep=0,text=owteal},"
            r"std/.style={circle,draw=owteal,fill=owteal,line width=1pt,minimum size=9mm,font=\bfseries,inner sep=0,text=white},"
            r"stc/.style={circle,draw=owochre,fill=owochre,line width=1.2pt,minimum size=11.5mm,font=\bfseries,inner sep=0,text=white},"
            r"stx/.style={circle,draw=owmuted,line width=1pt,minimum size=9mm,font=\bfseries,inner sep=0,text=owmuted},"
            r"stlbl/.style={font=\scriptsize,text=owmuted,align=center,text width=28mm},"
            r"stln/.style={draw=black!16,line width=1.4pt}]"
            + "".join(node) + "".join(lbl) + lines + r"\end{tikzpicture}}")

def beamer_slide(s, kicker="", parts=None, part_index=None):
    t = s.get("type")
    if t == "section":
        return (r"""\begin{frame}[plain]\centering\vfill
{\color{owochre}\rule{40pt}{2.5pt}}\par\vskip10pt
{\color{owdeep}\bfseries\Large %s\par}\vfill\end{frame}""" % _btext(s["title"]))
    if t == "agenda":
        return (r"\begin{frame}{%s}\vfill\centering %s\vfill\end{frame}"
                % (_btext(s.get("title") or "Roadmap"), _beamer_atlas(parts or [], None)))
    if t == "part":
        sub = (r"\\[10pt]{\color{owmuted}\large %s}" % _btext(s["subtitle"])) if s.get("subtitle") else ""
        return (r"""\begin{frame}[plain]\vfill\centering %s\\[24pt]
{\color{owochre}\bfseries\footnotesize PART %d}\\[5pt]
{\color{owdeep}\bfseries\Large %s}%s\vfill\end{frame}"""
                % (_beamer_atlas(parts or [], part_index), (part_index or 0) + 1, _btext(s["title"]), sub))
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
    if t == "stats":
        items = s["items"]
        w = 0.97 / len(items)
        cols = "".join(
            (r"\begin{column}{%.3f\textwidth}\centering{\color{%s}\fontsize{40}{44}\selectfont\bfseries %s}\\[6pt]"
             r"{\color{owmuted}\small %s}\end{column}"
             % (w, "owochre" if it.get("hi") else "owdeep", _btext(it["value"]), _btext(it["label"])))
            for it in items)
        return r"\begin{frame}{%s}\vfill\begin{columns}[c]%s\end{columns}\vfill\end{frame}" % (title, cols)
    if t == "flow":
        steps = s["steps"]
        nodes = "".join(
            (r"\node[sb%s] (n%d) {\textbf{%d}\\ %s};"
             % (("" if i == 0 else ",right=of n%d" % (i - 1)), i, i + 1, _btext(st)))
            for i, st in enumerate(steps))
        arrows = "".join(r"\draw[-{Latex[length=2mm]},owochre,line width=1.1pt] (n%d) -- (n%d);"
                         % (i - 1, i) for i in range(1, len(steps)))
        tw = 22 if len(steps) <= 5 else 18
        return (r"""\begin{frame}{%s}\vfill\centering
\resizebox{0.98\textwidth}{!}{\begin{tikzpicture}[node distance=5mm and 5mm,
  sb/.style={draw=owteal,line width=1pt,rounded corners=4pt,fill=white,inner sep=5pt,
  text width=%dmm,align=center,minimum height=17mm,font=\small,text=owdeep}]
%s
%s
\end{tikzpicture}}\vfill\end{frame}""" % (title, tw, nodes, arrows))
    if t == "compare":
        def side(d, color):
            items = "\n".join(r"\item %s" % _btext(x) for x in d["items"])
            return (r"{\color{%s}\bfseries\large %s}\par\medskip\begin{itemize}\small %s\end{itemize}"
                    % (color, _btext(d["head"]), items))
        return (r"""\begin{frame}{%s}\vfill
\begin{columns}[T]
\begin{column}{0.46\textwidth}%s\end{column}
\begin{column}{0.46\textwidth}%s\end{column}
\end{columns}\vfill\end{frame}""" % (title, side(s["left"], "owmuted"), side(s["right"], "owteal")))
    if t == "cards":
        cards = s["cards"]; w = 0.95 / len(cards)
        accent = ["owteal", "owochre", "owblue", "owmuted"]
        cols = "".join(
            (r"\begin{column}{%.3f\textwidth}{\color{%s}\rule{20pt}{3pt}}\par\smallskip"
             r"{\color{owdeep}\bfseries %s}\par\smallskip{\small %s}\end{column}"
             % (w, accent[i % 4], _btext(c["head"]), _btext(c.get("text", ""))))
            for i, c in enumerate(cards))
        return r"\begin{frame}{%s}\vfill\begin{columns}[T]%s\end{columns}\vfill\end{frame}" % (title, cols)
    if t == "anim":
        still = s.get("still")
        if not still:
            return r"\begin{frame}{%s}\end{frame}" % title
        return (r"\begin{frame}{%s}\centering\includegraphics[width=\linewidth,"
                r"height=0.74\textheight,keepaspectratio]{%s}\end{frame}" % (title, Path(still).name))
    return ""

def build_beamer(deck):
    body = "\n".join(beamer_slide(s, k, parts, pi) for s, k, pi, parts in _part_walk(deck["slides"]))
    meta = {k: _btext(deck[k]) for k in ("title", "subtitle", "author", "venue")}
    return (subst(BEAMER_PREAMBLE, C) + "\n\\begin{document}\n"
            + BEAMER_TITLE % meta + "\n" + body + "\n\\end{document}\n")

# =========================================================================================
# HTML  (self-contained: inline CSS + vendored slides.js)
# =========================================================================================
HTML_CSS = """
:root{--paper:#@paper@;--ink:#@ink@;--deep:#@deep@;--blue:#@blue@;--teal:#@teal@;--ochre:#@ochre@;--muted:#@muted@;--line:#@line@;}
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%;background:#0a1620;color:var(--ink);
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
.fig{display:flex;align-items:center;justify-content:center;width:100%;min-height:0}
.fig img{max-width:84vw;object-fit:contain;border-radius:8px;box-shadow:0 5px 26px rgba(11,46,79,.12)}
.cap{flex:0 0 auto;color:var(--muted);font-size:2.1vh;text-align:center;line-height:1.3}
.figbody{align-items:center}
.figbody .fig img{max-height:64vh}
.figbody.hasbul .fig img{max-height:44vh}
.figbody ul{flex:0 0 auto;max-width:82ch;width:100%}
.figbody li{font-size:2.45vh;margin:1vh 0}
.two{display:flex;gap:4vw;flex:1 1 auto;align-items:center;min-height:0;width:100%}
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
/* section kicker (small label above the title) */
.kick{flex:0 0 auto;color:var(--teal);font-size:1.8vh;font-weight:800;letter-spacing:.14em;
  text-transform:uppercase;margin-bottom:.7vh}
/* bullet slides -> centred stack of card-rows (balanced margins, not jammed left) */
.slide:has(.blist)>h2,.slide:has(.blist)>.kick{width:100%;max-width:72rem;margin-left:auto;margin-right:auto}
.blist{gap:0;justify-content:center}
.blist>ul{display:flex;flex-direction:column;gap:1.5vh;max-width:72rem;width:100%;margin:0 auto}
.blist li{background:#ffffff;border-radius:12px;border-left:6px solid var(--teal);
  padding:1.9vh 2.2vw;font-size:2.75vh;line-height:1.34;color:var(--ink);box-shadow:0 3px 15px rgba(11,46,79,.06)}
.blist li::before{display:none}
/* card grid (2-4 parallel points as cards) */
.cards{display:flex;gap:2.2vw;width:100%;align-items:stretch}
.card{flex:1;min-width:0;background:#ffffff;border-radius:14px;padding:3vh 1.8vw;
  box-shadow:0 5px 22px rgba(11,46,79,.08);border-top:6px solid var(--teal)}
.card h4{color:var(--deep);font-size:2.9vh;font-weight:800;margin-bottom:1vh;line-height:1.15}
.card p{color:var(--muted);font-size:2.1vh;line-height:1.36}
.card:nth-child(2){border-top-color:var(--ochre)}
.card:nth-child(3){border-top-color:var(--blue)}
.card:nth-child(4){border-top-color:var(--muted)}
/* icons */
.ic{display:block;flex:0 0 auto}
.card .ic{width:6vh;height:6vh;color:var(--teal);margin-bottom:1.6vh}
.card:nth-child(2) .ic{color:var(--ochre)}
.card:nth-child(3) .ic{color:var(--blue)}
.card:nth-child(4) .ic{color:var(--muted)}
.flow .step .ic{width:3.6vh;height:3.6vh;color:var(--teal);margin:0 auto .7vh}
.stat .ic{width:4.4vh;height:4.4vh;color:var(--teal);margin:0 auto 1.2vh}
.stat.hi .ic{color:var(--ochre)}
.compare .side h3 .ic{width:3.2vh;height:3.2vh;vertical-align:-.5vh;margin-right:.6vw;display:inline-block}
/* hero stats */
.stats{display:flex;gap:2.6vw;justify-content:center;align-items:stretch;width:100%}
.stat{flex:1 1 0;min-width:0;text-align:center;padding:3vh 1vw;border-radius:14px;background:#ffffff;
  box-shadow:0 5px 24px rgba(11,46,79,.08);border-top:6px solid var(--teal)}
.stat .v{color:var(--deep);font-size:7vh;font-weight:800;line-height:1;letter-spacing:-1px}
.stat .l{color:var(--muted);font-size:2.05vh;margin-top:1.4vh;line-height:1.28}
.stat.hi{border-top-color:var(--ochre)}
.stat.hi .v{color:var(--ochre)}
/* flow of steps */
.flow{display:flex;align-items:stretch;justify-content:center;flex-wrap:wrap;gap:.4vw;width:100%}
.flow .step{background:#ffffff;border:2px solid var(--teal);border-radius:12px;padding:2.2vh 1.1vw;
  text-align:center;color:var(--deep);font-weight:700;font-size:2.4vh;min-width:0;
  display:flex;flex-direction:column;justify-content:center;box-shadow:0 3px 14px rgba(11,46,79,.06)}
.flow .step .n{display:block;color:var(--teal);font-size:1.7vh;font-weight:800;margin-bottom:.5vh}
.flow .arw{display:flex;align-items:center;color:var(--ochre);font-size:3vh;font-weight:800;padding:0 .3vw}
/* side-by-side comparison */
.compare{display:flex;gap:3vw;align-items:stretch;width:100%}
.compare .side{flex:1;background:#ffffff;border-radius:14px;padding:2.6vh 2vw;box-shadow:0 5px 24px rgba(11,46,79,.08)}
.compare .side.a{border-top:6px solid var(--muted)}
.compare .side.b{border-top:6px solid var(--teal)}
.compare .side h3{font-size:2.9vh;color:var(--muted);margin-bottom:1.8vh;font-weight:800}
.compare .side.b h3{color:var(--teal)}
.compare .side ul{max-width:none}
.compare .side li{font-size:2.35vh;margin:1.2vh 0;padding-left:24px}
.compare .side.a li::before{background:var(--muted)}
/* the atlas / roadmap -- a themed journey of parts with a you-are-here marker */
.atlas{display:flex;align-items:flex-start;justify-content:center;width:100%;margin:0 auto}
.station{display:flex;flex-direction:column;align-items:center;text-align:center;flex:0 0 auto;padding:0 .3vw}
.station .dot{width:6.4vh;height:6.4vh;border-radius:50%;display:flex;align-items:center;justify-content:center;
  font-weight:800;font-size:2.5vh;border:2.5px solid var(--teal);color:var(--teal);background:#fff;
  box-shadow:0 3px 12px rgba(11,46,79,.08);transition:transform .2s}
.station .lbl{margin-top:1.3vh;font-size:1.95vh;color:var(--muted);line-height:1.22;max-width:11ch}
.atlas .line{flex:1 1 auto;height:3px;background:var(--line);border-radius:2px;margin-top:3.2vh;min-width:2rem}
.station.done .dot{background:var(--teal);border-color:var(--teal);color:#fff}
.station.done .lbl{color:var(--ink)}
.station.now .dot{background:var(--ochre);border-color:var(--ochre);color:#fff;transform:scale(1.28);
  box-shadow:0 6px 20px rgba(217,138,43,.4)}
.station.now .lbl{color:var(--deep);font-weight:800}
.station.next .dot{opacity:.55}.station.next .lbl{opacity:.6}
.atlas .line.donel{background:var(--teal)}
/* agenda slide */
.agenda .atlas{margin-top:2vh}
/* part divider -- map on top, big theme header below, faint mark behind */
.partslide{justify-content:center;align-items:center;gap:5.5vh;overflow:hidden}
.parthead{text-align:center;z-index:1}
.parthead .pnum{color:var(--ochre);font-size:2.1vh;font-weight:800;letter-spacing:.18em;text-transform:uppercase;margin-bottom:1.2vh}
.parthead .ptitle{color:var(--deep);font-size:5.4vh;font-weight:800;line-height:1.1;max-width:24ch}
.parthead .psub{color:var(--muted);font-size:2.5vh;margin-top:1.6vh;max-width:40ch}
.wm{position:absolute;right:-6vh;bottom:-8vh;width:44vh;height:44vh;opacity:.05;z-index:0}
.wm i{position:absolute;border-radius:8px}
/* nested-worlds mark */
.mark{display:inline-block;position:relative}
.mark i{position:absolute;border-radius:3px}
/* chrome (kept clear of content by the slide's bottom padding) */
#bar{position:fixed;left:0;bottom:0;height:5px;background:var(--teal);z-index:9;transition:width .25s}
#num{position:fixed;right:16px;bottom:12px;color:var(--muted);font-size:1.8vh;z-index:9;font-variant-numeric:tabular-nums}
#brand{position:fixed;left:16px;bottom:9px;z-index:9;display:flex;align-items:center;gap:8px;color:var(--muted);font-size:1.7vh;opacity:.85}
/* ---- animated concept diagrams ---- */
.dgm{width:100%;max-width:80rem;height:58vh}
.dgm .axis{stroke:#C9C2B4;stroke-width:2}
.dgm .axl{fill:var(--muted);font-size:18px;text-anchor:middle}
.dgm .code{fill:none;stroke:var(--teal);stroke-width:5;stroke-linecap:round}
.dgm .llm{fill:none;stroke:#9E2B25;stroke-width:5;stroke-linecap:round}
.dgm .err{fill:#9E2B25;opacity:0}
.dgm .dot{stroke:#fff;stroke-width:2}
.dgm .code-dot{fill:var(--teal)}.dgm .llm-dot{fill:#9E2B25}
.dgm .lbl-code{fill:var(--teal);font-weight:700;font-size:20px}
.dgm .lbl-llm{fill:#9E2B25;font-weight:700;font-size:20px}
.slide.on .draw{stroke-dasharray:1;stroke-dashoffset:1;animation:dashDraw 1.5s .3s cubic-bezier(.4,.1,.2,1) forwards}
.slide.on .err{opacity:0;animation:fadeIn .8s 1.55s both}
.slide.on .dgm .dot{opacity:0;animation:popIn .4s 1.75s both}
.slide.on .lbl-code,.slide.on .lbl-llm{opacity:0;animation:fadeIn .6s 1.6s both}
.dgmp{display:flex;flex-direction:column;gap:5.5vh;width:100%;max-width:72rem;margin:0 auto}
.plabel{font-size:1.75vh;font-weight:800;letter-spacing:.14em;text-transform:uppercase;color:var(--teal)}
.plabel.muted{color:var(--muted)}
.pnodes{display:flex;align-items:center;gap:1.3vw;margin-top:1.8vh}
.pnode{width:8.5vh;height:8.5vh;border-radius:16px;background:#fff;border:2.5px solid var(--teal);
  display:flex;align-items:center;justify-content:center;font-size:3.6vh;font-weight:800;color:var(--deep);
  box-shadow:0 5px 18px rgba(11,46,79,.09)}
.parrow{color:var(--ochre);font-size:3.2vh;font-weight:800}
.pwin{padding:1.6vh 2vw;border-radius:14px;background:var(--teal);color:#fff;font-size:3vh;font-weight:800;
  box-shadow:0 6px 20px rgba(15,140,140,.3)}
.pcap{margin-top:1.5vh;color:var(--muted);font-size:2.15vh}
.gauge{margin-top:1.8vh;height:5.4vh;background:#F0EADF;border-radius:12px;position:relative;overflow:hidden;max-width:54rem}
.gfill{height:100%;width:0;background:#9E2B25;border-radius:12px}
.gx{position:absolute;right:2vw;top:50%;transform:translateY(-50%);color:#fff;font-weight:800;font-size:2.1vh;opacity:0}
.slide.on .pnode,.slide.on .parrow,.slide.on .pwin{opacity:0;animation:popIn .45s both}
.slide.on .n1{animation-delay:.3s}.slide.on .a1{animation-delay:.55s}
.slide.on .n2{animation-delay:.8s}.slide.on .a2{animation-delay:1.05s}
.slide.on .n3{animation-delay:1.3s}.slide.on .a3{animation-delay:1.55s}
.slide.on .pwin{animation-delay:1.9s}
.slide.on .gfill{width:60%;animation:growW 1.5s .5s cubic-bezier(.3,.6,.2,1) both}
.slide.on .gx{animation:fadeIn .5s 2.1s both}
.dgmw{display:flex;align-items:flex-start;justify-content:center;gap:2.6vw;width:100%;max-width:82rem;margin:0 auto}
.wt-src{display:grid;grid-template-columns:repeat(3,1fr);gap:1.3vh 1.1vw}
.wt-w{width:6.4vh;height:6.4vh;border-radius:10px;border:2.5px solid var(--teal);position:relative;background:#fff}
.wt-w::after{content:"";position:absolute;inset:26%;border-radius:4px;background:var(--teal);opacity:.5}
.wt-hub{display:flex;flex-direction:column;align-items:center;margin-top:1.5vh}
.wt-core{width:13vh;height:13vh;border-radius:24px;background:var(--deep);position:relative;box-shadow:0 8px 30px rgba(11,46,79,.25)}
.wt-core::after{content:"";position:absolute;inset:34%;border-radius:8px;background:var(--teal)}
.wt-out{display:flex;flex-direction:column;align-items:center;margin-top:3vh}
.wt-new{width:9vh;height:9vh;border-radius:14px;border:3px solid var(--ochre);background:#fff;position:relative}
.wt-new::after{content:"";position:absolute;inset:28%;border-radius:5px;background:var(--ochre);opacity:.6}
.wt-cap{margin-top:1.3vh;color:var(--muted);font-size:1.9vh;text-align:center;max-width:15ch}
.wt-flow{flex:0 0 auto;align-self:flex-start;margin-top:6.5vh;color:var(--ochre);font-size:3.4vh;font-weight:800}
.wt-flow::before{content:"\\2192"}
.wt-arrow,.wt-src .wt-cap{}
.dgmc{display:flex;align-items:flex-end;justify-content:center;gap:9vw;height:54vh;max-width:56rem;
  margin:0 auto;border-bottom:3px solid var(--line)}
.cbar-wrap{display:flex;flex-direction:column;align-items:center;height:100%;justify-content:flex-end}
.cbar{width:13vw;border-radius:14px 14px 0 0;display:flex;justify-content:center;box-shadow:0 -6px 24px rgba(11,46,79,.12)}
.cbar-mono{background:#9E2B25;height:31%}.cbar-comp{background:var(--teal);height:92%}
.cval{color:#fff;font-weight:800;font-size:3.2vh;margin-top:1.6vh}
.clbl{margin-top:1.8vh;color:var(--deep);font-weight:800;font-size:2.5vh;text-align:center;line-height:1.25}
.clbl small{display:block;color:var(--muted);font-weight:500;font-size:1.95vh;margin-top:.5vh}
.dgmm{display:flex;flex-direction:column;align-items:center;gap:4.5vh;width:100%;max-width:82rem;margin:0 auto}
.mm-row{display:flex;align-items:center;justify-content:center;gap:2vw;flex-wrap:wrap}
.mm-in{display:flex;flex-direction:column;gap:1.6vh}
.mm-chip{padding:1.7vh 1.8vw;border-radius:14px;background:#fff;border:2.5px solid var(--teal);font-size:2.7vh;
  color:var(--deep);box-shadow:0 5px 16px rgba(11,46,79,.07);white-space:nowrap}
.mm-chip b{color:var(--teal);font-size:3.1vh}
.mm-a{border-color:var(--ochre)}.mm-a b{color:var(--ochre)}
.mm-ar{color:var(--muted);font-size:3.6vh;font-weight:800}
.mm-world{width:21vh;height:21vh;border-radius:26px;background:var(--deep);color:#fff;display:flex;flex-direction:column;
  align-items:center;justify-content:center;gap:.8vh;box-shadow:0 12px 36px rgba(11,46,79,.3)}
.mm-mark{width:6.5vh;height:6.5vh;border:3px solid var(--teal);border-radius:9px;position:relative}
.mm-mark::after{content:"";position:absolute;inset:30%;background:var(--ochre);border-radius:3px}
.mm-wl{font-size:2.8vh;font-weight:800}.mm-world small{font-size:1.8vh;opacity:.82}
.mm-sp{border-color:var(--deep)}.mm-sp b{color:var(--deep)}
.mm-cap{color:var(--muted);font-size:2.3vh;text-align:center;max-width:52ch}
.slide.on .mm-s{opacity:0;animation:popIn .5s .2s both}
.slide.on .mm-a{opacity:0;animation:popIn .5s .35s both}
.slide.on .mm-ar.a1{opacity:0;animation:fadeIn .4s .55s both}
.slide.on .mm-world{opacity:0;animation:popIn .55s .7s both}
.slide.on .mm-ar.a2{opacity:0;animation:fadeIn .4s 1.1s both}
.slide.on .mm-sp{opacity:0;animation:popIn .5s 1.25s both}
.slide.on .mm-cap{opacity:0;animation:fadeIn .6s 1.5s both}
@keyframes growUp{from{height:0}}
.slide.on .cbar-mono{animation:growUp 1s .3s cubic-bezier(.3,.6,.2,1) both}
.slide.on .cbar-comp{animation:growUp 1.2s .5s cubic-bezier(.3,.6,.2,1) both}
.slide.on .cval{opacity:0;animation:fadeIn .5s 1.3s both}
.slide.on .clbl{opacity:0;animation:riseIn .5s 1.1s both}
/* ---- entrance animations: content rises in, staggered, each time a slide is shown ---- */
@keyframes riseIn{from{opacity:0;transform:translateY(24px)}to{opacity:1;transform:none}}
@keyframes fadeIn{from{opacity:0}to{opacity:1}}
@keyframes popIn{from{opacity:0;transform:scale(.8)}to{opacity:1;transform:none}}
@keyframes growW{from{width:0}}
@keyframes dashDraw{to{stroke-dashoffset:0}}
.slide.on .ag>*{animation:riseIn .5s cubic-bezier(.2,.75,.3,1) both}
.slide.on .ag>*:nth-child(1){animation-delay:.04s}.slide.on .ag>*:nth-child(2){animation-delay:.12s}
.slide.on .ag>*:nth-child(3){animation-delay:.20s}.slide.on .ag>*:nth-child(4){animation-delay:.28s}
.slide.on .ag>*:nth-child(5){animation-delay:.36s}.slide.on .ag>*:nth-child(6){animation-delay:.44s}
.slide.on .ag>*:nth-child(7){animation-delay:.52s}.slide.on .ag>*:nth-child(8){animation-delay:.60s}
.slide.on .ag>*:nth-child(9){animation-delay:.68s}
.slide.on .fig,.slide.on .cap{animation:riseIn .55s both;animation-delay:.06s}
.slide.on .parthead{animation:riseIn .6s both;animation-delay:.18s}
.slide.on .statement .big,.slide.on .section .st{animation:riseIn .6s cubic-bezier(.2,.75,.3,1) both}
.slide.on h2,.slide.on .kick{animation:fadeIn .45s both}
/* part dividers appear at once -- no staggered/delayed reveal */
.partslide .atlas>*,.partslide .parthead,.partslide .wm{animation-delay:0s!important}
@media (prefers-reduced-motion:reduce){.slide.on *{animation:none!important}}
@media print{.slide{display:flex!important;opacity:1!important;position:relative;page-break-after:always;height:100vh}
  .slide *{animation:none!important}}
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

def _atlas_html(parts, current=None):
    """The roadmap journey: numbered stations, with done/now/next when `current` is set."""
    nodes = []
    for i, p in enumerate(parts):
        cls = "" if current is None else ("done" if i < current else "now" if i == current else "next")
        if i:
            lcls = "line donel" if (current is not None and i <= current) else "line"
            nodes.append(f'<div class="{lcls}"></div>')
        nodes.append(f'<div class="station {cls}"><span class="dot">{i+1}</span>'
                     f'<span class="lbl">{_h(p)}</span></div>')
    return f'<div class="atlas">{"".join(nodes)}</div>'

def _wm_html():
    return ('<span class="wm">'
            '<i style="inset:0;border:5px solid var(--deep)"></i>'
            '<i style="inset:22%;border:5px solid var(--blue)"></i>'
            '<i style="inset:40%;background:var(--teal)"></i></span>')

def html_slide(s, kicker="", parts=None, part_index=None):
    t = s.get("type")
    if t == "section":
        return f'<section class="slide section"><div class="rule"></div><div class="st">{_h(s["title"])}</div></section>'
    if t == "statement":
        return f'<section class="slide statement"><div class="big">{_h(s["text"])}</div></section>'
    if t == "agenda":
        h2 = f'<h2>{_h(s.get("title") or "What we will cover")}</h2>'
        return (f'<section class="slide agenda"><div class="kick">The roadmap</div>{h2}'
                f'<div class="body">{_atlas_html(parts or [], None)}</div></section>')
    if t == "part":
        sub = f'<div class="psub">{_h(s["subtitle"])}</div>' if s.get("subtitle") else ""
        return (f'<section class="slide partslide">{_wm_html()}{_atlas_html(parts or [], part_index)}'
                f'<div class="parthead"><div class="pnum">Part {(part_index or 0)+1}</div>'
                f'<div class="ptitle">{_h(s["title"])}</div>{sub}</div></section>')
    kick = f'<div class="kick">{_h(kicker)}</div>' if kicker else ""
    head = kick + f'<h2>{_h(s.get("title",""))}</h2>'
    def ul(bs):
        if not bs: return ""
        return "<ul>" + "".join(f"<li>{_h(b)}</li>" for b in bs) + "</ul>"
    if t == "bullets":
        blist_ul = "<ul class=\"ag\">" + "".join(f"<li>{_h(b)}</li>" for b in s.get("bullets",[])) + "</ul>"
        return f'<section class="slide">{head}<div class="body blist">{blist_ul}</div></section>'
    if t == "figure":
        cap = f'<div class="cap">{_h(s["caption"])}</div>' if s.get("caption") else ""
        bl = ul(s.get("bullets", []))
        img = f'<div class="fig"><img src="figs/{Path(s["image"]).name}" alt="{_h(s.get("caption") or s.get("title",""))}"></div>'
        hasb = " hasbul" if s.get("bullets") else ""
        return f'<section class="slide">{head}<div class="body figbody{hasb}">{img}{cap}{bl}</div></section>'
    if t == "twocol":
        return (f'<section class="slide">{head}<div class="body"><div class="two">'
                f'<div class="col">{ul(s.get("bullets",[]))}</div>'
                f'<div class="col img"><img src="figs/{Path(s["image"]).name}" alt="{_h(s.get("title",""))}"></div>'
                f'</div></div></section>')
    if t == "cards":
        cs = "".join(
            f'<div class="card">{_svg(c.get("icon",""))}<h4>{_h(c["head"])}</h4>'
            f'{("<p>"+_h(c["text"])+"</p>") if c.get("text") else ""}</div>'
            for c in s["cards"])
        return f'<section class="slide">{head}<div class="body"><div class="cards ag">{cs}</div></div></section>'
    if t == "stats":
        tiles = "".join(
            f'<div class="stat{" hi" if it.get("hi") else ""}">{_svg(it.get("icon",""))}'
            f'<div class="v">{_h(it["value"])}</div><div class="l">{_h(it["label"])}</div></div>'
            for it in s["items"])
        return f'<section class="slide">{head}<div class="body"><div class="stats ag">{tiles}</div></div></section>'
    if t == "flow":
        icons = s.get("icons", [])
        parts = []
        for i, st in enumerate(s["steps"]):
            if i:
                parts.append('<span class="arw">&rarr;</span>')
            ic = _svg(icons[i]) if i < len(icons) else ""
            parts.append(f'<div class="step">{ic}<span class="n">{i+1}</span>{_h(st)}</div>')
        return f'<section class="slide">{head}<div class="body"><div class="flow ag">{"".join(parts)}</div></div></section>'
    if t == "compare":
        def side(d, cls):
            items = "".join(f"<li>{_h(x)}</li>" for x in d["items"])
            return f'<div class="side {cls}"><h3>{_svg(d.get("icon",""))}{_h(d["head"])}</h3><ul>{items}</ul></div>'
        return (f'<section class="slide">{head}<div class="body"><div class="compare ag">'
                f'{side(s["left"],"a")}{side(s["right"],"b")}</div></div></section>')
    if t == "anim":
        return f'<section class="slide">{head}<div class="body">{DIAGRAMS.get(s.get("diagram",""),"")}</div></section>'
    return ""

def _part_walk(slides):
    """Yield (slide, kicker_theme, part_index) — content slides carry the current PART (theme)."""
    parts = [s["title"] for s in slides if s.get("type") == "part"]
    cur, pi = "", -1
    for s in slides:
        if s.get("type") == "part":
            pi += 1; cur = s["title"]
            yield s, cur, pi, parts
        elif s.get("type") == "agenda":
            yield s, "", None, parts
        else:
            yield s, cur, None, parts

def build_html(deck):
    title_slide = (f'<section class="slide title on">{_mark_html(1.3)}'
                   f'<div class="rule" style="margin-top:2vh"></div>'
                   f'<div class="t">{_h(deck["title"])}</div>'
                   f'<div class="s">{_h(deck["subtitle"])}</div>'
                   f'<div class="a">{_h(deck["author"])}</div>'
                   f'<div class="v">{_h(deck["venue"])}</div></section>')
    body = title_slide + "".join(html_slide(s, k, parts, pi) for s, k, pi, parts in _part_walk(deck["slides"]))
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
        img = s.get("image") or s.get("still")
        if not img: continue
        name = Path(img).name
        src = FIGSRC / name                          # paper figures
        if not src.exists():
            src = PRES / "assets" / name             # presentation-native charts
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
