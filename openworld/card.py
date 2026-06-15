"""Stunning, self-contained SVG model cards for world-model specs.

``render_card`` turns a world (or its spec) into a single standalone ``.svg`` ---
gradients, soft shadows, a designed composition graph (gradient-filled child
nodes, curved bridge edges with arrowheads, aggregator roll-ups), a typeset state
schema + action chips, and an area-fill rollout sparkline. One self-contained
file, no external references, so a card renders identically in a browser, a
GitHub README (``<img src="card.svg">``), or a slide.

``render_gallery`` builds an SVG contact sheet of clickable tiles (each links to
its ``<name>.svg``) --- the marketplace browse view.

Zero-dependency: standard library only.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .spec import SPEC_VERSION, to_spec
from .world import World

_THEMES = {
    "light": {
        "bg0": "#ffffff", "bg1": "#eef2f8", "text": "#0f172a", "muted": "#64748b",
        "accent": "#2563eb", "accent2": "#7c3aed", "node0": "#ffffff",
        "node1": "#eef2ff", "edge": "#2563eb", "chipbg": "#eef2ff",
        "chiptx": "#3730a3", "line": "#e2e8f0", "good": "#059669",
        "agg": "#7c3aed", "shadow": "#0f172a", "shadowO": "0.16",
        "panelStroke": "#e2e8f0",
    },
    "dark": {
        "bg0": "#0c111c", "bg1": "#141d2e", "text": "#e7ecf5", "muted": "#93a4bd",
        "accent": "#5b9dff", "accent2": "#b18bff", "node0": "#1b2435",
        "node1": "#141c2b", "edge": "#5b9dff", "chipbg": "#1d2740",
        "chiptx": "#c7d6ff", "line": "#26324a", "good": "#34d399",
        "agg": "#b18bff", "shadow": "#000000", "shadowO": "0.5",
        "panelStroke": "#26324a",
    },
}
_SERIES = ["#2563eb", "#0f9d8f", "#d97706", "#db2777", "#7c3aed"]

W = 900
PAD = 26
CX = 58
CW = W - 2 * CX
COLGAP = 38
COLW = (CW - COLGAP) // 2
LX = CX
RX = CX + COLW + COLGAP


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _as_spec(world_or_spec: Union[World, Dict[str, Any]]) -> Dict[str, Any]:
    return world_or_spec if isinstance(world_or_spec, dict) else to_spec(world_or_spec)


def _t(x, y, s, size, weight, fill, anchor="start", spacing=None, opacity=None):
    extra = f' text-anchor="{anchor}"' if anchor != "start" else ""
    if spacing is not None:
        extra += f' letter-spacing="{spacing}"'
    if opacity is not None:
        extra += f' opacity="{opacity}"'
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
            f'font-weight="{weight}" fill="{fill}"{extra}>{_esc(s)}</text>')


def _wrap(text: str, width: int, max_lines: int) -> List[str]:
    words, lines, cur = str(text).split(), [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if len(trial) <= width:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
        if len(lines) == max_lines:
            break
    if cur and len(lines) < max_lines:
        lines.append(cur)
    if len(lines) == max_lines and (len(" ".join(lines)) < len(text)):
        lines[-1] = lines[-1][:width - 1].rstrip() + "…"
    return lines or [""]


# --------------------------------------------------------------------------- #
# defs: gradients, shadow filter, arrowhead
# --------------------------------------------------------------------------- #
def _defs(c: Dict[str, str]) -> str:
    return (
        "<defs>"
        f'<linearGradient id="gbg" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0" stop-color="{c["bg0"]}"/>'
        f'<stop offset="1" stop-color="{c["bg1"]}"/></linearGradient>'
        f'<linearGradient id="gnode" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0" stop-color="{c["node0"]}"/>'
        f'<stop offset="1" stop-color="{c["node1"]}"/></linearGradient>'
        f'<linearGradient id="gacc" x1="0" y1="0" x2="1" y2="0">'
        f'<stop offset="0" stop-color="{c["accent"]}"/>'
        f'<stop offset="1" stop-color="{c["accent2"]}"/></linearGradient>'
        f'<linearGradient id="garea" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0" stop-color="{c["accent"]}" stop-opacity="0.34"/>'
        f'<stop offset="1" stop-color="{c["accent"]}" stop-opacity="0"/>'
        f'</linearGradient>'
        f'<filter id="sh" x="-30%" y="-30%" width="160%" height="160%">'
        f'<feDropShadow dx="0" dy="5" stdDeviation="9" flood-color="{c["shadow"]}" '
        f'flood-opacity="{c["shadowO"]}"/></filter>'
        f'<filter id="shsoft" x="-40%" y="-40%" width="180%" height="180%">'
        f'<feDropShadow dx="0" dy="10" stdDeviation="22" flood-color="{c["shadow"]}" '
        f'flood-opacity="{c["shadowO"]}"/></filter>'
        f'<marker id="arr" markerWidth="9" markerHeight="9" refX="6.5" refY="3" '
        f'orient="auto"><path d="M0,0 L6.5,3 L0,6 Z" fill="{c["edge"]}"/></marker>'
        "</defs>")


def _globe(x: float, y: float, r: float, c: Dict[str, str]) -> str:
    return (f'<g transform="translate({x:.1f},{y:.1f})">'
            f'<circle r="{r}" fill="url(#gacc)"/>'
            f'<circle r="{r}" fill="none" stroke="#ffffff" stroke-opacity="0.55" stroke-width="1.1"/>'
            f'<ellipse rx="{r*0.45:.1f}" ry="{r}" fill="none" stroke="#ffffff" stroke-opacity="0.55" stroke-width="1"/>'
            f'<line x1="{-r}" y1="0" x2="{r}" y2="0" stroke="#ffffff" stroke-opacity="0.55" stroke-width="1"/>'
            f'<line x1="{-r*0.86:.1f}" y1="{-r*0.5:.1f}" x2="{r*0.86:.1f}" y2="{-r*0.5:.1f}" stroke="#ffffff" stroke-opacity="0.4" stroke-width="0.9"/>'
            f'<line x1="{-r*0.86:.1f}" y1="{r*0.5:.1f}" x2="{r*0.86:.1f}" y2="{r*0.5:.1f}" stroke="#ffffff" stroke-opacity="0.4" stroke-width="0.9"/>'
            f'</g>')


def _pill(x, y, label, c, fg=None, bg=None, font=12) -> Tuple[str, float]:
    fg = fg or c["chiptx"]
    bg = bg or c["chipbg"]
    w = 16 + len(str(label)) * font * 0.62
    svg = (f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{font+10}" '
           f'rx="{(font+10)/2:.1f}" fill="{bg}"/>'
           + _t(x + 8, y + font + 2, label, font, "600", fg))
    return svg, w


# --------------------------------------------------------------------------- #
# graph: leaf state-transition automaton / composite dataflow, laid out + drawn
# --------------------------------------------------------------------------- #
def _gnode(x, y, w, h, lines, c, kind, initial, href=None) -> str:
    if kind == "agg":
        fill, stroke, sw = c["node1"], c["agg"], 1.4
    elif kind == "param":
        fill, stroke, sw = c["node1"], c["muted"], 1.2
    else:
        fill = "url(#gnode)"
        stroke, sw = (c["accent"], 2.2) if initial else (c["line"], 1.2)
    out = [f'<g filter="url(#sh)"><rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" '
           f'height="{h:.1f}" rx="13" fill="{fill}" stroke="{stroke}" '
           f'stroke-width="{sw}"/></g>']
    if kind == "world":                               # accent header strip on world nodes
        out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="5" '
                   f'rx="2.5" fill="url(#gacc)"/>')
    if initial:
        out.append(_t(x + 2, y - 7, "▶ start", 10, "700", c["accent"]))
    ty = y + (h - len(lines) * 15) / 2 + 12
    for i, ln in enumerate(lines):
        size = 13 if i == 0 else 11
        col = c["text"] if i == 0 else c["muted"]
        out.append(_t(x + w / 2, ty + i * 15, ln, size, "700" if i == 0 else "500",
                      col, anchor="middle"))
    inner = "".join(out)
    return f'<a href="{_esc(href)}">{inner}</a>' if href else inner


def _edge_label(x, y, text, c) -> str:
    if not text:
        return ""
    w = len(text) * 6.0 + 10
    return (f'<rect x="{x-w/2:.1f}" y="{y-8:.1f}" width="{w:.1f}" height="15" rx="7.5" '
            f'fill="{c["bg0"]}" stroke="{c["line"]}" stroke-width="0.8"/>'
            + _t(x, y + 3, text, 9.5, "600", c["edge"], anchor="middle"))


def _graph_layout(nodes, edges, x, y, w, c, min_h: float = 210) -> Tuple[str, float]:
    if not nodes:
        return _t(x, y + 30, "no graph", 12, "500", c["muted"]), 70
    ids = [n["id"] for n in nodes]
    nd = {n["id"]: n for n in nodes}
    if all(n.get("rank") is not None for n in nodes):
        layer = {n["id"]: n["rank"] for n in nodes}
    else:
        layer = {i: 0 for i in ids}
        fwd = [e for e in edges if e["src"] != e["dst"]]
        for _ in range(len(ids)):
            changed = False
            for e in fwd:
                if e["src"] in layer and e["dst"] in layer and \
                        layer[e["dst"]] < layer[e["src"]] + 1 <= len(ids):
                    layer[e["dst"]] = layer[e["src"]] + 1
                    changed = True
            if not changed:
                break
    base = min(layer.values())
    layer = {k: v - base for k, v in layer.items()}
    L = max(layer.values()) + 1
    layers: Dict[int, List[str]] = {}
    for i in ids:
        layers.setdefault(layer[i], []).append(i)

    maxchars = max((max((len(s) for s in nd[i]["label"]), default=3) for i in ids))
    maxlines = max(len(nd[i]["label"]) for i in ids)
    nw = max(122.0, min(196.0, maxchars * 7.2 + 30))
    if L > 1:                                   # guarantee columns never overlap
        nw = min(nw, (w - (L - 1) * 34) / L)
    nh = 26 + maxlines * 15
    vgap, top_pad, bot_pad = 26, 32, 16
    max_count = max(len(v) for v in layers.values())
    Hg = max_count * nh + (max_count - 1) * vgap
    area_h = max(Hg, min_h)
    colstep = (w - nw) / (L - 1) if L > 1 else 0.0
    y0 = y + top_pad + (area_h - Hg) / 2
    pos: Dict[str, Tuple[float, float]] = {}
    for lyr, members in layers.items():
        m = len(members)
        start = y0 + (Hg - (m * nh + (m - 1) * vgap)) / 2
        cx = x + lyr * colstep if L > 1 else x + (w - nw) / 2
        for j, i in enumerate(members):
            pos[i] = (cx, start + j * (nh + vgap))

    el: List[str] = []
    for e in edges:
        s, d = e["src"], e["dst"]
        if s not in pos or d not in pos:
            continue
        sx, sy = pos[s]
        dx, dy = pos[d]
        dash = ' stroke-dasharray="5 4"' if e.get("style") == "dash" else ""
        stroke = c["edge"]
        if s == d:                                            # self-loop (top)
            lx = sx + nw * 0.5
            el.append(f'<path d="M {lx-13:.1f} {sy:.1f} C {lx-24:.1f} {sy-26:.1f}, '
                      f'{lx+24:.1f} {sy-26:.1f}, {lx+13:.1f} {sy:.1f}" fill="none" '
                      f'stroke="{stroke}" stroke-width="1.5"{dash} '
                      f'marker-end="url(#arr)"/>')
            el.append(_edge_label(lx, sy - 24, e.get("action", ""), c))
            continue
        if dx > sx:                                           # forward
            x1, y1, x2, y2 = sx + nw, sy + nh / 2, dx, dy + nh / 2
            mx = (x1 + x2) / 2
            el.append(f'<path d="M {x1:.1f} {y1:.1f} C {mx:.1f} {y1:.1f}, '
                      f'{mx:.1f} {y2:.1f}, {x2-3:.1f} {y2:.1f}" fill="none" '
                      f'stroke="{stroke}" stroke-width="1.7" stroke-opacity="0.85"'
                      f'{dash} marker-end="url(#arr)"/>')
            lxm, lym = mx, (y1 + y2) / 2
        elif dx < sx:                                         # backward (curve below)
            x1, y1, x2, y2 = sx, sy + nh / 2, dx + nw, dy + nh / 2
            my = max(y1, y2) + nh / 2 + 14
            el.append(f'<path d="M {x1:.1f} {y1:.1f} C {x1-30:.1f} {my:.1f}, '
                      f'{x2+30:.1f} {my:.1f}, {x2+3:.1f} {y2:.1f}" fill="none" '
                      f'stroke="{stroke}" stroke-width="1.6" stroke-opacity="0.8"'
                      f'{dash} marker-end="url(#arr)"/>')
            lxm, lym = (x1 + x2) / 2, my
        else:                                                 # same column (right bow)
            x1, y1, x2, y2 = sx + nw, sy + nh / 2, dx + nw, dy + nh / 2
            off = 46
            el.append(f'<path d="M {x1:.1f} {y1:.1f} C {x1+off:.1f} {y1:.1f}, '
                      f'{x1+off:.1f} {y2:.1f}, {x2:.1f} {y2:.1f}" fill="none" '
                      f'stroke="{stroke}" stroke-width="1.7" stroke-opacity="0.85"'
                      f'{dash} marker-end="url(#arr)"/>')
            lxm, lym = x1 + off, (y1 + y2) / 2
        el.append(_edge_label(lxm, lym, e.get("action", ""), c))

    for i in ids:
        nx, ny = pos[i]
        el.append(_gnode(nx, ny, nw, nh, nd[i]["label"], c, nd[i].get("kind", "state"),
                         nd[i].get("initial", False), nd[i].get("href")))
    return "".join(el), top_pad + area_h + bot_pad


def _leaf_graph(spec: Dict[str, Any]):
    g = (spec.get("preview", {}) or {}).get("graph", {}) or {}
    nodes = [{"id": n["id"], "label": n["label"], "kind": "state",
              "initial": n.get("initial", False)} for n in g.get("nodes", [])]
    edges = [{"src": e["src"], "dst": e["dst"], "action": e.get("action", "")}
             for e in g.get("edges", [])]
    return nodes, edges


def _composite_graph(spec: Dict[str, Any]):
    comp = spec["composite"]
    children = comp.get("children", {})
    nodes, edges = [], []
    for ns, child in children.items():
        sub = ("composite world" if child.get("composite")
               else f'{len(child.get("actions", []))} actions')
        nodes.append({"id": f"c:{ns}", "label": [ns, sub], "kind": "world", "rank": 1,
                      "href": f'{child.get("name", ns)}.svg'})
    agg_ids = set()
    for a in comp.get("aggregators", []):
        nm = a.get("name", "agg")
        aid = f"a:{nm}"
        agg_ids.add(aid)
        nodes.append({"id": aid, "label": [f"Σ {nm}"], "kind": "agg", "rank": 2})
        for ns in children:
            edges.append({"src": f"c:{ns}", "dst": aid, "action": "", "style": "solid"})
    for b in comp.get("bridges", []):
        edges.append({"src": f'c:{b.get("a")}', "dst": f'c:{b.get("b")}',
                      "action": b.get("name", ""),
                      "style": "dash" if b.get("kind") == "route" else "solid"})
    params = False
    for bd in comp.get("bindings", []):
        sp, child = bd.get("source_path", []), f'c:{bd.get("child")}'
        if len(sp) >= 2 and sp[0] == "_agg" and f"a:{sp[1]}" in agg_ids:
            edges.append({"src": f"a:{sp[1]}", "dst": child,
                          "action": bd.get("key", ""), "style": "dash"})
        else:
            if not params:
                nodes.append({"id": "p:params", "label": ["params"],
                              "kind": "param", "rank": 0})
                params = True
            edges.append({"src": "p:params", "dst": child,
                          "action": bd.get("key", ""), "style": "dash"})
    return nodes, edges


def _composition(spec: Dict[str, Any], x: float, y: float, w: float,
                 c: Dict[str, str]) -> Tuple[str, float]:
    is_comp = bool(spec.get("composite"))
    title = "COMPOSITION GRAPH" if is_comp else "STATE-TRANSITION GRAPH"
    out = [_t(x, y + 2, title, 11, "700", c["muted"], spacing="1.6")]
    nodes, edges = _composite_graph(spec) if is_comp else _leaf_graph(spec)
    if not nodes:
        nodes = [{"id": "only", "label": [spec["name"]], "kind": "world",
                  "initial": True}]
        edges = []
    gsvg, gh = _graph_layout(nodes, edges, x, y + 18, w, c)
    return "".join(out) + gsvg, 18 + gh + 6


# --------------------------------------------------------------------------- #
# right column: schema, actions, dynamics, rollout
# --------------------------------------------------------------------------- #
def _schema_block(spec, x, y, w, c) -> Tuple[str, float]:
    out = [_t(x, y + 2, "STATE SCHEMA", 11, "700", c["muted"], spacing="1.6")]
    schema = {k: v for k, v in spec.get("state_schema", {}).items()
              if not k.startswith("_")}
    items = list(schema.items())[:6]
    ry = y + 22
    for k, ty in items:
        out.append(_t(x, ry + 11, k, 13, "600", c["text"]))
        pill, pw = _pill(x + w - (16 + len(str(ty)) * 12 * 0.62), ry - 1, ty, c, font=11)
        out.append(pill)
        out.append(f'<line x1="{x:.1f}" y1="{ry+20:.1f}" x2="{x+w:.1f}" y2="{ry+20:.1f}" '
                   f'stroke="{c["line"]}" stroke-width="1" stroke-dasharray="2 4"/>')
        ry += 27
    if len(schema) > 6:
        out.append(_t(x, ry + 10, f"+{len(schema) - 6} more", 11.5, "500", c["muted"]))
        ry += 22
    return "".join(out), ry - y + 6


def _chips_block(spec, x, y, w, c, title, items, limit) -> Tuple[str, float]:
    out = [_t(x, y + 2, title, 11, "700", c["muted"], spacing="1.6")]
    cx, cy = x, y + 16
    shown = items[:limit]
    for it in shown:
        pill, pw = _pill(cx, cy, it, c)
        if cx + pw > x + w:
            cx, cy = x, cy + 30
            pill, pw = _pill(cx, cy, it, c)
        out.append(pill)
        cx += pw + 8
    if len(items) > limit:
        pill, pw = _pill(cx, cy, f"+{len(items) - limit}", c)
        out.append(pill)
    return "".join(out), (cy + 30) - y + 6


def _dynamics_badge(spec, x, y, c) -> Tuple[str, float]:
    t = spec.get("transition", {}) or {}
    kind = t.get("kind", "none")
    verified = kind == "code" and not t.get("from_function")
    label = {"code": "verified code" if verified else "code (from function)",
             "function": "function (lossy)", "phased": "phased dynamics",
             "llm": "LLM dynamics", "composite": "composite dynamics",
             "none": "no dynamics"}.get(kind, kind)
    fg = c["good"] if kind == "code" else c["muted"]
    out = [_t(x, y + 2, "DYNAMICS", 11, "700", c["muted"], spacing="1.6")]
    check = f'<circle cx="{x+9:.1f}" cy="{y+27:.1f}" r="8" fill="{fg}"/>' \
            f'<path d="M{x+5.4:.1f},{y+27:.1f} l2.4,2.4 l4-4.4" fill="none" ' \
            f'stroke="#fff" stroke-width="1.8" stroke-linecap="round"/>' if kind == "code" \
            else f'<circle cx="{x+9:.1f}" cy="{y+27:.1f}" r="8" fill="none" stroke="{fg}" stroke-width="1.6"/>'
    out.append(check)
    out.append(_t(x + 24, y + 31, label, 13, "700", c["text"]))
    return "".join(out), 48


def _rollout_block(spec, x, y, w, c) -> Tuple[str, float]:
    out = [_t(x, y + 2, "SAMPLE ROLLOUT", 11, "700", c["muted"], spacing="1.6")]
    preview = spec.get("preview", {}) or {}
    series = {k: v for k, v in preview.get("series", {}).items()
              if isinstance(v, list) and len(v) > 1}
    cy = y + 16
    ch = 132
    out.append(f'<rect x="{x:.1f}" y="{cy:.1f}" width="{w:.1f}" height="{ch}" rx="14" '
               f'fill="{c["node1"]}" stroke="{c["line"]}" stroke-width="1"/>')
    if not series:
        out.append(_t(x + w / 2, cy + ch / 2 + 4, "no numeric preview", 12, "500",
                      c["muted"], anchor="middle"))
        return "".join(out), (cy + ch) - y + 6
    pad = 14
    names = sorted(series, key=lambda k: -(max(series[k]) - min(series[k])))[:4]
    base = cy + ch - pad

    def pts(vals):
        lo, hi = min(vals), max(vals)
        span = (hi - lo) or 1.0
        n = len(vals)
        return [(x + pad + j * (w - 2 * pad) / (n - 1),
                 cy + pad + (hi - v) / span * (ch - 2 * pad))
                for j, v in enumerate(vals)]

    # main series: gradient area + line
    main = names[0]
    p = pts(series[main])
    line = " ".join(f"{px:.1f},{py:.1f}" for px, py in p)
    area = (f'M {p[0][0]:.1f} {base:.1f} L '
            + " L ".join(f"{px:.1f} {py:.1f}" for px, py in p)
            + f' L {p[-1][0]:.1f} {base:.1f} Z')
    out.append(f'<path class="series" d="{area}" fill="url(#garea)"/>')
    out.append(f'<polyline class="series" points="{line}" fill="none" '
               f'stroke="{_SERIES[0]}" stroke-width="2.4" stroke-linejoin="round"/>')
    out.append(f'<circle cx="{p[-1][0]:.1f}" cy="{p[-1][1]:.1f}" r="3.2" fill="{_SERIES[0]}"/>')
    for i, nm in enumerate(names[1:], start=1):
        pl = " ".join(f"{px:.1f},{py:.1f}" for px, py in pts(series[nm]))
        out.append(f'<polyline class="series" points="{pl}" fill="none" '
                   f'stroke="{_SERIES[i % len(_SERIES)]}" stroke-width="1.8" '
                   f'stroke-opacity="0.9" stroke-linejoin="round"/>')
    # legend
    ly = cy + ch + 16
    lx = x
    for i, nm in enumerate(names):
        vals = series[nm]
        arrow = "↑" if vals[-1] > vals[0] else ("↓" if vals[-1] < vals[0] else "→")
        col = _SERIES[i % len(_SERIES)]
        out.append(f'<rect x="{lx:.1f}" y="{ly-9:.1f}" width="10" height="10" rx="2.5" fill="{col}"/>')
        label = f"{nm} {arrow}"
        out.append(_t(lx + 15, ly, label, 11, "600", c["muted"]))
        lx += 15 + len(label) * 6.6 + 14
    return "".join(out), (ly + 8) - y


def _details_block(spec, x, y, w, c) -> Tuple[str, float]:
    out = [_t(x, y + 2, "DETAILS", 11, "700", c["muted"], spacing="1.6")]
    facts = []
    nr = len(spec.get("rules", []))
    facts.append(f"{nr} rule" + ("" if nr == 1 else "s"))
    comp = spec.get("composite")
    if comp:
        facts.append(f'{len(comp.get("children", {}))} child worlds')
        nb = len(comp.get("bridges", []))
        facts.append(f"{nb} bridge" + ("" if nb == 1 else "s"))
        ag = comp.get("agents", {})
        if ag:
            facts.append(f'{len(ag)} agents: ' + ", ".join(list(ag)[:3]))
    gy = y + 20
    for i, fact in enumerate(facts):
        out.append(f'<circle cx="{x+4:.1f}" cy="{gy+i*19-4:.1f}" r="2.6" fill="{c["accent"]}"/>')
        out.append(_t(x + 14, gy + i * 19, fact, 12, "600", c["text"]))
    return "".join(out), 20 + len(facts) * 19 + 6


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def render_card(world_or_spec: Union[World, Dict[str, Any]],
                path: Optional[Union[str, Path]] = None, theme: str = "light") -> str:
    """Render a self-contained SVG model card. Returns the SVG; writes it to
    ``path`` when given."""
    spec = _as_spec(world_or_spec)
    c = _THEMES.get(theme, _THEMES["light"])
    card = spec.get("card", {})

    # header geometry
    desc = card.get("description") or spec.get("description", "")
    dlines = _wrap(desc, 86, 2)
    title_y = PAD + 52
    dy0 = title_y + 26
    tags = card.get("tags", [])
    tag_y = dy0 + len(dlines) * 19 + 6
    header_bottom = (tag_y + 26 if tags else dy0 + len(dlines) * 19 + 4)
    div_y = header_bottom + 8
    body_y = div_y + 24

    # hero graph (full content width) + a faint framing panel behind it
    comp_svg, lh = _composition(spec, CX, body_y, CW, c)
    graph_panel = (f'<rect x="{CX}" y="{body_y+14:.1f}" width="{CW}" height="{lh-14:.1f}" '
                   f'rx="16" fill="{c["node1"]}" fill-opacity="0.5" '
                   f'stroke="{c["line"]}" stroke-width="1"/>')

    # metadata row beneath: schema | actions+dynamics+details | rollout
    meta_y = body_y + lh + 16
    g = 20
    w1 = 250
    w2 = 222
    w3 = CW - w1 - w2 - 2 * g
    x1, x2, x3 = CX, CX + w1 + g, CX + w1 + w2 + 2 * g
    s_svg, sh = _schema_block(spec, x1, meta_y, w1, c)
    mid, my = [], meta_y
    a_svg, ah = _chips_block(spec, x2, my, w2, c, "ACTIONS",
                             list(spec.get("actions", [])), 10)
    mid.append(a_svg); my += ah + 14
    d_svg, dh = _dynamics_badge(spec, x2, my, c)
    mid.append(d_svg); my += dh + 10
    f_svg, fh = _details_block(spec, x2, my, w2, c)
    mid.append(f_svg); my += fh
    midh = my - meta_y
    r_svg, rh = _rollout_block(spec, x3, meta_y, w3, c)
    meta_bottom = meta_y + max(sh, midh, rh)
    foot_y = meta_bottom + 16
    H = foot_y + 30

    # header content
    badges = []
    bx = W - PAD - 18
    if card.get("license"):
        svg, w = _pill(0, 0, card["license"], c)  # measure
        bx -= w
        b, _ = _pill(bx, PAD + 22, card["license"], c, fg=c["good"], bg=c["chipbg"])
        badges.append(b)
        bx -= 8
    vlabel = f"v{card.get('version', '0.1')}"
    svg, w = _pill(0, 0, vlabel, c)
    bx -= w
    b, _ = _pill(bx, PAD + 22, vlabel, c)
    badges.append(b)

    head = [
        _globe(CX + 14, title_y - 6, 15, c),
        _t(CX + 38, title_y, spec["name"], 30, "800", c["text"]),
    ]
    for i, ln in enumerate(dlines):
        head.append(_t(CX, dy0 + i * 19, ln, 14, "500", c["muted"]))
    tx = CX
    for tg in tags[:6]:
        pill, pw = _pill(tx, tag_y - 14, tg, c)
        head.append(pill)
        tx += pw + 8

    lineage = card.get("lineage")
    foot = f"openworld_spec_version {spec.get('openworld_spec_version', SPEC_VERSION)}"
    if lineage:
        foot += f"  ·  lineage: {lineage}"

    body = (
        f'<rect x="0" y="0" width="{W}" height="{H}" fill="url(#gbg)"/>'
        f'<g filter="url(#shsoft)"><rect x="{PAD}" y="{PAD}" width="{W-2*PAD}" '
        f'height="{H-2*PAD}" rx="24" fill="url(#gbg)" stroke="{c["panelStroke"]}" '
        f'stroke-width="1"/></g>'
        f'<rect x="{PAD}" y="{PAD}" width="{W-2*PAD}" height="6" rx="3" fill="url(#gacc)"/>'
        + "".join(head) + "".join(badges)
        + f'<line x1="{CX}" y1="{div_y:.1f}" x2="{W-CX}" y2="{div_y:.1f}" '
          f'stroke="{c["line"]}" stroke-width="1"/>'
        + graph_panel + comp_svg + s_svg + "".join(mid) + r_svg
        + f'<line x1="{CX}" y1="{foot_y-8:.1f}" x2="{W-CX}" y2="{foot_y-8:.1f}" '
          f'stroke="{c["line"]}" stroke-width="1"/>'
        + _t(CX, foot_y + 12, foot, 11, "500", c["muted"])
        + _t(W - CX, foot_y + 12, "OpenWorld", 12, "800", c["accent"], anchor="end"))

    svg_doc = (
        f'<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H:.0f}" '
        f'width="{W}" height="{H:.0f}" font-family="-apple-system,BlinkMacSystemFont,'
        f'Segoe UI,Roboto,Helvetica,Arial,sans-serif">'
        + _defs(c) + body + "</svg>")
    if path is not None:
        Path(path).write_text(svg_doc, encoding="utf-8")
    return svg_doc


def render_gallery(specs: List[Dict[str, Any]], path: Optional[Union[str, Path]] = None,
                   theme: str = "light", title: str = "OpenWorld model gallery",
                   card_ext: str = ".svg") -> str:
    """Render an SVG contact sheet: clickable tiles linking to each ``<name>.svg``."""
    c = _THEMES.get(theme, _THEMES["light"])
    cols = 3
    tw, th, gap = 270, 150, 22
    rows = (len(specs) + cols - 1) // cols
    gx0, gy0 = PAD + 6, 92
    Hgal = gy0 + rows * (th + gap) + PAD
    Wgal = gx0 * 2 + cols * tw + (cols - 1) * gap

    tiles = []
    for i, spec in enumerate(specs):
        r, col = divmod(i, cols)
        x = gx0 + col * (tw + gap)
        y = gy0 + r * (th + gap)
        comp = spec.get("composite")
        card = spec.get("card", {})
        meta = ("leaf world" if not comp else
                f'{len(comp.get("children", {}))} worlds · '
                f'{len(comp.get("bridges", []))} bridges')
        tiles.append(f'<a href="{_esc(spec["name"] + card_ext)}">')
        tiles.append(f'<g filter="url(#sh)"><rect x="{x}" y="{y}" width="{tw}" '
                     f'height="{th}" rx="16" fill="url(#gnode)" stroke="{c["line"]}" '
                     f'stroke-width="1"/></g>')
        tiles.append(f'<rect x="{x}" y="{y}" width="{tw}" height="5" rx="2.5" fill="url(#gacc)"/>')
        tiles.append(_globe(x + 24, y + 36, 12, c))
        tiles.append(_t(x + 44, y + 40, spec["name"], 17, "800", c["text"]))
        tiles.append(_t(x + 18, y + 66, meta, 12, "600", c["muted"]))
        txx = x + 18
        for tg in card.get("tags", [])[:3]:
            pill, pw = _pill(txx, y + 80, tg, c, font=11)
            tiles.append(pill)
            txx += pw + 7
        tiles.append(_t(x + 18, y + th - 16, f"v{card.get('version', '0.1')}",
                        11.5, "700", c["muted"]))
        if card.get("license"):
            tiles.append(_t(x + tw - 18, y + th - 16, card["license"], 11.5, "700",
                            c["good"], anchor="end"))
        tiles.append("</a>")

    doc = (
        f'<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'viewBox="0 0 {Wgal} {Hgal}" width="{Wgal}" height="{Hgal}" '
        f'font-family="-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif">'
        + _defs(c)
        + f'<rect x="0" y="0" width="{Wgal}" height="{Hgal}" fill="url(#gbg)"/>'
        + _globe(gx0 + 12, 50, 16, c)
        + _t(gx0 + 38, 56, title, 26, "800", c["text"])
        + _t(gx0 + 38, 76, f"{len(specs)} world models", 13, "600", c["muted"])
        + "".join(tiles) + "</svg>")
    if path is not None:
        Path(path).write_text(doc, encoding="utf-8")
    return doc
