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
        "panelStroke": "#dfe4ec", "grid": "#9fb3d1",
    },
    "dark": {
        "bg0": "#0c111c", "bg1": "#141d2e", "text": "#e7ecf5", "muted": "#93a4bd",
        "accent": "#5b9dff", "accent2": "#b18bff", "node0": "#1b2435",
        "node1": "#141c2b", "edge": "#5b9dff", "chipbg": "#1d2740",
        "chiptx": "#c7d6ff", "line": "#26324a", "good": "#34d399",
        "agg": "#b18bff", "shadow": "#000000", "shadowO": "0.5",
        "panelStroke": "#26324a", "grid": "#33486b",
    },
}
_SERIES = ["#2563eb", "#0f9d8f", "#d97706", "#db2777", "#7c3aed"]
REACTFLOW_PLAYGROUND = "https://play.reactflow.dev"
MONO = "ui-monospace,SFMono-Regular,Menlo,Consolas,monospace"
SERIF = "'Iowan Old Style','Palatino Linotype',Palatino,Georgia,serif"
LEAFW, LEAFH = 176, 104

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


def _t(x, y, s, size, weight, fill, anchor="start", spacing=None, opacity=None,
       family=None):
    extra = f' text-anchor="{anchor}"' if anchor != "start" else ""
    if spacing is not None:
        extra += f' letter-spacing="{spacing}"'
    if opacity is not None:
        extra += f' opacity="{opacity}"'
    if family is not None:
        extra += f' font-family="{family}"'
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
        f'<marker id="arrUp" markerWidth="9" markerHeight="9" refX="3" refY="6.2" '
        f'orient="auto"><path d="M0,6.2 L3,0 L6,6.2 Z" fill="{c["agg"]}"/></marker>'
        f'<pattern id="grid" width="24" height="24" patternUnits="userSpaceOnUse">'
        f'<path d="M24 0 H0 V24" fill="none" stroke="{c["grid"]}" stroke-width="0.6" '
        f'opacity="0.35"/></pattern>'
        f'<pattern id="gridFine" width="6" height="6" patternUnits="userSpaceOnUse">'
        f'<circle cx="0.7" cy="0.7" r="0.7" fill="{c["grid"]}" opacity="0.18"/></pattern>'
        f'<filter id="glow" x="-60%" y="-60%" width="220%" height="220%">'
        f'<feGaussianBlur stdDeviation="2.4" result="b"/><feMerge>'
        f'<feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>'
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


# --- recursive "world of worlds": nested panels (paper-style composition) ----
def _depth_color(c: Dict[str, str], depth: int) -> str:
    return [c["accent"], c["accent2"], c["good"]][depth % 3]


def _compact_state_lines(spec: Dict[str, Any], n: int = 3) -> List[str]:
    schema = {k: v for k, v in spec.get("state_schema", {}).items()
              if not k.startswith("_")}
    out = [f"{k}: {v}" for k, v in list(schema.items())[:n]]
    if len(schema) > n:
        out.append(f"+{len(schema) - n} more")
    return out


def _leaf_card(spec: Dict[str, Any], x: float, y: float, c: Dict[str, str]):
    w, h = LEAFW, LEAFH
    acc = c["accent"]
    out = [f'<g filter="url(#sh)"><rect x="{x:.1f}" y="{y:.1f}" width="{w}" '
           f'height="{h}" rx="13" fill="url(#gnode)" stroke="{c["line"]}" '
           f'stroke-width="1.1"/></g>',
           f'<rect x="{x:.1f}" y="{y:.1f}" width="{w}" height="5" rx="2.5" fill="{acc}"/>',
           _t(x + 13, y + 25, spec["name"], 13.5, "700", c["text"], family=SERIF)]
    for i, ln in enumerate(_compact_state_lines(spec, 3)):
        out.append(_t(x + 13, y + 45 + i * 15, ln, 10, "500", c["muted"], family=MONO))
    out.append(_t(x + 13, y + h - 11, f'{len(spec.get("actions", []))} actions',
                  9.5, "600", acc, family=MONO))
    out.append(f'<circle cx="{x+w-15:.1f}" cy="{y+h-14:.1f}" r="3" fill="{c["good"]}"/>')
    return f'<a href="{_esc(spec["name"] + ".svg")}">' + "".join(out) + "</a>", w, h


def _measure_world(spec: Dict[str, Any]) -> Tuple[float, float]:
    comp = spec.get("composite")
    if not comp:
        return float(LEAFW), float(LEAFH)
    sizes = [_measure_world(ch) for ch in comp.get("children", {}).values()] \
        or [(float(LEAFW), float(LEAFH))]
    pad, head, gap, foot = 20, 52, 30, 16
    inner_w = sum(s[0] for s in sizes) + gap * (len(sizes) - 1)
    rowh = max(s[1] for s in sizes)
    return inner_w + 2 * pad, head + rowh + pad + foot


def _draw_world(spec: Dict[str, Any], x: float, y: float, c: Dict[str, str],
                depth: int = 0):
    comp = spec.get("composite")
    if not comp:
        return _leaf_card(spec, x, y, c)
    ch = comp.get("children", {})
    pad, head, gap, foot = 20, 52, 30, 16
    sizes = {ns: _measure_world(child) for ns, child in ch.items()}
    rowh = max(s[1] for s in sizes.values())
    inner_x, inner_y = x + pad, y + head
    cx, boxes, kids = inner_x, {}, []
    for ns, child in ch.items():
        w_i, h_i = sizes[ns]
        cy = inner_y + (rowh - h_i) / 2
        sub, _, _ = _draw_world(child, cx, cy, c, depth + 1)
        kids.append(sub)
        boxes[ns] = (cx, cy, w_i, h_i)
        cx += w_i + gap
    tw = (cx - gap - inner_x) + 2 * pad
    th = head + rowh + pad + foot
    acc = _depth_color(c, depth)
    over = [
        f'<g filter="url(#shsoft)"><rect x="{x:.1f}" y="{y:.1f}" width="{tw:.1f}" '
        f'height="{th:.1f}" rx="18" fill="{c["bg0"]}" fill-opacity="0.62" '
        f'stroke="{acc}" stroke-width="1.7"/></g>',
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{tw:.1f}" height="6" rx="3" fill="{acc}"/>',
        _t(x + pad, y + 31, spec["name"], 16, "700", c["text"], family=SERIF),
        _t(x + pad, y + 46, f'composite · {len(ch)} worlds', 9.5, "600",
           c["muted"], family=MONO),
    ] + kids

    # bridges between siblings (in the gap, mid-height, double-headed)
    for b in comp.get("bridges", []):
        a, bb = b.get("a"), b.get("b")
        if a in boxes and bb in boxes:
            xa, ya, wa, ha = boxes[a]
            xb, yb, wb, hb = boxes[bb]
            x1, y1, x2, y2 = xa + wa, ya + ha / 2, xb, yb + hb / 2
            mx = (x1 + x2) / 2
            head_marker = "" if b.get("kind") == "route" else ' marker-start="url(#arr)"'
            over.append(f'<path d="M {x1:.1f} {y1:.1f} C {mx:.1f} {y1:.1f}, '
                        f'{mx:.1f} {y2:.1f}, {x2:.1f} {y2:.1f}" fill="none" '
                        f'stroke="{c["edge"]}" stroke-width="1.9" '
                        f'marker-end="url(#arr)"{head_marker}/>')
            over.append(_edge_label(mx, (y1 + y2) / 2 - 11, b.get("name", ""), c))

    # aggregators: chip(s) top-right + dashed up-arrows from each child
    axr = x + tw - pad
    for a in reversed(comp.get("aggregators", [])):
        label = f'Σ {a.get("name", "agg")}'
        pw = len(label) * 6.4 + 16
        axr -= pw
        cyp = y + 14
        over.append(f'<rect x="{axr:.1f}" y="{cyp:.1f}" width="{pw:.1f}" height="21" '
                    f'rx="10.5" fill="{c["node1"]}" stroke="{c["agg"]}" stroke-width="1.1"/>')
        over.append(_t(axr + 8, cyp + 15, label, 10, "700", c["agg"], family=MONO))
        for (bx, by, bw, bh) in boxes.values():
            over.append(f'<path d="M {bx+bw/2:.1f} {by:.1f} L {axr+pw/2:.1f} {cyp+21:.1f}" '
                        f'fill="none" stroke="{c["agg"]}" stroke-width="1" '
                        f'stroke-dasharray="3 3" opacity="0.45" marker-end="url(#arrUp)"/>')
        axr -= 10

    # bindings: dashed downward influence into a child
    for bd in comp.get("bindings", []):
        child = bd.get("child")
        if child in boxes:
            bx, by, bw, bh = boxes[child]
            over.append(f'<path d="M {x+pad:.1f} {y+head-8:.1f} L {bx+bw*0.3:.1f} '
                        f'{by:.1f}" fill="none" stroke="{c["accent2"]}" '
                        f'stroke-width="1.1" stroke-dasharray="2 3" opacity="0.6" '
                        f'marker-end="url(#arr)"/>')
    return "".join(over), tw, th


def _composition(spec: Dict[str, Any], x: float, y: float, w: float,
                 c: Dict[str, str]) -> Tuple[str, float]:
    is_comp = bool(spec.get("composite"))
    title = "WORLD OF WORLDS" if is_comp else "STATE-TRANSITION GRAPH"
    out = [_t(x, y + 2, title, 11, "700", c["muted"], spacing="1.6", family=MONO)]
    gy = y + 20
    if not is_comp:
        nodes, edges = _leaf_graph(spec)
        if not nodes:
            nodes = [{"id": "only", "label": [spec["name"]], "kind": "world",
                      "initial": True}]
            edges = []
        gsvg, gh = _graph_layout(nodes, edges, x, gy, w, c)
        return "".join(out) + gsvg, 20 + gh + 6
    inner, gw, gh = _draw_world(spec, 0.0, 0.0, c, 0)
    s = min(1.0, w / gw) if gw > 0 else 1.0
    sw, sh = gw * s, gh * s
    tx = x + (w - sw) / 2
    wrapped = (f'<g transform="translate({tx:.1f},{gy:.1f}) scale({s:.4f})">'
               f'{inner}</g>')
    return "".join(out) + wrapped, 20 + sh + 10


# --------------------------------------------------------------------------- #
# right column: schema, actions, dynamics, rollout (consistent section style)
# --------------------------------------------------------------------------- #
def _section(x, y, label, c) -> str:
    """A consistent metadata-section header: accent tick + mono kicker."""
    return (f'<rect x="{x:.1f}" y="{y-9:.1f}" width="3" height="11" rx="1.5" '
            f'fill="{c["accent"]}"/>'
            + _t(x + 9, y, label, 10.5, "700", c["muted"], spacing="1.8", family=MONO))


def _fmt_num(v) -> str:
    return f"{v:g}"


def _schema_block(spec, x, y, w, c) -> Tuple[str, float]:
    out = [_section(x, y + 9, "STATE SCHEMA", c)]
    schema = {k: v for k, v in spec.get("state_schema", {}).items()
              if not k.startswith("_")}
    items = list(schema.items())[:6]
    ry = y + 26
    for idx, (k, ty) in enumerate(items):
        if idx % 2 == 0:                                  # zebra row tint
            out.append(f'<rect x="{x-4:.1f}" y="{ry-3:.1f}" width="{w+8:.1f}" '
                       f'height="24" rx="6" fill="{c["accent"]}" fill-opacity="0.035"/>')
        out.append(_t(x + 4, ry + 13, k, 12, "600", c["text"], family=MONO))
        tw = len(str(ty)) * 6.7 + 14
        out.append(f'<rect x="{x+w-tw:.1f}" y="{ry+1:.1f}" width="{tw:.1f}" height="17" '
                   f'rx="5" fill="{c["chipbg"]}"/>')
        out.append(_t(x + w - tw + 7, ry + 13, ty, 10.5, "700", c["accent"], family=MONO))
        ry += 26
    if len(schema) > 6:
        out.append(_t(x + 4, ry + 11, f"+ {len(schema) - 6} more fields", 11, "500",
                      c["muted"], family=MONO))
        ry += 20
    return "".join(out), ry - y + 6


def _chips_block(spec, x, y, w, c, title, items, limit) -> Tuple[str, float]:
    out = [_section(x, y + 9, title, c)]
    cx, cy, rowh = x, y + 24, 26
    for it in items[:limit]:
        cw = len(str(it)) * 6.5 + 18
        if cx + cw > x + w:
            cx, cy = x, cy + rowh
        out.append(f'<rect x="{cx:.1f}" y="{cy:.1f}" width="{cw:.1f}" height="20" rx="6" '
                   f'fill="{c["chipbg"]}" stroke="{c["accent"]}" stroke-width="1" '
                   f'stroke-opacity="0.16"/>')
        out.append(_t(cx + 9, cy + 14, it, 10.5, "600", c["chiptx"], family=MONO))
        cx += cw + 7
    if len(items) > limit:
        cw = 46
        if cx + cw > x + w:
            cx, cy = x, cy + rowh
        out.append(f'<rect x="{cx:.1f}" y="{cy:.1f}" width="{cw}" height="20" rx="6" '
                   f'fill="{c["chipbg"]}"/>')
        out.append(_t(cx + 9, cy + 14, f"+{len(items) - limit}", 10.5, "700",
                      c["muted"], family=MONO))
    return "".join(out), (cy + 20) - y + 8


def _dynamics_badge(spec, x, y, c) -> Tuple[str, float]:
    t = spec.get("transition", {}) or {}
    kind = t.get("kind", "none")
    verified = kind == "code" and not t.get("from_function")
    label = {"code": "verified code" if verified else "code (from function)",
             "function": "function (lossy)", "phased": "phased dynamics",
             "llm": "LLM dynamics", "composite": "composite dynamics",
             "none": "no dynamics"}.get(kind, kind)
    fg = c["good"] if kind == "code" else c["muted"]
    out = [_section(x, y + 9, "DYNAMICS", c)]
    yc = y + 30
    if kind == "code":
        out.append(f'<circle cx="{x+9:.1f}" cy="{yc:.1f}" r="8" fill="{fg}"/>'
                   f'<path d="M{x+5.4:.1f},{yc:.1f} l2.4,2.4 l4-4.4" fill="none" '
                   f'stroke="#fff" stroke-width="1.8" stroke-linecap="round" '
                   f'stroke-linejoin="round"/>')
    else:
        out.append(f'<circle cx="{x+9:.1f}" cy="{yc:.1f}" r="8" fill="none" '
                   f'stroke="{fg}" stroke-width="1.6"/>')
    out.append(_t(x + 24, yc + 4, label, 12.5, "700", c["text"]))
    return "".join(out), 44


def _rollout_block(spec, x, y, w, c) -> Tuple[str, float]:
    out = [_section(x, y + 9, "SAMPLE ROLLOUT", c)]
    series = {k: v for k, v in (spec.get("preview", {}) or {}).get("series", {}).items()
              if isinstance(v, list) and len(v) > 1}
    cy, ch = y + 22, 124
    out.append(f'<rect x="{x:.1f}" y="{cy:.1f}" width="{w:.1f}" height="{ch}" rx="12" '
               f'fill="{c["node1"]}" stroke="{c["line"]}" stroke-width="1"/>')
    if not series:
        out.append(_t(x + w / 2, cy + ch / 2 + 4, "no numeric preview", 12, "500",
                      c["muted"], anchor="middle", family=MONO))
        return "".join(out), (cy + ch) - y + 6
    pad = 16
    names = sorted(series, key=lambda k: -(max(series[k]) - min(series[k])))[:4]
    # faint horizontal gridlines
    for f in (0.25, 0.5, 0.75):
        gyl = cy + pad + f * (ch - 2 * pad)
        out.append(f'<line x1="{x+pad:.1f}" y1="{gyl:.1f}" x2="{x+w-pad:.1f}" '
                   f'y2="{gyl:.1f}" stroke="{c["line"]}" stroke-width="0.8" '
                   f'stroke-dasharray="1 5" opacity="0.7"/>')

    def pts(vals):
        lo, hi = min(vals), max(vals)
        span = (hi - lo) or 1.0
        n = len(vals)
        return [(x + pad + j * (w - 2 * pad) / (n - 1),
                 cy + pad + (hi - v) / span * (ch - 2 * pad))
                for j, v in enumerate(vals)]

    base = cy + ch - pad
    main = names[0]
    p = pts(series[main])
    line = " ".join(f"{px:.1f},{py:.1f}" for px, py in p)
    area = (f'M {p[0][0]:.1f} {base:.1f} L '
            + " L ".join(f"{px:.1f} {py:.1f}" for px, py in p)
            + f' L {p[-1][0]:.1f} {base:.1f} Z')
    out.append(f'<path class="series" d="{area}" fill="url(#garea)"/>')
    out.append(f'<polyline class="series" points="{line}" fill="none" '
               f'stroke="{_SERIES[0]}" stroke-width="2.4" stroke-linejoin="round"/>')
    out.append(f'<circle cx="{p[-1][0]:.1f}" cy="{p[-1][1]:.1f}" r="3.4" '
               f'fill="{_SERIES[0]}"/>')
    out.append(_t(p[-1][0] - 4, p[-1][1] - 8, _fmt_num(series[main][-1]), 9.5, "700",
                  _SERIES[0], anchor="end", family=MONO))
    for i, nm in enumerate(names[1:], start=1):
        pl = " ".join(f"{px:.1f},{py:.1f}" for px, py in pts(series[nm]))
        out.append(f'<polyline class="series" points="{pl}" fill="none" '
                   f'stroke="{_SERIES[i % len(_SERIES)]}" stroke-width="1.8" '
                   f'stroke-opacity="0.9" stroke-linejoin="round"/>')
    # legend with end values
    ly, lx = cy + ch + 16, x + 2
    for i, nm in enumerate(names):
        vals = series[nm]
        arrow = "↑" if vals[-1] > vals[0] else ("↓" if vals[-1] < vals[0] else "→")
        col = _SERIES[i % len(_SERIES)]
        out.append(f'<rect x="{lx:.1f}" y="{ly-9:.1f}" width="9" height="9" rx="2" fill="{col}"/>')
        label = f"{nm} {arrow}"
        out.append(_t(lx + 14, ly, label, 10.5, "600", c["muted"], family=MONO))
        lx += 14 + len(label) * 6.3 + 14
    return "".join(out), (ly + 8) - y


def _details_block(spec, x, y, w, c) -> Tuple[str, float]:
    out = [_section(x, y + 9, "DETAILS", c)]
    facts = []
    nr = len(spec.get("rules", []))
    facts.append(f"{nr} rule" + ("" if nr == 1 else "s"))
    comp = spec.get("composite")
    if comp:
        facts.append(f'{len(comp.get("children", {}))} child worlds')
        nb = len(comp.get("bridges", []))
        if nb:
            facts.append(f"{nb} bridge" + ("" if nb == 1 else "s"))
        na = len(comp.get("aggregators", []))
        if na:
            facts.append(f"{na} aggregator" + ("" if na == 1 else "s"))
        ag = comp.get("agents", {})
        if ag:
            facts.append(f'{len(ag)} agents: ' + ", ".join(list(ag)[:3]))
    gy = y + 24
    for i, fact in enumerate(facts):
        out.append(f'<circle cx="{x+4:.1f}" cy="{gy+i*19-4:.1f}" r="2.4" fill="{c["accent"]}"/>')
        out.append(_t(x + 13, gy + i * 19, fact, 11.5, "600", c["text"], family=MONO))
    return "".join(out), 24 + len(facts) * 19 + 6


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
        _globe(CX + 15, title_y - 2, 15, c),
        _t(CX + 40, title_y - 20, "OPENWORLD · MODEL CARD", 9.5, "700",
           c["accent"], spacing="2.4", family=MONO),
        _t(CX + 40, title_y + 6, spec["name"], 29, "700", c["text"], family=SERIF),
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

    pw, ph = W - 2 * PAD, H - 2 * PAD
    o = PAD + 13
    tick = (lambda px, py, dx, dy:
            f'<path d="M {px:.0f} {py+dy:.0f} L {px:.0f} {py:.0f} L {px+dx:.0f} {py:.0f}" '
            f'fill="none" stroke="{c["accent"]}" stroke-width="1.4" opacity="0.55"/>')
    ticks = (tick(o, o, 12, 12) + tick(W - o, o, -12, 12)
             + tick(o, H - o, 12, -12) + tick(W - o, H - o, -12, -12))
    body = (
        f'<rect x="0" y="0" width="{W}" height="{H}" fill="url(#gbg)"/>'
        f'<g filter="url(#shsoft)"><rect x="{PAD}" y="{PAD}" width="{pw}" '
        f'height="{ph}" rx="24" fill="url(#gbg)" stroke="{c["panelStroke"]}" '
        f'stroke-width="1"/></g>'
        f'<clipPath id="pc"><rect x="{PAD}" y="{PAD}" width="{pw}" height="{ph}" rx="24"/></clipPath>'
        f'<rect x="{PAD}" y="{PAD}" width="{pw}" height="{ph}" fill="url(#gridFine)" '
        f'clip-path="url(#pc)"/>'
        f'<rect x="{PAD}" y="{PAD}" width="{pw}" height="6" rx="3" fill="url(#gacc)"/>'
        + ticks + "".join(head) + "".join(badges)
        + f'<line x1="{CX}" y1="{div_y:.1f}" x2="{W-CX}" y2="{div_y:.1f}" '
          f'stroke="{c["line"]}" stroke-width="1"/>'
        + graph_panel + comp_svg + s_svg + "".join(mid) + r_svg
        + f'<line x1="{CX}" y1="{foot_y-8:.1f}" x2="{W-CX}" y2="{foot_y-8:.1f}" '
          f'stroke="{c["line"]}" stroke-width="1"/>'
        + _t(CX, foot_y + 12, foot, 11, "500", c["muted"])
        + f'<a href="{REACTFLOW_PLAYGROUND}" target="_blank">'
        + _t(W / 2, foot_y + 12, "▸ open in React Flow", 10.5, "700", c["accent"],
             anchor="middle", family=MONO) + "</a>"
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


def _rf_world(spec, nid, parent, x, y, nodes, edges):
    comp = spec.get("composite")
    w, h = _measure_world(spec)
    node = {"id": nid, "position": {"x": round(x, 1), "y": round(y, 1)},
            "data": {"label": spec["name"]},
            "style": {"width": int(w), "height": int(h)}}
    if parent:
        node["parentNode"] = parent
        node["extent"] = "parent"
    if not comp:
        node["type"] = "default"
        lines = _compact_state_lines(spec, 2)
        node["data"]["label"] = spec["name"] + (("\n" + ", ".join(lines)) if lines else "")
        nodes.append(node)
        return
    node["type"] = "group"
    nodes.append(node)
    ch = comp.get("children", {})
    pad, head, gap = 20, 52, 30
    sizes = {ns: _measure_world(c) for ns, c in ch.items()}
    rowh = max(s[1] for s in sizes.values())
    cx, cids = pad, {}
    for ns, child in ch.items():
        w_i, h_i = sizes[ns]
        cid = f"{nid}/{ns}"
        cids[ns] = cid
        _rf_world(child, cid, nid, cx, head + (rowh - h_i) / 2, nodes, edges)
        cx += w_i + gap
    for a in comp.get("aggregators", []):
        aid = f"{nid}/agg/{a.get('name', 'agg')}"
        nodes.append({"id": aid, "parentNode": nid, "extent": "parent",
                      "type": "output", "position": {"x": int(w - 150), "y": 12},
                      "data": {"label": "Σ " + a.get("name", "agg")},
                      "style": {"width": 130, "height": 30}})
        for cid in cids.values():
            edges.append({"id": f"e:{cid}->{aid}", "source": cid, "target": aid,
                          "animated": True})
    for b in comp.get("bridges", []):
        sa, sb = cids.get(b.get("a")), cids.get(b.get("b"))
        if sa and sb:
            e = {"id": f"e:{sa}~{sb}", "source": sa, "target": sb,
                 "label": b.get("name", ""), "type": "smoothstep"}
            if b.get("kind") == "route":
                e["animated"] = True
            edges.append(e)
    for bd in comp.get("bindings", []):
        cid = cids.get(bd.get("child"))
        if cid:
            edges.append({"id": f"e:bind:{cid}:{bd.get('key', '')}", "source": nid,
                          "target": cid, "label": bd.get("key", ""), "animated": True,
                          "style": {"strokeDasharray": "4 3"}})


def to_reactflow(world_or_spec: Union[World, Dict[str, Any]]) -> Dict[str, Any]:
    """Export the world's graph as React Flow ``{nodes, edges, playground}`` with
    positions (nested composites use ``parentNode`` group nodes; leaves use the
    state-transition automaton). ``playground`` is the React Flow playground URL
    (https://play.reactflow.dev) where the exported nodes/edges can be pasted to
    explore interactively. Pure JSON data, no React dependency."""
    spec = _as_spec(world_or_spec)
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    if spec.get("composite"):
        _rf_world(spec, spec["name"], None, 0.0, 0.0, nodes, edges)
        return {"nodes": nodes, "edges": edges, "playground": REACTFLOW_PLAYGROUND}
    g = (spec.get("preview", {}) or {}).get("graph", {}) or {}
    gn, ge = g.get("nodes", []), g.get("edges", [])
    if not gn:
        return {"nodes": [{"id": "n0", "position": {"x": 0, "y": 0},
                           "type": "input", "data": {"label": spec["name"]}}],
                "edges": [], "playground": REACTFLOW_PLAYGROUND}
    ids = [n["id"] for n in gn]
    layer = {i: 0 for i in ids}
    fwd = [e for e in ge if e["src"] != e["dst"]]
    for _ in range(len(ids)):
        changed = False
        for e in fwd:
            if layer[e["dst"]] < layer[e["src"]] + 1 <= len(ids):
                layer[e["dst"]] = layer[e["src"]] + 1
                changed = True
        if not changed:
            break
    rows: Dict[int, int] = {}
    for n in gn:
        lyr = layer[n["id"]]
        r = rows.get(lyr, 0)
        rows[lyr] = r + 1
        nodes.append({"id": f"n{n['id']}", "position": {"x": lyr * 240, "y": r * 110},
                      "type": "input" if n.get("initial") else "default",
                      "data": {"label": "\n".join(n.get("label", []))}})
    for k, e in enumerate(ge):
        edges.append({"id": f"e{k}", "source": f"n{e['src']}",
                      "target": f"n{e['dst']}", "label": e.get("action", "")})
    return {"nodes": nodes, "edges": edges, "playground": REACTFLOW_PLAYGROUND}


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
