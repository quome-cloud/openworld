"""Beautiful, self-contained HTML model cards for world-model specs.

``render_card`` turns a world (or its spec) into a single ``.html`` file with
inline CSS and inline SVG --- no external references, so a card is a drop-in,
embeddable artifact for a "marketplace for world models". ``render_gallery``
writes a responsive grid of tiles linking to each card.

Layout (the approved model-card design): a metadata header band; a left column
with an SVG composition diagram (nested child boxes, bridge arrows, aggregator
roll-ups, counts); a right column with the state schema, action chips, and an SVG
rollout sparkline drawn from the spec's preview series.

Zero-dependency: standard library only.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .spec import SPEC_VERSION, to_spec
from .world import World

# palette per theme: (background tokens live in CSS; these drive the SVG)
_THEMES = {
    "light": {"text": "#1f2933", "muted": "#64748b", "accent": "#2563eb",
              "parent": "#c7d2fe", "child": "#eef2ff", "edge": "#2563eb",
              "agg": "#7c3aed"},
    "dark": {"text": "#e6e9ef", "muted": "#94a3b8", "accent": "#60a5fa",
             "parent": "#27314a", "child": "#1b2233", "edge": "#60a5fa",
             "agg": "#a78bfa"},
}
_SERIES_COLORS = ["#2563eb", "#0f766e", "#d97706", "#b91c1c", "#7c3aed"]


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _as_spec(world_or_spec: Union[World, Dict[str, Any]]) -> Dict[str, Any]:
    return world_or_spec if isinstance(world_or_spec, dict) else to_spec(world_or_spec)


# --------------------------------------------------------------------------- #
# SVG pieces
# --------------------------------------------------------------------------- #
def _box(x, y, w, h, fill, stroke, title, lines, colors, rx=10):
    parts = [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>',
        f'<text x="{x + 12}" y="{y + 22}" font-size="13" font-weight="700" '
        f'fill="{colors["text"]}">{_esc(title)}</text>',
    ]
    for i, line in enumerate(lines):
        parts.append(
            f'<text x="{x + 12}" y="{y + 40 + i * 15}" font-size="10.5" '
            f'fill="{colors["muted"]}">{_esc(line)}</text>')
    return "".join(parts)


def _state_preview_lines(spec: Dict[str, Any], limit: int = 3) -> List[str]:
    schema = spec.get("state_schema", {})
    keys = [k for k in schema if not k.startswith("_")][:limit]
    extra = max(0, len([k for k in schema if not k.startswith("_")]) - limit)
    lines = [f"{k}: {schema[k]}" for k in keys]
    if extra:
        lines.append(f"+{extra} more")
    return lines


def _composition_svg(spec: Dict[str, Any], colors: Dict[str, str]) -> str:
    comp = spec.get("composite")
    width = 340
    if not comp:                                  # leaf world: one box
        h = 86
        body = _box(12, 12, width - 24, h, colors["child"], colors["accent"],
                    spec["name"], _state_preview_lines(spec), colors)
        return _svg_wrap(width, h + 24, body)

    children = comp.get("children", {})
    names = list(children)
    cw, ch, gap, top, pad = 188, 70, 18, 56, 16
    inner_left = pad + 12
    height = top + len(names) * (ch + gap) + 64
    parent = (f'<rect x="{pad}" y="{pad}" width="{width - 2 * pad}" '
              f'height="{height - 2 * pad}" rx="14" fill="{colors["parent"]}" '
              f'fill-opacity="0.35" stroke="{colors["accent"]}" stroke-width="2"/>'
              f'<text x="{pad + 14}" y="{pad + 26}" font-size="13.5" '
              f'font-weight="800" fill="{colors["text"]}">{_esc(spec["name"])}</text>')

    boxes, centers = [], {}
    for i, ns in enumerate(names):
        y = top + i * (ch + gap)
        boxes.append(_box(inner_left, y, cw, ch, colors["child"], colors["accent"],
                          ns, _state_preview_lines(children[ns], 2), colors))
        centers[ns] = (inner_left + cw, y + ch / 2, inner_left, y + ch / 2)

    # bridge arrows down the right gutter
    gutter = inner_left + cw + 16
    edges = []
    for b in comp.get("bridges", []):
        a, bb = b.get("a"), b.get("b")
        if a in centers and bb in centers:
            _, ya, _, _ = centers[a]
            xr, yb, _, _ = centers[bb]
            edges.append(
                f'<path d="M {inner_left + cw} {ya} C {gutter + 18} {ya}, '
                f'{gutter + 18} {yb}, {inner_left + cw} {yb}" fill="none" '
                f'stroke="{colors["edge"]}" stroke-width="1.6" '
                f'marker-end="url(#arrow)"/>'
                f'<text x="{gutter + 22}" y="{(ya + yb) / 2}" font-size="9.5" '
                f'fill="{colors["edge"]}">{_esc(b.get("name", "bridge"))}</text>')

    # aggregator roll-up pills along the bottom
    aggs = comp.get("aggregators", [])
    pills, px = [], inner_left
    py = top + len(names) * (ch + gap) + 4
    for a in aggs:
        label = f'Σ {a.get("name", "agg")}'
        w = 12 + len(label) * 6.4
        pills.append(
            f'<rect x="{px}" y="{py}" width="{w:.0f}" height="22" rx="11" '
            f'fill="none" stroke="{colors["agg"]}" stroke-width="1.3"/>'
            f'<text x="{px + 8}" y="{py + 15}" font-size="10" '
            f'fill="{colors["agg"]}">{_esc(label)}</text>')
        px += w + 10

    defs = (f'<defs><marker id="arrow" markerWidth="8" markerHeight="8" '
            f'refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" '
            f'fill="{colors["edge"]}"/></marker></defs>')
    body = defs + parent + "".join(boxes) + "".join(edges) + "".join(pills)
    return _svg_wrap(width, height, body)


def _svg_wrap(w: int, h: float, body: str) -> str:
    return (f'<svg viewBox="0 0 {w} {h:.0f}" width="100%" '
            f'preserveAspectRatio="xMinYMin meet" '
            f'xmlns="http://www.w3.org/2000/svg" role="img">{body}</svg>')


def _sparkline_svg(preview: Dict[str, Any], colors: Dict[str, str],
                   max_series: int = 4) -> str:
    series = preview.get("series", {}) if preview else {}
    series = {k: v for k, v in series.items() if isinstance(v, list) and len(v) > 1}
    if not series:
        return ""
    names = sorted(series, key=lambda k: -(max(series[k]) - min(series[k])))[:max_series]
    w, h, pad = 320, 96, 10
    rows = []
    legend = []
    for i, name in enumerate(names):
        vals = series[name]
        lo, hi = min(vals), max(vals)
        span = (hi - lo) or 1.0
        n = len(vals)
        pts = " ".join(
            f"{pad + j * (w - 2 * pad) / (n - 1):.1f},"
            f"{h - pad - (v - lo) / span * (h - 2 * pad):.1f}"
            for j, v in enumerate(vals))
        col = _SERIES_COLORS[i % len(_SERIES_COLORS)]
        rows.append(f'<polyline points="{pts}" fill="none" stroke="{col}" '
                    f'stroke-width="2" stroke-linejoin="round"/>')
        arrow = "↑" if vals[-1] > vals[0] else ("↓" if vals[-1] < vals[0] else "→")
        legend.append(
            f'<span class="leg"><i style="background:{col}"></i>'
            f'{_esc(name)} {arrow}</span>')
    svg = _svg_wrap(w, h, "".join(rows))
    return f'<div class="spark">{svg}<div class="legend">{"".join(legend)}</div></div>'


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #
_CSS = """
:root{--bg:#f6f7f9;--card:#fff;--text:#1f2933;--muted:#64748b;--accent:#2563eb;
--chip:#eef2ff;--chiptx:#3349b5;--line:#e6e9ef}
body.dark{--bg:#0f1117;--card:#161a22;--text:#e6e9ef;--muted:#94a3b8;
--accent:#60a5fa;--chip:#1e2533;--chiptx:#9db5ff;--line:#222a38}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);
font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.card{max-width:760px;margin:32px auto;background:var(--card);border:1px solid var(--line);
border-radius:16px;overflow:hidden;box-shadow:0 6px 24px rgba(0,0,0,.06)}
.card-head{padding:20px 24px;border-bottom:1px solid var(--line)}
.title{font-size:22px;font-weight:800;display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.ver{font-size:12px;font-weight:700;color:var(--muted)}
.badge{font-size:11px;font-weight:700;color:var(--chiptx);background:var(--chip);
padding:2px 8px;border-radius:999px}
.desc{margin:8px 0 12px;color:var(--muted);font-size:14px;line-height:1.5}
.tags{display:flex;gap:6px;flex-wrap:wrap}
.tag{font-size:11px;font-weight:600;color:var(--chiptx);background:var(--chip);
padding:3px 9px;border-radius:999px}
.body{display:grid;grid-template-columns:1fr 1fr;gap:8px;padding:18px 24px}
@media(max-width:620px){.body{grid-template-columns:1fr}}
.col h3{font-size:11px;letter-spacing:.08em;color:var(--muted);margin:12px 0 8px}
.col h3:first-child{margin-top:0}
.counts{font-size:11px;color:var(--muted);margin-top:6px}
.schema{list-style:none;margin:0;padding:0;font-size:13px}
.schema li{display:flex;justify-content:space-between;padding:3px 0;border-bottom:1px dashed var(--line)}
.schema .ty{color:var(--accent);font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px}
.chips{display:flex;gap:6px;flex-wrap:wrap}
.chip{font-size:12px;font-weight:600;background:var(--chip);color:var(--chiptx);
padding:3px 10px;border-radius:8px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.spark{margin-top:4px}
.legend{display:flex;gap:12px;flex-wrap:wrap;margin-top:4px;font-size:11px;color:var(--muted)}
.leg i{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:4px;vertical-align:middle}
.card-foot{padding:12px 24px;border-top:1px solid var(--line);font-size:11px;color:var(--muted)}
.grid{max-width:1040px;margin:32px auto;padding:0 16px;display:grid;
grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:16px}
.gh{max-width:1040px;margin:32px auto 0;padding:0 16px}
.tile{display:block;text-decoration:none;color:inherit;background:var(--card);
border:1px solid var(--line);border-radius:14px;padding:16px;transition:transform .08s,box-shadow .08s}
.tile:hover{transform:translateY(-2px);box-shadow:0 8px 22px rgba(0,0,0,.08)}
.tile .tn{font-size:16px;font-weight:800}
.tile .tm{font-size:12px;color:var(--muted);margin:6px 0}
"""


def _header(spec: Dict[str, Any]) -> str:
    card = spec.get("card", {})
    ver = _esc(card.get("version", "0.1"))
    lic = card.get("license")
    badge = f'<span class="badge">{_esc(lic)}</span>' if lic else ""
    desc = _esc(card.get("description") or spec.get("description", ""))
    tags = "".join(f'<span class="tag">{_esc(t)}</span>' for t in card.get("tags", []))
    return (f'<header class="card-head"><div class="title">\U0001F30D '
            f'{_esc(spec["name"])} <span class="ver">v{ver}</span>{badge}</div>'
            f'<p class="desc">{desc}</p><div class="tags">{tags}</div></header>')


def _counts(spec: Dict[str, Any]) -> str:
    comp = spec.get("composite")
    if not comp:
        return '<div class="counts">leaf world</div>'
    return (f'<div class="counts">worlds: {len(comp.get("children", {}))} · '
            f'bridges: {len(comp.get("bridges", []))} · '
            f'agents: {len(comp.get("agents", {}))}</div>')


def _schema_html(spec: Dict[str, Any], limit: int = 10) -> str:
    schema = {k: v for k, v in spec.get("state_schema", {}).items()
              if not k.startswith("_")}
    items = list(schema.items())[:limit]
    rows = "".join(f'<li><span>{_esc(k)}</span><span class="ty">{_esc(t)}</span></li>'
                   for k, t in items)
    if len(schema) > limit:
        rows += f'<li><span>+{len(schema) - limit} more</span><span></span></li>'
    return f'<ul class="schema">{rows}</ul>'


def _actions_html(spec: Dict[str, Any], limit: int = 14) -> str:
    actions = spec.get("actions", [])
    chips = "".join(f'<span class="chip">{_esc(a)}</span>' for a in actions[:limit])
    if len(actions) > limit:
        chips += f'<span class="chip">+{len(actions) - limit}</span>'
    return f'<div class="chips">{chips}</div>'


def render_card(world_or_spec: Union[World, Dict[str, Any]], path: Optional[Union[str, Path]] = None,
                theme: str = "light") -> str:
    """Render a self-contained HTML model card. Returns the HTML; writes it to
    ``path`` when given."""
    spec = _as_spec(world_or_spec)
    colors = _THEMES.get(theme, _THEMES["light"])
    spark = _sparkline_svg(spec.get("preview", {}), colors)
    spark_html = spark or '<div class="counts">no numeric preview available</div>'
    lineage = spec.get("card", {}).get("lineage")
    foot = f'openworld_spec_version {_esc(spec.get("openworld_spec_version", SPEC_VERSION))}'
    if lineage:
        foot += f' · lineage: {_esc(lineage)}'
    body = (
        f'<main class="card">{_header(spec)}<section class="body">'
        f'<div class="col col-left"><h3>COMPOSITION</h3>'
        f'{_composition_svg(spec, colors)}{_counts(spec)}</div>'
        f'<div class="col col-right"><h3>STATE SCHEMA</h3>{_schema_html(spec)}'
        f'<h3>ACTIONS</h3>{_actions_html(spec)}'
        f'<h3>SAMPLE ROLLOUT</h3>{spark_html}</div>'
        f'</section><footer class="card-foot">{foot}</footer></main>')
    doc = (f'<!DOCTYPE html>\n<html lang="en"><head><meta charset="utf-8">'
           f'<meta name="viewport" content="width=device-width, initial-scale=1">'
           f'<title>{_esc(spec["name"])} · OpenWorld model card</title>'
           f'<style>{_CSS}</style></head><body class="{_esc(theme)}">{body}</body></html>')
    if path is not None:
        Path(path).write_text(doc, encoding="utf-8")
    return doc


def render_gallery(specs: List[Dict[str, Any]], path: Optional[Union[str, Path]] = None,
                   theme: str = "light", title: str = "OpenWorld model gallery",
                   card_dir: str = "") -> str:
    """Render a responsive grid of tiles linking to each world's ``<name>.html``."""
    tiles = []
    for spec in specs:
        card = spec.get("card", {})
        comp = spec.get("composite")
        meta = ("leaf world" if not comp else
                f'{len(comp.get("children", {}))} worlds · '
                f'{len(comp.get("bridges", []))} bridges')
        tags = " · ".join(card.get("tags", [])[:3])
        tag_html = "".join(f'<span class="tag">{_esc(t)}</span>'
                           for t in card.get("tags", [])[:4])
        href = f'{card_dir}{spec["name"]}.html'
        tiles.append(
            f'<a class="tile" href="{_esc(href)}"><div class="tn">\U0001F30D '
            f'{_esc(spec["name"])}</div><div class="tm">{_esc(meta)}</div>'
            f'<div class="tags">{tag_html}</div>'
            f'<div class="tm">{_esc(tags)}</div></a>')
    doc = (f'<!DOCTYPE html>\n<html lang="en"><head><meta charset="utf-8">'
           f'<meta name="viewport" content="width=device-width, initial-scale=1">'
           f'<title>{_esc(title)}</title><style>{_CSS}</style></head>'
           f'<body class="{_esc(theme)}"><div class="gh"><h1>{_esc(title)}</h1></div>'
           f'<div class="grid">{"".join(tiles)}</div></body></html>')
    if path is not None:
        Path(path).write_text(doc, encoding="utf-8")
    return doc
