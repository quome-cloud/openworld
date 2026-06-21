"""E80 law fit (workhorse): test the world-time-compute scaling law across domains.

Law: per-world accuracy under world-time compute saturates, acc(c)=base+(C-base)(1-e^{-c/tau}).
Clause 1 (scaling) is testable OFFLINE from the base/light/heavy arms we already ran: a small
dose (light) should predict the heavy asymptote, and that relation should hold LEAVE-ONE-DOMAIN-
OUT (the only honest universality test). We predict the headroom-normalized heavy lift
  y = (heavy - base) / (1 - base)
from cheap features, and report within-domain Spearman + leave-one-domain-out R^2/Spearman.

Pure numpy, no GPU. Domains: ARC (grids), List Functions (lists), CLRS (algorithms),
Bongard (vision). Run: python3 e80_law_fit.py
"""

import json
from pathlib import Path

import numpy as np

RES = Path(__file__).resolve().parent / "results"
# domain -> (ttt_file, base_arm)
DOMAINS = {
    "listfn":  ("e80_text_listfn.json", "zeroshot"),
    "clrs":    ("e80_text_clrs.json",   "zeroshot"),
    "arc":     ("e80_arc_ttt.json",     "zeroshot"),
    "bongard": ("e80_bongard.json",     "prototype"),
}
EPS = 0.05


def _pw(d):
    return d.get("per_world") or d.get("per_task") or {}


def load_ll(dom):
    """Zero-training likelihood-identifiability features per world (clause 2), if measured:
    ll_slope = answer log-prob gain from 0 to max demos; ll_base = zero-shot log-prob; depth."""
    f = RES / f"e80_ll_{dom}.json"
    if not f.exists():
        return {}
    out = {}
    for w, r in json.load(open(f)).get("per_world", {}).items():
        kll = r.get("kll", {})
        kk = sorted(int(k) for k in kll)
        if len(kk) >= 2:
            out[w] = {"ll_slope": kll[str(kk[-1])] - kll[str(kk[0])],
                      "ll_base": kll[str(kk[0])], "depth": r.get("depth", 0)}
    return out


def load_points():
    pts = []
    for dom, (tf, base_arm) in DOMAINS.items():
        if not (RES / tf).exists():
            continue
        pw = _pw(json.load(open(RES / tf)))
        ll = load_ll(dom)                      # zero-training likelihood features (clause 2)
        base, light, heavy = pw.get(base_arm, {}), pw.get("light", {}), pw.get("heavy", {})
        for w in heavy:
            if w not in base or w not in light:
                continue
            b, l, h = base[w], light[w], heavy[w]
            if None in (b, l, h):
                continue
            if b >= 0.9:                       # no headroom -> world-time compute cannot help
                continue
            head = 1.0 - b
            y = (h - b) / head                 # fraction of headroom captured
            p = dict(domain=dom, world=w, base=b,
                     light_lift=l - b, heavy_lift=h - b,
                     headroom=head, y=float(np.clip(y, -0.5, 1.0)))
            if w in ll:
                p.update(ll[w])                # ll_slope, ll_base, depth
            pts.append(p)
    return pts


def _spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() < 1e-9 or b.std() < 1e-9:
        return None
    return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])


def _r2(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return float(1 - ((y - p) ** 2).sum() / max(1e-9, ((y - y.mean()) ** 2).sum()))


def fit_lodo(pts, feats):
    """Leave-one-domain-out: fit normalized-lift ~ feats on all but one domain, predict it."""
    doms = sorted({p["domain"] for p in pts})
    X = np.column_stack([np.ones(len(pts))] + [[p[f] for p in pts] for f in feats])
    y = np.array([p["y"] for p in pts])
    dom = np.array([p["domain"] for p in pts])
    pred = np.full(len(pts), np.nan)
    per = {}
    for D in doms:
        tr, te = dom != D, dom == D
        if tr.sum() < 5 or te.sum() < 2:
            continue
        beta, *_ = np.linalg.lstsq(X[tr], y[tr], rcond=None)
        pred[te] = X[te] @ beta
        per[D] = {"n": int(te.sum()),
                  "spearman": _spearman(pred[te], y[te]),
                  "pred_mean": round(float(pred[te].mean()), 3),
                  "actual_mean": round(float(y[te].mean()), 3)}
    m = ~np.isnan(pred)
    return {"features": feats, "lodo_r2": round(_r2(y[m], pred[m]), 3) if m.sum() > 2 else None,
            "lodo_spearman": _spearman(pred[m], y[m]), "per_domain": per}


def main():
    pts = load_points()
    doms = sorted({p["domain"] for p in pts})
    out = {"n_worlds": len(pts), "domains": doms,
           "per_domain_n": {d: sum(p["domain"] == d for p in pts) for d in doms}}

    # within-domain: does light-lift rank worlds by heavy-lift? (clause 1, ordinal)
    out["within_domain_light_vs_heavy_spearman"] = {
        d: _spearman([p["light_lift"] for p in pts if p["domain"] == d],
                     [p["heavy_lift"] for p in pts if p["domain"] == d]) for d in doms}

    # CLAUSE 1: light probe predicts heavy asymptote (cheap probe), LODO
    out["models"] = {
        "light_only": fit_lodo(pts, ["light_lift"]),
        "light+base": fit_lodo(pts, ["light_lift", "base"]),
    }
    # CLAUSE 2: predict the asymptote from ZERO-TRAINING likelihood signals (no probe at all)
    pts_ll = [p for p in pts if "ll_slope" in p]
    out["n_with_ll"] = len(pts_ll)
    out["ll_domains"] = sorted({p["domain"] for p in pts_ll})
    if len(pts_ll) >= 8 and len({p["domain"] for p in pts_ll}) >= 2:
        s = np.array([p["ll_slope"] for p in pts_ll])
        out["ll_slope_vs_y_spearman"] = _spearman(s, [p["y"] for p in pts_ll])
        out["within_domain_ll_spearman"] = {
            d: _spearman([p["ll_slope"] for p in pts_ll if p["domain"] == d],
                         [p["y"] for p in pts_ll if p["domain"] == d])
            for d in sorted({p["domain"] for p in pts_ll})}
        out["models"]["ll_freeproxy"] = fit_lodo(pts_ll, ["ll_slope", "base"])
        out["models"]["ll+depth"] = fit_lodo(pts_ll, ["ll_slope", "base", "depth"])
    (RES / "e80_law_fit.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
