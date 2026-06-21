"""E80 law: fit and validate the predictive law for WHEN world-time compute pays off.

Joins per-world cheap proxies (in-context slope, headroom, depth from e80_proxy_*.json) with the
MEASURED per-world TTT lift (heavy - base, from the e80_*_ttt / e80_arc_ttt / e80_bongard runs),
then fits  lift ~ slope x headroom x depth-discount  and reports leave-one-DOMAIN-out
predictivity -- i.e. can proxies measured with no fine-tuning forecast the lift on a held-out
domain? Offline (numpy). Writes results/e80_law.json for the paper pipeline.

  python3 e80_law.py
"""

import json
from pathlib import Path

import numpy as np

RES = Path(__file__).resolve().parent / "results"

# domain -> (proxy_file, ttt_file, base_arm)
DOMAINS = {
    "listfn":  ("e80_proxy_listfn.json", "e80_text_listfn.json", "zeroshot"),
    "clrs":    ("e80_proxy_clrs.json",   "e80_text_clrs.json",   "zeroshot"),
    "arc":     ("e80_proxy_arc.json",    "e80_arc_ttt.json",     "zeroshot"),
    "bongard": ("e80_proxy_bongard.json", "e80_bongard.json",    "prototype"),
}


def _pw(d):
    return d.get("per_world") or d.get("per_task") or {}


def load_points():
    pts = []  # dict per world
    for dom, (pf, tf, base) in DOMAINS.items():
        if not (RES / pf).exists() or not (RES / tf).exists():
            continue
        pj = json.load(open(RES / pf))
        proxy = pj["per_world"]
        pw = _pw(json.load(open(RES / tf)))
        heavy, zero = pw.get("heavy", {}), pw.get(base, {})
        for w, pr in proxy.items():
            kacc = pr.get("kacc", {})
            kk = sorted(int(k) for k in kacc)
            if len(kk) < 2 or w not in heavy or w not in zero:
                continue
            if heavy[w] is None or zero[w] is None:
                continue
            slope = kacc[str(kk[-1])] - kacc[str(kk[0])]
            head = 1.0 - kacc[str(kk[-1])]
            pts.append(dict(domain=dom, world=w, slope=slope, headroom=head,
                            depth=pr["depth"], lift=heavy[w] - zero[w]))
    return pts


def _design(pts):
    s = np.array([p["slope"] for p in pts])
    h = np.array([p["headroom"] for p in pts])
    d = np.array([p["depth"] for p in pts])
    X = np.column_stack([np.ones(len(pts)), s, h, s * h, -np.log1p(d)])
    y = np.array([p["lift"] for p in pts])
    return X, y, s


def _r2(y, pred):
    return float(1 - ((y - pred) ** 2).sum() / max(1e-9, ((y - y.mean()) ** 2).sum()))


def _spearman(a, b):
    if len(a) < 3 or np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return None
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    pts = load_points()
    domains = sorted({p["domain"] for p in pts})
    out = {"n_worlds": len(pts), "domains": domains, "per_domain_n":
           {d: sum(1 for p in pts if p["domain"] == d) for d in domains}}
    if len(pts) >= 8 and len(domains) >= 2:
        X, y, s = _design(pts)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        out["r2_in_sample"] = _r2(y, X @ beta)
        out["slope_vs_lift_corr"] = float(np.corrcoef(s, y)[0, 1])
        out["slope_vs_lift_spearman"] = _spearman(s, y)
        # within-domain: does the slope rank worlds by lift INSIDE each domain?
        wd = {}
        for D in domains:
            idx = [i for i, p in enumerate(pts) if p["domain"] == D]
            wd[D] = _spearman(s[idx], y[idx])
        out["within_domain_spearman"] = wd
        out["within_domain_spearman_mean"] = float(np.mean([v for v in wd.values() if v is not None]))
        out["beta"] = {"intercept": float(beta[0]), "slope": float(beta[1]),
                       "headroom": float(beta[2]), "slope_x_headroom": float(beta[3]),
                       "neg_log_depth": float(beta[4])}
        # leave-one-domain-out: fit without domain D, predict its worlds
        lodo = {}
        preds = np.full(len(pts), np.nan)
        for D in domains:
            tr = [i for i, p in enumerate(pts) if p["domain"] != D]
            te = [i for i, p in enumerate(pts) if p["domain"] == D]
            if len(tr) < 5 or len(te) < 2:
                continue
            b, *_ = np.linalg.lstsq(X[tr], y[tr], rcond=None)
            pr = X[te] @ b
            preds[te] = pr
            yt = y[te]
            lodo[D] = {"n": len(te), "pred_mean": float(pr.mean()), "actual_mean": float(yt.mean()),
                       "corr": float(np.corrcoef(pr, yt)[0, 1]) if yt.std() > 1e-6 else None}
        out["leave_one_domain_out"] = lodo
        m = ~np.isnan(preds)
        out["lodo_overall_r2"] = _r2(y[m], preds[m]) if m.sum() > 2 else None
        out["scatter"] = [{"domain": p["domain"], "actual": p["lift"],
                           "pred": float(preds[i])} for i, p in enumerate(pts) if not np.isnan(preds[i])]
    (RES / "e80_law.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in out if k != "scatter"}, indent=2))


if __name__ == "__main__":
    main()
