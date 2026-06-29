"""E119 Phase 0 — deterministic probe: does any macro selection signal carry directional
information on the zero-reward procedure-walls? No LLM; reuses existing perception/search/DSL."""
import numpy as np
from e119 import slm


def enumerate_predicates(frames):
    """All reach/count/align predicates over the colors and per-color counts observed in `frames`."""
    grids = [np.asarray(f).reshape(64, 64) for f in frames]
    colors = sorted({int(c) for g in grids for c in np.unique(g)})
    preds = [{"type": "reach", "color": c} for c in colors]
    for c in colors:
        for k in sorted({int((g == c).sum()) for g in grids}):
            for op in ("==", ">=", "<="):
                preds.append({"type": "count", "color": c, "op": op, "k": k})
    for i, a in enumerate(colors):
        for b in colors[i + 1:]:
            preds.append({"type": "align", "a": a, "b": b})
    return preds


def scan_satisfiable(preds, frames):
    """Subset of `preds` ever true on some frame in `frames`."""
    return [p for p in preds if slm.satisfiable(p, frames)]
