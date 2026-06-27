"""E122 -- ONLINE surprise monitor: detect rule changes DURING play (causal), not retrospectively.

E121 segmented a finished trace (its threshold used the whole-trace median -- lookahead). For the monitor to
drive a solver it must fire the MOMENT the rules change, using only frames seen so far. OnlineRegimeMonitor
keeps a running baseline of the typical in-regime board delta and declares a regime change when the current
delta jumps far above it (a board reload). It is fully causal: raw (unmasked) board deltas, a trailing
window, a warmup and a refractory gap -- no future information. On a regime change a solver should drop the
stale dynamics and re-explore the new regime (the surprise-driven reset wired into the expert panel).

Validation: stream the monitor over each banked trace and score the boundaries it raised CAUSALLY against the
ground-truth level-ups (used only to score, never to detect), and emit the same self-rebuilding
PhasedTransition World (E121) but built from the online segmentation.

    ~/.arcv/bin/python experiments/e122_online_regimes.py ka59,m0r0
"""
import os, sys, json
from collections import deque
from pathlib import Path
import numpy as np

ROOT = Path("/Users/jim/Desktop/openworld")
sys.path.insert(0, str(ROOT / "experiments"))
import arc_agi
import openworld as O
import e120_expert_consensus as E120
import e121_surprise_regimes as E121
import e119.perceive as P

ARCH = json.loads((ROOT / "experiments/results/arc3_fullgame_sourcefree.json").read_text())
MAPS = ROOT / "papers/arc-3/maps"; MAPS.mkdir(parents=True, exist_ok=True)
OUT = ROOT / "experiments/results/e122_online_regimes.json"
BOARD = 64 * 64


class OnlineRegimeMonitor:
    """Causal regime-change detector. feed(delta) -> True iff the rules just changed (declared NOW, from
    past deltas only). delta = board cells changed this step. A reload spikes far above the in-regime
    baseline; normal moves change a handful of cells."""

    def __init__(self, k=5.0, warmup=5, min_gap=8, floor_frac=0.03):
        self.k = k; self.warmup = warmup; self.min_gap = min_gap
        self.floor = floor_frac * BOARD          # a rule change moves at least this many cells
        self.recent = deque(maxlen=20)           # trailing in-regime deltas
        self.t = 0; self.last = 0; self.boundaries = [0]; self.regime = 0

    def feed(self, delta):
        base = float(np.median(self.recent)) if self.recent else 1.0
        fired = (self.t - self.last >= self.min_gap and len(self.recent) >= self.warmup
                 and delta >= self.k * max(base, 1.0) and delta >= self.floor)
        if fired:
            self.boundaries.append(self.t); self.last = self.t; self.regime += 1
            self.recent.clear()                  # fresh baseline for the new regime
        else:
            self.recent.append(delta)
        self.t += 1
        return fired


def build(game, arc):
    sol = ARCH.get("solutions", {}).get(game)
    if not sol:
        print(f"  {game}: no banked trace, skip"); return None
    env = arc.make(game)
    frames, acts, levels = E121.replay_with_levels(env, sol)
    # CAUSAL detection: stream raw board deltas (no mask -> no future info) through the monitor.
    mon = OnlineRegimeMonitor(); events = []
    for t in range(len(acts)):
        d = int((frames[t] != frames[t + 1]).sum())
        if mon.feed(d):
            events.append(t)
    online_bounds = mon.boundaries
    sc = E121.score(online_bounds, levels)

    # world-building (post-hoc, fine): masked sigs + the ONLINE segmentation -> self-rebuilding World
    mask = P.status_mask(frames)
    sigs, *_ = E121.surprise_signals(frames, acts, mask)
    w, nreg, reg_states = E121.regime_world(game, sigs, acts, online_bounds)
    w.name = f"arc3-online-{game}"
    spec = O.to_spec(w, preview_steps=18)
    O.render_card(w, str(MAPS / f"{game}_online.svg"))
    (MAPS / f"{game}_online.spec.json").write_text(json.dumps(spec, indent=2))

    # parity check vs E121 retrospective boundaries
    retro = E121.detect_boundaries(
        E121.surprise_signals(frames, acts, mask)[3])
    print(f"  {game}: ONLINE detected {len(events)} changes at {events} (causal) | "
          f"recall={sc['recall']} precision={sc['precision']} vs retro={retro}")
    return {"game": game, "steps": len(acts), "online_events": events,
            "online_boundaries": online_bounds, "retro_boundaries": retro,
            "n_regimes": nreg, **sc, "card_svg": f"maps/{game}_online.svg"}


def main():
    games = (sys.argv[1].split(",") if len(sys.argv) > 1
             else os.environ.get("EXPERT_GAMES", "ka59,m0r0").split(","))
    print(f"[e122] ONLINE (causal) regime monitor for {games}", flush=True)
    arc = arc_agi.Arcade(); results = {}
    for g in games:
        try:
            r = build(g, arc)
            if r:
                results[g] = r
        except Exception as e:
            print(f"  {g}: ERROR {type(e).__name__}: {str(e)[:160]}")
    OUT.write_text(json.dumps({"experiment": "e122_online_regimes",
                               "method": "causal OnlineRegimeMonitor over raw board deltas; scored vs level-ups",
                               "games": results}, indent=2))
    print(f"[e122] wrote {OUT.relative_to(ROOT)}  ({len(results)} games)")
    rec = [r["recall"] for r in results.values() if r.get("recall") is not None]
    if rec:
        print(f"[e122] mean ONLINE recall of rule-changes (causal detection): {sum(rec)/len(rec):.2f}")


if __name__ == "__main__":
    main()
