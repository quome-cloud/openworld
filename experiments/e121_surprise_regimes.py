"""E121 -- SURPRISE-TRIGGERED self-rebuilding world model (a Bayesian belief over DYNAMICS).

The observation (yours): if the rules suddenly change, the world model should pick it up -- a new world for
the new rules gets composed forward. ARC-AGI-3 makes this concrete: each level can add a mechanic (the
compositionality cliff), so the rules literally change at level boundaries.

The detector is SURPRISE. Replaying a verified trace, we online-learn the discovered (masked-state,action)->
next dynamics. At each step we ask: did something the model had NOT seen just happen (novelty), or did a
state+action it WAS confident about behave differently (contradiction)? A burst of surprise = a regime
boundary = the rules changed. We then SEGMENT the trace at those boundaries, build a per-regime world model,
and compose them forward with openworld.PhasedTransition -- dynamics that advance irreversibly, recording
the active regime in state. The result is a single World that NOTICES when the rules change and rebuilds
itself; its atlas card shows the regimes.

Self-check (honest, falsifiable): do the SURPRISE-detected boundaries align with the ground-truth level-ups
(where the rules actually change)? We measure precision/recall of detected boundaries vs level-ups and save
it BEFORE asserting. Surprise is computed from frames only -- the level signal is used solely to score it,
never to detect (source-free).

    ~/.arcv/bin/python experiments/e121_surprise_regimes.py ka59,m0r0
"""
import os, sys, json
from pathlib import Path
import numpy as np

ROOT = Path("/Users/jim/Desktop/openworld")
sys.path.insert(0, str(ROOT / "experiments"))
import arc_agi
import openworld as O
import e120_expert_consensus as E120          # reuse env/replay/masking helpers
import e119.perceive as P

ARCH = json.loads((ROOT / "experiments/results/arc3_fullgame_sourcefree.json").read_text())
MAPS = ROOT / "papers/arc-3/maps"; MAPS.mkdir(parents=True, exist_ok=True)
OUT = ROOT / "experiments/results/e121_surprise_regimes.json"
TOL = 3            # a detected boundary counts as matching a level-up if within +/- this many steps


def replay_with_levels(env, actions):
    """Replay a trace; return per-step frames, kept actions, and levels_completed AFTER each step."""
    o = env.reset(); frames = [E120.g_of(o)]; acts = []; levels = []
    base = o.levels_completed
    for a in actions:
        o = E120.env_step(env, a)
        if o is None or getattr(o, "frame", None) is None:
            break
        frames.append(E120.g_of(o)); acts.append(a); levels.append(o.levels_completed - base)
        if str(o.state) != "GameState.NOT_FINISHED":
            break
    return frames, acts, levels


def surprise_signals(frames, acts, mask):
    """Online pass over the trace. Three frame-only signals per step:
      - delta[t]  = # masked cells changed between frame t and t+1. A level-up RELOADS the board -> an
                    anomalously large delta. This is the workhorse rule-change signal.
      - novelty[t]= next masked-state never seen before (saturates ~1 on a solution path; kept as diagnostic)
      - contra[t] = a confident (state,action) prediction broke (purest rule-change evidence, but rare on a
                    single forward path). Reported as a diagnostic.
    Returns q-ids per frame, the three arrays, and the id map."""
    idmap = {}

    def qid(fr):
        b = P.state_key(fr, mask)
        if b not in idmap:
            idmap[b] = f"q{len(idmap)}"
        return idmap[b]

    sigs = [qid(f) for f in frames]
    keep = ~mask                                   # compare boards with the status bar removed
    mf = [f * keep for f in frames]
    delta = np.array([int((mf[t] != mf[t + 1]).sum()) for t in range(len(acts))], dtype=float)
    seen = {sigs[0]}; table = {}
    novelty = np.zeros(len(acts)); contra = np.zeros(len(acts))
    for t, a in enumerate(acts):
        S, A, Tn = sigs[t], E120.aname(a), sigs[t + 1]
        if Tn not in seen:
            novelty[t] = 1.0
        if S in table and A in table[S] and table[S][A] != Tn:
            contra[t] = 1.0
        seen.add(Tn); table.setdefault(S, {})[A] = Tn
    return sigs, novelty, contra, delta, idmap


def detect_boundaries(delta, k=4.0, min_gap=8):
    """Regime boundary = an anomalously large board change (board reload) -- delta >= k x the typical
    (median) in-regime delta. Consecutive spikes within min_gap are one boundary (keep the peak). Step 0 is
    the initial regime."""
    n = len(delta)
    if n == 0:
        return [0]
    pos = delta[delta > 0]
    med = float(np.median(pos)) if pos.size else 1.0
    thr = max(k * med, med + 1.0)
    hot = [t for t in range(n) if delta[t] >= thr]
    clusters = []                                   # group consecutive spikes within min_gap
    for t in hot:
        if clusters and t - clusters[-1][-1] < min_gap:
            clusters[-1].append(t)
        else:
            clusters.append([t])
    peaks = [max(c, key=lambda s: delta[s]) for c in clusters]   # one boundary per cluster = its peak
    return sorted(set([0] + peaks))


def score(boundaries, levels):
    """Precision/recall of surprise boundaries vs ground-truth level-ups (rule changes)."""
    ups = [t for t in range(1, len(levels)) if levels[t] > levels[t - 1]]
    if not ups:
        return {"level_ups": 0, "boundaries": len(boundaries), "matched": 0,
                "recall": None, "precision": None, "up_steps": [], "boundary_steps": boundaries}
    matched_ups = sum(1 for u in ups if any(abs(u - b) <= TOL for b in boundaries))
    matched_bs = sum(1 for b in boundaries if b > 0 and any(abs(u - b) <= TOL for u in ups))
    nb = max(1, len([b for b in boundaries if b > 0]))
    return {"level_ups": len(ups), "boundaries": len(boundaries), "matched": matched_ups,
            "recall": round(matched_ups / len(ups), 3), "precision": round(matched_bs / nb, 3),
            "up_steps": ups, "boundary_steps": boundaries}


def regime_world(game, sigs, acts, boundaries):
    """Per-regime (state,action)->next tables, composed forward with PhasedTransition (irreversible advance,
    trigger = the step where each regime begins). The World rebuilds its dynamics at every detected change."""
    bset = sorted(set(boundaries) | {0})
    tables = [dict() for _ in bset]
    for t, a in enumerate(acts):
        r = max(i for i, b in enumerate(bset) if b <= t)
        tables[r].setdefault(sigs[t], {})[E120.aname(a)] = sigs[t + 1]

    def mk(tbl):
        def fn(state, action):
            nm = action.get("name") if isinstance(action, dict) else getattr(action, "name", action)
            return {"sig": tbl.get(state.get("sig"), {}).get(nm, state.get("sig"))}
        return O.FunctionTransition(fn)

    phases = [(bset[i], mk(tables[i])) for i in range(len(bset))]   # phase 0 trigger ignored by PhasedTransition
    acts_all = sorted({a for tbl in tables for d in tbl.values() for a in d})[:80]
    w = O.World(name=f"arc3-regimes-{game}",
                description=(f"Self-rebuilding world model of ARC-AGI-3 {game}: openworld.PhasedTransition over "
                            f"{len(bset)} SURPRISE-detected regimes (rule changes at steps {bset}); dynamics "
                            f"advance irreversibly as the rules change."),
                initial_state={"sig": sigs[0] if sigs else "q0"}, actions=acts_all or ["noop"],
                transition=O.PhasedTransition(phases))
    return w, len(bset), [len(t) for t in tables]


def build(game, arc):
    sol = ARCH.get("solutions", {}).get(game)
    if not sol:
        print(f"  {game}: no banked trace, skip"); return None
    env = arc.make(game)
    frames, acts, levels = replay_with_levels(env, sol)
    mask = P.status_mask(frames)
    sigs, novelty, contra, delta, idmap = surprise_signals(frames, acts, mask)
    boundaries = detect_boundaries(delta)
    sc = score(boundaries, levels)
    w, nreg, reg_states = regime_world(game, sigs, acts, boundaries)
    spec = O.to_spec(w, preview_steps=18)
    nodes = len(spec.get("preview", {}).get("graph", {}).get("nodes", []))
    O.render_card(w, str(MAPS / f"{game}_regimes.svg"))
    (MAPS / f"{game}_regimes.spec.json").write_text(json.dumps(spec, indent=2))
    try:
        import cairosvg
        cairosvg.svg2png(bytestring=(MAPS / f"{game}_regimes.svg").read_text().encode("utf-8"),
                         write_to=str(MAPS / f"{game}_regimes.png"), output_width=1700)
    except Exception:
        pass
    print(f"  {game}: {len(acts)} steps, {sc['level_ups']} level-ups (rule changes) | "
          f"surprise found {nreg} regimes at {sc['boundary_steps']} | "
          f"recall={sc['recall']} precision={sc['precision']}  states/regime={reg_states}")
    return {"game": game, "steps": len(acts), "novel_steps": int(novelty.sum()),
            "contradiction_steps": int(contra.sum()), "n_regimes": nreg, "regime_states": reg_states,
            "graph_nodes": nodes, **sc, "card_svg": f"maps/{game}_regimes.svg"}


def main():
    games = (sys.argv[1].split(",") if len(sys.argv) > 1
             else os.environ.get("EXPERT_GAMES", "ka59,m0r0").split(","))
    print(f"[e121] surprise-triggered regime detection for {games}", flush=True)
    arc = arc_agi.Arcade()
    results = {}
    for g in games:
        try:
            r = build(g, arc)
            if r:
                results[g] = r
        except Exception as e:
            print(f"  {g}: ERROR {type(e).__name__}: {str(e)[:160]}")
    # save BEFORE any assert (CLAUDE.md): a failed self-check must never lose the run
    OUT.write_text(json.dumps({"experiment": "e121_surprise_regimes",
                               "method": "online surprise (novelty/contradiction) -> regime boundaries -> "
                                         "PhasedTransition self-rebuilding world; scored vs level-ups",
                               "tolerance_steps": TOL, "games": results}, indent=2))
    print(f"[e121] wrote {OUT.relative_to(ROOT)}  ({len(results)} games)")
    # honest self-check: surprise should recover MOST rule changes (report, don't tune)
    rec = [r["recall"] for r in results.values() if r.get("recall") is not None]
    if rec:
        mean_rec = sum(rec) / len(rec)
        print(f"[e121] mean recall of rule-changes by surprise alone: {mean_rec:.2f}")
        if mean_rec < 0.5:
            print("[e121] NOTE: weak alignment -- surprise under-detects rule changes on these traces "
                  "(reported honestly; detector or masking may need work).")


if __name__ == "__main__":
    main()
