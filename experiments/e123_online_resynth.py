"""E123 -- ONLINE per-regime RE-SYNTHESIS loop (replay-to-boundary, then rebuild).

The mechanism you described: because the env is replay-only (no snapshot), re-synthesizing a regime mid-play
means -- when surprise says the rules changed -- REPLAY from reset() to the last completed frame (the regime
boundary), then build a FRESH world model for the new regime from there, independent of the old rules, and
append it. The earlier regimes stay frozen and verified; only the current one is (re)built.

This is the programmatic loop that E121/E122 implied but did not run: E121 built one global table then sliced
it; here each regime gets its OWN replay-to-boundary + fresh model, exactly the workflow. We then compose the
per-regime models with openworld.PhasedTransition and ROUND-TRIP verify that the resynthesized composite
reproduces the actual play (fidelity = fraction of steps whose next-state it predicts correctly).

Honest scope: exploration here is GUIDED by the verified trace (the action source), so we can reach deep
regimes and isolate the re-synthesis mechanism; autonomous deep search to FIND those actions is the separate
goal-inference wall (E88-E90/E119), not what this measures. Detection is causal (E122 OnlineRegimeMonitor).

    ~/.arcv/bin/python experiments/e123_online_resynth.py ka59,m0r0
"""
import os, sys, json
from pathlib import Path
import numpy as np

ROOT = Path("/Users/jim/Desktop/openworld")
sys.path.insert(0, str(ROOT / "experiments"))
import arc_agi
import openworld as O
import e120_expert_consensus as E120
import e121_surprise_regimes as E121
import e122_online_regimes as E122
import e119.perceive as P

ARCH = json.loads((ROOT / "experiments/results/arc3_fullgame_sourcefree.json").read_text())
MAPS = ROOT / "papers/arc-3/maps"; MAPS.mkdir(parents=True, exist_ok=True)
OUT = ROOT / "experiments/results/e123_online_resynth.json"


def detect_online(env, actions):
    """Causal forward pass: stream raw board deltas through the monitor; return regime boundaries + frames."""
    o = env.reset(); frames = [E120.g_of(o)]; acts = []; levels = []; base = o.levels_completed
    mon = E122.OnlineRegimeMonitor()
    for a in actions:
        o = E120.env_step(env, a)
        if o is None or getattr(o, "frame", None) is None:
            break
        fr = E120.g_of(o)
        mon.feed(int((frames[-1] != fr).sum()))
        frames.append(fr); acts.append(a); levels.append(o.levels_completed - base)
        if str(o.state) != "GameState.NOT_FINISHED":
            break
    return mon.boundaries, frames, acts, levels


def _valid(o):
    return o is not None and getattr(o, "frame", None) is not None and np.asarray(o.frame).size > 0


def resynth_regime(arc, game, actions, lo, hi, mask, idmap, true_sig_lo):
    """REPLAY-TO-BOUNDARY then build a FRESH model for [lo,hi). The env is replay-only and reset() pollutes a
    progressed multi-level game, so we take a FRESH env (clean level-0 reset), replay actions[:lo] to reach
    the regime's start frame (the last completed frame), then learn that regime's (state,action)->next table
    by stepping through its own actions only -- independent of every other regime. We verify the replayed
    boundary frame matches the originally observed one (replay-to-boundary landed where it should)."""
    env = arc.make(game)                        # fresh env => clean reset to level 0 (avoids reset pollution)
    o = env.reset()
    for a in actions[:lo]:                      # replay to the last completed frame (regime start)
        no = E120.env_step(env, a)
        if not _valid(no):
            break
        o = no
    if not _valid(o):
        return {}, "q0", False
    def qid(fr):
        b = P.state_key(fr, mask)
        if b not in idmap:
            idmap[b] = f"q{len(idmap)}"
        return idmap[b]

    s = qid(E120.g_of(o)); start = s; landed = (s == true_sig_lo); table = {}
    for t in range(lo, hi):
        o = E120.env_step(env, actions[t])
        if not _valid(o):
            break
        ns = qid(E120.g_of(o))
        table.setdefault(s, {})[E120.aname(actions[t])] = ns
        s = ns
    return table, start, landed


def compose(game, regimes, s0):
    """regimes: ordered list of (lo, table). Compose into a PhasedTransition self-rebuilding World."""
    def mk(tbl):
        def fn(state, action):
            nm = action.get("name") if isinstance(action, dict) else getattr(action, "name", action)
            return {"sig": tbl.get(state.get("sig"), {}).get(nm, state.get("sig"))}
        return O.FunctionTransition(fn)
    phases = [(lo, mk(tbl)) for lo, tbl in regimes]            # trigger = regime start step; phase 0 ignored
    acts = sorted({a for _, tbl in regimes for d in tbl.values() for a in d})[:80]
    w = O.World(name=f"arc3-resynth-{game}",
                description=(f"Online re-synthesized world model of ARC-AGI-3 {game}: {len(regimes)} regimes, "
                            f"each built by REPLAY-TO-BOUNDARY then a fresh model, composed with "
                            f"openworld.PhasedTransition (irreversible advance as the rules change)."),
                initial_state={"sig": s0}, actions=acts or ["noop"], transition=O.PhasedTransition(phases))
    return w


def roundtrip(w, actions, true_sigs):
    """Does the resynthesized composite reproduce play EXACTLY? Step the World over the trace's actions and
    compare each predicted next-state to the actually-observed one. Fidelity = fraction matched (a complete,
    correctly-segmented per-regime model -> 1.0)."""
    st = dict(w.initial_state); ok = 0; n = 0
    for i, a in enumerate(actions):
        st = w.transition.step(st, O.Action(E120.aname(a)))   # FunctionTransition.step calls action.to_dict()
        if i + 1 < len(true_sigs) and st.get("sig") == true_sigs[i + 1]:
            ok += 1
        n += 1
    return round(ok / max(n, 1), 3)


def build(game, arc):
    sol = ARCH.get("solutions", {}).get(game)
    if not sol:
        print(f"  {game}: no banked trace, skip"); return None
    env = arc.make(game)
    bounds, frames, acts, levels = detect_online(env, sol)
    mask = P.status_mask(frames)
    idmap = {}                                  # ONE master id-map: replayed regime states reuse these ids

    def qid(fr):
        b = P.state_key(fr, mask)
        if b not in idmap:
            idmap[b] = f"q{len(idmap)}"
        return idmap[b]

    true_sigs = [qid(f) for f in frames]        # the clean, full sig sequence (ground truth for round-trip)
    bset = sorted(set(bounds) | {0})
    spans = [(bset[i], bset[i + 1] if i + 1 < len(bset) else len(acts)) for i in range(len(bset))]
    regimes = []; landed = []
    for lo, hi in spans:                        # <-- each regime: fresh env, replay-to-boundary, rebuild
        table, _start, ok = resynth_regime(arc, game, acts, lo, hi, mask, idmap, true_sigs[lo])
        regimes.append((lo, table)); landed.append(ok)
    w = compose(game, regimes, true_sigs[0])
    fid = roundtrip(w, acts, true_sigs)
    sc = E121.score(bset, levels)
    O.render_card(w, str(MAPS / f"{game}_resynth.svg"))
    (MAPS / f"{game}_resynth.spec.json").write_text(json.dumps(O.to_spec(w, preview_steps=18), indent=2))
    print(f"  {game}: {len(spans)} regimes re-synthesized (replay-to-boundary) | "
          f"boundary landed {sum(landed)}/{len(landed)} | round-trip fidelity={fid} | "
          f"recall={sc['recall']} | states/regime={[len(t) for _, t in regimes]}")
    return {"game": game, "n_regimes": len(spans), "boundaries": bset,
            "replay_to_boundary_landed": f"{sum(landed)}/{len(landed)}", "roundtrip_fidelity": fid,
            "regime_states": [len(t) for _, t in regimes], "recall_vs_levelups": sc["recall"],
            "card_svg": f"maps/{game}_resynth.svg"}


def main():
    games = (sys.argv[1].split(",") if len(sys.argv) > 1
             else os.environ.get("EXPERT_GAMES", "ka59,m0r0").split(","))
    print(f"[e123] online per-regime re-synthesis (replay-to-boundary) for {games}", flush=True)
    arc = arc_agi.Arcade(); results = {}
    for g in games:
        try:
            r = build(g, arc)
            if r:
                results[g] = r
        except Exception as e:
            print(f"  {g}: ERROR {type(e).__name__}: {str(e)[:160]}")
    OUT.write_text(json.dumps({"experiment": "e123_online_resynth",
                               "method": "per-regime replay-to-boundary + fresh model -> PhasedTransition; "
                                         "round-trip verified",
                               "games": results}, indent=2))
    print(f"[e123] wrote {OUT.relative_to(ROOT)}  ({len(results)} games)")
    fid = [r["roundtrip_fidelity"] for r in results.values()]
    if fid:
        print(f"[e123] mean round-trip fidelity of re-synthesized composites: {sum(fid)/len(fid):.2f}")


if __name__ == "__main__":
    main()
