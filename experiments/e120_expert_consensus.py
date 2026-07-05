"""E120 -- Bayesian-experts CONSENSUS world model (ConsensusTransition over expert LENSES).

The expert-panel router (scripts/arc_experts.py) runs differently-primed Claude subagents -- topology /
temporal / force / color-algebra / symmetry / perception-reframe / counting -- on a stuck game. Each lens
that produced a verified action trace is replayed here into ITS OWN openworld.World: the discovered
(masked-state, action) -> next-state dynamics as a FunctionTransition. We then COMBINE the experts with
openworld.ConsensusTransition(mode="vote"): per-state majority over the experts' predicted next-states,
each member weighted by its ENV-VERIFIED fidelity (levels reached). The result is ONE consensus world
model per game -- the panel's agreed dynamics -- serialized to an atlas card (the vote IS the map).

This is the principled MSA setup (arXiv 2507.12547) with subagents as experts rather than different
models: N experts -> N candidate Worlds -> a ConsensusTransition vote -> one consensus World. It makes the
expert frames genuine OpenWorld worlds, not just prompts. Source-free + env-grounded throughout (fidelity =
real levels_completed; identical masked states share a node across experts so the vote aligns).

Run (needs the arc venv + arc_agi):
    ~/.arcv/bin/python experiments/e120_expert_consensus.py ka59,m0r0
    DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib ~/.arcv/bin/python experiments/e120_expert_consensus.py  # +PNG
"""
import os, sys, json, glob, re, io
from pathlib import Path
import numpy as np

ROOT = Path("/Users/jim/Desktop/openworld")
sys.path.insert(0, str(ROOT / "experiments"))
import arc_agi
from arcengine import GameAction
import openworld as O
from openworld.transition import FunctionTransition
import e119.perceive as P

ARCH = ROOT / "experiments/results/arc3_fullgame_sourcefree.json"
TRACES = ROOT / "experiments/results/arc3_traces"
MAPS = ROOT / "papers/arc-3/maps/multiworld"; MAPS.mkdir(parents=True, exist_ok=True)
OUT = ROOT / "experiments/results/e120_expert_consensus.json"
SIMPLE = {1: GameAction.ACTION1, 2: GameAction.ACTION2, 3: GameAction.ACTION3,
          4: GameAction.ACTION4, 5: GameAction.ACTION5, 7: GameAction.ACTION7}


def g_of(o):
    a = np.asarray(o.frame); return (a[-1] if a.ndim == 3 else a).reshape(64, 64)


def aname(a):
    """Trace element -> stable action name. [a]=directional 's<a>'; [6,x,y]=click 'c<x>_<y>'."""
    return f"c{a[1]}_{a[2]}" if a[0] == 6 else f"s{a[0]}"


def env_step(env, a):
    return env.step(GameAction.ACTION6, {"x": a[1], "y": a[2]}) if a[0] == 6 else env.step(SIMPLE[a[0]])


def lens_of_rid(rid):
    """Recover which expert lens drove a run from its captured prompt ('--- STRATEGY LENS (expert: X)')."""
    p = TRACES / "prompts" / f"{rid}.md"
    if p.exists():
        m = re.search(r"expert:\s*(\w+)", p.read_text(encoding="utf-8", errors="ignore"))
        if m:
            return m.group(1)
    return None


def gather(game):
    """Every verified trace for a game: the banked incumbent + each per-expert run. Dedup by lens,
    keeping the deepest (highest-fidelity) trace per lens -> list of (lens, actions, fidelity)."""
    arch = json.loads(ARCH.read_text())
    cand = []
    banked = arch.get("solutions", {}).get(game)
    if banked:
        cand.append(("banked", banked, int(arch["per_game"].get(game, {}).get("levels", 0))))
    for f in sorted(glob.glob(str(TRACES / "solutions" / f"{game}__*.json"))):
        try:
            d = json.loads(open(f).read())
        except Exception:
            continue
        if not d.get("actions"):
            continue
        lens = lens_of_rid(Path(f).stem) or "open"
        cand.append((lens, d["actions"], int(d.get("levels", 0))))
    best = {}
    for lens, acts, fid in cand:
        if lens not in best or fid > best[lens][1]:
            best[lens] = (acts, fid)
    return [(lens, a, f) for lens, (a, f) in best.items()]


def replay_frames(env, actions):
    """Replay a verified trace; return the list of observed 64x64 frames (start + after each step)."""
    o = env.reset(); frames = [g_of(o)]; acts = []
    for a in actions:
        o = env_step(env, a)
        if o is None or getattr(o, "frame", None) is None:
            break
        frames.append(g_of(o)); acts.append(a)
        if str(o.state) != "GameState.NOT_FINISHED":
            break
    return frames, acts


def replay_table(env, actions, mask, idmap):
    """Replay a trace under a SHARED game mask, building the (state,action)->next table over SHARED q-ids
    (idmap: masked-bytes->qN, shared across ALL of a game's experts so identical masked states vote on the
    same node -- this is what makes the ConsensusTransition vote semantically correct). Returns (table,
    s0_id)."""
    frames, acts = replay_frames(env, actions)

    def qid(fr):
        b = P.state_key(fr, mask)
        if b not in idmap:
            idmap[b] = f"q{len(idmap)}"
        return idmap[b]

    sigs = [qid(f) for f in frames]
    table = {}
    for i, a in enumerate(acts):
        table.setdefault(sigs[i], {})[aname(a)] = sigs[i + 1]
    return table, (sigs[0] if sigs else "q0")


def table_transition(table):
    def fn(state, action):
        nm = action.get("name") if isinstance(action, dict) else getattr(action, "name", action)
        return {"sig": table.get(state.get("sig"), {}).get(nm, state.get("sig"))}
    return FunctionTransition(fn)


def consensus_world(game, members_data, s0):
    """members_data: list of (lens, table, fidelity). Build the ConsensusTransition(mode='vote') World."""
    members = [(table_transition(tbl), float(fid)) for (lens, tbl, fid) in members_data if tbl] \
        or [(table_transition({}), 0.0)]
    cons = O.ConsensusTransition(members, mode="vote")
    acts = sorted({a for (lens, tbl, fid) in members_data for d in tbl.values() for a in d})[:80]
    panel = ", ".join(f"{l}(L{f})" for l, t, f in members_data)
    w = O.World(name=f"arc3-consensus-{game}",
                description=(f"Bayesian-experts CONSENSUS world model of ARC-AGI-3 {game}: "
                            f"openworld.ConsensusTransition vote over {len(members)} expert lenses "
                            f"[{panel}], each weighted by env-verified levels reached."),
                initial_state={"sig": s0}, actions=acts or ["noop"], transition=cons)
    return w, len(members)


def build(game, arc):
    members_data = gather(game)
    if not members_data:
        print(f"  {game}: no verified traces yet, skip"); return None
    env = arc.make(game); idmap = {}; s0 = None
    # ONE shared mask per game: derive it from the deepest (most informative) trace, then reuse for every
    # lens so the same physical state -> same masked bytes -> same q-id across experts (aligned voting).
    deepest = max(members_data, key=lambda m: len(m[1]))[1]
    mask = P.status_mask(replay_frames(env, deepest)[0])
    built = []
    for lens, acts, fid in members_data:
        tbl, s0i = replay_table(env, acts, mask, idmap)
        s0 = s0 or s0i
        built.append((lens, tbl, fid))
        print(f"    expert {lens:<18} fidelity L{fid}  -> {len(tbl)} states")
    w, nmem = consensus_world(game, built, s0)
    spec = O.to_spec(w, preview_steps=16)
    nodes = len(spec.get("preview", {}).get("graph", {}).get("nodes", []))
    edges = len(spec.get("preview", {}).get("graph", {}).get("edges", []))
    svg = MAPS / f"{game}_consensus.svg"
    O.render_card(w, str(svg))
    (ROOT / "experiments/results").mkdir(exist_ok=True)
    (MAPS / f"{game}_consensus.spec.json").write_text(json.dumps(spec, indent=2))
    png = None
    try:  # optional raster if cairosvg + libcairo are available
        import cairosvg
        png = str(svg).replace(".svg", ".png")
        cairosvg.svg2png(bytestring=svg.read_text(encoding="utf-8").encode("utf-8"),
                         write_to=png, output_width=1700)
    except Exception:
        png = None
    print(f"  {game}: consensus over {nmem} experts -> {nodes} nodes / {edges} edges  card={svg.name}"
          + (f" (+png)" if png else ""))
    return {"game": game, "n_experts": nmem,
            "experts": [{"lens": l, "fidelity": f, "states": len(t)} for l, t, f in built],
            "shared_states": len(idmap), "graph_nodes": nodes, "graph_edges": edges,
            "card_svg": f"maps/{game}_consensus.svg", "card_png": (os.path.basename(png) if png else None)}


def main():
    games = (sys.argv[1].split(",") if len(sys.argv) > 1
             else os.environ.get("EXPERT_GAMES", "ka59,m0r0").split(","))
    print(f"[e120] expert-consensus world models for {games}", flush=True)
    arc = arc_agi.Arcade()       # make ONCE; reuse arc.make per game (arc.make is slow)
    results = {}
    for g in games:
        try:
            r = build(g, arc)
            if r:
                results[g] = r
        except Exception as e:
            print(f"  {g}: ERROR {type(e).__name__}: {str(e)[:160]}")
    OUT.write_text(json.dumps({"experiment": "e120_expert_consensus",
                               "method": "ConsensusTransition(mode=vote) over expert-lens world models",
                               "games": results}, indent=2))
    print(f"[e120] wrote {OUT.relative_to(ROOT)}  ({len(results)} games)")


if __name__ == "__main__":
    main()
