"""Source-FREE auto-banker -- the OpenWorld-verified gate.

For each scratch_arc/sb_<game>/solved.json (the deepest source-free solve), bank it ONLY if it clears
THREE independent gates, in order:

  1. AUDIT (source-free).  scripts/audit_sandbox.audit -> the agent's working dir must reference NO game
     source (no environment_files, no inspect.getsource / spec_from_file_location / arc_agi import). A
     tainted dir is refused outright -- it is not a fair-agent solve.
  2. REAL-ENV DEPTH.  Replay the action trace from reset() in the real arc_agi engine; it must raise
     levels_completed by the claimed amount (the solution actually works).
  3. OPENWORLD WORLD (the load-bearing piece).  Treat the discovered masked-frame state-transition graph
     as an OpenWorld `World` (masked-frame perceptor -> symbolic state, FunctionTransition over the learned
     table -> dynamics, induced CodeObjective reward=levels_completed), REPLAY THE SOLUTION THROUGH
     world.step, and require world-depth == real-depth, 0 misses, validate_spec()==[] and render_card()
     succeeds. This is exactly E121's round-trip; we reuse its machinery so a source-free solve is banked
     iff it provably runs THROUGH the OpenWorld framework, not only through arc_agi.

Only a solve that clears all three, AND is strictly deeper than the banked depth, is written to
experiments/results/arc3_fullgame_sourcefree.json. We also promote solved.json -> solved_best.json as the
best-keeper seed for the next round. Deterministic (no LLM); safe to run repeatedly.

Run with the arc venv python (needs arc_agi); openworld is imported from the repo root.
"""
import json, sys, glob, os, shutil
from pathlib import Path

ROOT = Path("/Users/jim/Desktop/openworld")
sys.path.insert(0, str(ROOT))                              # openworld (zero-dep core)
sys.path.insert(0, str(ROOT / "experiments"))             # e121 machinery
sys.path.insert(0, str(ROOT / "scratch_arc" / "agent"))   # arc3_harness.Game (real-env verifier)
sys.path.insert(0, str(ROOT / "scripts"))                 # audit_sandbox

from audit_sandbox import audit
import e121_openworld_roundtrip as e121

ARCH = ROOT / "experiments" / "results" / "arc3_fullgame_sourcefree.json"


def load_arch():
    if ARCH.exists():
        return json.loads(ARCH.read_text())
    return {"protocol": "source-free (process-isolated SandboxGame; audit-gated; OpenWorld-verified)",
            "verification": "audit_sandbox (no source access) + real-env replay + E121 OpenWorld World "
                            "round-trip (world.step reproduces depth, valid spec, renderable card)",
            "per_game": {}, "solutions": {}, "roundtrip": {}}


def sb_best(game):
    """Deepest (levels, dict, path) across this game's source-free solved files."""
    best = None
    wd = ROOT / "scratch_arc" / f"sb_{game}"
    for fn in ("solved_best.json", "solved.json"):
        p = wd / fn
        if p.exists():
            try:
                d = json.loads(p.read_text())
                lv = int(d.get("levels", 0))
                if best is None or lv > best[0]:
                    best = (lv, d, p)
            except Exception:
                pass
    return best


def openworld_roundtrip(game, actions):
    """Build the OpenWorld World from the solution trajectory and replay THROUGH it (E121 machinery).
    Returns dict with depth_real, depth_through_world, misses, spec_valid, card_renders, pass."""
    frames, deltas, keys, depth = e121.trace(game, actions)
    sids = e121.state_ids(frames, indexed=False)
    table, det = e121.build_table(sids, deltas, keys)
    indexed = False
    if not det:                                            # masking too coarse -> path world (flagged)
        sids = e121.state_ids(frames, indexed=True)
        table, _ = e121.build_table(sids, deltas, keys)
        indexed = True
    world = e121.make_world(game, sids[0], table)
    wlevels, miss = e121.run_through_world(world, keys)
    from openworld import to_spec, validate_spec, render_card
    spec_valid = card_ok = False
    try:
        spec = to_spec(world)
        spec_valid = (validate_spec(spec) == [])
        card_ok = bool(render_card(spec))
    except Exception as ex:
        return {"depth_real": depth, "depth_through_world": wlevels, "misses": miss,
                "spec_valid": False, "card_renders": False, "indexed_fallback": indexed,
                "n_states": 0, "n_transitions": len(table), "pass": False, "error": str(ex)[:200]}
    return {"depth_real": depth, "depth_through_world": wlevels, "misses": miss,
            "spec_valid": spec_valid, "card_renders": card_ok, "indexed_fallback": indexed,
            "n_states": len({s for k in table for s in (k[0],)} | {v[0] for v in table.values()}),
            "n_transitions": len(table),
            "pass": (wlevels == depth and miss == 0 and spec_valid and card_ok)}


def main():
    arch = load_arch()
    arch.setdefault("roundtrip", {})
    changed = []
    games = sorted(os.path.basename(d).replace("sb_", "")
                   for d in glob.glob(str(ROOT / "scratch_arc" / "sb_*"))
                   if os.path.isdir(d))
    for g in games:
        wd = ROOT / "scratch_arc" / f"sb_{g}"
        sb = sb_best(g)
        if not sb:
            continue
        lv, d, p = sb
        banked = int(arch["per_game"].get(g, {}).get("levels", 0))
        if lv <= banked:
            continue
        # GATE 1: source-free audit
        findings = audit(str(wd))
        if findings:
            print(f"[sf-bank] {g}: TAINTED -> refuse to bank: {findings[:2]}", flush=True)
            continue
        actions = d.get("actions") or []
        if not actions:
            continue
        # GATE 2 + 3: real-env depth AND OpenWorld World round-trip
        try:
            rt = openworld_roundtrip(g, actions)
        except Exception as ex:
            print(f"[sf-bank] {g}: roundtrip error: {ex}", flush=True)
            continue
        if rt["depth_real"] < lv:
            print(f"[sf-bank] {g}: real-env depth {rt['depth_real']} < claim {lv}; skip", flush=True)
            continue
        if not rt["pass"]:
            print(f"[sf-bank] {g}: OpenWorld round-trip FAIL "
                  f"(world {rt['depth_through_world']}/{rt['depth_real']} miss={rt['misses']} "
                  f"spec_valid={rt['spec_valid']} card={rt['card_renders']}); skip", flush=True)
            continue
        # passed all three gates -> bank the (real-env) verified depth
        depth = rt["depth_real"]
        win = int(d.get("win", 0) or arch["per_game"].get(g, {}).get("win", 0))
        arch["per_game"][g] = {"levels": depth, "win": win}
        arch["solutions"][g] = actions
        arch["roundtrip"][g] = rt
        shutil.copy(str(p), str(wd / "solved_best.json"))      # best-keeper seed for next round
        changed.append((g, banked, depth, win))
        print(f"[sf-bank] {g}: {banked} -> {depth}/{win} CLEAN + replay-verified + "
              f"OpenWorld-World-verified ({rt['n_states']} states, "
              f"{'indexed' if rt['indexed_fallback'] else 'masked'})", flush=True)

    # recompute totals + write
    arch["full_games"] = sorted(gm for gm, v in arch["per_game"].items()
                                if v.get("win") and v["levels"] >= v["win"])
    arch["n_full_games"] = len(arch["full_games"])
    arch["total_levels"] = sum(v["levels"] for v in arch["per_game"].values())
    arch["total_possible"] = sum(v.get("win", 0) for v in arch["per_game"].values())
    arch["n_games_started"] = len(arch["per_game"])
    ARCH.write_text(json.dumps(arch, indent=1))
    if changed:
        print("[sf-bank] banked: " + "; ".join(f"{g} {b}->{l}/{w}" for g, b, l, w in changed)
              + f"  =>  {arch['n_full_games']} full, {arch['total_levels']}/{arch['total_possible']} levels",
              flush=True)
    else:
        print("[sf-bank] no new clean + OpenWorld-verified gains", flush=True)


if __name__ == "__main__":
    main()
