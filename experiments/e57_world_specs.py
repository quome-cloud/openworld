"""E57 - Portable world-model specs + a marketplace gallery of model cards.

Every world model in the framework should be expressible as a single portable
JSON spec (a publishable unit, like a HuggingFace model card) and renderable to a
beautiful, self-contained HTML card. This experiment exercises that on real
worlds already in the repo --- three leaf worlds and a composed multi-world ---
and proves the spec is a *lossless* artifact:

  1. to_spec(world)            -> a JSON-friendly spec dict
  2. validate_spec(spec)       -> [] (passes the marketplace publish gate)
  3. from_spec(spec, allow_code=True) -> a runnable world whose rollout exactly
     reproduces the original (behavioral round-trip)
  4. render_card / render_gallery -> a browsable gallery/ (the marketplace seed)

Deterministic/offline; standard library only.
"""

from pathlib import Path

from openworld import (Aggregator, Bridge, CodeTransition, CompositeWorld,
                       from_spec, render_card, render_gallery, spec_to_json,
                       to_spec, validate_spec)
from openworld.state import Action

from common import make_oracle_world, save_results

GALLERY = Path(__file__).resolve().parent.parent / "gallery"


# A module-level aggregator + bridge so inspect.getsource recovers them (the spec
# stays round-trippable rather than lossy).
def total_treated(children):
    return sum(int(c.get("treated", 0)) for c in children.values())


TRANSFER_CODE = """
def transition(state, action):
    a = dict(state["a"]); b = dict(state["b"])
    if a["critical_waiting"] > b["critical_waiting"] + 1:
        a["critical_waiting"] -= 1
        b["critical_waiting"] += 1
    return {"a": a, "b": b}
"""


def hospital_network():
    """A composite of two triage clinics that load-balance critical patients."""
    return CompositeWorld(
        name="hospital-network",
        children={"north": make_oracle_world("triage"),
                  "south": make_oracle_world("triage")},
        bridges=[Bridge(name="transfer", a="north", b="south",
                        transition=CodeTransition(TRANSFER_CODE),
                        description="balance critical load north->south")],
        aggregators=[Aggregator(name="total_treated", fn=total_treated)],
        default_actions={"north": "treat_critical", "south": "treat_moderate"},
        timescales={"north": 1, "south": 1},
        description="Two triage clinics that transfer critical patients to "
                    "balance load; treated counts roll up to the network.",
    )


def rollout(world, actions, agent=None):
    states, s = [], world.initial_state.copy()
    for a in actions:
        s = world.transition.step(s, Action(a, agent=agent))
        states.append(dict(s))
    return states


# (world factory, replay actions, agent, card metadata)
WORLDS = [
    (lambda: make_oracle_world("sprint"), ["ship", "ship", "fix", "refactor", "ship"],
     None, {"version": "1.0", "license": "MIT", "lineage": "E34 sprint world",
            "tags": ["engineering", "verified", "leaf"]}),
    (lambda: make_oracle_world("triage"),
     ["treat_critical", "wait", "treat_moderate", "wait", "treat_critical"], None,
     {"version": "1.0", "license": "MIT", "lineage": "E08/E26 triage world",
      "tags": ["healthcare", "queue", "leaf"]}),
    (lambda: make_oracle_world("orchard"), ["pick", "pick", "wait", "pick"], "alice",
     {"version": "1.0", "license": "MIT", "lineage": "E01 orchard world",
      "tags": ["resource", "multi-agent", "leaf"]}),
    (hospital_network, ["tick", "tick", "north:treat_critical", "tick"], None,
     {"version": "0.9", "license": "MIT", "lineage": "composed from triage",
      "tags": ["healthcare", "composite", "multi-world"]}),
]


def main():
    GALLERY.mkdir(exist_ok=True)
    specs, rows = [], []
    for factory, actions, agent, card in WORLDS:
        world = factory()
        spec = to_spec(world, card=card)
        problems = validate_spec(spec)
        reloaded = from_spec(spec, allow_code=True)
        try:
            exact = rollout(world, actions, agent) == rollout(reloaded, actions, agent)
        except Exception:
            exact = False
        spec_json = spec_to_json(spec)
        (GALLERY / f"{spec['name']}.json").write_text(spec_json, encoding="utf-8")
        render_card(spec, path=GALLERY / f"{spec['name']}.svg")
        specs.append(spec)
        rows.append({
            "name": spec["name"],
            "kind": "composite" if "composite" in spec else "leaf",
            "n_children": len(spec.get("composite", {}).get("children", {})),
            "validated": problems == [],
            "round_trip_exact": exact,
            "spec_bytes": len(spec_json),
            "n_actions": len(spec["actions"]),
            "problems": problems,
        })

    render_gallery(specs, path=GALLERY / "index.svg")
    results = {
        "n_worlds": len(rows),
        "all_validated": all(r["validated"] for r in rows),
        "all_round_trip_exact": all(r["round_trip_exact"] for r in rows),
        "rows": rows,
        "gallery_dir": str(GALLERY.relative_to(GALLERY.parent)),
    }
    save_results("e57_world_specs", results)

    print("E57 - world-model specs + marketplace gallery\n")
    print(f"  {'world':<18}{'kind':<11}{'valid':>7}{'round-trip':>12}{'bytes':>8}")
    for r in rows:
        print(f"  {r['name']:<18}{r['kind']:<11}{str(r['validated']):>7}"
              f"{str(r['round_trip_exact']):>12}{r['spec_bytes']:>8}")
    print(f"\n  gallery written to {GALLERY}/ (index.svg + per-world .svg/.json)")

    # --- self-checks ---
    assert all(r["validated"] for r in rows), "every spec must pass validate_spec"
    assert all(r["round_trip_exact"] for r in rows), \
        "every spec must round-trip behaviorally with allow_code=True"
    assert (GALLERY / "index.svg").exists(), "gallery index must be written"
    print("\nchecks pass: every world serializes, validates, and round-trips losslessly.")


if __name__ == "__main__":
    main()
