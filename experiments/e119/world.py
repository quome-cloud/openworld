"""Emit a solved ARC game as an openworld.World (CLAUDE.md: build solvers as OpenWorld)."""
import openworld as O


def action_name(act):
    if act[0] == 6:
        return f"click_{act[1]}_{act[2]}"
    return f"a{act[0]}"


def solver_world(game_name, chain):
    """Materialize the solved path as a World: masked-frame key = state; the learned
    (key, action_name) -> (next_key, levels) table = FunctionTransition dynamics."""
    table = {(t["key"], action_name(t["action"])): (t["next_key"], t["levels"]) for t in chain}
    actions = sorted({action_name(t["action"]) for t in chain})
    start_key = chain[0]["key"] if chain else ""

    def fn(state, action):
        nxt = table.get((state.get("key"), action.get("name")))
        return dict(state) if nxt is None else {"key": nxt[0], "levels": nxt[1]}

    return O.World(
        name=f"arc_{game_name}",
        description=f"Learned state-graph solver for ARC-AGI-3 game {game_name}.",
        initial_state={"key": start_key, "levels": 0},
        actions=actions,
        rules=[f"Masked-frame key = state; raising 'levels' wins. {len(chain)} learned transitions."],
        transition=O.FunctionTransition(fn),
    )
