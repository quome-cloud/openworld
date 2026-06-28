"""Assemble a synthesized object-state predict() + goal energy into an openworld.World.

FunctionTransition-style semantics over object state (carrying level_up), a CodePerceptor
(frame -> object state) and a CodeObjective (goal energy) attached for to_spec.
to_spec(world).preview.graph is the MAP; render_card the atlas; serve /view the UI.
round_trip_ok checks lossless serialization -- from_spec(to_spec(w)) reproduces the rollout
under a fixed action cycle. Structural integrity, independent of real-data fidelity.

API adaptations vs task-3-brief (see task-3-report.md for full details):
- _ObjTransition(CodeTransition) instead of FunctionTransition: serializes as kind="code"
  (lossless round-trip), AND accepts plain dict states/actions (required by test API).
- Embeds predict_src in the transition code with sandbox-safe _parse_act helper (no eval).
- CodePerceptor: produces=["objects"] (List[str] not dict), modality="image" (in MODALITIES).
- round_trip_ok: uses Action objects for both w and w2 (w2 has standard CodeTransition.step).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import openworld as O
from openworld import CodeObjective, CodePerceptor
from openworld.transition import CodeTransition
from openworld.state import WorldState, Action as _Action
from openworld.sandbox import run_transition_code
from e125 import objstate


# ---------------------------------------------------------------------------
# Sandbox-safe action parser + self-contained transition wrapper.
# Appended to embedded predict_src so the whole thing compiles in one exec().
# ---------------------------------------------------------------------------
_TRANS_SUFFIX = '''

def _parse_act(s):
    """Parse "[4]" -> [4], "[1,2]" -> [1,2]; fallback wraps s in a list."""
    s = str(s).strip()
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1].strip()
        return [int(x.strip()) for x in inner.split(",")] if inner else []
    return [s]


def transition(state, action):
    nm = action.get("name", "") if isinstance(action, dict) else str(action)
    act = _parse_act(nm)
    s = {"bg": state.get("bg", 0),
         "objects": [dict(o) for o in state.get("objects", [])]}
    ns, lu = predict(s, act)
    return {"bg": ns["bg"],
            "objects": [dict(o) for o in ns["objects"]],
            "level_up": bool(lu)}
'''


def _make_transition_code(predict_src: str) -> str:
    """Embed predict_src + sandbox-safe wrapper into a single transition code string."""
    return predict_src.rstrip() + "\n" + _TRANS_SUFFIX


class _ObjTransition(CodeTransition):
    """CodeTransition that also accepts plain-dict state/action inputs.

    Standard CodeTransition.step calls action.to_dict() which fails for plain
    dicts. This override normalises both state and action before delegating to
    the sandbox runner, making tests that pass {'name': '[4]'} directly work
    while keeping serialisation identical to CodeTransition (kind='code').
    """

    def step(self, state, action):
        sd = dict(state.copy() if hasattr(state, "copy") else state)
        ad = action.to_dict() if hasattr(action, "to_dict") else dict(action)
        result = run_transition_code(self.code, sd, ad, self.func_name)
        return WorldState(result)


def build_world(predict_src: str, goal_src: str, initial_state: dict,
                actions, game: str) -> O.World:
    """Build an openworld.World from a synthesized predict() + goal energy function.

    Args:
        predict_src: source for ``predict(state, action) -> (next_state, level_up)``
                     where state is ``{bg, objects}`` and action is a list like ``[4]``.
        goal_src:    source for ``reward(state, action, next_state) -> float``
        initial_state: dict with at least ``{bg, objects}``; ``level_up=False`` added.
        actions:     iterable of actions; stored as str representations on the World.
        game:        game identifier (used in world name/description).

    Returns:
        An openworld.World whose transition is an _ObjTransition (CodeTransition
        subclass) with the predict logic embedded, a CodePerceptor on
        ``world.perceptors``, and a CodeObjective on ``world.objectives``.
    """
    code = _make_transition_code(predict_src)
    transition = _ObjTransition(code, func_name="transition")

    w = O.World(
        name=f"e125-{game}",
        description=f"E125 object-state world model for {game}",
        initial_state={**initial_state, "level_up": False},
        actions=[str(a) for a in actions],
        transition=transition,
    )
    # Attach perception boundary (frame -> object state) for to_spec.
    # modality="image" keeps it within MODALITIES; produces=["objects"] is List[str].
    w.perceptors = [CodePerceptor(
        objstate.PERCEIVE_SRC,
        produces=["objects"],
        modality="image",
    )]
    # Attach goal energy for to_spec (serialised as an objectives descriptor).
    w.objectives = [CodeObjective(goal_src, name="goal_energy")]
    return w


def round_trip_ok(w: O.World, steps: int = 8) -> bool:
    """Return True iff from_spec(to_spec(w), allow_code=True) reproduces w's rollout.

    Uses Action objects (not plain dicts) so that the reconstructed standard
    CodeTransition.step (which calls action.to_dict()) works correctly.
    Compares the decision-relevant state_key and level_up flag at each step.
    """
    spec = O.to_spec(w)
    w2 = O.from_spec(spec, allow_code=True)
    if w2 is None:
        return False
    acts = w.actions
    s1: WorldState = WorldState(dict(w.initial_state))
    s2: WorldState = WorldState(dict(w2.initial_state))
    for i in range(steps):
        act = _Action(str(acts[i % len(acts)]))
        s1 = w.transition.step(s1, act)
        s2 = w2.transition.step(s2, act)
        if (objstate.state_key(dict(s1)) != objstate.state_key(dict(s2))
                or bool(s1.get("level_up")) != bool(s2.get("level_up"))):
            return False
    return True
