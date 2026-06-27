"""Codex synthesizes predict(frame,action)->(next_frame,level_up); accepted ONLY via verify.check on a
held-out split. On a miss, the counterexample is appended and codex re-proposes (bounded retries). Source-free
+ telemetry-captured. Codex is a proposal engine inside the verifier loop -- never an authority."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "scripts"))
import numpy as np
from e125 import verify
from e124 import codex_iso
import capture_lib

SCHEMA = {"type": "object", "additionalProperties": False, "required": ["predict_src", "rationale"],
          "properties": {"predict_src": {"type": "string"}, "rationale": {"type": "string"}}}


def _grid(frame, mask):
    fr = verify._masked(frame, mask)
    return "\n".join("".join(f"{int(c):x}" for c in row) for row in np.asarray(fr).reshape(64, 64))


def render_transitions(transitions, mask, k=12):
    out = []
    for t in transitions[:k]:
        out.append(f"action={t['action']} level_up={bool(t['level_up'])}\nFROM:\n{_grid(t['frame'],mask)}\n"
                   f"TO:\n{_grid(t['next_frame'],mask)}")
    return "\n---\n".join(out)


def _prompt(transitions, action_api, mask, counterexample):
    base = (f"You are reverse-engineering an unknown 64x64 grid game's dynamics from observed transitions. "
            f"Do NOT run shell commands or read files. Write a Python function "
            f"`predict(frame, action) -> (next_frame, level_up)` using numpy as np only (no imports/IO), where "
            f"`frame` is a 64x64 int array, `action` is a list like [1] or [6,x,y], `next_frame` is the "
            f"predicted next 64x64 array, and `level_up` is a bool (did the level advance).\n\nActions: "
            f"{action_api}\n\nObserved transitions (hex grids, status bar masked):\n{render_transitions(transitions, mask)}")
    if counterexample is not None:
        base += (f"\n\nYour previous predict() FAILED on this transition (fix it):\naction="
                 f"{counterexample['action']} level_up={bool(counterexample['level_up'])}\nFROM:\n"
                 f"{_grid(counterexample['frame'],mask)}\nTO:\n{_grid(counterexample['next_frame'],mask)}")
    return base + "\n\nReturn JSON {predict_src, rationale}."


def synthesize(transitions, action_api, game, mask, model="gpt-5.5", n_retries=3, traces_dir=None, _runner=None):
    run = _runner or codex_iso.run
    if len(transitions) < 2:
        return None, None                      # cannot form a disjoint held-out set
    split = max(1, min(len(transitions) - 1, int(len(transitions) * 0.7)))
    train, held = transitions[:split], transitions[split:]
    ce = None
    for attempt in range(n_retries):
        prompt = _prompt(train, action_api, mask, ce)
        res = run(prompt, SCHEMA, model, game)
        final = res.get("final") or {}
        src = final.get("predict_src")
        tainted = bool(res.get("tainted"))
        fn = None if tainted else verify.compile_predict(src or "")
        ok, ce = verify.check(fn, held, mask) if fn else (False, held[0] if held else None)
        if fn is None: ce = None
        if traces_dir:
            capture_lib.codex_record(traces_dir, {"game": game, "level": 0, "regime": attempt, "model": model,
                "model_version": res.get("model_version", ""), "prompt": prompt, "raw": res.get("raw", ""),
                "events": res.get("events", []), "parsed": {"subgoals": [], "macros": []},
                "decision": ("accept" if ok else "reject"), "tainted": tainted})
        if ok:
            return src, fn
    return None, None
