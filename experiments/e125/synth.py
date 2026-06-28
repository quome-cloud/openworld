"""Codex synthesizes predict(frame,action)->(next_frame,level_up), accepted ONLY via verify.check on a
held-out split. FunSearch-style EVOLUTION: keep the best program and mutate it by showing codex the EXACT
cells it mispredicts (a far stronger signal than a bare counterexample) -- reliably reaching a gate-pass where
one-shot synthesis is a coin-flip. Source-free + telemetry-captured. Codex is a proposal engine inside the
verifier loop -- never an authority."""
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


def score_predict(predict_fn, transitions, mask):
    """Return (n_matched, fails) where n_matched counts exact (masked next-frame + level_up) reproductions and
    fails is a list of (transition, predicted_next_frame|None) for the misses -- a continuous gate signal."""
    if predict_fn is None:
        return 0, [(t, None) for t in transitions]
    matched, fails = 0, []
    for t in transitions:
        try:
            nf, lu = predict_fn(np.asarray(t["frame"]), list(t["action"]))
        except Exception:
            fails.append((t, None)); continue
        if (verify._masked(nf, mask) == verify._masked(t["next_frame"], mask)).all() and bool(lu) == bool(t["level_up"]):
            matched += 1
        else:
            fails.append((t, nf))
    return matched, fails


def _mut_prompt(src, fail, action_api, mask):
    """Mutation prompt: show codex its current predict() and the EXACT cells it mispredicts on a failing
    transition (far stronger signal than a bare counterexample), asking it to fix that while keeping the rest."""
    t, nf = fail
    if nf is None:
        diff = "it raised an exception / failed to compile on this input"
    else:
        mp, mr = verify._masked(nf, mask), verify._masked(t["next_frame"], mask)
        ys, xs = np.where(mp != mr)
        cells = "; ".join(f"({int(y)},{int(x)}) you={int(mp[y, x])} real={int(mr[y, x])}"
                          for y, x in list(zip(ys, xs))[:20])
        diff = f"on action={t['action']} it mispredicts these masked cells: {cells or '(the level_up flag is wrong)'}"
    return (f"Your predict() is CLOSE but not exact. Here it is:\n```python\n{src}\n```\n"
            f"{diff}.\nFIX predict() to reproduce that transition exactly while keeping everything it already "
            f"gets right (numpy as np only, no imports/IO). Actions: {action_api}. Return JSON {{predict_src, rationale}}.")


def synthesize(transitions, action_api, game, mask, model="gpt-5.5", n_retries=4, traces_dir=None, _runner=None):
    """Evolve a predict() that exact-matches every held-out transition (FunSearch-style: keep the best program,
    mutate it by showing codex the exact cells it mispredicts). The verifier gate is the exact evaluator.
    Returns (src, fn) on a full gate-pass, else (None, None) -- only a fully-verified model is accepted."""
    run = _runner or codex_iso.run
    if len(transitions) < 2:
        return None, None                      # cannot form a disjoint held-out set
    split = max(1, min(len(transitions) - 1, int(len(transitions) * 0.7)))
    train, held = transitions[:split], transitions[split:]
    best_src = best_fn = best_fails = None; best_score = -1
    for attempt in range(n_retries):
        prompt = (_prompt(train, action_api, mask, None) if best_src is None
                  else _mut_prompt(best_src, best_fails[0], action_api, mask))
        res = run(prompt, SCHEMA, model, game)
        final = res.get("final") or {}
        src = final.get("predict_src")
        tainted = bool(res.get("tainted"))
        fn = None if tainted else verify.compile_predict(src or "")
        sc, fails = score_predict(fn, held, mask)
        if sc > best_score:
            best_src, best_fn, best_score, best_fails = src, fn, sc, fails
        full = (best_score == len(held))
        if traces_dir:
            capture_lib.codex_record(traces_dir, {"game": game, "level": 0, "regime": attempt, "model": model,
                "model_version": res.get("model_version", ""), "prompt": prompt, "raw": res.get("raw", ""),
                "events": res.get("events", []), "parsed": {"subgoals": [], "macros": []},
                "decision": ("accept" if full else f"evolve {sc}/{len(held)}"), "tainted": tainted})
        if full:
            return best_src, best_fn
    return None, None
