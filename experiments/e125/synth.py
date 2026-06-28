"""Codex synthesizes predict(frame,action)->(next_frame,level_up), accepted ONLY via the verifier gate
(verify.check) on a held-out split. The search layer is a faithful FunSearch (google-deepmind/funsearch):
a programs DATABASE clusters proposals by score-signature; each new prompt samples prior programs by
Boltzmann(score) (and prefers shorter ones), renders them ASCENDING BY SCORE as predict_v0/v1, and asks for
a better predict_v2 -- the cross-program improvement trajectory is the engine. Two domain augmentations on top
of vanilla FunSearch: the exact mispredicted cells of the best program, and an anti-repeat MEMORY of approaches
already tried that failed. The algorithm layer is swappable (FunSearch today; AlphaEvolve / DEAP-GP / PySR are
candidate alternatives) -- the harness owns the evaluator (the gate) and acceptance, the searcher only
proposes. Source-free + telemetry-captured. Codex is a proposal engine inside the verifier loop, never an
authority."""
import os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "scripts"))
import numpy as np
from e125 import verify
from e124 import codex_iso
import capture_lib

SCHEMA = {"type": "object", "additionalProperties": False,
          "required": ["predict_src", "goal_score_src", "rationale"],
          "properties": {"predict_src": {"type": "string"}, "goal_score_src": {"type": "string"},
                         "rationale": {"type": "string"}}}


def _grid(frame, mask):
    fr = verify._masked(frame, mask)
    return "\n".join("".join(f"{int(c):x}" for c in row) for row in np.asarray(fr).reshape(64, 64))


def render_transitions(transitions, mask, k=12):
    out = []
    for t in transitions[:k]:
        out.append(f"action={t['action']} level_up={bool(t['level_up'])}\nFROM:\n{_grid(t['frame'],mask)}\n"
                   f"TO:\n{_grid(t['next_frame'],mask)}")
    return "\n---\n".join(out)


_GOAL_INSTR = (
    "\n\nIMPORTANT -- the win condition is NOT given to you and the observed transitions show NO level-up yet. "
    "You must HYPOTHESISE the goal yourself from the visual structure of the frames (what configuration the "
    "game is plausibly asking the player to reach -- e.g. a movable object reaching a target marker, a region "
    "filled, a pattern matched). Then:\n"
    "1. Bake that hypothesis into predict()'s `level_up`: return level_up=True exactly when the next_frame "
    "satisfies your hypothesised win configuration (it must still be False on every observed transition above, "
    "since none of them won).\n"
    "2. Also write `goal_score(frame) -> float`: a SYMBOLIC energy that is LOWER the closer `frame` is to your "
    "hypothesised goal and 0 (or minimal) at the goal (e.g. Manhattan distance of the movable object to the "
    "target). A planner will DESCEND this energy, so it must vary smoothly with progress -- not be flat. "
    "numpy as np only, no imports/IO.")


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
    return base + _GOAL_INSTR + "\n\nReturn JSON {predict_src, goal_score_src, rationale}."


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


def score_program(predict_fn, transitions, mask):
    """Like score_predict but also returns the per-test pass/fail SIGNATURE FunSearch clusters on (programs with
    the same signature solve the same subset). Returns (n_matched, signature: tuple[bool], fails)."""
    n, fails = score_predict(predict_fn, transitions, mask)
    failed_ids = {id(t) for (t, _) in fails}
    signature = tuple(id(t) not in failed_ids for t in transitions)
    return n, signature, fails


# --- FunSearch program database: clusters by score-signature, Boltzmann cluster sampling, length preference,
#     and a k-shot prompt that shows prior programs ascending by score (predict_v0, predict_v1) asking for the
#     next, better version (predict_v2). The cross-program improvement TRAJECTORY is the engine of FunSearch;
#     we add the exact mispredicted cells of the best-shown program as a domain-specific feedback augmentation.


def _softmax(logits, temperature):
    a = np.asarray(logits, dtype=float)
    a = (a - a.max()) / max(temperature, 1e-6)
    e = np.exp(a)
    return e / e.sum()


def _rename_fn(src, old, new):
    """Rename `def old(` -> `def new(` for display as a versioned program."""
    return re.sub(rf"\bdef\s+{re.escape(old)}\s*\(", f"def {new}(", src or "")


def _cells_diff(fails, mask, k=3, per_t=20):
    """Human-readable list of the exact masked cells the program still mispredicts (domain feedback)."""
    out = []
    for t, nf in fails[:k]:
        if nf is None:
            out.append(f"action={t['action']}: raised/failed to compile"); continue
        mp, mr = verify._masked(nf, mask), verify._masked(t["next_frame"], mask)
        ys, xs = np.where(mp != mr)
        cells = "; ".join(f"({int(y)},{int(x)}) you={int(mp[y, x])} real={int(mr[y, x])}"
                          for y, x in list(zip(ys, xs))[:per_t])
        out.append(f"action={t['action']}: {cells or '(level_up flag wrong)'}")
    return "\n".join(out)


def _funsearch_prompt(samples, action_api, mask, goal_src, failed=None):
    """k-shot FunSearch prompt: prior programs rendered ascending by score as predict_v0..v{k-1}; ask for an
    improved predict_v{k}. Includes the exact cells the best-shown program mispredicts AND a memory of
    already-tried approaches that FAILED, so the model does not re-walk known-bad trajectories."""
    progs = sorted(samples, key=lambda p: p["score"])
    blocks = []
    for i, p in enumerate(progs):
        blocks.append(f"# predict_v{i} (score {p['score']})\n```python\n{_rename_fn(p['src'], 'predict', f'predict_v{i}')}\n```")
    nextv = len(progs)
    best = progs[-1]
    diff = _cells_diff(best.get("fails") or [], mask)
    goal_block = f"\nCurrent goal_score():\n```python\n{goal_src}\n```\n" if goal_src else ""
    fail_block = ""
    if failed:
        fail_block = ("\n\nAlready tried and FAILED -- do not repeat these approaches (each made things worse or "
                      "no better):\n" + "\n".join(f"- {f}" for f in failed) + "\n")
    return ("These are successive versions of a predict() world model, ordered by increasing score (how many "
            f"observed transitions each reproduces exactly):\n\n" + "\n\n".join(blocks) +
            f"\n\nWrite an IMPROVED `predict_v{nextv}` that scores HIGHER than all above -- continue the trend. "
            f"predict_v{nextv-1} still mispredicts these masked cells:\n{diff}\n{goal_block}{fail_block}"
            f"Name the function `predict` (numpy as np only, no imports/IO). Keep (or improve) the win-condition "
            f"hypothesis in level_up and the goal_score energy. Actions: {action_api}. "
            f"Return JSON {{predict_src, goal_score_src, rationale}}.")


class _Database:
    """A FunSearch programs database (single island): programs grouped into clusters by score-signature; a new
    prompt samples functions_per_prompt clusters by Boltzmann(score) and one program per cluster by
    Boltzmann(-length), so shorter & higher-scoring programs are favored. Tracks the best program seen."""

    def __init__(self, functions_per_prompt=2, cluster_temp=0.1, rng=None):
        self.fpp = functions_per_prompt
        self.cluster_temp = cluster_temp
        self.rng = rng if rng is not None else np.random.RandomState(0)
        self.clusters = {}          # signature -> {"score": int, "progs": [prog]}
        self.best = None
        self.failures = []          # {"rationale","score"} for non-improving attempts (anti-repeat memory)

    def register(self, src, fn, score, signature, fails, goal_src, rationale=""):
        prog = {"src": src, "fn": fn, "score": score, "fails": fails, "goal_src": goal_src,
                "len": len(src or "")}
        improved = self.best is None or score > self.best["score"]
        if improved:
            self.best = prog
        elif rationale:
            self.failures.append({"rationale": rationale.strip(), "score": score})
        cl = self.clusters.setdefault(tuple(signature), {"score": score, "progs": []})
        cl["progs"].append(prog)

    def failed_summaries(self, limit=6):
        """Most-recent distinct failed approaches as 'rationale -> scored N' lines for the anti-repeat block."""
        seen, out = set(), []
        for f in reversed(self.failures):
            r = f["rationale"]
            if r and r not in seen:
                seen.add(r); out.append(f"{r} -> scored {f['score']}")
            if len(out) >= limit:
                break
        return list(reversed(out))

    def sample(self):
        """Return up to fpp prior programs (one per sampled cluster) for the next k-shot prompt."""
        sigs = list(self.clusters)
        if not sigs:
            return []
        scores = np.array([self.clusters[s]["score"] for s in sigs], dtype=float)
        k = min(self.fpp, len(sigs))
        probs = _softmax(scores, self.cluster_temp)
        chosen = self.rng.choice(len(sigs), size=k, replace=False, p=probs)
        picks = []
        for ci in chosen:
            progs = self.clusters[sigs[ci]]["progs"]
            lens = np.array([p["len"] for p in progs], dtype=float)
            span = lens.max() - lens.min()
            norm = (lens - lens.min()) / span if span > 0 else np.zeros_like(lens)
            lp = _softmax(-norm, 1.0)
            picks.append(progs[self.rng.choice(len(progs), p=lp)])
        return sorted(picks, key=lambda p: p["score"])


def synthesize(transitions, action_api, game, mask, model="gpt-5.5", n_retries=4, traces_dir=None, _runner=None,
               functions_per_prompt=2, seed=0, seed_src=None):
    """FunSearch over predict() world models: codex proposes programs; the verifier gate (exact masked-frame +
    level_up match on held-out transitions) is the evaluator; programs enter a database clustered by
    score-signature; each new prompt shows prior programs ASCENDING BY SCORE as predict_v0/v1 and asks for a
    better predict_v2 (the improvement trajectory), plus the exact cells the best still mispredicts and a memory
    of failed approaches. An autonomous win hypothesis is baked into level_up and a symbolic goal_score(frame)
    energy rides along. `seed_src` carries a prior round's verified program forward WITHIN a level: it is
    registered first so the search extends it instead of re-climbing from scratch (reset is per-LEVEL, not
    per-round). Returns (src, predict_fn, goal_fn) on a full gate-pass, else (None, None, None)."""
    run = _runner or codex_iso.run
    if len(transitions) < 2:
        return None, None, None                # cannot form a disjoint held-out set
    split = max(1, min(len(transitions) - 1, int(len(transitions) * 0.7)))
    train, held = transitions[:split], transitions[split:]
    db = _Database(functions_per_prompt=functions_per_prompt, rng=np.random.RandomState(seed))

    def _accept(prog):
        goal_fn = verify.compile_goal(prog["goal_src"]) if prog.get("goal_src") else None
        if traces_dir:                                  # persist for offline plan/goal debugging (no re-spend)
            try:
                with open(os.path.join(traces_dir, f"{game}_verified.py"), "w") as fh:
                    fh.write(f"# E125 verified predict()+goal_score() for {game}\n{prog['src'] or ''}\n\n{prog.get('goal_src') or ''}\n")
            except Exception:
                pass
        return prog["src"], prog["fn"], goal_fn

    if seed_src:                                        # carry-forward: seed the database with the prior program
        sfn = verify.compile_predict(seed_src)
        if sfn is not None:
            sc, sig, fails = score_program(sfn, held, mask)
            db.register(seed_src, sfn, sc, sig, fails, None, rationale="carried-forward best")
            if sc == len(held):                         # still verifies on the (grown) held set -> reuse, no codex
                return _accept(db.best)

    for attempt in range(n_retries):
        samples = db.sample()
        prompt = (_prompt(train, action_api, mask, None) if not samples
                  else _funsearch_prompt(samples, action_api, mask, db.best["goal_src"],
                                         failed=db.failed_summaries()))
        res = run(prompt, SCHEMA, model, game)
        final = res.get("final") or {}
        src = final.get("predict_src")
        goal_src = final.get("goal_score_src")
        rationale = final.get("rationale") or ""
        tainted = bool(res.get("tainted"))
        fn = None if tainted else verify.compile_predict(src or "")
        sc, sig, fails = score_program(fn, held, mask)
        if src and not tainted:
            db.register(src, fn, sc, sig, fails, goal_src, rationale=rationale)
        best_full = db.best is not None and db.best["score"] == len(held) and db.best["fn"] is not None
        if traces_dir:
            capture_lib.codex_record(traces_dir, {"game": game, "level": 0, "regime": attempt, "model": model,
                "model_version": res.get("model_version", ""), "prompt": prompt, "raw": res.get("raw", ""),
                "events": res.get("events", []), "parsed": {"subgoals": [], "macros": []},
                "decision": ("accept" if best_full else f"evolve {sc}/{len(held)} (best {db.best['score'] if db.best else 0})"),
                "tainted": tainted})
        if best_full:
            return _accept(db.best)
    return None, None, None


# --- object-state synthesis path (predict(state,action)->(next_state,level_up)); reuses the FunSearch
#     _Database/_softmax/_rename_fn/failed_summaries machinery, but renders/scoring is over OBJECT state ---
from e125 import objstate as _objstate_s


def _objs(s):
    return (f"bg={s.get('bg')} objects=["
            + ", ".join(f"(c{o['color']} y{o['y']} x{o['x']} s{o['size']})" for o in s.get("objects", []))
            + "]")


def render_obj_transitions(transitions, k=12):
    out = []
    for t in transitions[:k]:
        out.append(f"action={t['action']} level_up={bool(t['level_up'])}\n"
                   f"FROM: {_objs(t['state'])}\nTO:   {_objs(t['next_state'])}")
    return "\n---\n".join(out)


_OBJ_GOAL_INSTR = (
    "\n\nIMPORTANT -- the win condition is NOT given and no observed transition won. HYPOTHESISE the goal from "
    "the object configurations (e.g. a movable object reaching a target object's position). Bake it into "
    "predict()'s level_up (True only when next_state matches your hypothesised win; it must be False on every "
    "observed transition above). Also write goal_score(state) -> float: a SYMBOLIC energy LOWER nearer the goal "
    "(e.g. Manhattan distance between the mover and the target object), varying smoothly. Operate on the object "
    "dict only (state['objects'] = list of {color,size,y,x}); pure Python, no imports, no numpy.")


def _obj_contract(action_api):
    return (f"You are reverse-engineering an unknown grid game's dynamics from observed OBJECT-state transitions. "
            f"Do NOT run shell commands or read files. Write `predict(state, action) -> (next_state, level_up)` "
            f"in pure Python (NO imports, NO numpy), where `state` is a dict "
            f"{{'bg': int, 'objects': [{{'color','size','y','x'}}, ...]}}, `action` is a list like [4] or [6,x,y], "
            f"`next_state` is the predicted next state dict (same shape), `level_up` a bool. Actions: {action_api}")


def _obj_prompt(transitions, action_api, counterexample=None):
    base = (_obj_contract(action_api) + "\n\nObserved transitions:\n"
            + render_obj_transitions(transitions))
    if counterexample is not None:
        base += (f"\n\nYour previous predict() FAILED on:\naction={counterexample['action']} "
                 f"level_up={bool(counterexample['level_up'])}\nFROM: {_objs(counterexample['state'])}\n"
                 f"TO:   {_objs(counterexample['next_state'])}")
    return base + _OBJ_GOAL_INSTR + "\n\nReturn JSON {predict_src, goal_score_src, rationale}."


def _obj_diff(fails, fields=("color", "y", "x"), k=3):
    out = []
    for t, ns in fails[:k]:
        if ns is None:
            out.append(f"action={t['action']}: predict raised/failed"); continue
        pk = _objstate_s.state_key(ns, fields)[1]
        rk = _objstate_s.state_key(t["next_state"], fields)[1]
        msg = f"you->{pk} real->{rk}" if pk != rk else "(objects match -- the level_up flag is wrong)"
        out.append(f"action={t['action']}: {msg}")
    return "\n".join(out)


def _obj_funsearch_prompt(samples, action_api, failed=None):
    progs = sorted(samples, key=lambda p: p["score"])
    blocks = [f"# predict_v{i} (score {p['score']})\n```python\n{_rename_fn(p['src'], 'predict', f'predict_v{i}')}\n```"
              for i, p in enumerate(progs)]
    nextv = len(progs)
    diff = _obj_diff(progs[-1].get("fails") or [])
    fail_block = ""
    if failed:
        fail_block = ("\n\nAlready tried and FAILED -- do not repeat these approaches:\n"
                      + "\n".join(f"- {f}" for f in failed) + "\n")
    return ("These are successive predict() object-state world models, ordered by increasing score:\n\n"
            + "\n\n".join(blocks)
            + f"\n\nWrite an IMPROVED `predict_v{nextv}` scoring HIGHER than all above. predict_v{nextv-1} "
            f"still mispredicts:\n{diff}\n{fail_block}Name the function `predict` (pure Python, no imports, no numpy). "
            f"Keep/improve the win hypothesis in level_up and the goal_score(state) energy. Actions: {action_api}. "
            f"Return JSON {{predict_src, goal_score_src, rationale}}.")
