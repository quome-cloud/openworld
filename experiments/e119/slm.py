"""SLM proposer: per-family decoding, a tiny predicate DSL with an executable grader, abstaining subgoal."""
import json, re
import numpy as np
from e119 import abstain

# Per-family decoding (spec §5): pin every family; Gemma differs sharply from Qwen.
_FAMILY = [
    ("qwen3", {"temperature": 0.6, "top_p": 0.95, "top_k": 20}),   # thinking default; see thinking flag
    ("qwen",  {"temperature": 0.7, "top_p": 0.8,  "top_k": 20, "repeat_penalty": 1.05}),
    ("gemma", {"temperature": 1.0, "top_p": 0.95, "top_k": 64}),
    ("llama", {"temperature": 0.6, "top_p": 0.9,  "top_k": 40}),
    ("phi",   {"temperature": 0.7, "top_p": 0.9,  "top_k": 40}),
]


def llm_options(model, thinking=False):
    name = model.lower()
    for key, opts in _FAMILY:
        if key in name:
            o = dict(opts)
            if key == "qwen3" and not thinking:
                o.update({"temperature": 0.7, "top_p": 0.8})       # non-thinking Qwen3
            o["num_predict"] = 4096 if thinking else 1024          # generous; thinking needs room
            return o
    return {"temperature": 0.7, "top_p": 0.9, "top_k": 40, "num_predict": 1024}


def compile_predicate(pred):
    t = pred.get("type")
    if t == "reach":
        c = pred["color"]
        return lambda f: bool((np.asarray(f).reshape(64, 64) == c).any())
    if t == "count":
        c, k, op = pred["color"], pred["k"], pred.get("op", "==")
        def fn(f):
            n = int((np.asarray(f).reshape(64, 64) == c).sum())
            return {"==": n == k, ">=": n >= k, "<=": n <= k}.get(op, False)
        return fn
    if t == "align":
        import arc3_graph
        a, b = pred["a"], pred["b"]
        def fn(f):
            objs, _ = arc3_graph.objects(np.asarray(f).reshape(64, 64))
            ca = [o for o in objs if o["color"] == a]
            cb = [o for o in objs if o["color"] == b]
            return bool(ca and cb and round(ca[0]["centroid"][1]) == round(cb[0]["centroid"][1]))
        return fn
    return lambda f: False


def satisfiable(pred, frames):
    fn = compile_predicate(pred)
    return any(fn(f) for f in frames)


def _canon(pred):
    return tuple(sorted((k, str(v)) for k, v in pred.items()))


def _parse(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no json")
    return json.loads(m.group(0))


_PROMPT = (
    "You pick the GOAL of one ARC level as a JSON predicate. Objects (relational):\n{oj}\n"
    'Allowed: {{"type":"reach","color":N}} | {{"type":"count","color":N,"op":"==|>=|<=","k":N}} '
    '| {{"type":"align","a":N,"b":N}}. Output ONLY the JSON.'
)


def propose_subgoal(llm, obj_json, frames, n=6, tau=0.5):
    prompt = _PROMPT.format(oj=json.dumps(obj_json)[:1500])

    def sample():
        return _parse(llm.ask(prompt))

    def behavior(pred):
        # behavior signature = (is it satisfiable on observed frames, canonical predicate)
        return (satisfiable(pred, frames), _canon(pred))

    winner, _meta = abstain.best_of_n(sample, behavior, n=n, tau=tau)
    return winner
