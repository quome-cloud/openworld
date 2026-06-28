import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
import numpy as np
from e125 import synth

F0 = np.zeros((64,64), dtype=int); F1 = F0.copy(); F1[0,0]=1; F2=F1.copy(); F2[0,0]=2
def _t(f,a,nf,lu): return {"frame":f,"action":a,"next_frame":nf,"level_up":lu}
TRANS = [_t(F0,[1],F1,False), _t(F1,[1],F2,False)]
GOOD = "def predict(frame, action):\n    nf=frame.copy(); nf[0,0]=frame[0,0]+1\n    return nf, bool(nf[0,0]==3)"
GOAL = "def goal_score(frame):\n    return float(3 - frame[0,0])"

def _runner_giving(src, goal=None):
    def run(prompt, schema, model, game, **kw):
        final = {"predict_src": src, "rationale": "x"}
        if goal is not None:
            final["goal_score_src"] = goal
        return {"final": final, "events": [], "tainted": False,
                "raw": "", "model_version": "gpt-5.5-test"}
    return run

def test_synthesize_accepts_passing_model(tmp_path):
    src, fn, goal = synth.synthesize(TRANS, "actions=[1]", "g", mask=None, n_retries=1,
                                     traces_dir=str(tmp_path), _runner=_runner_giving(GOOD))
    assert fn is not None and src is not None
    nf, lu = fn(F0, [1]); assert nf[0,0] == 1

def test_synthesize_returns_none_when_model_never_passes(tmp_path):
    bad = "def predict(frame, action):\n    return frame.copy(), False"
    src, fn, goal = synth.synthesize(TRANS, "actions=[1]", "g", mask=None, n_retries=2,
                                     traces_dir=str(tmp_path), _runner=_runner_giving(bad))
    assert fn is None

def test_synthesize_writes_telemetry(tmp_path):
    synth.synthesize(TRANS, "a", "g", mask=None, n_retries=1, traces_dir=str(tmp_path),
                     _runner=_runner_giving(GOOD))
    assert os.path.exists(tmp_path/"calls.jsonl")


def test_synthesize_evolves_from_bad_to_good(tmp_path):
    """FunSearch-style: a stateful mock returns a BAD predict first, a GOOD one on the next mutation;
    synthesize must keep evolving and accept the GOOD model (one-shot would have failed on the BAD first try)."""
    bad = "def predict(frame, action):\n    return frame.copy(), False"
    calls = {"n": 0}
    def run(prompt, schema, model, game, **kw):
        calls["n"] += 1
        src = bad if calls["n"] == 1 else GOOD
        return {"final": {"predict_src": src, "rationale": ""}, "events": [], "tainted": False,
                "raw": "", "model_version": ""}
    src, fn, goal = synth.synthesize(TRANS, "actions=[1]", "g", mask=None, n_retries=4,
                                     traces_dir=str(tmp_path), _runner=run)
    assert fn is not None and calls["n"] >= 2          # it had to evolve past the bad first attempt
    nf, lu = fn(F0, [1]); assert nf[0, 0] == 1


def test_synthesize_reuses_full_seed_without_codex(tmp_path):
    """A carried-forward seed that still verifies on the (possibly grown) held set is reused with NO codex call
    -- within a level we don't re-climb from scratch each round."""
    calls = {"n": 0}
    def run(*a, **k):
        calls["n"] += 1
        return {"final": {}, "events": [], "tainted": False, "raw": "", "model_version": ""}
    src, fn, goal = synth.synthesize(TRANS, "actions=[1]", "g", mask=None, n_retries=3,
                                     traces_dir=str(tmp_path), _runner=run, seed_src=GOOD)
    assert fn is not None and calls["n"] == 0
    nf, lu = fn(F0, [1]); assert nf[0, 0] == 1

def test_synthesize_seeds_kshot_from_partial_seed(tmp_path):
    """A partial seed (not full) seeds the database, so the FIRST prompt is a k-shot improving the carried
    program (predict_v0) rather than a cold cold-start prompt."""
    partial = "def predict(frame, action):\n    return frame.copy(), False"
    prompts = []
    def run(prompt, schema, model, game, **kw):
        prompts.append(prompt)
        return {"final": {"predict_src": GOOD, "goal_score_src": GOAL, "rationale": ""}, "events": [],
                "tainted": False, "raw": "", "model_version": ""}
    src, fn, goal = synth.synthesize(TRANS, "actions=[1]", "g", mask=None, n_retries=2,
                                     traces_dir=str(tmp_path), _runner=run, seed_src=partial)
    assert fn is not None and "predict_v0" in prompts[0]

def test_synthesize_persists_verified_program(tmp_path):
    """On a gate-pass the accepted predict()+goal_score() are written to traces_dir for offline plan debugging."""
    import glob
    src, fn, goal = synth.synthesize(TRANS, "actions=[1]", "gx", mask=None, n_retries=1,
                                     traces_dir=str(tmp_path), _runner=_runner_giving(GOOD, GOAL))
    assert fn is not None
    files = glob.glob(str(tmp_path / "*verified*"))
    assert files
    content = open(files[0]).read()
    assert "def predict" in content and "def goal_score" in content

def test_synthesize_returns_goal_fn_when_provided(tmp_path):
    """Codex emits a goal_score energy alongside predict(); synthesize compiles + returns it for planning."""
    src, fn, goal = synth.synthesize(TRANS, "actions=[1]", "g", mask=None, n_retries=1,
                                     traces_dir=str(tmp_path), _runner=_runner_giving(GOOD, GOAL))
    assert callable(goal)
    assert goal(F0) == 3.0 and goal(F2) == 1.0          # energy descends toward the hypothesized goal

def test_synthesize_goal_none_when_absent(tmp_path):
    """No goal_score_src in the proposal -> goal_fn is None (planning falls back to BFS), predict still returned."""
    src, fn, goal = synth.synthesize(TRANS, "actions=[1]", "g", mask=None, n_retries=1,
                                     traces_dir=str(tmp_path), _runner=_runner_giving(GOOD))
    assert fn is not None and goal is None

def test_schema_requires_goal_score_src():
    """The codex output schema is OpenAI-strict (all-required + no extra props) and includes goal_score_src."""
    assert synth.SCHEMA["additionalProperties"] is False
    assert "goal_score_src" in synth.SCHEMA["required"]
    assert "goal_score_src" in synth.SCHEMA["properties"]


# ---------------- faithful FunSearch: database + clusters + k-shot ascending versioned prompt ----------------

def test_score_program_returns_per_test_signature():
    """score_program gives the per-test pass/fail signature FunSearch clusters on (plus n_matched + fails)."""
    bad = synth.verify.compile_predict("def predict(frame, action):\n    return frame.copy(), False")
    n, sig, fails = synth.score_program(synth.verify.compile_predict(GOOD), TRANS, mask=None)
    assert n == 2 and sig == (True, True) and fails == []
    n, sig, fails = synth.score_program(bad, TRANS, mask=None)
    assert n == 0 and sig == (False, False) and len(fails) == 2

def test_rename_fn_versions_the_def():
    out = synth._rename_fn("def predict(frame, action):\n    return frame, False", "predict", "predict_v0")
    assert "def predict_v0(frame, action):" in out and "def predict(" not in out

def test_funsearch_prompt_orders_ascending_and_asks_next_version():
    lo = {"src": "def predict(f, a):\n    return f, False", "score": 2, "fails": []}
    hi = {"src": "def predict(f, a):\n    return f, True",  "score": 5, "fails": []}
    p = synth._funsearch_prompt([hi, lo], "actions=[1]", None, None)   # pass unsorted on purpose
    assert "predict_v0" in p and "predict_v1" in p and "predict_v2" in p   # k-shot + asks for next version
    assert p.index("score 2") < p.index("score 5")                        # rendered ascending by score

def test_database_tracks_best_and_clusters_by_signature():
    db = synth._Database(functions_per_prompt=2, rng=np.random.RandomState(0))
    db.register("a", None, 1, (True, False, False), [], None)
    db.register("bb", None, 2, (True, True, False), [], None)
    db.register("c", None, 2, (True, True, False), [], None)   # same signature as bb -> same cluster
    assert db.best["score"] == 2 and len(db.clusters) == 2

def test_database_sample_returns_programs_ascending_by_score():
    db = synth._Database(functions_per_prompt=2, rng=np.random.RandomState(0))
    db.register("a", None, 1, (True, False), [], None)
    db.register("b", None, 2, (True, True), [], None)
    s = db.sample()
    assert [p["score"] for p in s] == [1, 2]                   # ascending trajectory for the prompt

def test_database_remembers_failed_attempts_excluding_best():
    """Non-improving attempts go into a failure memory (with their rationale + score); the best does not."""
    db = synth._Database(rng=np.random.RandomState(0))
    db.register("a", None, 5, (True, True, True, True, True), [], None, rationale="approach A")   # best
    db.register("b", None, 2, (True, True, False, False, False), [], None, rationale="approach B") # worse
    db.register("c", None, 0, (False,)*5, [], None, rationale="approach C")                         # broke
    fs = db.failed_summaries()
    joined = " | ".join(fs)
    assert "approach B" in joined and "approach C" in joined
    assert "approach A" not in joined            # the best is never in the do-not-repeat memory

def test_funsearch_prompt_lists_known_failed_approaches():
    p = synth._funsearch_prompt([{"src": "def predict(f,a):\n    return f, False", "score": 3, "fails": []}],
                                "actions=[1]", None, None, failed=["tried X -> 2/18", "tried Y -> 0/18"])
    assert "tried X -> 2/18" in p and "tried Y -> 0/18" in p
    assert "do not repeat" in p.lower()

def test_synthesize_threads_failed_rationale_into_later_prompt(tmp_path):
    """A worse attempt's rationale enters the failure memory and appears in a subsequent prompt (anti-repeat)."""
    partial = "def predict(frame, action):\n    return frame.copy(), False"   # scores 0 -> a failure
    prompts = []; calls = {"n": 0}
    seq = [(partial, "bad approach Z"), (partial, "bad approach Z2"), (GOOD, "good approach")]
    def run(prompt, schema, model, game, **kw):
        prompts.append(prompt)
        src, rat = seq[min(calls["n"], len(seq)-1)]; calls["n"] += 1
        return {"final": {"predict_src": src, "goal_score_src": GOAL, "rationale": rat}, "events": [],
                "tainted": False, "raw": "", "model_version": ""}
    synth.synthesize(TRANS, "actions=[1]", "g", mask=None, n_retries=3, traces_dir=str(tmp_path), _runner=run)
    assert any("bad approach Z" in p for p in prompts[1:])   # the failed approach is remembered & shown

def test_synthesize_uses_multishot_referencing_prior_versions(tmp_path):
    """After the seed program is registered, later prompts are FunSearch k-shot prompts citing predict_v0."""
    partial = "def predict(frame, action):\n    return frame.copy(), False"   # scores 0 -> not accepted
    prompts = []; calls = {"n": 0}
    def run(prompt, schema, model, game, **kw):
        prompts.append(prompt)
        src = partial if calls["n"] == 0 else GOOD
        calls["n"] += 1
        return {"final": {"predict_src": src, "goal_score_src": GOAL, "rationale": ""}, "events": [],
                "tainted": False, "raw": "", "model_version": ""}
    src, fn, goal = synth.synthesize(TRANS, "actions=[1]", "g", mask=None, n_retries=4,
                                     traces_dir=str(tmp_path), _runner=run)
    assert fn is not None                                       # evolved to the GOOD program
    assert any("predict_v0" in p for p in prompts[1:])          # 2nd+ prompt is a k-shot FunSearch prompt


def test_score_predict_counts_and_collects_fails():
    good = synth.verify.compile_predict(GOOD)
    bad = synth.verify.compile_predict("def predict(frame, action):\n    return frame.copy(), False")
    m, fails = synth.score_predict(good, TRANS, mask=None); assert m == 2 and fails == []
    m, fails = synth.score_predict(bad, TRANS, mask=None); assert m == 0 and len(fails) == 2
