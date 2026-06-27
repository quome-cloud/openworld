import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
import numpy as np
from e125 import synth

F0 = np.zeros((64,64), dtype=int); F1 = F0.copy(); F1[0,0]=1; F2=F1.copy(); F2[0,0]=2
def _t(f,a,nf,lu): return {"frame":f,"action":a,"next_frame":nf,"level_up":lu}
TRANS = [_t(F0,[1],F1,False), _t(F1,[1],F2,False)]
GOOD = "def predict(frame, action):\n    nf=frame.copy(); nf[0,0]=frame[0,0]+1\n    return nf, bool(nf[0,0]==3)"

def _runner_giving(src):
    def run(prompt, schema, model, game, **kw):
        return {"final": {"predict_src": src, "rationale": "x"}, "events": [], "tainted": False,
                "raw": "", "model_version": "gpt-5.5-test"}
    return run

def test_synthesize_accepts_passing_model(tmp_path):
    src, fn = synth.synthesize(TRANS, "actions=[1]", "g", mask=None, n_retries=1,
                               traces_dir=str(tmp_path), _runner=_runner_giving(GOOD))
    assert fn is not None and src is not None
    nf, lu = fn(F0, [1]); assert nf[0,0] == 1

def test_synthesize_returns_none_when_model_never_passes(tmp_path):
    bad = "def predict(frame, action):\n    return frame.copy(), False"
    src, fn = synth.synthesize(TRANS, "actions=[1]", "g", mask=None, n_retries=2,
                               traces_dir=str(tmp_path), _runner=_runner_giving(bad))
    assert fn is None

def test_synthesize_writes_telemetry(tmp_path):
    synth.synthesize(TRANS, "a", "g", mask=None, n_retries=1, traces_dir=str(tmp_path),
                     _runner=_runner_giving(GOOD))
    assert os.path.exists(tmp_path/"calls.jsonl")
