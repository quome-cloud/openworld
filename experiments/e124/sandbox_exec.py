"""Run a codex-generated predicate/score_fn on a frame in a SEPARATE subprocess with a hard timeout. This is
ROBUSTNESS (codex is not adversarial), not a security sandbox: a buggy/looping function degrades to None
(-> the hypothesis is dropped), never crashing or hanging the search."""
import os, sys, json, subprocess, tempfile, base64
import numpy as np

_RUNNER = r'''
import sys, json, base64, numpy as np
src=base64.b64decode(sys.argv[1]).decode(); name=sys.argv[2]
arr=np.frombuffer(base64.b64decode(sys.argv[3]), dtype=np.int64).reshape(64,64)
ns={"np": np, "__builtins__": __builtins__}
try:
    exec(src, ns)
    v=ns[name](arr)
    print(json.dumps({"v": float(v)}))
except Exception as e:
    print(json.dumps({"v": None}))
'''

def eval_fn(src, fn_name, frame, timeout=2.0):
    arr = np.asarray(frame).astype(np.int64).reshape(64, 64)
    b_src = base64.b64encode(src.encode()).decode()
    b_arr = base64.b64encode(arr.tobytes()).decode()
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(_RUNNER); runner = f.name
    try:
        p = subprocess.run([sys.executable, runner, b_src, fn_name, b_arr],
                           capture_output=True, text=True, timeout=timeout)
        out = json.loads(p.stdout.strip().splitlines()[-1])
        return out["v"]
    except Exception:
        return None
    finally:
        try: os.unlink(runner)
        except Exception: pass
