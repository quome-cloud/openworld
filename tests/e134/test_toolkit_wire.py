from pathlib import Path

import numpy as np


def test_toolkit_reexports_composite_api():
    # the EWM toolkit must surface the composite perception API for the agent
    from experiments.e133.ewm_toolkit import composite_key, select_lens, LENSES
    assert callable(composite_key) and callable(select_lens)
    assert isinstance(LENSES, dict) and len(LENSES) >= 6


def test_composite_key_distinguishes_via_toolkit():
    from experiments.e133.ewm_toolkit import composite_key
    a = np.zeros((8, 8), dtype=int)
    b = a.copy(); b[0, 0] = 7
    assert composite_key(a) != composite_key(b)


def test_harness_copies_composite_into_workspace():
    sh = Path("scripts/run_arc_agent_ewm_toolkit.sh").read_text()
    assert "composite.py" in sh and "perceptors.py" in sh
    # the agent is instructed to perceive via the composite
    assert "composite_key(frame)" in sh


def test_flat_workspace_import_resolves(tmp_path):
    # PROVE the agent's FLAT workspace import works: copy the helper files side-by-side
    # (as the harness does) and import them with bare module names in a clean subprocess.
    import shutil, subprocess, sys
    root = Path(__file__).resolve().parents[2]
    for rel in ["experiments/e125/objstate.py", "experiments/e133/ewm_toolkit.py",
                "experiments/e134/perceptors.py", "experiments/e134/composite.py"]:
        shutil.copy(root / rel, tmp_path / Path(rel).name)
    code = ("import numpy as np;"
            "from ewm_toolkit import composite_key, select_lens, LENSES;"
            "a=np.zeros((8,8),dtype=int); b=a.copy(); b[0,0]=7;"
            "assert composite_key(a)!=composite_key(b);"
            "assert len(LENSES)>=6; print('OK')")
    r = subprocess.run([sys.executable, "-c", code], cwd=tmp_path,
                       capture_output=True, text=True)
    assert r.returncode == 0 and "OK" in r.stdout, r.stderr
