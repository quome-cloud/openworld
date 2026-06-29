# tests/e127/test_audit.py
from experiments.e127 import audit

def test_clean_dir(tmp_path):
    (tmp_path / "engine.py").write_text("class Engine:\n    def reset(self): return None\n")
    assert audit.audit_dir(str(tmp_path)) == []
    assert audit.audit_clean(str(tmp_path)) is True

def test_flags_environment_files(tmp_path):
    (tmp_path / "cheat.py").write_text("p = 'environment_files/dc22/dc22.py'\n")
    findings = audit.audit_dir(str(tmp_path))
    assert findings and audit.audit_clean(str(tmp_path)) is False

def test_flags_getsource(tmp_path):
    (tmp_path / "x.py").write_text("import inspect\ninspect.getsource(env._game)\n")
    assert audit.audit_clean(str(tmp_path)) is False

def test_flags_spec_from_file_location(tmp_path):
    (tmp_path / "y.py").write_text("import importlib.util\nimportlib.util.spec_from_file_location('g','g.py')\n")
    assert audit.audit_clean(str(tmp_path)) is False

def test_flags_fstring_open(tmp_path):
    # f-string / variable filename: open(f"{gid}.py") has no literal id before .py"
    (tmp_path / "z.py").write_text("gid = 'dc22'\nsrc = open(f'{gid}.py').read()\n")
    assert audit.audit_clean(str(tmp_path)) is False

def test_flags_pathlib_read_text(tmp_path):
    # pathlib source read: no open(), no literal id
    (tmp_path / "w.py").write_text("import pathlib\nsrc = pathlib.Path(p).read_text()\n")
    assert audit.audit_clean(str(tmp_path)) is False
