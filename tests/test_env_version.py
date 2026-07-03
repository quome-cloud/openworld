"""capture_lib.env_version resolves the ARC-AGI-3 engine version hash from the on-disk env cache
(directory name only -- source-free). CI-safe: uses a synthetic environment_files tree."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import capture_lib as c


def test_resolves_hash_from_sandbox_env_cache(tmp_path):
    d = tmp_path / "experiments" / ".sandbox_env" / "lf52" / "environment_files" / "lf52" / "271a04aa"
    d.mkdir(parents=True)
    (d / "lf52.py").write_text("# not read")                 # source present but only the DIR name is used
    assert c.env_version("lf52", root=str(tmp_path)) == "271a04aa"


def test_falls_back_to_experiments_and_root_caches(tmp_path):
    d = tmp_path / "experiments" / "environment_files" / "dc22" / "fdcac232"
    d.mkdir(parents=True)
    assert c.env_version("dc22", root=str(tmp_path)) == "fdcac232"


def test_none_when_no_cache(tmp_path):
    assert c.env_version("nope", root=str(tmp_path)) is None
