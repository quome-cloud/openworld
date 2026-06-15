"""Tests for the `openworld` CLI."""

import json

import pytest

pytest.importorskip("click")
from click.testing import CliRunner                            # noqa: E402

from openworld import to_spec                                  # noqa: E402
from openworld.cli import main                                 # noqa: E402
from tests.test_spec import counter_world, economy_world       # noqa: E402


def _write_specs(d):
    for w in (counter_world(), economy_world()):
        (d / f"{w.name}.json").write_text(json.dumps(to_spec(w)))


def test_ls_lists_specs(tmp_path):
    _write_specs(tmp_path)
    res = CliRunner().invoke(main, ["ls", str(tmp_path)])
    assert res.exit_code == 0
    assert "counter" in res.output and "economy" in res.output


def test_card_writes_svg(tmp_path):
    spec = tmp_path / "counter.json"
    spec.write_text(json.dumps(to_spec(counter_world())))
    out = tmp_path / "c.svg"
    res = CliRunner().invoke(main, ["card", str(spec), "--out", str(out)])
    assert res.exit_code == 0 and out.exists()
    assert out.read_text().startswith("<?xml")


def test_serve_help_lists_options():
    res = CliRunner().invoke(main, ["serve", "--help"])
    assert res.exit_code == 0
    assert "--allow-code" in res.output and "--port" in res.output


def test_build_degrades_to_manual_without_claude(monkeypatch):
    # neither tmux nor claude -> scaffold + manual mode
    monkeypatch.setattr("openworld._tmux.tmux_available", lambda: False)
    monkeypatch.setattr("openworld._tmux.claude_available", lambda: False)
    with CliRunner().isolated_filesystem():
        res = CliRunner().invoke(main, ["build", "a tiny counter world", "--name", "demo"])
        assert res.exit_code == 0
        import pathlib
        assert pathlib.Path("build/BUILD.md").exists()
        assert pathlib.Path("build/demo.py").exists()
        assert "manual mode" in res.output


def test_build_uses_headless_claude_without_tmux(monkeypatch):
    # claude present, tmux absent -> headless path (claude -p), no manual fallback
    monkeypatch.setattr("openworld._tmux.tmux_available", lambda: False)
    monkeypatch.setattr("openworld._tmux.claude_available", lambda: True)
    counter_spec = to_spec(counter_world())

    def fake_headless(prompt, cwd, wait_for, timeout=1800.0):
        import json as _j
        wait_for.parent.mkdir(parents=True, exist_ok=True)
        wait_for.write_text(_j.dumps(counter_spec))      # pretend Claude wrote it
        return True, "authored the world"

    monkeypatch.setattr("openworld._tmux.claude_headless", fake_headless)
    with CliRunner().isolated_filesystem():
        res = CliRunner().invoke(main, ["build", "a counter", "--name", "counter"])
        assert res.exit_code == 0
        assert "headlessly" in res.output
        import pathlib
        assert pathlib.Path("specs/counter.json").exists()
