"""E142 integrity + wiring tests: the generalization-critique must be GENERIC (no banked game content)
and actually wired into the harness."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CRITIQUE = ROOT / "experiments" / "e142" / "critique.md"
HARNESS = ROOT / "scripts" / "run_arc_agent_ewm_toolkit.sh"

# the 25 public game ids -- the critique must mention NONE of them (it must be game-agnostic)
GAME_IDS = ["ar25", "bp35", "cd82", "cn04", "dc22", "ft09", "g50t", "ka59", "lf52", "lp85", "ls20",
            "m0r0", "r11l", "re86", "s5i5", "sb26", "sc25", "sk48", "sp80", "su15", "tn36", "tr87",
            "tu93", "vc33", "wa30"]


def test_critique_is_solution_free_and_source_free():
    text = CRITIQUE.read_text()
    low = text.lower()
    # no game-specific content (would make it banked answers, not generic methodology)
    leaked = [g for g in GAME_IDS if g in low]
    assert not leaked, f"critique leaks game-specific ids: {leaked}"
    # explicitly forbids source/banked solutions
    assert "source-free" in low and "solution-free" in low
    assert "do not read game source" in low or "do not read game source" in low.replace("\n", " ")
    # it is an adversarial generalization review, not a helper
    assert "generaliz" in low and ("critic" in low or "skeptic" in low or "adversarial" in low)


def test_critique_targets_overfitting_and_next_level():
    low = CRITIQUE.read_text().lower()
    for must in ("hard-cod", "ontology", "next", "unseen"):
        assert must in low, f"critique missing the '{must}' generalization check"


def test_harness_wires_the_critique_step():
    sh = HARNESS.read_text()
    assert "critique.md" in sh                       # copied into the workspace
    assert "CRITIQUE PROTOCOL" in sh                 # the protocol is injected into TASK.md
    assert "BEFORE banking" in sh                    # gating point is correct
