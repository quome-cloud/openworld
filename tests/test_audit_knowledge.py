"""The knowledge auditor (scripts/audit_sandbox.py :: audit_knowledge) must flag ONLY genuinely
source-DERIVED content in memory notes -- never the integrity/methodology notes that merely *discuss*
or *forbid* source access. Regression for the false-positive taint that mislabeled 200+ source-free
runs as memory_tainted because the notes contain the words 'source-faithful' / 'read source'."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from audit_sandbox import audit_knowledge

# Real lines from the auto-memory: these DISCUSS or FORBID source access as methodology/rules.
# The auditor must treat them as CLEAN (they contain no laundered game mechanics).
CLEAN_METHODOLOGY = [
    ("arc3-reconstruct-simulator.md",
     "The idea: three protocols on ARC-3. (1) source-faithful = read `<game>.py`, "
     "(2) source-free = act only; we take the source-FREE path (no real source)."),
    ("arc3-no-banked-solutions.md",
     "research-integrity rule: seeding/solving from banked solution traces (the answer key) is "
     "CHEATING; the agent must be SOLUTION-free as well as source-free."),
    ("arc3-falsify-unwinnable.md",
     "BEFORE re-running dead sweeps, replay any prior fair pure-click trace (skip read_source) "
     "in the isolated sandbox as a winnability diagnostic."),
    ("arc3-s5i5-rulers-walls.md",
     "closes part of the source-faithful(24/25) vs source-free(8/25) gap honestly."),
    ("arc3-dc22-keybox-bridge.md",
     "FABLE ARM: INDEPENDENT full re-solve 6/6 source-free AND solution-free -- 491 acts, fresh-verified."),
]

# Genuinely source-DERIVED content (laundered answer key): must be flagged.
REAL_TAINT = [
    ("arc3-cheat1.md",
     "spec_from_file_location('environment_files/dc22/dc22.py'); the engine sets _score>=6 when b(11) covers (6,46)."),
    ("arc3-cheat2.md",
     "I read dc22.py directly: the level-up fires when levels_completed reaches win."),
]


def _write(tmp, files):
    for name, txt in files:
        (tmp / name).write_text(txt)


def test_methodology_and_rule_notes_are_clean(tmp_path):
    _write(tmp_path, CLEAN_METHODOLOGY)
    findings = audit_knowledge(memory_dir=str(tmp_path))
    assert findings == [], f"methodology/rule notes must not be flagged as taint, got: {findings}"


def test_source_loading_signature_is_tainted(tmp_path):
    _write(tmp_path, REAL_TAINT[:1])
    assert audit_knowledge(memory_dir=str(tmp_path)), "spec_from_file_location + game .py must flag"


def test_real_gameid_source_read_is_tainted(tmp_path):
    _write(tmp_path, REAL_TAINT[1:])
    assert audit_knowledge(memory_dir=str(tmp_path)), "reading <gameid>.py must flag"


def test_negated_gameid_reference_is_clean(tmp_path):
    (tmp_path / "a.md").write_text("Integrity: never read dc22.py -- that would be cheating.")
    assert audit_knowledge(memory_dir=str(tmp_path)) == [], "a forbidden-context <gameid>.py must not flag"
