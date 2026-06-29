"""Audit a sandbox agent working dir for SOURCE ACCESS. Exit 0 = clean (no game source touched),
1 = tainted. A solve is bankable as 'source-free' only if its dir passes this audit.

  python scripts/audit_sandbox.py scratch_arc/sb_<game>
"""
import sys, os, glob, re

# The integrity invariant: a solver may ACT on the env, but must never read the game's SOURCE.
# SOURCE_READ = reading <game>.py (the cheat); ENGINE_IMPORT = importing arc_agi (legitimate acting).
SOURCE_READ = re.compile(r"environment_files|inspect\.getsource|spec_from_file_location|"
                         r"importlib\.util\.spec_from_file_location")
ENGINE_IMPORT = re.compile(r"import\s+arc_agi|from\s+arc_agi")


def audit(wd, mode="strict"):
    """Audit a working dir for SOURCE access.
    mode="strict"  (open-ended AGENT tier): no source reads AND no arc_agi import -- the agent must act
                   only through the process-isolated SandboxGame pipe (fair by construction).
    mode="source_only" (fixed CHEAP solver tier): no source reads; importing arc_agi to STEP the env is
                   allowed -- the algorithm is statically verifiable to consume only frames (fair by audit).
    """
    findings = []
    # 1) no game source physically present in the working dir (both modes)
    src = [p for p in glob.glob(os.path.join(wd, "**", "*.py"), recursive=True)
           if "environment_files" in p]
    if src:
        findings.append(f"game source present: {src[:3]}")
    if os.path.isdir(os.path.join(wd, "environment_files")):
        findings.append("environment_files/ dir present")
    # 2) no script reads source (both modes); arc_agi import forbidden only in strict mode
    for p in glob.glob(os.path.join(wd, "*.py")):
        if os.path.basename(p) == "arc3_sandbox.py":
            continue                                   # the sanctioned client harness
        try:
            txt = open(p, errors="ignore").read()
        except Exception:
            continue
        for m in set(SOURCE_READ.findall(txt)):
            findings.append(f"{os.path.basename(p)}: SOURCE-READ '{m}'")
        if mode == "strict":
            for m in set(ENGINE_IMPORT.findall(txt)):
                findings.append(f"{os.path.basename(p)}: engine import '{m.strip()}' (agent must use SandboxGame)")
    return findings


# Source-DERIVED knowledge: content in memory notes / CLAUDE.md extracted from a game's source (a
# laundered answer key). PRECISE signals only -- must NOT false-positive on legit project scripts
# (serve.py, card.py, solver files) or openworld's own `inspect.getsource` spec serialization, or
# ENGINE/harness notes (`arcengine`, `_game.full_reset()`). The decisive signals: the game-source dir,
# loading a specific game's <gameid>.py, or explicit "source-derived/faithful / read ... source" labels.
_GAME_IDS = ("ar25|bp35|cd82|cn04|dc22|ft09|g50t|ka59|lf52|lp85|ls20|m0r0|r11l|re86|s5i5|sb26|sc25|"
             "sk48|sp80|su15|tn36|tr87|tu93|vc33|wa30")
KNOWLEDGE_TAINT = re.compile(
    r"environment_files|spec_from_file_location|source-derived|source-faithful|"
    rf"(?:{_GAME_IDS})\.py\b|read (?:the |env )?(?:game )?source", re.IGNORECASE)


def audit_knowledge(memory_dir=None, claude_md=None):
    """Scan the agent's loaded KNOWLEDGE sources (auto memory notes + CLAUDE.md) for source-DERIVED content.
    Returns a list of findings; [] means no laundered-source knowledge is in scope. This makes the
    memory/CLAUDE.md contamination self-detecting on every run instead of relying on manual review."""
    findings = []
    targets = []
    if memory_dir and os.path.isdir(memory_dir):
        targets += [p for p in glob.glob(os.path.join(memory_dir, "*.md"))
                    if os.path.basename(p) != "MEMORY.md"]
    if claude_md and os.path.isfile(claude_md):
        targets.append(claude_md)
    for p in targets:
        try:
            txt = open(p, errors="ignore").read()
        except Exception:
            continue
        # ignore the explicit source-free RULE line in CLAUDE.md (it names the patterns to forbid)
        hits = set(m.group(0) for m in KNOWLEDGE_TAINT.finditer(txt)
                   if "never read" not in txt[max(0, m.start() - 60):m.start()].lower()
                   and "do not" not in txt[max(0, m.start() - 40):m.start()].lower())
        for h in sorted(hits):
            findings.append(f"{os.path.basename(p)}: source-derived knowledge '{h}'")
    return findings


def audit_files(paths, mode="source_only"):
    """Audit specific script files (e.g. the fixed cheap solvers) for SOURCE reads. Returns findings."""
    findings = []
    for p in paths:
        try:
            txt = open(p, errors="ignore").read()
        except Exception:
            continue
        for m in set(SOURCE_READ.findall(txt)):
            findings.append(f"{os.path.basename(p)}: SOURCE-READ '{m}'")
        if mode == "strict":
            for m in set(ENGINE_IMPORT.findall(txt)):
                findings.append(f"{os.path.basename(p)}: engine import '{m.strip()}'")
    return findings


if __name__ == "__main__":
    # `audit_sandbox.py --knowledge` audits the agent's loaded memory + CLAUDE.md for source-derived
    # content; `audit_sandbox.py <dir>` audits a working dir for source reads.
    if len(sys.argv) >= 2 and sys.argv[1] == "--knowledge":
        mem = "/Users/jim/.claude/projects/-Users-jim-Desktop-openworld/memory"
        cmd = "/Users/jim/Desktop/openworld/CLAUDE.md"
        findings = audit_knowledge(memory_dir=mem, claude_md=cmd)
        if findings:
            print("TAINTED (memory + CLAUDE.md): source-derived knowledge present:")
            for f in findings:
                print("  -", f)
            sys.exit(1)
        print("CLEAN (memory + CLAUDE.md): no source-derived knowledge detected")
        sys.exit(0)
    wd = sys.argv[1]
    findings = audit(wd)
    if findings:
        print(f"TAINTED ({wd}):")
        for f in findings:
            print("  -", f)
        sys.exit(1)
    print(f"CLEAN ({wd}): no source access detected")
    sys.exit(0)
