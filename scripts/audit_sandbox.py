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
    wd = sys.argv[1]
    findings = audit(wd)
    if findings:
        print(f"TAINTED ({wd}):")
        for f in findings:
            print("  -", f)
        sys.exit(1)
    print(f"CLEAN ({wd}): no source access detected")
    sys.exit(0)
