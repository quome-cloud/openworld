"""Audit a sandbox agent working dir for SOURCE ACCESS. Exit 0 = clean (no game source touched),
1 = tainted. A solve is bankable as 'source-free' only if its dir passes this audit.

  python scripts/audit_sandbox.py scratch_arc/sb_<game>
"""
import sys, os, glob, re

TAINT = re.compile(r"environment_files|inspect\.getsource|spec_from_file_location|"
                   r"importlib\.util\.spec_from_file_location|import\s+arc_agi|from\s+arc_agi")


def audit(wd):
    findings = []
    # 1) no game source physically present in the agent dir
    src = [p for p in glob.glob(os.path.join(wd, "**", "*.py"), recursive=True)
           if "environment_files" in p]
    if src:
        findings.append(f"game source present: {src[:3]}")
    if os.path.isdir(os.path.join(wd, "environment_files")):
        findings.append("environment_files/ dir present")
    # 2) no agent script reads source / imports the engine
    for p in glob.glob(os.path.join(wd, "*.py")):
        if os.path.basename(p) == "arc3_sandbox.py":
            continue                                   # the sanctioned client harness
        try:
            txt = open(p, errors="ignore").read()
        except Exception:
            continue
        for m in set(TAINT.findall(txt)):
            findings.append(f"{os.path.basename(p)}: references '{m}'")
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
