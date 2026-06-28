"""Structural no-source-read audit of a model working dir. Source access = reading a game's <gid>.py
(the answer key). This makes 'source-simulated' auditable: a clean model workdir proves the engine
was authored from observed play, not from the real source."""
import os, glob, re

SOURCE_READ = re.compile(
    r"environment_files|inspect\.getsource|spec_from_file_location|"
    r"importlib\.util\.spec_from_file_location|[a-z0-9]{4,8}\.py['\"]")


def audit_dir(wd):
    findings = []
    if os.path.isdir(os.path.join(wd, "environment_files")):
        findings.append("environment_files/ dir present")
    for p in glob.glob(os.path.join(wd, "**", "*.py"), recursive=True):
        try:
            txt = open(p, errors="ignore").read()
        except Exception:
            continue
        m = SOURCE_READ.search(txt)
        if m:
            findings.append(f"{os.path.basename(p)}: source-read pattern {m.group(0)!r}")
    return findings


def audit_clean(wd):
    return audit_dir(wd) == []
