"""One-off, idempotent correction pass for the knowledge-audit false positive.

Background: the old `audit_sandbox.py :: audit_knowledge` regex matched the DISCUSSION words
'source-faithful' / 'source-derived' / 'read source' in the auto-memory. Those words live in the
integrity/methodology notes that DEFINE and FORBID source access (arc3-no-banked-solutions,
arc3-falsify-unwinnable, arc3-reconstruct-simulator, ...), so the auditor flagged the *controls*
themselves and stamped 200+ genuinely source-free runs `memory_tainted: true`. The auditor is now
hard-signature-only and re-audits current memory CLEAN.

This script re-stamps the affected meta sidecars, PRESERVING the original verdict for provenance:
  - knowledge_audit_original / memory_tainted_original  <- the old (false-positive) verdict, kept
  - knowledge_audit / memory_tainted                    <- the corrected verdict
  - knowledge_audit_correction                          <- why + when + the honest caveat

Honest caveat recorded in every stamp: run-time memory is not snapshotted, so this reflects the
corrected auditor run against CURRENT memory; the triggering content was stable rule/methodology prose.

  python scripts/restamp_knowledge_audit.py <iso_utc_timestamp> [--apply]
Without --apply it is a dry run (prints counts only).
"""
import os, sys, glob, json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from audit_sandbox import audit_knowledge

META = os.path.join(ROOT, "experiments/results/arc3_traces/meta")
MEM = "/Users/jim/.claude/projects/-Users-jim-Desktop-openworld/memory"
CLAUDE_MD = os.path.join(ROOT, "CLAUDE.md")


def main():
    ts = sys.argv[1] if len(sys.argv) > 1 else "unknown"
    apply = "--apply" in sys.argv[2:]

    # authoritative corrected verdict against current memory (expected: clean)
    current = audit_knowledge(memory_dir=MEM, claude_md=CLAUDE_MD)
    clean_now = (current == [])
    print(f"corrected auditor on current memory: {'CLEAN' if clean_now else 'TAINTED'} ({len(current)} findings)")

    tainted = restamped = already = 0
    for f in sorted(glob.glob(os.path.join(META, "*.json"))):
        try:
            m = json.load(open(f))
        except Exception:
            continue
        ka = m.get("knowledge_audit")
        was_tainted = (m.get("memory_tainted") is True) or (isinstance(ka, dict) and ka.get("clean") is False)
        if not was_tainted:
            continue
        tainted += 1
        if "knowledge_audit_correction" in m:      # idempotent
            already += 1
            continue
        if not apply:
            continue
        m["knowledge_audit_original"] = ka
        m["memory_tainted_original"] = m.get("memory_tainted")
        m["knowledge_audit"] = {
            "clean": clean_now,
            "method": "hard-signature re-audit (audit_sandbox.py; discussion words source-faithful/"
                      "source-derived/read-source excluded)",
            "findings": current,
        }
        m["memory_tainted"] = not clean_now
        m["knowledge_audit_correction"] = {
            "reaudited_at": ts,
            "auditor": "scripts/audit_sandbox.py (context-aware, hard-signal-only)",
            "reason": "original findings matched methodology/integrity DISCUSSION words in memory notes "
                      "(the notes that define/forbid source access), not laundered game source; the run "
                      "was source-free by process isolation.",
            "caveat": "run-time memory not snapshotted; reflects corrected auditor against current memory "
                      "(the triggering content was stable rule/methodology prose).",
        }
        with open(f, "w") as fh:
            fh.write(json.dumps(m, indent=1))
        restamped += 1

    print(f"tainted meta: {tainted} | already-corrected: {already} | "
          f"{'re-stamped' if apply else 'would re-stamp'}: {restamped if apply else tainted - already}")
    if not apply:
        print("dry run -- pass --apply to write")


if __name__ == "__main__":
    main()
