"""Source-free isolation for codex calls: audit the codex exec --json event stream for any read of game
source, and (Task 2) wrap the codex exec invocation. The read-only sandbox still permits disk reads, so the
audit is the real gate -- a tainted call is discarded (CLAUDE.md cardinal rule)."""
import json, re

# any of these appearing in a shell command or file path = a source read
_SOURCE_RE = re.compile(
    r"(arc_agi|environment_files|arcengine|experiments/[a-z0-9]{4}\.py|/[a-z0-9]{4}\.py\b)", re.I)


def audit_events(events, game):
    """Return True if any event reads game source (the game's <id>.py, arc_agi, environment_files)."""
    gid = re.escape(game)
    needles = re.compile(rf"({_SOURCE_RE.pattern}|{gid}\.py)", re.I)
    for e in events or []:
        blob = " ".join(str(e.get(k, "")) for k in ("command", "path", "cmd", "args", "text"))
        if needles.search(blob):
            # an agent_message merely mentioning the id is not a read; only exec/file events count
            if e.get("type") in ("agent_message", "reasoning", "token_count"):
                continue
            return True
    return False
