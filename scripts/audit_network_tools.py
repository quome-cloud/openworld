"""Leak-channel audit: did any SOURCE-FREE ARC-AGI-3 solver run reach the internet, or invoke a
web/network tool? A source-free solver must discover a game's rules by ACTING; fetching anything over the
network is a leakage channel (it could pull a walkthrough, the game's source, or a solution). This audits
the captured transcripts for that channel and writes experiments/results/arc3_tool_audit.json, which
scripts/make_arc3_assets.py turns into the paper's audit macros.

What it checks, per source-free run (joined from the run meta):
  * Tool INVENTORY + web-tool invocations -- from each Claude-arm stream-json transcript's tool_use blocks
    (opus sb_, fable sbfable_). Any WebSearch/WebFetch/fetch/browser tool call is a hit.
  * NETWORK commands in every Bash command -- curl/wget/http(s)/urllib/requests/socket/pip install/git
    clone|pull|fetch/ssh/scp/package installs. Applied to Claude Bash commands AND to the codex plaintext
    command logs (.codex.log.gz), so all three arms are covered.

Honest by construction: transcripts are the immutable per-run record; this only READS them. A clean result
(0 network hits, 0 web-tool calls) is evidence the leak channel was unused; a non-zero result would name
the offending run. Run where the transcripts live (they are gitignored, kept on disk for the dataset):

    python scripts/audit_network_tools.py
"""
import os, sys, json, glob, gzip, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TR = ROOT / "experiments" / "results" / "arc3_traces"
OUT = ROOT / "experiments" / "results" / "arc3_tool_audit.json"

# UNAMBIGUOUS network/internet signals only. Deliberately NOT bare nc/dig/host/ping/socket -- those
# short words appear constantly inside the Python the agent writes to local files via heredocs
# (`cat > sim.py <<'PY' ... PY`), which is data, not a command, and would be pure false positives.
# We require a real fetch/transfer: a URL, an HTTP client, a socket CONNECT, a package install that
# downloads, a git remote op, or ssh/scp to a host. This is the leak channel that actually matters.
NET = re.compile(
    r'(?:https?://|ftp://|'
    r'\bcurl\b|\bwget\b|\bhttpie\b|\bhttpx\b|\baiohttp\b|\bwebsocket\b|'
    r'requests\.(?:get|post|put|head|patch|delete|request|Session)|'
    r'urllib\.request|urlopen|http\.client|socket\.socket|\.connect\(\(|'
    r'\bgit\s+(?:clone|pull|fetch|remote)\b|'
    r'\b(?:pip3?|conda|npm|yarn|apt|apt-get|brew)\s+install\b|'
    r'\bssh\s+[A-Za-z0-9_.-]+@|\bscp\s+\S+@)', re.I)
WEB_TOOLS = {"websearch", "webfetch", "fetch", "browser", "navigate", "webbrowser"}


def net_hits(cmd):
    """Return unambiguous network-activity tokens in a shell command (see NET)."""
    return [m.group(0).strip() for m in NET.finditer(cmd or "")]


def scan_jsonl(path):
    """(tool_counter, n_bash, web_tool_calls, [network_examples]) for a Claude stream-json transcript."""
    from collections import Counter
    tools = Counter(); nbash = 0; web = 0; net = []
    for line in open(path, errors="ignore"):
        if '"tool_use"' not in line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        content = (d.get("message") or {}).get("content")
        for blk in content if isinstance(content, list) else []:
            if not (isinstance(blk, dict) and blk.get("type") == "tool_use"):
                continue
            name = blk.get("name", "?")
            tools[name] += 1
            if name.lower() in WEB_TOOLS:
                web += 1
            if name == "Bash":
                nbash += 1
                for h in net_hits((blk.get("input") or {}).get("command", "")):
                    net.append(h)
    return tools, nbash, web, net


def scan_codex_gz(path):
    """(n_lines, [network_examples]) for a codex plaintext .log.gz command trace."""
    net = []; nlines = 0
    try:
        with gzip.open(path, "rt", errors="ignore") as f:
            for line in f:
                nlines += 1
                for h in net_hits(line):
                    net.append(h)
    except Exception:
        pass
    return nlines, net


def main():
    from collections import Counter
    arms = {}          # label -> aggregate
    tool_total = Counter()
    net_examples = []
    n_runs = n_bash = n_web = 0

    for m in sorted(glob.glob(str(TR / "meta" / "*.json"))):
        try:
            d = json.load(open(m))
        except Exception:
            continue
        if not d.get("source_free", False):
            continue
        model = str((d.get("model_config") or {}).get("requested_model")
                    or (d.get("model_config") or {}).get("model") or "").lower()
        label = ("fable" if "fable" in model else
                 "codex" if ("codex" in model or "gpt" in model) else
                 "opus" if ("opus" in model or "claude" in model) else "other")
        a = arms.setdefault(label, {"runs": 0, "bash_cmds": 0, "web_tool_calls": 0,
                                    "network_hits": 0, "transcripts": 0})
        tf = TR / (d.get("transcript_file") or "")
        a["runs"] += 1; n_runs += 1
        if str(tf).endswith(".jsonl") and tf.exists():
            tools, nb, web, net = scan_jsonl(tf)
            tool_total.update(tools)
            a["transcripts"] += 1; a["bash_cmds"] += nb; a["web_tool_calls"] += web
            a["network_hits"] += len(net); n_bash += nb; n_web += web
            net_examples += [(d.get("run_id"), h) for h in net]
        elif str(tf).endswith(".gz") and tf.exists():
            nl, net = scan_codex_gz(tf)
            a["transcripts"] += 1; a["network_hits"] += len(net)
            net_examples += [(d.get("run_id"), h) for h in net]

    payload = {
        "note": "Source-free ARC-AGI-3 leak-channel audit. Scanned every source-free run's captured "
                "transcript for internet access (network commands in Bash / codex logs) and web-tool "
                "invocations (WebSearch/WebFetch/browser). Read-only over the immutable per-run records.",
        "n_runs": n_runs,
        "n_transcripts_scanned": sum(a["transcripts"] for a in arms.values()),
        "n_bash_commands": n_bash,
        "network_hits": sum(a["network_hits"] for a in arms.values()),
        "web_tool_calls": n_web,
        "network_examples": net_examples[:20],
        "tool_inventory": dict(tool_total.most_common()),
        "arms": arms,
        "web_tools_watched": sorted(WEB_TOOLS),
    }
    json.dump(payload, open(OUT, "w"), indent=1)
    print(f"wrote {OUT.name}: runs={n_runs} transcripts={payload['n_transcripts_scanned']} "
          f"bash_cmds={n_bash} network_hits={payload['network_hits']} web_tool_calls={n_web}")
    print("  per-arm:", {k: (v["runs"], v["network_hits"], v["web_tool_calls"]) for k, v in arms.items()})
    if payload["network_hits"] or n_web:
        print("  !! NON-ZERO leak signal:", net_examples[:5])


if __name__ == "__main__":
    main()
