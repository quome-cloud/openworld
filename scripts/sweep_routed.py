"""Overnight ROUTED source-free sweep -- the hybrid-world-models pipeline (E116 made real + source-free).

Phase 1 (CHEAP, fast): a fixed pixel-only frontier search (E107) attempts every game source-free
                       (fairness BY AUDIT). Captured as deterministic run records.
Phase 2 (AGENT, deep): every game the cheap tier did NOT fully solve is routed to the live coding agent in
                       the process-isolated sandbox (fairness BY CONSTRUCTION), best-keeper across rounds.
                       Pinned model/effort; full prompt+transcript captured.

After every batch: finalize_traces (verified outcomes -> runs.jsonl) + bank_from_runs (deepest verified per
game -> archive, tier-tagged). Local commit per phase/round; NO push (the human reviews + pushes).

Plain python3 (stdlib). Banker/finalizer/cheap run under the arc venv (need arc_agi + openworld).
    python3 scripts/sweep_routed.py
"""
import subprocess, time, os, sys, json, signal, glob, re
from pathlib import Path

ROOT = Path("/Users/jim/Desktop/openworld")
# Durable arc venv (rebuildable; do NOT hard-code a session scratchpad path -- it is reaped per session).
# Override with ARC_VENV=... ; default lives in $HOME so it survives across Claude sessions.
ARC_VENV = os.environ.get("ARC_VENV", os.path.expanduser("~/.arcv/bin/python"))
RUNNER = str(ROOT / "scripts" / "run_arc_agent_sandbox.sh")
ARCH = ROOT / "experiments" / "results" / "arc3_fullgame_sourcefree.json"
LOGDIR = ROOT / "scratch_arc"
TRACES = ROOT / "experiments" / "results" / "arc3_traces"

GAMES = ["ar25", "bp35", "cd82", "cn04", "dc22", "ft09", "g50t", "ka59", "lf52", "lp85", "ls20", "m0r0",
         "r11l", "re86", "s5i5", "sb26", "sc25", "sk48", "sp80", "su15", "tn36", "tr87", "tu93", "vc33",
         "wa30"]

POOL = int(os.environ.get("POOL", "4"))
PER_AGENT_S = int(os.environ.get("PER_AGENT_S", "2700"))     # 45 min wall-clock per agent per round
ROUNDS = int(os.environ.get("ROUNDS", "4"))
MODEL = os.environ.get("MODEL", "claude-opus-4-8")
EFFORT = os.environ.get("EFFORT", "high")


def log(m):
    print(f"[routed {time.strftime('%H:%M:%S')}] {m}", flush=True)


def venv(script, *args, timeout=1800):
    try:
        r = subprocess.run([ARC_VENV, str(ROOT / "scripts" / script), *args], cwd=str(ROOT),
                           capture_output=True, text=True, timeout=timeout)
        for ln in (r.stdout or "").splitlines():
            if any(tag in ln for tag in ("[finalize]", "[bank]", "[cheap]")):
                log(ln.strip())
        if r.returncode != 0:
            log(f"{script} rc={r.returncode}: {(r.stderr or '')[-300:]}")
    except Exception as ex:
        log(f"{script} error: {ex}")


def finalize_and_bank():
    venv("finalize_traces.py")
    venv("bank_from_runs.py")


def fully_solved(g):
    if not ARCH.exists():
        return False
    try:
        v = json.loads(ARCH.read_text()).get("per_game", {}).get(g)
        return bool(v and v.get("win") and v["levels"] >= v["win"])
    except Exception:
        return False


def launch_agent(g):
    out = open(LOGDIR / f"sb_{g}_sweep.out", "a")
    env = dict(os.environ, MODEL=MODEL, EFFORT=EFFORT)
    return subprocess.Popen(["bash", RUNNER, g, "agent"], cwd=str(ROOT), env=env,
                            stdout=out, stderr=subprocess.STDOUT, start_new_session=True)


def kill_tree(p):
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
    except Exception:
        pass


def rate_limit_reset():
    """If the most-recent agent runs were REJECTED by the usage cap (tiny stub transcripts carrying a
    'rateLimitType'/'session limit' marker), return the reset epoch (seconds) so the caller can wait it
    out instead of spinning. Only considers freshly-written stubs whose reset is still in the future."""
    tdir = TRACES / "transcripts"
    files = sorted(glob.glob(str(tdir / "*__agent__*.jsonl")),
                   key=lambda f: os.path.getmtime(f), reverse=True)[:max(POOL + 1, 4)]
    now = time.time(); reset = None
    for f in files:
        try:
            if now - os.path.getmtime(f) > 900:      # only recent runs (<15 min) count
                continue
            if os.path.getsize(f) > 60000:           # big transcript => real work, not a rejection
                continue
            txt = open(f, encoding="utf-8", errors="ignore").read()
            if not re.search(r"rateLimitType|session limit|usage limit|\"rejected\"", txt):
                continue
            m = re.search(r'"resetsAt":\s*(\d+)', txt)
            if m:
                reset = max(reset or 0, int(m.group(1)))
            else:
                reset = max(reset or 0, int(now + 1800))   # unknown reset -> back off 30 min
        except Exception:
            pass
    return reset if (reset and reset > now + 5) else None


def wait_if_rate_limited():
    """Block until the usage cap resets (plus a small buffer) if a rejection was just detected. Returns
    True if it waited. Caps the wait at 6h as a safety. This converts wasteful reject-spin into productive
    waiting: when the window resets, the very next agent launch does real work again."""
    reset = rate_limit_reset()
    if not reset:
        return False
    wait = min(reset - time.time() + 30, 6 * 3600)
    if wait <= 0:
        return False
    log(f"RATE-LIMITED: usage cap hit; pausing {int(wait)}s until reset "
        f"~{time.strftime('%H:%M', time.localtime(reset))} (no agents launched meanwhile)")
    time.sleep(wait)
    log("rate-limit window reset; resuming agent launches")
    return True


def commit(tag):
    a = json.loads(ARCH.read_text()) if ARCH.exists() else {}
    by_tier = a.get("by_tier", {})
    msg = (f"Routed source-free sweep [{tag}]: {a.get('n_full_games', 0)} full, "
           f"{a.get('total_levels', 0)}/{a.get('total_possible', 0)} levels "
           f"(cheap+agent; audit-clean + OpenWorld-World-verified).\n\n"
           f"by tier: { {k: len(v) for k, v in by_tier.items()} }. Every banked solve is a captured, "
           "timestamped run in arc3_traces/runs.jsonl (prompt+transcript+model/effort metadata), "
           "replay-verified and round-tripped through an OpenWorld World. No push (human reviews).\n\n"
           "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>")
    subprocess.run(["git", "-C", str(ROOT), "add", str(ARCH),
                    "experiments/results/arc3_traces/runs.jsonl",
                    "experiments/results/arc3_traces/meta",
                    "experiments/results/arc3_traces/prompts",
                    "experiments/results/arc3_traces/solutions",
                    "scripts/"], check=False)
    subprocess.run(["git", "-C", str(ROOT), "commit", "-q", "-m", msg], check=False)
    log(f"committed [{tag}] (local, not pushed)")


def preflight_knowledge_audit():
    """Before any run, verify the agent's loaded knowledge (memory notes + CLAUDE.md) is free of
    source-DERIVED content. Warn loudly if not -- those runs will be flagged memory_tainted and excluded
    from the fair count, so a contaminated memory state should be cleaned before trusting the sweep."""
    try:
        sys.path.insert(0, str(ROOT / "scripts"))
        from audit_sandbox import audit_knowledge
        mem = "/Users/jim/.claude/projects/-Users-jim-Desktop-openworld/memory"
        f = audit_knowledge(memory_dir=mem, claude_md=str(ROOT / "CLAUDE.md"))
        if f:
            log("!!! KNOWLEDGE AUDIT TAINTED — source-derived content in memory/CLAUDE.md; runs will be "
                "flagged memory_tainted and EXCLUDED from the fair count. Clean these first:")
            for x in f[:12]:
                log(f"    - {x}")
        else:
            log("knowledge audit CLEAN (memory + CLAUDE.md free of source-derived content)")
    except Exception as ex:
        log(f"knowledge audit error (non-fatal): {ex}")


def main():
    log(f"START routed sweep: {len(GAMES)} games | pool={POOL} per-agent={PER_AGENT_S}s rounds={ROUNDS} "
        f"| model={MODEL} effort={EFFORT}")
    preflight_knowledge_audit()

    # ---- Phase 1: cheap tier (fast, all games) ---- (skip on resume via SKIP_CHEAP=1)
    if os.environ.get("SKIP_CHEAP") == "1":
        log("Phase 1 (cheap) SKIPPED (resume)")
    else:
        log("Phase 1 (cheap pixel-search, source-free by audit) over all games")
        venv("run_cheap_tier.py", *GAMES, timeout=1800)
        finalize_and_bank()
        commit("cheap")

    # ---- Phase 2: agent rounds on not-fully-solved games ----
    for r in range(ROUNDS):
        queue = [g for g in GAMES if not fully_solved(g)]
        log(f"round {r}: {len(queue)} games not fully solved -> agent: {queue}")
        if not queue:
            log("all games fully solved; stopping early")
            break
        running = {}
        qi = 0
        while qi < len(queue) or running:
            if len(running) < POOL and qi < len(queue):
                wait_if_rate_limited()      # don't launch agents into a usage-cap rejection window
            while len(running) < POOL and qi < len(queue):
                g = queue[qi]; qi += 1
                running[g] = (launch_agent(g), time.time())
                log(f"launch agent {g}  ({len(running)} active, {len(queue) - qi} queued)")
            time.sleep(20)
            done = []
            for g, (p, st) in running.items():
                if p.poll() is not None:
                    done.append((g, "exited"))
                elif time.time() - st > PER_AGENT_S:
                    kill_tree(p); done.append((g, "timeout"))
            for g, why in done:
                del running[g]
                log(f"agent {g} {why}")
                finalize_and_bank()
        commit(f"round{r}")

    finalize_and_bank()
    commit("final")
    a = json.loads(ARCH.read_text()) if ARCH.exists() else {}
    log(f"DONE: {a.get('n_full_games', 0)} full games, "
        f"{a.get('total_levels', 0)}/{a.get('total_possible', 0)} levels; "
        f"by tier { {k: len(v) for k, v in a.get('by_tier', {}).items()} }")


if __name__ == "__main__":
    main()
