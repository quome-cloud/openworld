"""SOURCE-FREE sweep over all 25 ARC-AGI-3 games, CODEX (gpt-5.5) variant -- the model-ablation twin of
sweep_sourcefree.py. Identical protocol (process-isolated SandboxGame agent; no source; banked ONLY if it
audits clean + replay-verifies + round-trips through an OpenWorld World), but the agent is `codex exec`
instead of `claude -p`, the workdirs are scratch_arc/sbcodex_<game>, and results go to a SEPARATE archive
(arc3_fullgame_sourcefree_codex.json) so the Claude run is never clobbered. This isolates the MODEL variable:
source-free Claude (8/25) vs source-free codex (?) decomposes the 8->16 (vs source-faithful codex) gap.

    python3 scripts/sweep_sourcefree_codex.py
"""
import subprocess, time, os, sys, json, signal
from pathlib import Path

ROOT = Path("/Users/jim/Desktop/openworld")
ARC_VENV = os.environ.get("ARC_VENV", os.path.expanduser("~/.arcv/bin/python"))
RUNNER = str(ROOT / "scripts" / "run_arc_agent_sandbox_codex.sh")
BANKER = str(ROOT / "scripts" / "autobank_sourcefree.py")
ARCH = ROOT / "experiments" / "results" / "arc3_fullgame_sourcefree_codex.json"
LOGDIR = ROOT / "scratch_arc"

# the banker is parametrized by env: scan sbcodex_* workdirs, write the codex archive (defaults preserve
# the Claude pipeline, so this leaves arc3_fullgame_sourcefree.json untouched).
BANK_ENV = {**os.environ, "SF_WD_PREFIX": "sbcodex_", "SF_ARCH": str(ARCH)}

GAMES = ["ar25", "bp35", "cd82", "cn04", "dc22", "ft09", "g50t", "ka59", "lf52", "lp85", "ls20", "m0r0",
         "r11l", "re86", "s5i5", "sb26", "sc25", "sk48", "sp80", "su15", "tn36", "tr87", "tu93", "vc33",
         "wa30"]

POOL = 2               # gentle: a source-faithful codex sweep may also be running
PER_AGENT_S = 2700     # 45 min wall-clock budget per agent per round
ROUNDS = 3


def log(m):
    print(f"[sweep-codex {time.strftime('%H:%M:%S')}] {m}", flush=True)


def fully_solved(g):
    if not ARCH.exists():
        return False
    try:
        v = json.loads(ARCH.read_text()).get("per_game", {}).get(g)
        return bool(v and v.get("win") and v["levels"] >= v["win"])
    except Exception:
        return False


def bank():
    try:
        r = subprocess.run([ARC_VENV, BANKER], cwd=str(ROOT), capture_output=True, text=True,
                           timeout=1200, env=BANK_ENV)
        for ln in (r.stdout or "").splitlines():
            if "[sf-bank]" in ln:
                log(ln.strip())
    except Exception as ex:
        log(f"bank error: {ex}")


def launch(g):
    out = open(LOGDIR / f"sbcodex_{g}_sweep.out", "a")
    return subprocess.Popen(["bash", RUNNER, g], cwd=str(ROOT),
                            stdout=out, stderr=subprocess.STDOUT,
                            start_new_session=True)


def kill_tree(p):
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
    except Exception:
        pass


def commit(round_idx):
    a = json.loads(ARCH.read_text()) if ARCH.exists() else {}
    msg = (f"Source-free CODEX sweep round {round_idx}: {a.get('n_full_games', 0)} full, "
           f"{a.get('total_levels', 0)}/{a.get('total_possible', 0)} levels "
           f"(audit-clean + OpenWorld-World-verified; model ablation vs Claude source-free).\n\n"
           "codex (gpt-5.5) through the SAME source-free SandboxGame pipeline as the Claude run: "
           "process-isolated, audited for source access, replay-verified, OpenWorld-World round-tripped. "
           "Separate archive (arc3_fullgame_sourcefree_codex.json); no push.\n\n"
           "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>")
    subprocess.run(["git", "-C", str(ROOT), "add", str(ARCH),
                    "scripts/run_arc_agent_sandbox_codex.sh", "scripts/sweep_sourcefree_codex.py",
                    "scripts/autobank_sourcefree.py"], check=False)
    subprocess.run(["git", "-C", str(ROOT), "commit", "-q", "-m", msg], check=False)
    log(f"committed round {round_idx} (local, not pushed)")


def main():
    log(f"START source-free CODEX sweep: {len(GAMES)} games, pool={POOL}, "
        f"per-agent={PER_AGENT_S}s, rounds={ROUNDS}")
    for r in range(ROUNDS):
        queue = [g for g in GAMES if not fully_solved(g)]
        log(f"round {r}: {len(queue)} games unsolved -> {queue}")
        if not queue:
            log("all games fully solved; stopping early")
            break
        running = {}
        qi = 0
        while qi < len(queue) or running:
            while len(running) < POOL and qi < len(queue):
                g = queue[qi]; qi += 1
                running[g] = (launch(g), time.time())
                log(f"launch {g}  ({len(running)} active, {len(queue) - qi} queued)")
            time.sleep(20)
            done = []
            for g, (p, st) in running.items():
                if p.poll() is not None:
                    done.append((g, "exited"))
                elif time.time() - st > PER_AGENT_S:
                    kill_tree(p); done.append((g, "timeout"))
            for g, why in done:
                del running[g]
                log(f"{g} {why}")
                bank()
        bank()
        commit(r)
    bank()
    commit(ROUNDS)
    a = json.loads(ARCH.read_text()) if ARCH.exists() else {}
    log(f"DONE: {a.get('n_full_games', 0)} full games, "
        f"{a.get('total_levels', 0)}/{a.get('total_possible', 0)} levels across "
        f"{a.get('n_games_started', 0)} games started")


if __name__ == "__main__":
    main()
