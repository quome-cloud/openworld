"""Overnight SOURCE-FREE sweep over all 25 ARC-AGI-3 games.

Each game is solved by a process-isolated SandboxGame agent (claude -p; no game source, numpy-only
interpreter -> fair by construction). After every agent exits we run autobank_sourcefree.py, which banks a
solve ONLY if it (1) audits clean (no source access), (2) replay-verifies on the real engine, AND (3)
ROUND-TRIPS THROUGH AN OPENWORLD WORLD (world.step reproduces depth, 0 misses, valid spec, renderable card).
Best-keeper across rounds (each round reseeds from the deepest banked solution and pushes further). Local
git commit per round; NO push (concurrent cloud agents share the remote -- the human reviews in the morning).

Plain python3 (stdlib only). The banker is invoked with the arc venv (needs arc_agi + openworld).
    python3 scripts/sweep_sourcefree.py
"""
import subprocess, time, os, sys, json, signal
from pathlib import Path

ROOT = Path("/Users/jim/Desktop/openworld")
ARC_VENV = ("/private/tmp/claude-501/-Users-jim-Desktop-openworld/"
            "71e8c8de-fcca-4c0d-b13e-d3aae6071546/scratchpad/arcv/bin/python")
RUNNER = str(ROOT / "scripts" / "run_arc_agent_sandbox.sh")
BANKER = str(ROOT / "scripts" / "autobank_sourcefree.py")
ARCH = ROOT / "experiments" / "results" / "arc3_fullgame_sourcefree.json"
LOGDIR = ROOT / "scratch_arc"

GAMES = ["ar25", "bp35", "cd82", "cn04", "dc22", "ft09", "g50t", "ka59", "lf52", "lp85", "ls20", "m0r0",
         "r11l", "re86", "s5i5", "sb26", "sc25", "sk48", "sp80", "su15", "tn36", "tr87", "tu93", "vc33",
         "wa30"]

POOL = 4               # concurrent claude -p agents
PER_AGENT_S = 2700     # 45 min wall-clock budget per agent per round
ROUNDS = 4             # best-keeper deepening passes


def log(m):
    print(f"[sweep {time.strftime('%H:%M:%S')}] {m}", flush=True)


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
        r = subprocess.run([ARC_VENV, BANKER], cwd=str(ROOT), capture_output=True, text=True, timeout=1200)
        for ln in (r.stdout or "").splitlines():
            if "[sf-bank]" in ln:
                log(ln.strip())
    except Exception as ex:
        log(f"bank error: {ex}")


def launch(g):
    out = open(LOGDIR / f"sb_{g}_sweep.out", "a")
    return subprocess.Popen(["bash", RUNNER, g], cwd=str(ROOT),
                            stdout=out, stderr=subprocess.STDOUT,
                            start_new_session=True)             # own process group -> killable as a tree


def kill_tree(p):
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
    except Exception:
        pass


def commit(round_idx):
    a = json.loads(ARCH.read_text()) if ARCH.exists() else {}
    msg = (f"Source-free sweep round {round_idx}: {a.get('n_full_games', 0)} full, "
           f"{a.get('total_levels', 0)}/{a.get('total_possible', 0)} levels "
           f"(audit-clean + OpenWorld-World-verified).\n\n"
           "Every banked solve: process-isolated source-free agent, replay-verified on the real engine, "
           "and round-tripped through an OpenWorld World (FunctionTransition over the discovered "
           "masked-frame graph; reward=levels). No push (human reviews).\n\n"
           "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>")
    subprocess.run(["git", "-C", str(ROOT), "add",
                    str(ARCH), "scripts/autobank_sourcefree.py", "scripts/sweep_sourcefree.py"], check=False)
    subprocess.run(["git", "-C", str(ROOT), "commit", "-q", "-m", msg], check=False)
    log(f"committed round {round_idx} (local, not pushed)")


def main():
    log(f"START source-free sweep: {len(GAMES)} games, pool={POOL}, "
        f"per-agent={PER_AGENT_S}s, rounds={ROUNDS}")
    for r in range(ROUNDS):
        queue = [g for g in GAMES if not fully_solved(g)]
        log(f"round {r}: {len(queue)} games unsolved -> {queue}")
        if not queue:
            log("all games fully solved; stopping early")
            break
        running = {}       # game -> (proc, start_time)
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
                bank()         # cheap: only games whose depth increased are re-verified
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
