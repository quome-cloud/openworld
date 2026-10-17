"""Memory-across-attempts experiment ("trap-chest").

K passes over the full 25-game set per task, CLEAN interface, with a persistent
per-game ledger built MECHANICALLY from the agent's own clean episode logs:
  - an episode that ends in a loss (before the step cap) immediately after `take X`
    ledgers X as a fatal take for that game;
  - an episode that ends won immediately after `take X` ledgers X as the winning take.
Every ledger entry cites the episode file + step it derives from. The ledger starts
EMPTY and the code contains no game-specific constants; entries derive only from
clean-condition episode logs produced by this same script.

Pass 1 runs with an empty ledger (memoryless baseline); passes 2+ consume it.
BALROG's own protocol scores episodes independently and cannot reward this; the
experiment measures what that protocol leaves on the table.

Full transitions (obs, cmd, obs', reward, done) are logged for every episode under
results/transitions/ for the later source-blind induction leg.
"""
import json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fable_tw.harness import TextWorldFactory, TASKS, MAX_EPISODE_STEPS
from fable_tw.cleanagents import make_agent

K = int(sys.argv[1]) if len(sys.argv) > 1 else 3
OUT = os.path.join("results", "memory")
TRANS = os.path.join("results", "transitions")
LEDGER_PATH = os.path.join("memory", "ledger.json")
REPORT = "FABLE_TEXTWORLD_REPORT.md"
os.makedirs("memory", exist_ok=True)
os.makedirs(OUT, exist_ok=True)
os.makedirs(TRANS, exist_ok=True)


def log(msg):
    line = f"- `{time.strftime('%H:%M:%S')} {msg}`"
    print(line, flush=True)
    with open(REPORT, "a") as f:
        f.write(line + "\n")


def load_ledger():
    if os.path.exists(LEDGER_PATH):
        d = json.load(open(LEDGER_PATH))
    else:
        d = {}
    return d


def save_ledger(ledger):
    with open(LEDGER_PATH, "w") as f:
        json.dump(ledger, f, indent=1)


def memory_view(ledger, game_id):
    """Generic agent-facing view of the ledger for one game."""
    avoid, target = set(), None
    for e in ledger.get(game_id, []):
        if e["type"] == "fatal_take":
            avoid.add(e["object"])
        elif e["type"] == "winning_take":
            target = e["object"]
    return {"avoid": avoid, "target": target}


def learn(ledger, game_id, transitions, episode_file, won, steps):
    """Mechanical fact extraction from this game's own clean episode log."""
    added = 0
    for i, t in enumerate(transitions):
        if not t["done"] or not t["cmd"].startswith("take "):
            continue
        obj = t["cmd"][5:].split(" from ")[0]
        if won:
            kind = "winning_take"
        elif steps < MAX_EPISODE_STEPS:
            kind = "fatal_take"  # ended lost before the cap: the take was fatal
        else:
            continue  # truncation at the cap tells us nothing about the last take
        entry = {"type": kind, "object": obj,
                 "source": {"episode_file": episode_file, "step": i}}
        if entry not in ledger.setdefault(game_id, []):
            ledger[game_id].append(entry)
            added += 1
    return added


def run_episode(env, gamefile, task, ep_idx, pass_no, memory):
    t0 = time.time()
    obs = env.reset()
    agent = make_agent(task, memory=memory)
    text = obs["text"]["long_term_context"]
    transitions = []
    done, info = False, None
    for _ in range(200):
        cmd = agent.act(text)
        o, r, done, info = env.step(cmd)
        ntext = o["text"]["long_term_context"]
        transitions.append({"obs": text, "cmd": cmd, "obs_next": ntext,
                            "reward": r, "done": done})
        text = ntext
        if done:
            break
    rec = {"task": task, "episode": ep_idx, "pass": pass_no, "protocol": "clean+memory",
           "game": os.path.basename(gamefile),
           "memory_in": {"avoid": sorted(memory.get("avoid", [])), "target": memory.get("target")},
           "steps_executed": len(transitions), "done": done,
           "won": bool(info["won"]) if info else False,
           "score": info["score"] if info else 0,
           "max_score": info["max_score"] if info else None,
           "progression": env.get_stats()["progression"],
           "wall_s": round(time.time() - t0, 3)}
    return rec, transitions


def main():
    if os.path.exists(LEDGER_PATH):
        os.remove(LEDGER_PATH)  # operator mandate: the ledger starts EMPTY
    ledger = {}
    factory = TextWorldFactory()
    first_solve = {}   # (task, game) -> pass number
    per_pass = {}
    for p in range(1, K + 1):
        per_pass[p] = {}
        for task in TASKS:
            eps = []
            for ep in range(25):
                env, gamefile = factory.get_env(task, seed=ep)
                gid = os.path.basename(gamefile)
                mem = memory_view(ledger, gid)
                rec, transitions = run_episode(env, gamefile, task, ep, p, mem)
                env.close()
                d = os.path.join(OUT, f"pass{p}", task)
                os.makedirs(d, exist_ok=True)
                ep_file = os.path.join(d, f"ep_{ep:02d}.json")
                with open(ep_file, "w") as f:
                    json.dump(rec, f, indent=1)
                td = os.path.join(TRANS, f"pass{p}", task)
                os.makedirs(td, exist_ok=True)
                with open(os.path.join(td, gid.rsplit(".", 1)[0] + ".json"), "w") as f:
                    json.dump({"game": gid, "pass": p, "won": rec["won"],
                               "transitions": transitions}, f, indent=1)
                added = learn(ledger, gid, transitions, ep_file, rec["won"], rec["steps_executed"])
                save_ledger(ledger)
                if rec["won"] and (task, gid) not in first_solve:
                    first_solve[(task, gid)] = p
                eps.append(rec)
                if task == "treasure_hunter" or not rec["won"]:
                    log(f"[mem pass{p}] {task} {gid} -> prog={rec['progression']:.2f} won={rec['won']} "
                        f"steps={rec['steps_executed']} mem_avoid={len(rec['memory_in']['avoid'])} "
                        f"mem_target={'Y' if rec['memory_in']['target'] else '-'} ledger+={added}")
            mean = sum(e["progression"] for e in eps) / len(eps)
            per_pass[p][task] = mean
            log(f"[mem pass{p}] {task} MEAN: {mean*100:.1f}%")
        overall = sum(per_pass[p][t] for t in TASKS) / len(TASKS)
        per_pass[p]["_overall"] = overall
        log(f"[mem pass{p}] OVERALL: {overall*100:.2f}%")
    summary = {
        "passes": {str(p): per_pass[p] for p in per_pass},
        "attempts_to_first_solve": {f"{t}/{g}": p for (t, g), p in sorted(first_solve.items())},
        "unsolved_after_all_passes": [f"{t}/{os.path.basename(gf)}"
                                      for t in TASKS for gf in factory.gamefiles[t]
                                      if (t, os.path.basename(gf)) not in first_solve],
        "ledger_entries": sum(len(v) for v in ledger.values()),
        "ledger_games": len(ledger),
    }
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    log(f"[mem] DONE: ledger {summary['ledger_entries']} entries / {summary['ledger_games']} games; "
        f"unsolved after {K} passes: {summary['unsolved_after_all_passes']}")


if __name__ == "__main__":
    main()
