"""CLEAN protocol suite runner: agents receive only the BALROG-served observation text
and the done flag. Closed loop, replanned every step, pure code."""
import json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fable_tw.harness import TextWorldFactory, TASKS, NUM_EPISODES_OFFICIAL
from fable_tw.cleanagents import make_agent

MODE = sys.argv[1] if len(sys.argv) > 1 else "official"  # official | full
ONLY = sys.argv[2] if len(sys.argv) > 2 else None
OUT = os.path.join("results", "clean", MODE)
os.makedirs(OUT, exist_ok=True)
REPORT = "FABLE_TEXTWORLD_REPORT.md"


def log(msg):
    line = f"- `{time.strftime('%H:%M:%S')} {msg}`"
    print(line, flush=True)
    with open(REPORT, "a") as f:
        f.write(line + "\n")


def run_episode(env, gamefile, task, ep_idx):
    t0 = time.time()
    obs = env.reset()
    agent = make_agent(task)
    rec = {"task": task, "episode": ep_idx, "protocol": "clean", "game": os.path.basename(gamefile),
           "trace": [], "steps_executed": 0}
    done = False
    info = None
    text = obs["text"]["long_term_context"]
    for step in range(200):  # env truncates at 80 anyway
        cmd = agent.act(text)
        o, r, done, info = env.step(cmd)
        rec["steps_executed"] += 1
        text = o["text"]["long_term_context"]
        rec["trace"].append(cmd)
        if done:
            break
    rec["done"] = done
    rec["won"] = bool(info["won"]) if info else False
    rec["score"] = info["score"] if info else 0
    rec["max_score"] = info["max_score"] if info else None
    rec["progression"] = env.get_stats()["progression"]
    rec["wall_s"] = round(time.time() - t0, 3)
    return rec


def main():
    factory = TextWorldFactory()
    all_results = {}
    tasks = [t for t in TASKS if ONLY is None or t == ONLY]
    for task in tasks:
        episodes = []
        n = NUM_EPISODES_OFFICIAL if MODE == "official" else 25
        for ep in range(n):
            if MODE == "official":
                env, gamefile = factory.get_env(task)
            else:
                env, gamefile = factory.get_env(task, seed=ep)
            rec = run_episode(env, gamefile, task, ep)
            env.close()
            episodes.append(rec)
            d = os.path.join(OUT, task)
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, f"ep_{ep:02d}.json"), "w") as f:
                json.dump(rec, f, indent=1)
            log(f"[clean/{MODE}] {task} ep{ep} game={rec['game']} -> prog={rec['progression']:.3f} won={rec['won']} "
                f"score={rec['score']}/{rec['max_score']} steps={rec['steps_executed']} {rec['wall_s']}s")
        mean = sum(e["progression"] for e in episodes) / len(episodes)
        all_results[task] = {"episodes": len(episodes), "mean_progression": mean}
        log(f"[clean/{MODE}] {task} TASK MEAN over {len(episodes)} eps: {mean*100:.1f}%")
    if len(tasks) == 3:
        overall = sum(v["mean_progression"] for v in all_results.values()) / 3
        all_results["_overall"] = overall
        log(f"[clean/{MODE}] OVERALL TextWorld score: {overall*100:.2f}% (SOTA 75.7)")
    with open(os.path.join(OUT, f"summary{('_'+ONLY) if ONLY else ''}.json"), "w") as f:
        json.dump(all_results, f, indent=1)


if __name__ == "__main__":
    main()
