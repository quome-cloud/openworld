"""PRIVILEGED protocol suite runner.

Test-time channels: the game's .json spec (initial state + quest) for planning, and the
BALROG-served observation channel for execution. No walkthrough/policy_commands, no
facts/admissible_commands at test time. Plans executed closed-loop-lite: each feedback is
checked against error markers; any hit is recorded as a misprediction (expected: zero).
"""
import json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fable_tw.harness import TextWorldFactory, TASKS, NUM_EPISODES_OFFICIAL
from fable_tw.worldmodel import World, plan, verify_plan

ERROR_MARKERS = [
    "You can't see any such thing",
    "You have to",
    "That's already",
    "I don't understand",
    "You can't go that way",
    "is locked",
    "You need to take",
    "That seems to fit the lock",  # unlock with wrong key variants
    "not open",
    "You can't reach",
]

MODE = sys.argv[1] if len(sys.argv) > 1 else "official"  # official | full
OUT = os.path.join("results", "privileged", MODE)
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
    world = World(gamefile.rsplit(".", 1)[0] + ".json")
    t_plan0 = time.time()
    cmds = plan(world, task)
    plan_s = time.time() - t_plan0
    ok, err, steps = verify_plan(world, task, cmds)
    rec = {
        "task": task, "episode": ep_idx, "protocol": "privileged", "game": os.path.basename(gamefile),
        "plan_len": len(cmds), "plan_time_s": round(plan_s, 3), "model_verified": ok, "verify_err": err,
        "commands": cmds, "mispredictions": [], "steps_executed": 0,
    }
    done = False
    info = None
    for c in cmds:
        o, r, done, info = env.step(c)
        rec["steps_executed"] += 1
        text = o["text"]["long_term_context"]
        if any(m in text for m in ERROR_MARKERS):
            rec["mispredictions"].append({"cmd": c, "obs": text[:300]})
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
    for task in TASKS:
        episodes = []
        n = NUM_EPISODES_OFFICIAL if MODE == "official" else 25
        for ep in range(n):
            if MODE == "official":
                env, gamefile = factory.get_env(task)  # official cycling
            else:
                env, gamefile = factory.get_env(task, seed=ep)
            rec = run_episode(env, gamefile, task, ep)
            env.close()
            episodes.append(rec)
            d = os.path.join(OUT, task)
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, f"ep_{ep:02d}.json"), "w") as f:
                json.dump(rec, f, indent=1)
            log(f"[priv/{MODE}] {task} ep{ep} game={rec['game']} -> prog={rec['progression']:.3f} won={rec['won']} "
                f"score={rec['score']}/{rec['max_score']} steps={rec['steps_executed']}/{rec['plan_len']} "
                f"mispred={len(rec['mispredictions'])} {rec['wall_s']}s")
        mean = sum(e["progression"] for e in episodes) / len(episodes)
        all_results[task] = {"episodes": len(episodes), "mean_progression": mean}
        log(f"[priv/{MODE}] {task} TASK MEAN over {len(episodes)} eps: {mean*100:.1f}%")
    overall = sum(v["mean_progression"] for v in all_results.values()) / len(all_results)
    all_results["_overall"] = overall
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(all_results, f, indent=1)
    log(f"[priv/{MODE}] OVERALL TextWorld score: {overall*100:.2f}% (SOTA 75.7)")


if __name__ == "__main__":
    main()
