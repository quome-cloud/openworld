"""BALROG-Crafter clean-protocol suite runner.

Scored protocol (mirrors BALROG evaluator, balrog/evaluator.py + config.yaml):
  - crafter.Env(area=(64,64), view=(9,9), size=(256,256), reward=True) +
    CrafterLanguageWrapper(max_episode_steps=2000, unique_items=True,
    precise_location=False) — the wrapper's reset() performs one Noop step,
    then the agent acts for up to 2000 steps; episode ends early on death.
  - per-episode progression = unlocked achievements / 22 (wrapper
    get_stats()); suite score = mean over episodes * 100.
  - 10 episodes (config eval.num_episodes.crafter).

CLEAN BOUNDARY: the agent consumes ONLY what the wrapper serves —
obs["text"]["long_term_context"] and obs["text"]["short_term_context"].
The env object is touched by the agent code through reset()/step() alone.
Everything else in this file (get_stats, info dict, obs["image"]) is used
strictly harness-side for scoring, logging and rendering, exactly as
BALROG's own evaluator uses it. See the source-leak audit in the report.

Seeding: BALROG's evaluator seeds via reset(seed=...), which for Crafter is
a silent no-op (deprecated gym.Wrapper.seed never reaches crafter.Env._seed;
BALROG leaves the constructor seed null => random worlds). We construct
crafter.Env(seed=S) per episode for reproducibility and record S. Note:
even seeded, crafter is only reproducible up to Python set-iteration order
of live objects in its chunk-balancing code (env-internal).

Conditions:
  A — memoryless: fresh agent per episode (leaderboard-comparable).
  B — memory: policy parameters derived from a cross-episode ledger built
      ONLY from clean episode results (memory.py; provenance recorded).

Transition logs: every episode writes results/transitions/<name>.jsonl.gz
with (obs_t, action, obs_t+1 fields, reward, done, info-as-served incl. the
semantic map) for the source-blind induction leg. Videos: every episode
writes an .mp4 with a per-frame overlay strip (step, subgoal, action,
achievements) rendered from the served obs["image"].
"""

import base64
import gzip
import json
import os
import sys
import time
import warnings

warnings.simplefilter('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import crafter  # noqa: E402
from balrog_text_env import CrafterLanguageWrapper, ACTIONS  # noqa: E402
from belief import TextBelief  # noqa: E402
from brain import Brain  # noqa: E402
import memory as memmod  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, 'results')
SEEDS = list(range(9001, 9011))     # official suite seeds (fresh; the dev
                                    # seeds 7001-7010 were used for tuning)
MAX_STEPS = 2000

CRAFTER_TO_BALROG = {
    'noop': 'Noop', 'move_left': 'Move West', 'move_right': 'Move East',
    'move_up': 'Move North', 'move_down': 'Move South', 'do': 'Do',
    'sleep': 'Sleep', 'place_stone': 'Place Stone',
    'place_table': 'Place Table', 'place_furnace': 'Place Furnace',
    'place_plant': 'Place Plant', 'make_wood_pickaxe': 'Make Wood Pickaxe',
    'make_stone_pickaxe': 'Make Stone Pickaxe',
    'make_iron_pickaxe': 'Make Iron Pickaxe',
    'make_wood_sword': 'Make Wood Sword',
    'make_stone_sword': 'Make Stone Sword',
    'make_iron_sword': 'Make Iron Sword'}


def make_env(seed):
    env = crafter.Env(area=(64, 64), view=(9, 9), size=(256, 256),
                      reward=True, seed=seed)
    return CrafterLanguageWrapper(env, '', max_episode_steps=MAX_STEPS)


def classify_death(belief, brain, died, step):
    if not died:
        return None
    phase = step % 300
    night = 148 <= phase <= 272
    z = belief.mob_dist('zombie')
    s = belief.mob_dist('skeleton')
    a = belief.mob_dist('arrow')
    food = belief.inventory.get('food', 9)
    drink = belief.inventory.get('drink', 9)
    if z <= 2 and night:
        return 'zombie_night' + ('' if brain.home else '_no_home')
    if z <= 2:
        return 'zombie_day'
    if s <= 5 or a <= 3:
        return 'skeleton_arrows'
    if food == 0 or drink == 0:
        return 'starvation'
    return 'unknown'


class TransitionLogger:
    def __init__(self, path, meta):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = gzip.open(path, 'wt')
        self.f.write(json.dumps(dict(meta=meta)) + '\n')

    def log(self, t, action, reward, done, obs, info):
        sem = info.get('semantic')
        rec = dict(
            t=t, action=action, reward=float(reward), done=bool(done),
            text=obs['text'],
            info=dict(
                inventory=dict(info['inventory']),
                achievements={k: int(v) for k, v in
                              info['achievements'].items() if v},
                discount=float(info['discount']),
                player_pos=[int(v) for v in info['player_pos']],
                semantic_b64=base64.b64encode(
                    np.asarray(sem, dtype=np.uint8).tobytes()).decode()
                if sem is not None else None))
        self.f.write(json.dumps(rec) + '\n')

    def close(self):
        self.f.close()


class VideoLogger:
    def __init__(self, path, fps=12):
        import imageio
        from PIL import ImageFont
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.w = imageio.get_writer(path, fps=fps, macro_block_size=8)
        self.font = ImageFont.load_default()

    def log(self, img, step, goal, action, n_ach):
        from PIL import Image, ImageDraw
        frame = img.convert('RGB')
        strip = Image.new('RGB', (frame.width, 40), (16, 16, 16))
        d = ImageDraw.Draw(strip)
        phase = step % 300
        tod = 'NIGHT' if 148 <= phase <= 272 else 'day'
        d.text((4, 3), f'step {step:4d}  {tod}  ach {n_ach}/22',
               fill=(240, 240, 240), font=self.font)
        d.text((4, 21), f'{goal} | {action}',
               fill=(160, 220, 160), font=self.font)
        canvas = Image.new('RGB', (frame.width, frame.height + 40))
        canvas.paste(frame, (0, 0))
        canvas.paste(strip, (0, frame.height))
        self.w.append_data(np.asarray(canvas))

    def close(self):
        self.w.close()


def run_episode(seed, condition='A', tag='', params=None, video=True,
                transitions=True):
    name = f'{condition}_seed{seed}{tag}'
    wrapper = make_env(seed)
    obs = wrapper.reset()
    belief = TextBelief()
    brain = Brain(belief, privileged=False, params=params)
    belief.observe(obs['text']['long_term_context'],
                   obs['text']['short_term_context'], None)
    tlog = TransitionLogger(
        os.path.join(RESULTS, 'transitions', name + '.jsonl.gz'),
        dict(seed=seed, condition=condition, protocol='clean_text',
             actions=ACTIONS, params=params or {},
             note='obs text = agent input; info/semantic logged harness-side '
                  'for the induction dataset')) if transitions else None
    vlog = VideoLogger(os.path.join(RESULTS, 'animations', name + '.mp4')) \
        if video else None
    t0 = time.time()
    unlock_steps = {}
    goal_counts = {}
    action_log = []
    steps = 0
    died = False
    info = {}
    for step in range(MAX_STEPS):
        action, goal = brain.act()
        obs, reward, done, info = wrapper.step(CRAFTER_TO_BALROG[action])
        steps += 1
        belief.observe(obs['text']['long_term_context'],
                       obs['text']['short_term_context'], action)
        goal_counts[goal] = goal_counts.get(goal, 0) + 1
        action_log.append(action)
        n_ach = sum(1 for v in info['achievements'].values() if v > 0)
        for aname, cnt in info['achievements'].items():
            if cnt > 0 and aname not in unlock_steps:
                unlock_steps[aname] = steps
        if tlog:
            tlog.log(steps, action, reward, done, obs, info)
        if vlog:
            vlog.log(obs['image'], steps, goal, action, n_ach)
        if done:
            died = info['discount'] == 0.0
            break
    if tlog:
        tlog.close()
    if vlog:
        vlog.close()
    stats = wrapper.get_stats()
    ach = {k: v for k, v in (stats['achievements'] or {}).items() if v > 0}
    result = dict(
        seed=seed, condition=condition, protocol='clean_text', steps=steps,
        died=died, death_cause=classify_death(belief, brain, died, steps),
        score=stats['score'], progression=stats['progression'],
        achievements=sorted(ach.keys()), unlock_steps=unlock_steps,
        wall_time=round(time.time() - t0, 1), goal_counts=goal_counts,
        action_log=action_log, params=params or {},
        diagnostics=dict(
            ambiguous_steps=belief.ambiguous_steps,
            relocalizations=belief.relocalizations,
            believed_ach=sorted(belief.ach),
            map_known_cells=int((belief.map != None).sum()),  # noqa: E711
            home_established=brain.home is not None))
    return result


def summarize(results, label):
    scores = [r['progression'] for r in results]
    mean = sum(scores) / len(scores)
    se = (sum((s - mean) ** 2 for s in scores) / max(1, len(scores) - 1)
          / len(scores)) ** 0.5
    return dict(label=label, episodes=len(results),
                progression_pct=round(100 * mean, 2),
                standard_error_pct=round(100 * se, 2),
                per_episode=[round(100 * s, 2) for s in scores],
                mean_achievements=round(
                    sum(r['score'] for r in results) / len(results), 2),
                deaths=sum(1 for r in results if r['died']),
                death_causes={r['seed']: r['death_cause'] for r in results})


def run_condition(condition, seeds=SEEDS):
    outdir = os.path.join(RESULTS, f'condition_{condition}')
    os.makedirs(outdir, exist_ok=True)
    results = []
    ledger = memmod.load_ledger() if condition == 'B' else None
    for i, seed in enumerate(seeds):
        path = os.path.join(outdir, f'ep_{i:02d}_seed{seed}.json')
        if os.path.exists(path):
            results.append(json.load(open(path)))
            print(f'[skip] {condition} ep{i} seed={seed}')
            continue
        params, fired = (memmod.derive_params(ledger) if condition == 'B'
                         else (None, []))
        r = run_episode(seed, condition=condition, params=params)
        r['memory_entries_fired'] = fired
        json.dump(r, open(path, 'w'), indent=1)
        results.append(r)
        if condition == 'B':
            memmod.record_episode(ledger, r, path)
            memmod.save_ledger(ledger)
        print(f'{time.strftime("%H:%M:%S")} {condition} ep{i} seed={seed} -> '
              f'{r["score"]:.0f}/22 ({100*r["progression"]:.1f}%) '
              f'steps={r["steps"]} died={r["died"]} '
              f'cause={r["death_cause"]} fired={[f["rule"] for f in fired]} '
              f'{r["wall_time"]}s')
        sys.stdout.flush()
    summary = summarize(results, f'condition_{condition}')
    json.dump(summary, open(
        os.path.join(RESULTS, f'summary_{condition}.json'), 'w'), indent=1)
    print(json.dumps({k: v for k, v in summary.items()
                      if k != 'death_causes'}, indent=1))
    return summary


if __name__ == '__main__':
    cond = sys.argv[1] if len(sys.argv) > 1 else 'A'
    seeds = [int(s) for s in sys.argv[2].split(',')] if len(sys.argv) > 2 \
        else SEEDS
    run_condition(cond, seeds)
