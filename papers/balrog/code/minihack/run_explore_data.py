"""Exploration dataset for the source-blind induction leg.

Per task, 3 episodes of deliberately diverse behavior, logged as full
transitions under results/transitions/exploration/:
  policy 'random'  : uniform random legal actions
  policy 'sweep'   : cycles through the whole action list in order
                     (systematic action probing incl. no-ops and prompts)
  policy 'drunkard': random walk biased to movement, occasional non-move
                     actions; bumps walls, wanders into monsters, etc.

These cover dynamics optimal play never exercises (wall bumps, futile
kicks/open/search, prompt states, combat with weak roles, lava deaths).
"""

import random

import numpy as np

import mh_harness as H
from transitions import TransitionLogger
from run_suite import log


def policy_random(space, t, rng):
    return rng.choice(space)


def policy_sweep(space, t, rng):
    return space[t % len(space)]


def policy_drunkard(space, t, rng):
    moves = [a for a in space if a in ("north", "south", "east", "west",
                                       "northeast", "southeast", "southwest",
                                       "northwest")]
    if rng.random() < 0.8 and moves:
        return rng.choice(moves)
    return rng.choice(space)


POLICIES = [("random", policy_random), ("sweep", policy_sweep),
            ("drunkard", policy_drunkard)]


def main():
    for ti, task in enumerate(H.MINIHACK_TASKS):
        for pi, (pname, pol) in enumerate(POLICIES):
            seed = 9000 + ti * 100 + pi
            rng = random.Random(seed)
            random.seed(seed)
            np.random.seed(seed)
            env = H.make_env(task)
            obs, info = env.reset(seed=seed)
            space = list(env.env.language_action_space)
            tlog = TransitionLogger(task, pi, seed, "exploration", pname,
                                    space, "exploration")
            tlog.log_reset(obs)
            done = False
            t = 0
            while not done and t < 105:
                a = pol(space, t, rng)
                obs, r, term, trunc, info = env.step(a)
                done = term or trunc
                tlog.log_step(a, obs, r, done, info)
                t += 1
            fn = tlog.close()
            env.close()
            log(f"[explore] {task} {pname} seed={seed}: {t} steps -> {fn}")


if __name__ == "__main__":
    main()
