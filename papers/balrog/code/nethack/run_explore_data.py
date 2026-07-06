"""Labeled exploration dataset for the quarantined source-blind induction
leg: policies chosen to cover dynamics optimal play never touches (wall
bumps, futile commands, prompt states entered blind, deaths by misadventure).

3 policies x 4 episodes x <=1500 steps, dev-range seeds (300+), logged with
the same streaming transition format as scored runs.
"""

import random
import time

import numpy as np

import nh_harness as H
from nh_transitions import TransitionLogger

MOVES = ["north", "south", "east", "west", "northeast", "northwest",
         "southeast", "southwest"]
SAFE_CMDS = ["search", "look", "wait", "open", "kick", "pickup", "eat",
             "down", "up", "esc", "more", "inventory", "space"]
LETTERS = [chr(c) for c in range(ord("a"), ord("z") + 1)]


def policy_random(space, rng, step):
    return rng.choice(list(space))


def policy_drunkard(space, rng, step):
    if rng.random() < 0.85:
        return rng.choice(MOVES)
    return rng.choice(SAFE_CMDS)


def policy_sweep(space, rng, step):
    # cycles commands to trigger prompt states, answers with random letters
    seq = SAFE_CMDS + MOVES
    if rng.random() < 0.25:
        return rng.choice(LETTERS)
    return seq[step % len(seq)]


POLICIES = {"random": policy_random, "drunkard": policy_drunkard,
            "sweep": policy_sweep}


def main():
    for pname, pol in POLICIES.items():
        for ep in range(4):
            seed = 300 + ep * 10 + list(POLICIES).index(pname)
            rng = random.Random(seed)
            env = H.make_env()
            np.random.seed(seed)
            obs, info = env.reset(seed=seed)
            # NOTE (interface wart, disclosed in report): BALROG's NLE
            # action list includes double-digit strings ("05") that its own
            # wrapper cannot map to a key — stepping them raises ValueError.
            # Policies sample from the steppable subset.
            space = [a for a in env.env.language_action_space
                     if not (len(a) == 2 and a.isdigit())]
            tlog = TransitionLogger(f"explore_{pname}", ep, seed,
                                    "exploration", pname, space,
                                    "exploration")
            tlog.log_reset(obs)
            done, steps = False, 0
            t0 = time.time()
            while not done and steps < 1500:
                a = pol(space, rng, steps)
                if a not in space:
                    a = "search"
                try:
                    obs, r, term, trunc, info = env.step(a)
                except ValueError:
                    # unmappable action string served by the wrapper's own
                    # list; substitute a no-op-ish legal action
                    a = "search"
                    obs, r, term, trunc, info = env.step(a)
                tlog.log_step(a, obs, r, term or trunc, info)
                done = term or trunc
                steps += 1
            env.close()
            fn = tlog.close()
            print(f"{pname} ep{ep}: {steps} steps "
                  f"{time.time()-t0:.0f}s -> {fn}", flush=True)


if __name__ == "__main__":
    main()
