"""Cartpole-swingup as a verified OpenWorld world -- the continuous-control benchmark.

This expresses the *standard* cartpole dynamics (the Florian-corrected equations of
motion used by OpenAI Gym / Gymnasium `CartPole`, here driven by a continuous force
for the swing-up task) as explicit, verified code over symbolic state. The SAME
world is then run by (a) OpenWorld's verified model with planning (CEM-MPC, 0 data,
here) and (b) learned/perceptual world models on GPU -- DreamerV3 (learns the model
from pixels) and V-JEPA-2 (frozen video features + learned latent dynamics + MPC).

Unlike the gridworld, continuous physics is the learned/perceptual models' home
turf, so this is a *fairer* head-to-head: if OpenWorld's 0-shot verified model still
matches them, it wins on their terrain.

The dynamics are an INDEPENDENT, pre-existing spec (Gym's documented equations), not
something we invent -- so "verified, exact" is anchored to an external reference
(validated to <1e-9 against gymnasium's CartPole on the GPU instance, mirroring the
MiniGrid bit-for-bit check; offline we validate the CodeTransition against a separate
reference + vectorized executor of the same equations).

State is symbolic: x (cart pos), x_dot, theta (pole angle from UPRIGHT; theta=0 is
up, theta=pi is hanging down), theta_dot. Action: a continuous force on the cart in
[-FORCE_MAG, FORCE_MAG], carried in action params {"force": F}.
"""

import math

from openworld import World, CodeTransition

# --- standard Gym/Florian cartpole constants ---
GRAVITY = 9.8
MASS_CART = 1.0
MASS_POLE = 0.1
TOTAL_MASS = MASS_CART + MASS_POLE      # 1.1
LENGTH = 0.5                            # half the pole length
POLEMASS_LENGTH = MASS_POLE * LENGTH    # 0.05
FORCE_MAG = 10.0
TAU = 0.02                              # integration timestep (50 Hz)

X_THRESHOLD = 2.4                       # rail limit (cart must stay within)
ANGLE_SUCCESS = 0.2                     # rad from upright counted as "balanced" (~11.5 deg)

CARTPOLE_ACTIONS = ["push"]            # one continuous action: force on the cart
CARTPOLE_INITIAL = {"x": 0.0, "x_dot": 0.0, "theta": math.pi, "theta_dot": 0.0}  # hanging down

CARTPOLE_RULES = [
    "Continuous cartpole-swingup with the standard Gym/Florian equations of motion.",
    "theta is the pole angle from UPRIGHT (0=up, pi=hanging down); state is "
    "(x, x_dot, theta, theta_dot), all continuous.",
    "Action 'push' applies a continuous force F (clipped to [-10, 10] N) to the cart.",
    "Dynamics integrate one TAU=0.02s Euler step of the cart-pole ODE.",
    "Goal: swing the pole up and balance it upright (|angle|<0.2 rad) with the cart "
    "on the rail (|x|<2.4) and hold it.",
]

# Self-contained verified transition: pure (state, action) -> state, math only.
CARTPOLE_CODE = '''
def transition(state, action):
    s = dict(state)
    g, mc, mp, mt, l, pml, tau, fmag = 9.8, 1.0, 0.1, 1.1, 0.5, 0.05, 0.02, 10.0
    F = float(action["params"]["force"])
    if F > fmag:
        F = fmag
    elif F < -fmag:
        F = -fmag
    x, xd, th, thd = s["x"], s["x_dot"], s["theta"], s["theta_dot"]
    ct, st = math.cos(th), math.sin(th)
    temp = (F + pml * thd * thd * st) / mt
    thacc = (g * st - ct * temp) / (l * (4.0 / 3.0 - mp * ct * ct / mt))
    xacc = temp - pml * thacc * ct / mt
    # Euler integration (Gym "euler" order)
    s["x"] = x + tau * xd
    s["x_dot"] = xd + tau * xacc
    s["theta"] = th + tau * thd
    s["theta_dot"] = thd + tau * thacc
    return s
'''


def step_ref(state, force):
    """Independent plain-Python reference of the SAME equations (no sandbox)."""
    g, mc, mp, mt, l, pml, tau, fmag = 9.8, 1.0, 0.1, 1.1, 0.5, 0.05, 0.02, 10.0
    F = max(-fmag, min(fmag, float(force)))
    x, xd, th, thd = state["x"], state["x_dot"], state["theta"], state["theta_dot"]
    ct, st = math.cos(th), math.sin(th)
    temp = (F + pml * thd * thd * st) / mt
    thacc = (g * st - ct * temp) / (l * (4.0 / 3.0 - mp * ct * ct / mt))
    xacc = temp - pml * thacc * ct / mt
    return {"x": x + tau * xd, "x_dot": xd + tau * xacc,
            "theta": th + tau * thd, "theta_dot": thd + tau * thacc}


def step_batch(X, F):
    """Vectorized executor of the verified dynamics, for fast CEM-MPC planning.

    X: (N,4) array [x, x_dot, theta, theta_dot]; F: (N,) forces.
    Validated bit-identical (<1e-9) to the canonical CodeTransition in e65.
    Returns (N,4) next states.
    """
    import numpy as np
    g, mp, mt, l, pml, tau, fmag = 9.8, 0.1, 1.1, 0.5, 0.05, 0.02, 10.0
    F = np.clip(F, -fmag, fmag)
    x, xd, th, thd = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    ct, st = np.cos(th), np.sin(th)
    temp = (F + pml * thd * thd * st) / mt
    thacc = (g * st - ct * temp) / (l * (4.0 / 3.0 - mp * ct * ct / mt))
    xacc = temp - pml * thacc * ct / mt
    out = np.empty_like(X)
    out[:, 0] = x + tau * xd
    out[:, 1] = xd + tau * xacc
    out[:, 2] = th + tau * thd
    out[:, 3] = thd + tau * thacc
    return out


def wrap_angle(theta):
    """Map an angle to (-pi, pi]; |wrap(theta)| is distance from upright."""
    return (theta + math.pi) % (2 * math.pi) - math.pi


def step_reward(x, theta, force):
    """Dense swing-up reward in [0,1]: upright * cart-centered, tiny control cost.

    upright = (1+cos theta)/2 (1 up, 0 down); centered tolerates |x| up to the rail.
    """
    upright = (1.0 + math.cos(theta)) / 2.0
    centered = max(0.0, 1.0 - (x / X_THRESHOLD) ** 2)
    return upright * centered - 1e-3 * (force / FORCE_MAG) ** 2


def is_balanced(state):
    """A single state is upright-and-centered."""
    return abs(wrap_angle(state["theta"])) < ANGLE_SUCCESS and abs(state["x"]) < X_THRESHOLD


def swingup_success(states, hold=50):
    """Episode succeeds if the pole is balanced+centered for the final `hold` steps."""
    if len(states) < hold:
        return False
    return all(is_balanced(s) for s in states[-hold:])


def render(state, size=64):
    """Minimal numpy RGB render (cart as a bar, pole as a line) for the GPU models."""
    import numpy as np
    img = np.full((size, size, 3), 235, dtype=np.uint8)            # light gray bg
    cx = int((state["x"] / (X_THRESHOLD * 1.1) * 0.5 + 0.5) * (size - 1))
    cy = int(size * 0.65)
    img[cy - 1:cy + 2, max(0, cx - 6):min(size, cx + 7)] = (40, 40, 40)   # cart
    L = size * 0.30
    th = state["theta"]
    tx = int(cx + L * math.sin(th))
    ty = int(cy - L * math.cos(th))
    n = max(abs(tx - cx), abs(ty - cy), 1)                          # rasterize pole line
    for i in range(n + 1):
        px = int(cx + (tx - cx) * i / n)
        py = int(cy + (ty - cy) * i / n)
        if 0 <= px < size and 0 <= py < size:
            img[py, px] = (200, 60, 40)
    return img


def build_cartpole_world():
    return World(name="cartpole-swingup",
                 description="Continuous cartpole-swingup as a verified symbolic world.",
                 initial_state=dict(CARTPOLE_INITIAL),
                 actions=list(CARTPOLE_ACTIONS),
                 rules=list(CARTPOLE_RULES),
                 transition=CodeTransition(CARTPOLE_CODE))


if __name__ == "__main__":
    from openworld.state import Action
    w = build_cartpole_world()
    s = dict(CARTPOLE_INITIAL)
    # apply a constant push and confirm the CodeTransition and reference agree
    for _ in range(5):
        a = Action("push", params={"force": 3.0})
        s_code = dict(w.transition.step(s, a))
        s_ref = step_ref(s, 3.0)
        assert max(abs(s_code[k] - s_ref[k]) for k in s_ref) < 1e-12, (s_code, s_ref)
        s = s_code
    print("ok: cartpole CodeTransition == reference; state after 5 pushes:", s)
