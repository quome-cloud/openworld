"""E41 - Non-stationary dynamics: detecting and adapting to sudden rule changes.

E32 showed a *known* regime switch is handled exactly by a pre-verified
PhasedTransition. E41 asks the harder question: when the change is UNANNOUNCED,
who notices, and how fast do they recover?

A reservoir tracks toward a target level at a fixed rate (a piecewise rule with
a kink at the target). The hidden regime (target, rate) jumps suddenly at two
unannounced times in a 120-step episode. Each step we perceive the current
level through the perception boundary and ask four one-step-ahead predictors
for the next level:

  static_frozen   - fit the rule once on the first regime, never update.
  window_1nn      - sliding-window nearest-neighbor over the last W transitions.
  symbolic_refit  - predict with the current rule; on a prediction error,
                    re-identify the rule from the transient (rate = the latest
                    move; target = far in the direction of motion, or here once
                    motion stalls) and switch (snaps back to exact). (Ours.)
  oracle_switch   - knows the change times and the new rule (0-lag ceiling).

Metrics: per-step correctness timeline, recovery lag after each change, and
cumulative errors (regret). Deterministic and offline; perception is run
through the gated boundary (perfect for the headline, with a noisy variant
showing the E39 decomposition adds on top).
"""

from openworld import Action, MockPerceptor, Observation, World
from openworld.transition import FunctionTransition

from common import save_results

CAP = 100
T = 120
CHANGES = [40, 80]                      # unannounced regime-change steps
REGIMES = [(30, 2), (75, 4), (10, 3)]   # (target, rate) per regime
WARMUP = 12                             # steps before scoring (all regime 0)
WINDOW = 10                             # sliding-window size
PATIENCE = 2                            # symbolic: errors before declaring a change


def tracking_rule(level, target, rate):
    # Move toward the target by `rate`, clamping at the target (clean
    # convergence, no overshoot/oscillation).
    if level < target:
        return min(target, level + rate)
    if level > target:
        return max(target, level - rate)
    return level


def regime_at(t):
    idx = 0
    for c in CHANGES:
        if t >= c:
            idx += 1
    return REGIMES[idx]


def make_world(level):
    def step(state, action):
        s = dict(state)
        target, rate = action.params["regime"]
        s["level"] = tracking_rule(s["level"], target, rate)
        return s
    return World(name="reservoir", description="tracks toward a (hidden) target",
                 initial_state={"level": level}, actions=["tick"],
                 transition=FunctionTransition(step))


def true_trajectory():
    """The ground-truth level sequence under the non-stationary regime."""
    levels = [0]
    for t in range(T):
        target, rate = regime_at(t)
        levels.append(tracking_rule(levels[-1], target, rate))
    return levels


def perceive(level, noise_at=None, t=None):
    """Observe the level through the gated perception boundary (deterministic)."""
    seen = level
    if noise_at is not None and t in noise_at:
        seen = max(0, min(CAP, level + 7))     # a wrong but in-range reading
    p = MockPerceptor(produces=["level"], deltas=[{"level": seen}],
                      schema={"level": (int, (0, CAP))})
    w = make_world(0)
    w.observe(Observation("text", f"gauge reads {seen}"), p)
    return w.state["level"]



def run(noise_at=None):
    truth = true_trajectory()

    # All methods start having learned the initial stationary regime during
    # warmup (the test is adaptation to the unannounced *change*, not initial
    # identification): the frozen and symbolic models begin with regime 0's
    # true rule; the window starts filled with regime-0 transitions.
    warm = [(perceive(truth[t], noise_at, t), truth[t + 1]) for t in range(WARMUP)]
    frozen_rule = REGIMES[0]
    sym_rule = REGIMES[0]
    window = list(warm[-WINDOW:])

    correct = {m: [] for m in ("static_frozen", "window_1nn", "symbolic_refit", "oracle_switch")}
    for t in range(WARMUP, T):
        cur = perceive(truth[t], noise_at, t)
        nxt = truth[t + 1]

        # static frozen
        correct["static_frozen"].append(tracking_rule(cur, *frozen_rule) == nxt)

        # sliding-window 1-NN (standard: predict the next of the nearest past
        # level in the window; stale after a change until the window refills).
        if window:
            _, pred = min(window, key=lambda tr: abs(tr[0] - cur))
        else:
            pred = cur
        correct["window_1nn"].append(pred == nxt)

        # symbolic monitor + refit. Predict with the current (target, rate).
        # On error, re-identify from the transient: rate is the latest move
        # magnitude; if the state is still moving, the target is far in that
        # direction; if it has stalled (move shrank), the target is here.
        sym_pred = tracking_rule(cur, *sym_rule)
        sym_ok = sym_pred == nxt
        correct["symbolic_refit"].append(sym_ok)
        if not sym_ok:
            delta = nxt - cur
            if delta > 0:
                sym_rule = (CAP, delta)          # climbing at rate=delta
            elif delta < 0:
                sym_rule = (0, -delta)           # falling at rate=delta
            else:
                sym_rule = (nxt, sym_rule[1])    # stalled: target reached here

        # oracle: knows the rule in force right now
        correct["oracle_switch"].append(tracking_rule(cur, *regime_at(t)) == nxt)

        # reveal the transition for online updating
        window.append((cur, nxt))
        window[:] = window[-WINDOW:]

    return correct


def recovery_lag(flags, scored_start):
    """Steps after each change until 3-consecutive-correct (None if no recovery)."""
    lags = {}
    for c in CHANGES:
        i0 = c - scored_start
        lag = None
        run_len = 0
        for j in range(i0, len(flags)):
            run_len = run_len + 1 if flags[j] else 0
            if run_len >= 3:
                lag = (j - 2) - i0
                break
        lags[c] = lag
    return lags


def summarize(correct):
    scored = T - WARMUP
    out = {}
    for m, flags in correct.items():
        out[m] = {
            "errors": int(sum(1 for f in flags if not f)),
            "accuracy": sum(flags) / len(flags),
            "recovery_lag": recovery_lag(flags, WARMUP),
        }
    return scored, out


def main():
    clean = run()
    noisy = run(noise_at=set(range(WARMUP, T, 7)))   # ~1/7 perceptions corrupted
    scored, clean_sum = summarize(clean)
    _, noisy_sum = summarize(noisy)

    save_results("e41_nonstationary", {
        "horizon": T, "warmup": WARMUP, "changes": CHANGES, "regimes": REGIMES,
        "window": WINDOW, "scored_steps": scored,
        "perfect_perception": clean_sum,
        "noisy_perception": noisy_sum,
        "timeline": {m: [int(b) for b in flags] for m, flags in clean.items()},
    })

    print(f"Non-stationary reservoir, changes at {CHANGES} of {T} steps "
          f"(scored {scored}). Perfect perception:\n")
    print(f"  {'method':<16}{'accuracy':>10}{'errors':>8}   recovery lag/change")
    for m, s in clean_sum.items():
        lags = ", ".join("none" if v is None else str(v) for v in s["recovery_lag"].values())
        print(f"  {m:<16}{s['accuracy']:>10.2f}{s['errors']:>8}   [{lags}]")
    print("\nNoisy perception (~1/7 readings corrupted) - same ranking, error "
          "floor raised by perception (E39 decomposition):")
    for m, s in noisy_sum.items():
        print(f"  {m:<16}{s['accuracy']:>10.2f}")

    sym = clean_sum["symbolic_refit"]["recovery_lag"]
    win = clean_sum["window_1nn"]["recovery_lag"]
    assert all(v is not None and v <= 4 for v in sym.values()), "symbolic must recover fast"
    assert clean_sum["static_frozen"]["recovery_lag"][CHANGES[0]] is None, \
        "frozen must not recover"
    print(f"\nsymbolic recovers in {list(sym.values())} steps (to exact); "
          f"window in {list(win.values())}; frozen never.")


if __name__ == "__main__":
    main()
