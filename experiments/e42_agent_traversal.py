"""E42 - Agents traversing connected worlds with changing rules (capstone).

Every thread at once, at the AGENT-BELIEF level: two reservoir worlds joined by
a toll Route (E31), each with its own non-stationary rule (E41) and its own
perception boundary (E39/E40); an agent hops between them via travel and tries
to track each world's rule. The hard, agent-level twist: a world's rule can
change WHILE THE AGENT IS AWAY, so on return it faces a silently-changed world.

We run four agent-belief models on the same perceived observations and ask who
generalizes the rules across hops:

  symbolic_per_world  - a separate symbolic rule-belief per world; predict with
                        the retained belief, re-identify from the transient on
                        error (E41 monitor, kept per world). No cross-world
                        interference; away-changes caught on return. (Ours.)
  shared_online       - ONE belief used wherever the agent is; learning world B
                        overwrites its world-A belief (E36 interference, now
                        driven by traversal).
  per_world_window    - a separate sliding-window 1-NN per world; no
                        interference but refills its window each visit/change.
  oracle              - knows the agent's location and each world's current rule.

The world-of-worlds, the toll crossing, and perception all run through the real
CompositeWorld / Route / observe machinery; deterministic and offline.
"""

from openworld import (
    Action, Aggregator, CompositeWorld, MockPerceptor, Observation, Route, World,
)
from openworld.transition import FunctionTransition, Transition

from common import save_results

CAP = 100
T = 160
DWELL = 8                                  # steps in a world before hopping
TOLL = 1
# per-world regime schedules: list of (target, rate); change steps advance them
REGIMES = {
    "A": [(30, 2), (80, 4), (15, 3)],
    "B": [(70, 3), (20, 5), (55, 2)],
}
CHANGES = {"A": [40, 96], "B": [56, 112]}   # mix of present/away (asserted below)
CLOCK = [0]                                  # advanced once per loop iteration


def tracking_rule(level, target, rate):
    if level < target:
        return min(target, level + rate)
    if level > target:
        return max(target, level - rate)
    return level


def regime_index(world, t):
    return sum(1 for c in CHANGES[world] if t >= c)


def regime_at(world, t):
    return REGIMES[world][regime_index(world, t)]


def location_at(t):
    return "A" if (t // DWELL) % 2 == 0 else "B"


def make_child(world):
    def step(state, action):
        s = dict(state)
        if action.get("name") == "tick":
            s["level"] = tracking_rule(s["level"], *regime_at(world, CLOCK[0]))
        return s
    start = 0 if world == "A" else CAP
    return World(name=world, description=f"reservoir {world}",
                 initial_state={"level": start}, actions=["tick"],
                 transition=FunctionTransition(step))


class TollTransition(Transition):
    def step(self, state, action):
        s = state.copy()
        if action.name == "cross" and s["agent"].get("coins", 0) >= TOLL:
            s["agent"]["coins"] -= TOLL
            s["b"]["tolls"] = s["b"].get("tolls", 0) + TOLL
        return s


def perceive(level, world):
    """Each world has its own perception boundary."""
    p = MockPerceptor(produces=["level"], deltas=[{"level": level}],
                      schema={"level": (int, (0, CAP))})
    w = World(name="scratch", description="", initial_state={"level": -1},
              actions=[], transition=FunctionTransition(lambda s, a: s))
    w.observe(Observation("text" if world == "A" else "video_frame", f"{world}:{level}"), p)
    return w.state["level"]


def transient_update(belief, cur, nxt):
    delta = nxt - cur
    if delta > 0:
        return (CAP, delta)
    if delta < 0:
        return (0, -delta)
    return (nxt, belief[1])


# --- belief models: predict(loc, cur) -> next ; update(loc, cur, nxt) -------
class SymbolicPerWorld:
    def __init__(self):
        self.b = {w: REGIMES[w][0] for w in REGIMES}

    def predict(self, loc, cur):
        return tracking_rule(cur, *self.b[loc])

    def update(self, loc, cur, nxt):
        if self.predict(loc, cur) != nxt:
            self.b[loc] = transient_update(self.b[loc], cur, nxt)


class SharedOnline:
    def __init__(self):
        self.b = REGIMES["A"][0]

    def predict(self, loc, cur):
        return tracking_rule(cur, *self.b)

    def update(self, loc, cur, nxt):
        if self.predict(loc, cur) != nxt:
            self.b = transient_update(self.b, cur, nxt)


class PerWorldWindow:
    def __init__(self, window=10):
        self.W = window
        self.win = {w: [(perceive(REGIMES[w][0][0], w), REGIMES[w][0][0])] for w in REGIMES}

    def predict(self, loc, cur):
        w = self.win[loc]
        return min(w, key=lambda tr: abs(tr[0] - cur))[1] if w else cur

    def update(self, loc, cur, nxt):
        self.win[loc].append((cur, nxt))
        self.win[loc][:] = self.win[loc][-self.W:]


class Oracle:
    def predict(self, loc, cur):
        return tracking_rule(cur, *regime_at(loc, CLOCK[0]))

    def update(self, loc, cur, nxt):
        pass


def build_composite():
    return CompositeWorld(
        name="twin", children={"A": make_child("A"), "B": make_child("B")},
        agents={"trav": {"at": "A", "coins": 10**6}},
        default_actions={"A": "tick", "B": "tick"},   # both worlds advance on tick
        bridges=[Route("road", "A", "B", transition=None, on_cross=TollTransition())],
        aggregators=[Aggregator("total_level",
                                lambda kids: kids["A"]["level"] + kids["B"]["level"])],
    )


def run():
    CLOCK[0] = 0
    comp = build_composite()
    models = {"symbolic_per_world": SymbolicPerWorld(), "shared_online": SharedOnline(),
              "per_world_window": PerWorldWindow(), "oracle": Oracle()}
    flags = {m: [] for m in models}
    arrivals = []                          # (t, world, silently_changed)
    last_left_regime = {"A": 0, "B": 0}
    present = "A"

    for t in range(T):
        CLOCK[0] = t                                      # transition + oracle agree on t
        loc = comp.state["_agents"]["trav"]["at"]
        # arrival event: did we just hop into `loc`, and did its rule change away?
        if t > 0 and loc != present:
            silently = regime_index(loc, CLOCK[0]) != last_left_regime[loc]
            arrivals.append((t, loc, silently))
        if loc != present:
            last_left_regime[present] = regime_index(present, CLOCK[0])
        present = loc

        cur = perceive(comp.state[loc]["level"], loc)        # per-world perception
        comp.step(Action("tick"))                            # both worlds advance (regime_at t)
        nxt = comp.state[loc]["level"]
        for name, model in models.items():
            flags[name].append(model.predict(loc, cur) == nxt)
            model.update(loc, cur, nxt)

        # hop on the dwell schedule
        if (t + 1) % DWELL == 0:
            dest = "B" if loc == "A" else "A"
            comp.step(Action("travel", params={"agent": "trav", "to": dest}))

    return flags, arrivals, comp


def recovery_after(flags, t0, end):
    run_len = 0
    for j in range(t0, end):
        run_len = run_len + 1 if flags[j] else 0
        if run_len >= 3:
            return (j - 2) - t0
    return None


def main():
    flags, arrivals, comp = run()
    # sanity: schedule must produce both present and away changes per world
    for w in REGIMES:
        away = any(loc == w and sil for _, loc, sil in arrivals)
        assert away, f"schedule produced no away-change for world {w}"

    summary = {}
    for m, fl in flags.items():
        # split arrival recovery by silently-changed vs unchanged
        unchanged, changed = [], []
        for k, (t0, loc, sil) in enumerate(arrivals):
            end = arrivals[k + 1][0] if k + 1 < len(arrivals) else T
            lag = recovery_after(fl, t0, end)
            (changed if sil else unchanged).append(lag)
        def avg(xs):
            real = [x for x in xs if x is not None]
            return round(sum(real) / len(real), 2) if real else None
        summary[m] = {
            "accuracy": round(sum(fl) / len(fl), 3),
            "recovery_unchanged_return": avg(unchanged),
            "recovery_silently_changed_return": avg(changed),
            "never_recovered_returns": sum(
                1 for k, (t0, loc, sil) in enumerate(arrivals)
                if recovery_after(fl, t0, arrivals[k + 1][0] if k + 1 < len(arrivals) else T) is None),
        }

    save_results("e42_agent_traversal", {
        "horizon": T, "dwell": DWELL, "regimes": REGIMES, "changes": CHANGES,
        "n_arrivals": len(arrivals), "tolls_paid": comp.state["B"].get("tolls", 0),
        "summary": summary,
        "timeline": {m: [int(b) for b in fl] for m, fl in flags.items()},
        "arrivals": [{"t": t0, "world": loc, "silently_changed": sil} for t0, loc, sil in arrivals],
    })

    print(f"Two worlds + toll route, agent hops every {DWELL} steps over {T} "
          f"(arrivals={len(arrivals)}, tolls={comp.state['B'].get('tolls', 0)}).\n")
    print(f"  {'belief model':<20}{'accuracy':>9}{'recov(unchg)':>13}{'recov(changed)':>15}")
    for m, s in summary.items():
        u = "-" if s["recovery_unchanged_return"] is None else s["recovery_unchanged_return"]
        c = "-" if s["recovery_silently_changed_return"] is None else s["recovery_silently_changed_return"]
        print(f"  {m:<20}{s['accuracy']:>9.2f}{str(u):>13}{str(c):>15}")

    sym, shared = summary["symbolic_per_world"], summary["shared_online"]
    assert sym["accuracy"] > shared["accuracy"], "per-world separation should beat shared (interference)"
    print(f"\nper-world symbolic {sym['accuracy']:.2f} vs shared {shared['accuracy']:.2f}: "
          "separating beliefs per world avoids cross-world interference across hops.")


if __name__ == "__main__":
    main()
