"""E62 - A branch-covering acceptance gate catches faults a single-state smoke-run misses.

The verifier's behavioral check smoke-runs each action once, from the initial
state. A transition can be correct there yet violate an invariant (or crash) only
on a state reached later in a rollout -- exactly the gap E3 exposes (the gate's
branch-level false-accept rate). This experiment quantifies the gap and shows that
running each action over a small set of branch-covering states closes it, while
honestly marking what verification still cannot catch (silent correctness errors,
which need an oracle, not a gate).

Deterministic and offline (no LLM): a hand-built panel of transition programs run
through two gates -- single-state (probe_states=[]) and branch-covering.
"""

from openworld.state import Action, WorldState
from openworld.verify import Verifier

from common import SPRINT_INITIAL, save_results

INVARIANTS = [("counters never negative", lambda s: all(v >= 0 for v in s.values()))]
ACTIONS = [Action("ship"), Action("fix"), Action("refactor")]

# Branch-covering states: each exercises a guard the initial state cannot
# (empty backlog; one outstanding bug; one unit of debt; a high-debt regime).
PROBE_STATES = [
    {"backlog": 0, "shipped": 8, "bugs": 0, "debt": 3},   # ship guard: backlog == 0
    {"backlog": 4, "shipped": 3, "bugs": 1, "debt": 1},   # fix/refactor guards: bugs/debt == 1
    {"backlog": 1, "shipped": 5, "bugs": 2, "debt": 7},   # backlog == 1; high debt
]

CORRECT = '''
def transition(state, action):
    s = dict(state)
    name = action["name"]
    if name == "ship" and s["backlog"] > 0:
        s["backlog"] -= 1; s["shipped"] += 1; s["debt"] += 1
        s["bugs"] += s["debt"] // 4
    elif name == "fix":
        s["bugs"] = max(0, s["bugs"] - 2)
    elif name == "refactor":
        s["debt"] = max(0, s["debt"] - 2)
    return s
'''

# Faults that are invariant-clean from the initial state but violate "counters
# never negative" on a reachable non-initial state (branch-only faults).
SHIP_NO_GUARD = CORRECT.replace('if name == "ship" and s["backlog"] > 0:',
                                'if name == "ship":')                       # backlog 0 -> -1
FIX_OFF_BY = CORRECT.replace('s["bugs"] = max(0, s["bugs"] - 2)',
                             'if s["bugs"] > 0:\n            s["bugs"] -= 2')  # bugs 1 -> -1
REFACTOR_OFF_BY = CORRECT.replace('s["debt"] = max(0, s["debt"] - 2)',
                                  'if s["debt"] > 0:\n            s["debt"] -= 2')  # debt 1 -> -1
BRANCH_FAULTS = {"ship_no_guard": SHIP_NO_GUARD, "fix_off_by_one": FIX_OFF_BY,
                 "refactor_off_by_one": REFACTOR_OFF_BY}

# A *silent* correctness error: invariant-clean everywhere, but wrong (uses //5).
# Neither gate can catch this without a ground-truth oracle -- the honest limit.
SILENT = CORRECT.replace("s[\"debt\"] // 4", "s[\"debt\"] // 5")


def accepts(code, probe_states):
    v = Verifier(initial_state=WorldState(dict(SPRINT_INITIAL)),
                 sample_actions=ACTIONS, invariants=INVARIANTS,
                 probe_states=probe_states)
    ok, _ = v.check_behavior(code)
    return ok


def main():
    single = lambda code: accepts(code, [])
    branch = lambda code: accepts(code, PROBE_STATES)

    correct = {"single_state": single(CORRECT), "branch_covering": branch(CORRECT)}
    faults = {name: {"single_state": single(code), "branch_covering": branch(code)}
              for name, code in BRANCH_FAULTS.items()}
    silent = {"single_state": single(SILENT), "branch_covering": branch(SILENT)}

    n = len(BRANCH_FAULTS)
    single_false_accept = sum(1 for f in faults.values() if f["single_state"]) / n
    branch_false_accept = sum(1 for f in faults.values() if f["branch_covering"]) / n

    results = {
        "n_branch_faults": n, "n_probe_states": len(PROBE_STATES),
        "correct_program": correct,
        "branch_faults": faults,
        "silent_correctness_error": silent,
        "single_state_false_accept_rate": round(single_false_accept, 3),
        "branch_covering_false_accept_rate": round(branch_false_accept, 3),
    }
    save_results("e62_branch_gate", results)

    print("E62 - branch-covering acceptance gate\n")
    print(f"  correct program: single-state={correct['single_state']}  "
          f"branch-covering={correct['branch_covering']}")
    print(f"  {'branch-only fault':<22} {'single-state':>12} {'branch-cov':>11}")
    for name, r in faults.items():
        print(f"  {name:<22} {str(r['single_state']):>12} {str(r['branch_covering']):>11}")
    print(f"  silent correctness err  {str(silent['single_state']):>12} "
          f"{str(silent['branch_covering']):>11}")
    print(f"\n  single-state false-accept on branch faults = {single_false_accept:.0%}")
    print(f"  branch-covering false-accept on branch faults = {branch_false_accept:.0%}")

    # --- honest self-checks (save first) ---
    assert correct["single_state"] and correct["branch_covering"], "correct program must pass both gates"
    assert single_false_accept == 1.0, "single-state gate should miss every branch-only fault"
    assert branch_false_accept == 0.0, "branch-covering gate should catch every branch-only fault"
    assert silent["single_state"] and silent["branch_covering"], \
        "a silent correctness error passes both gates (verification != correctness; needs an oracle)"
    print("\nchecks pass: branch-covering verification rejects every branch-only fault the "
          "single-state smoke-run accepts; both still pass a silent correctness error, the "
          "honest limit of gate-without-oracle verification.")


if __name__ == "__main__":
    main()
