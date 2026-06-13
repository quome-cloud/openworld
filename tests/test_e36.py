"""Fast offline sanity tests for E36 (tiny k, tiny epochs)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))

from openworld import Action  # noqa: E402

import e36_representations as e  # noqa: E402


def _step_composite(comp, joint, active, action, k):
    st = comp.initial_state.copy()
    for i in range(k):
        st[f"s{i}"] = dict(joint[f"s{i}"])
    out = comp.transition.step(st, Action(f"s{active}:{action}"))
    return {f"s{i}": dict(out[f"s{i}"]) for i in range(k)}


def test_composite_matches_oracle():
    """Symbolic composite step == joint_oracle on random joints (the ceiling)."""
    k = 2
    comp = e.build_composite(k)
    rng = np.random.RandomState(0)
    for _ in range(30):
        joint = {f"s{i}": {f: int(rng.randint(0, e.G + 1)) for f in e.FIELDS}
                 for i in range(k)}
        active = int(rng.randint(0, k))
        action = e.ACTIONS[int(rng.randint(0, len(e.ACTIONS)))]
        got = _step_composite(comp, joint, active, action, k)
        assert got == e.joint_oracle(joint, active, action, k)


def test_mlp_training_loss_decreases():
    """An MLP must fit a trivial target: first loss > last loss."""
    rng = np.random.RandomState(0)
    x = rng.randn(40, 5)
    w = rng.randn(5, 3)
    y = x @ w
    mlp = e.MLP(5, 3, hidden=16, seed=0)
    first, last = mlp.train(x, y, epochs=300, lr=1e-2)
    assert last < first
    assert last < 0.5 * first


def test_split_marginal_cover_joint_novel():
    """Each child sees its full local space; test is cross-sector off-diagonal."""
    k = 2
    train = e.make_train(k)
    test = e.make_test(k, n=60)
    # every single-sector local slice appears in training (full marginal cover)
    combos = set(
        (tx["joint"]["s0"]["stock"], tx["joint"]["s0"]["output"], tx["joint"]["s0"]["waste"])
        for tx in train if tx["active"] == 0
    )
    assert len(combos) == (e.G + 1) ** 3
    # training is on the cross-sector diagonal; test is overwhelmingly off it
    assert e.fraction_off_diagonal(train) == 0.0
    assert e.fraction_off_diagonal(test) > 0.8


def test_symbolic_ceiling_assert_runs():
    """The startup oracle guard passes for small k."""
    e.assert_symbolic_ceiling()
