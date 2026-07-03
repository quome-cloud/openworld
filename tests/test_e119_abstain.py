from e119 import abstain


def test_best_of_n_returns_majority_behavior_when_agree():
    seq = iter(["a", "a", "a", "b"])
    winner, meta = abstain.best_of_n(lambda: next(seq), behavior_fn=lambda c: c, n=4, tau=0.5)
    assert winner == "a"
    assert meta["agreement"] >= 0.5


def test_best_of_n_abstains_when_no_cluster_clears_tau():
    seq = iter(["a", "b", "c", "d"])
    winner, meta = abstain.best_of_n(lambda: next(seq), behavior_fn=lambda c: c, n=4, tau=0.5)
    assert winner is None
    assert meta["agreement"] < 0.5


def test_best_of_n_clusters_by_behavior_not_text():
    # different text, same behavior signature -> they agree
    cand = iter(["x=1", "x = 1", "x= 1", "y=9"])
    winner, meta = abstain.best_of_n(
        lambda: next(cand),
        behavior_fn=lambda c: c.replace(" ", "").split("=")[0],  # variable name = behavior
        n=4, tau=0.5)
    assert winner is not None
    assert winner.replace(" ", "").startswith("x")
