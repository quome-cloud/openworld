from experiments.e130.weights import ExpertWeights


def test_starts_uniform():
    w = ExpertWeights(["a", "b"])
    assert abs(w.weight("a") - 0.5) < 1e-9 and abs(w.weight("b") - 0.5) < 1e-9


def test_rewarded_expert_gains_weight():
    w = ExpertWeights(["a", "b"], eta=0.5)
    for _ in range(5):
        w.reward("a", 1.0); w.reward("b", 0.0)
    assert w.weight("a") > 0.8                       # the productive expert dominates
    assert abs(sum(w.as_dict().values()) - 1.0) < 1e-9


def test_unknown_name_is_noop():
    w = ExpertWeights(["a"]); w.reward("zzz", 1.0)   # must not crash
    assert abs(w.weight("a") - 1.0) < 1e-9
