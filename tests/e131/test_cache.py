from experiments.e131.lookahead import FrontierCache, value


def test_value_prefers_level_delta_then_novelty():
    seen = {("a",)}
    assert value(2, 3, ("b",), seen) == (1, 1)     # +1 level, novel
    assert value(2, 2, ("b",), seen) == (0, 1)     # no level, novel
    assert value(2, 2, ("a",), seen) == (0, 0)     # no level, seen
    assert value(2, 3, ("a",), seen) > value(2, 2, ("b",), seen)   # level-delta dominates novelty


def test_cache_roundtrip_and_seen():
    c = FrontierCache()
    assert c.get(("s0",), 1) is None
    c.put(("s0",), 1, ("s1",), 5, [[1]])
    assert c.get(("s0",), 1) == (("s1",), 5)
    assert ("s1",) in c.seen
    assert c.path_to[("s1",)] == [[1]]
    c.put(("s0",), 1, ("s1",), 5, [[9]])           # path_to not overwritten once set
    assert c.path_to[("s1",)] == [[1]]
