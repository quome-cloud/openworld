"""E147 AutoMem memory-structure test: fast invariants for CI (deterministic, no LLM)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
import e147_automem_memory_structure as e


def test_structure_ladder_at_long_horizon():
    """P2 keyed-dedup > P1 append > P0 context on a long-horizon, duplicate-heavy episode."""
    ev = e.make_episode(n_keys=8, gap_scale=4.0, dup_rate=1.0, seed=0)
    p0 = e.simulate("p0_context", ev)
    p1 = e.simulate("p1_append", ev)
    p2 = e.simulate("p2_keyed", ev)
    assert p2 > p1 > p0


def test_memory_is_a_long_horizon_lever():
    """No memory advantage when the gap fits the window; full advantage once it doesn't."""
    seeds = range(6)
    def prog(sc, gs):
        return sum(e.simulate(sc, e.make_episode(8, gs, 0.5, s)) for s in seeds) / 6
    short = prog("p2_keyed", 0.25) - prog("p0_context", 0.25)     # gap < window
    long = prog("p2_keyed", 4.0) - prog("p0_context", 4.0)        # gap >> window
    assert short <= 0.05 < 0.4 <= long


def test_dedup_only_helps_with_duplicates():
    """P1 degrades as duplicate rate rises; P2 (dedup) is duplicate-invariant; equal when dup=0."""
    def prog(sc, dr):
        return sum(e.simulate(sc, e.make_episode(8, 4.0, dr, s)) for s in range(6)) / 6
    assert abs(prog("p1_append", 0.0) - prog("p2_keyed", 0.0)) < 1e-9      # no dups -> structure moot
    assert prog("p1_append", 4.0) < prog("p1_append", 0.0) - 0.05          # dups degrade the flat log
    assert abs(prog("p2_keyed", 0.0) - prog("p2_keyed", 4.0)) < 1e-9       # dedup invariant


def test_world_roundtrips_and_matches_simulator():
    """Each scaffold's OpenWorld World round-trips its rollout and equals the reference simulator."""
    from openworld import to_spec, from_spec, validate_spec
    ev = e.make_episode(8, 4.0, 1.0, seed=0)
    for sc in e.SCAFFOLDS:
        w = e.build_memory_world(sc, ev)
        spec = to_spec(w)
        assert validate_spec(spec) == []
        prog_world = e.rollout_world(w)
        prog_round = e.rollout_world(from_spec(spec, allow_code=True))
        prog_sim = e.simulate(sc, ev)
        assert abs(prog_world - prog_round) < 1e-12
        assert abs(prog_world - prog_sim) < 1e-12
