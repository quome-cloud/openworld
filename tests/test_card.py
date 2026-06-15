"""Tests for HTML+SVG model cards and the gallery index."""

from openworld import render_card, render_gallery, to_spec
from tests.test_spec import counter_world, economy_world


def test_card_is_self_contained_svg():
    svg = render_card(counter_world())
    assert svg.startswith("<?xml")
    assert "<svg" in svg
    assert "counter" in svg
    assert "ACTIONS" in svg
    for action in ("inc", "dec", "noop"):
        assert action in svg
    # self-contained: no fetched external resources (xmlns namespace URIs are
    # declarations, not network fetches, so they are allowed).
    for external in ("src=", "<script", "<image", 'href="http', "url(http", "@import"):
        assert external not in svg


def test_card_renders_composition_graph_and_writes_file(tmp_path):
    out = tmp_path / "economy.svg"
    svg = render_card(economy_world(), path=out, theme="dark")
    assert out.exists() and out.read_text(encoding="utf-8") == svg
    assert "#0c111c" in svg                             # dark theme background
    assert "shop" in svg and "market" in svg           # child nodes
    assert "restock" in svg                             # bridge edge label
    assert "total_value" in svg                         # aggregator node


def test_leaf_renders_state_transition_graph():
    svg = render_card(counter_world())
    assert "STATE-TRANSITION GRAPH" in svg
    assert "▶ start" in svg                             # initial-state node marked
    assert "<path" in svg                               # transition edges drawn


def test_card_accepts_a_spec_dict_too():
    spec = to_spec(counter_world(), card={"tags": ["toy", "verified"], "license": "MIT"})
    svg = render_card(spec)
    assert "toy" in svg and "verified" in svg and "MIT" in svg


def test_sparkline_present_when_preview_has_series():
    svg = render_card(counter_world())
    assert "polyline" in svg                            # rollout sparkline drawn


def test_gallery_links_to_each_card(tmp_path):
    specs = [to_spec(counter_world()), to_spec(economy_world())]
    out = tmp_path / "index.svg"
    svg = render_gallery(specs, path=out)
    assert out.exists()
    assert 'href="counter.svg"' in svg
    assert 'href="economy.svg"' in svg
    assert "counter" in svg and "economy" in svg
