"""Tests for HTML+SVG model cards and the gallery index."""

from openworld import render_card, render_gallery, to_spec
from tests.test_spec import counter_world, economy_world


def test_card_is_self_contained_html():
    html = render_card(counter_world())
    assert html.startswith("<!DOCTYPE html>")
    assert "counter" in html
    assert "<svg" in html
    assert "ACTIONS" in html
    for action in ("inc", "dec", "noop"):
        assert action in html
    # self-contained: no fetched external resources (the SVG xmlns namespace URI
    # is a declaration, not a network fetch, so it is allowed).
    for external in ("src=", "<script", "<link", 'href="http', "url(http", "@import"):
        assert external not in html


def test_card_renders_composition_and_writes_file(tmp_path):
    out = tmp_path / "economy.html"
    html = render_card(economy_world(), path=out, theme="dark")
    assert out.exists() and out.read_text(encoding="utf-8") == html
    assert 'body class="dark"' in html
    assert "shop" in html and "market" in html        # child boxes
    assert "restock" in html                           # bridge label
    assert "total_value" in html                       # aggregator pill


def test_card_accepts_a_spec_dict_too():
    spec = to_spec(counter_world(), card={"tags": ["toy", "verified"], "license": "MIT"})
    html = render_card(spec)
    assert "toy" in html and "verified" in html and "MIT" in html


def test_sparkline_present_when_preview_has_series():
    html = render_card(counter_world())
    assert "polyline" in html                          # rollout sparkline drawn


def test_gallery_links_to_each_card(tmp_path):
    specs = [to_spec(counter_world()), to_spec(economy_world())]
    out = tmp_path / "index.html"
    html = render_gallery(specs, path=out)
    assert out.exists()
    assert 'href="counter.html"' in html
    assert 'href="economy.html"' in html
    assert "counter" in html and "economy" in html
