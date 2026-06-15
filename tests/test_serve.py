"""Tests for the FastAPI world-model inference server."""

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient                       # noqa: E402

from openworld import (CodePerceptor, CodeTransition, World, to_spec)         # noqa: E402
from openworld.serve import serve_app                          # noqa: E402
from tests.test_spec import counter_world, economy_world       # noqa: E402

INTAKE_PERCEIVE = """
def perceive(data):
    out = {}
    for line in str(data).splitlines():
        if ':' in line:
            k, v = line.split(':', 1)
            k = k.strip(); v = v.strip()
            if k in ('priority', 'load'):
                out[k] = int(v) if v.lstrip('-').isdigit() else 0
    return out
"""
INTAKE_STEP = """
def transition(state, action):
    s = dict(state)
    if action["name"] == "work" and s["load"] > 0:
        s["load"] = s["load"] - 1
        s["done"] = s["done"] + 1
    return s
"""


def intake_world():
    w = World(name="intake", description="ticket intake",
              initial_state={"priority": 0, "load": 0, "done": 0},
              actions=["work"], rules=["'work' clears one load, increments done."],
              transition=CodeTransition(INTAKE_STEP))
    w.perceptors = [CodePerceptor(code=INTAKE_PERCEIVE, produces=["priority", "load"],
                                  schema={"priority": (int, (0, 9)),
                                          "load": (int, (0, 99))}, modality="text")]
    w.emit = [{"modality": "report", "fields": ["priority", "load", "done"],
               "report": "priority {priority}: cleared {done}, {load} left"}]
    return w


@pytest.fixture
def client():
    specs = [to_spec(counter_world()), to_spec(economy_world()), to_spec(intake_world())]
    return TestClient(serve_app(specs, allow_code=True))


def test_health_and_list(client):
    assert client.get("/healthz").json()["runnable"] is True
    names = {w["name"] for w in client.get("/worlds").json()["worlds"]}
    assert names == {"counter", "economy", "intake"}


def test_step_returns_next_state_and_node(client):
    r = client.post("/worlds/counter/step",
                    json={"state": {"n": 3, "label": "x", "history": []},
                          "action": {"name": "inc"}}).json()
    assert r["next_state"]["n"] == 4
    assert "n" in r["changed"]


def test_batch_predict(client):
    body = {"inputs": [
        {"state": {"n": 1, "label": "x", "history": []}, "action": {"name": "inc"}},
        {"state": {"n": 1, "label": "x", "history": []}, "action": {"name": "dec"}}]}
    outs = client.post("/worlds/counter/predict", json=body).json()["outputs"]
    assert [o["next_state"]["n"] for o in outs] == [2, 0]


def test_composite_rollout_runs_bridge(client):
    # economy: shop(counter) -inc-> bumps shop.n; bridge 'restock' feeds market.volume
    r = client.post("/worlds/economy/rollout",
                    json={"actions": [{"name": "tick"}, {"name": "tick"}]}).json()
    final = r["final_state"]
    assert final["market"]["volume"] > 10          # bridge moved shop.n into volume
    assert "_agg" in final                          # aggregator recomputed


def test_actions_and_metrics(client):
    acts = client.get("/worlds/economy/actions").json()
    assert "tick" in acts["actions"] and "restock" in acts["bridges"]
    m = client.get("/worlds/counter/metrics").json()
    assert m["dynamics"] == "code" and m["reachable_states"] >= 1


def test_run_perception_pipeline(client):
    r = client.post("/worlds/intake/run",
                    json={"input": {"modality": "text", "data": "priority: 7\nload: 3"},
                          "steps": 3}).json()
    assert r["perceived_delta"] == {"priority": 7, "load": 3}
    assert r["output"]["emitted"][0]["report"] == "priority 7: cleared 3, 0 left"
    assert len(r["trajectory"]) == 4                # perceive + 3 steps


def test_observe_gate_rejects_out_of_range(client):
    r = client.post("/worlds/intake/observe",
                    json={"delta": {"priority": 999}})       # out of [0,9]
    assert r.status_code == 422


def test_reactflow_and_view_and_card(client):
    assert client.get("/worlds/intake/reactflow").json()["nodes"]
    assert client.get("/worlds/intake/view").status_code == 200
    assert "image/svg+xml" in client.get("/worlds/counter/card.svg").headers["content-type"]


def test_websocket_streams_run(client):
    with client.websocket_connect("/worlds/intake/live") as ws:
        ws.send_json({"input": {"modality": "text", "data": "priority: 2\nload: 2"},
                      "steps": 2})
        frames, done = 0, None
        while True:
            m = ws.receive_json()
            if "frame" in m:
                frames += 1
            if m.get("done"):
                done = m
                break
        assert frames == 3 and done["output"]["emitted"][0]["fields"]["done"] == 2


def test_code_gated_when_not_allowed():
    c = TestClient(serve_app([to_spec(counter_world())], allow_code=False))
    assert c.get("/worlds/counter").json()["runnable"] is False
    r = c.post("/worlds/counter/step",
               json={"state": {"n": 1, "label": "x", "history": []},
                     "action": {"name": "inc"}})
    assert r.status_code == 403
