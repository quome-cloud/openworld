"""A 'showcase' world that exercises every world-model spec component at once.

Built to be the legend-figure for the framework: external inputs -> perceptors (of
several modalities) -> child worlds + verified transitions -> composite (bridge +
aggregator) -> emit -> act. Rendering its spec gives one diagram that names every part a
user wires up when prototyping a code world model.

  perception: JSON (structured), Code (telemetry), Vision (image), Transcript (audio),
              DAG (causal graph)
  worlds:     intake (tickets arrive)  +  ops (engineers resolve)
  composite:  a Bridge (hand off work intake->ops) + an Aggregator (open items)
  emit:       a verified CodeEmitter (status report) + a ToolEmitter (page on-call)
"""

from openworld import (Aggregator, Bridge, CodeEmitter, CodePerceptor,
                       CodeTransition, CompositeWorld, DAGPerceptor, Dial, JSONPerceptor,
                       MockLLM, Objective, ToolEmitter, ToolRegistry, TranscriptPerceptor,
                       VisionPerceptor, World)

INTAKE_CODE = '''
def transition(state, action):
    s = dict(state)
    if action["name"] == "receive":
        s["queue"] = s["queue"] + 1
    elif action["name"] == "triage" and s["queue"] > 0:
        s["severity"] = min(5, s["severity"] + 1)
    return s
'''

OPS_CODE = '''
def transition(state, action):
    s = dict(state)
    if action["name"] == "work" and s["load"] > 0:
        s["load"] = s["load"] - 1
        s["resolved"] = s["resolved"] + 1
    return s
'''

# Bridge coupling: hand one queued ticket from intake to ops if there is capacity.
HANDOFF_CODE = '''
def transition(state, action):
    a, b = dict(state["a"]), dict(state["b"])
    if a["queue"] > 0 and b["load"] < b["capacity"]:
        a["queue"] -= 1
        b["load"] += 1
    return {"a": a, "b": b}
'''

# Verified emitter: turn world state into a status report (no LLM).
REPORT_CODE = '''
def emit(state):
    load = state.get("ops", {}).get("load", 0)
    queue = state.get("intake", {}).get("queue", 0)
    return {"status": "busy" if load + queue > 4 else "ok", "open_items": load + queue}
'''

# Tool emitter: page the on-call engineer when overloaded.
PAGE_CODE = '''
def choose(state):
    if state.get("ops", {}).get("load", 0) >= state.get("ops", {}).get("capacity", 3):
        return {"tool": "page_oncall", "args": {"reason": "ops at capacity"}}
    return {"tool": "noop", "args": {}}
'''


def build_showcase_world():
    intake = World(name="intake", description="Incident tickets arrive and are triaged.",
                   initial_state={"queue": 3, "severity": 2},
                   actions=["receive", "triage"],
                   rules=["'receive' enqueues a ticket; 'triage' raises severity (cap 5)."],
                   transition=CodeTransition(INTAKE_CODE))
    ops = World(name="ops", description="Engineers resolve handed-off work.",
                initial_state={"load": 0, "resolved": 0, "capacity": 3},
                actions=["work"],
                rules=["'work' resolves one loaded item if any."],
                transition=CodeTransition(OPS_CODE))

    composite = CompositeWorld(
        name="operations",
        children={"intake": intake, "ops": ops},
        bridges=[Bridge(name="handoff", a="intake", b="ops",
                        transition=CodeTransition(HANDOFF_CODE),
                        description="Hand a queued ticket to ops when capacity allows.",
                        rules=["move intake.queue -> ops.load while ops.load < capacity"])],
        aggregators=[Aggregator(name="open_items",
                                fn=lambda kids: kids["intake"]["queue"] + kids["ops"]["load"])],
        description="An incident-operations world: intake feeds ops via a verified bridge.",
        rules=["Namespaced child actions plus 'tick'; open_items aggregates the leaves."])

    # perception boundary: one real perceptor per modality -> symbolic state. Structured
    # text (JSON), runnable code (telemetry), image, audio, and a causal graph (DAG). This is
    # an illustrative open set, not the full catalogue (Text/Regex/Mock perceptors omitted).
    _llm = MockLLM()
    composite.perceptors = [
        JSONPerceptor(paths={"queue": "queue", "severity": "sev"},
                      schema={"queue": (int, (0, 99)), "severity": (int, (0, 5))},
                      modality="text"),
        CodePerceptor(code='def perceive(data):\n    return {"load": int(data.get("cpu_pct", 0)) // 25}',
                      produces=["load"], schema={"load": (int, (0, 4))}, modality="text"),
        VisionPerceptor(_llm, produces=["queue"], schema={"queue": (int, (0, 99))}),
        TranscriptPerceptor(_llm, produces=["severity"], schema={"severity": (int, (0, 5))}),
        DAGPerceptor(mode="schema"),
    ]
    # objectives + a tunable dial: steer between throughput and care at inference time
    composite.objectives = [
        Objective(name="throughput", fn=lambda s, a, ns: ns.get("ops", {}).get("resolved", 0),
                  weight=Dial(name="speed_vs_care", value=0.6, minimum=0.0, maximum=1.0),
                  description="resolve more items"),
        Objective(name="care", fn=lambda s, a, ns: -ns.get("intake", {}).get("severity", 0),
                  weight=0.4, description="keep severity low"),
    ]
    # emit boundary: a verified report + a tool action
    composite.emit = [
        CodeEmitter(code=REPORT_CODE, reads=["intake", "ops"]),
        ToolEmitter(code=PAGE_CODE, registry=ToolRegistry(), reads=["ops"]),
    ]
    return composite


if __name__ == "__main__":
    from openworld.spec import to_spec, validate_spec
    w = build_showcase_world()
    spec = to_spec(w)
    problems = validate_spec(spec)
    print("components:",
          {"perceptors": len(spec.get("perception", [])),
           "children": len(spec.get("composite", {}).get("children", [])),
           "bridges": len(spec.get("composite", {}).get("bridges", [])),
           "aggregators": len(spec.get("composite", {}).get("aggregators", [])),
           "objectives": len(spec.get("objectives", [])),
           "emit": len(spec.get("emit", []))})
    print("validate_spec problems:", problems)
    assert not problems, problems
    print("ok: showcase world serializes + validates")
