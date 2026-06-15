"""A perception-driven demo world: paste a ticket, watch it get worked, get a report.

Run it to write `specs/intake.json`, then:

    openworld serve specs/ --allow-code
    # open http://127.0.0.1:8080/worlds/intake/view and paste:
    #   priority: 7
    #   load: 4
"""
from pathlib import Path

from openworld import (CodePerceptor, CodeTransition, World, spec_to_json,
                       to_spec, validate_spec)

PERCEIVE = """
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

STEP = """
def transition(state, action):
    s = dict(state)
    if action["name"] == "work" and s["load"] > 0:
        s["load"] = s["load"] - 1
        s["done"] = s["done"] + 1
    return s
"""


def build():
    w = World(
        name="intake",
        description="A support-ticket intake: perceive a ticket, work it down, "
                    "emit a resolution report.",
        initial_state={"priority": 0, "load": 0, "done": 0},
        actions=["work"],
        rules=["'work' clears one unit of load and increments done (until load is 0)."],
        transition=CodeTransition(STEP),
    )
    w.perceptors = [CodePerceptor(
        code=PERCEIVE, produces=["priority", "load"], modality="text",
        schema={"priority": (int, (0, 9)), "load": (int, (0, 99))})]
    w.emit = [{"modality": "report", "fields": ["priority", "load", "done"],
               "report": "priority {priority}: cleared {done}, {load} remaining"}]
    w.objectives = [{"name": "clear the queue", "goal": "max done"}]
    return w


if __name__ == "__main__":
    spec = to_spec(build(), card={"tags": ["perception", "demo", "leaf"],
                                  "license": "MIT", "version": "1.0",
                                  "lineage": "examples/intake_world.py"})
    assert not validate_spec(spec), validate_spec(spec)
    Path("specs").mkdir(exist_ok=True)
    Path("specs/intake.json").write_text(spec_to_json(spec), encoding="utf-8")
    print("wrote specs/intake.json")
