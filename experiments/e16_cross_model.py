"""E16 - Cross-model-family synthesis replication (round-2 review item R1).

Replicates the synthesis-reliability protocol with generators from two other
model families (Meta Llama-3.1-8B, Google Gemma-2-9B) across all three
worlds, three sampling seeds each. Records verified acceptance, generator
iterations used, ground-truth probe accuracy of accepted code, and synthesis
wall-clock time (review item R6).
"""

import time

from openworld import OllamaLLM, World
from openworld.verify import SynthesisError

from common import WORLD_SPECS, require_ollama, save_results, wilson_ci
from e02_synthesis import world_probes
from common import probe_accuracy

MODELS = ["llama3.1:8b", "gemma2:9b"]
ATTEMPTS = 3
MAX_ITERS = 4


class CountingLLM(OllamaLLM):
    """Counts chat calls so generator iterations are observable."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = 0

    def chat(self, messages, **options):
        self.calls += 1
        return super().chat(messages, **options)


def main():
    for model in MODELS:
        require_ollama(model, timeout=600)

    runs = []
    for model in MODELS:
        for spec_name, spec in WORLD_SPECS.items():
            for attempt in range(ATTEMPTS):
                llm = CountingLLM(model=model, temperature=0.7, timeout=600,
                                  options={"seed": 9000 + attempt})
                world = World(
                    name=spec_name, description=spec["description"],
                    initial_state=dict(spec["initial"]),
                    actions=list(spec["actions"]), rules=list(spec["rules"]),
                    llm=llm,
                )
                start = time.perf_counter()
                record = {"model": model, "world": spec_name, "attempt": attempt}
                try:
                    transition = world.compile(max_iters=MAX_ITERS)
                    record["accepted"] = True
                    record["probe_accuracy"] = probe_accuracy(
                        transition, world_probes(spec_name), spec["oracle"])
                except SynthesisError as exc:
                    record["accepted"] = False
                    record["probe_accuracy"] = 0.0
                    record["last_feedback"] = str(exc)[:200]
                record["iterations"] = llm.calls
                record["wall_seconds"] = time.perf_counter() - start
                runs.append(record)
                print(f"  {model} {spec_name} #{attempt}: accepted={record['accepted']} "
                      f"acc={record['probe_accuracy']:.2f} iters={llm.calls} "
                      f"{record['wall_seconds']:.1f}s")

    summary = []
    for model in MODELS:
        rows = [r for r in runs if r["model"] == model]
        accepted = sum(r["accepted"] for r in rows)
        summary.append({
            "model": model,
            "n": len(rows),
            "acceptance_rate": accepted / len(rows),
            "acceptance_ci": list(wilson_ci(accepted, len(rows))),
            "mean_probe_accuracy_accepted": (
                sum(r["probe_accuracy"] for r in rows if r["accepted"]) / accepted
                if accepted else 0.0),
            "mean_iterations": sum(r["iterations"] for r in rows) / len(rows),
            "mean_wall_seconds": sum(r["wall_seconds"] for r in rows) / len(rows),
        })
    save_results("e16_cross_model", {
        "attempts_per_cell": ATTEMPTS, "max_iters": MAX_ITERS,
        "summary": summary, "runs": runs,
    })
    for s in summary:
        print(f"{s['model']}: accept {s['acceptance_rate']:.0%}, "
              f"probe {s['mean_probe_accuracy_accepted']:.2f}, "
              f"{s['mean_wall_seconds']:.0f}s avg")


if __name__ == "__main__":
    main()
