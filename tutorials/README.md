# OpenWorld Tutorials

> **New to "worlds" and "oracles"? Start here →
> [OpenWorld for scikit-learn users](from_scikit_learn.md).** It maps every
> concept to one you already know from `fit`/`predict`/`score` and clears up
> the three things that trip up ML practitioners most.

Five domain walkthroughs, each pairing a guide with a runnable script, plus the
scikit-learn bridge above. Read them in any order — the healthcare tutorial is
the gentlest introduction to dynamics synthesis.

| Tutorial | Domain | What it teaches |
|---|---|---|
| [OpenWorld for scikit-learn users](from_scikit_learn.md) | Onboarding / dev-rel | Concept bridge from ML: world vs. dataset, what an oracle is, verification vs. test accuracy — with side-by-side `fit`/`predict` vs. `compile`/`step` code |
| [Healthcare: ICU triage](healthcare_triage.md) | Clinical operations | LLM-synthesized dynamics, safety invariants, a stewardship dial over outcomes vs spend |
| [Legaltech: settlement negotiation](legaltech_settlement.md) | Litigation strategy | Multi-agent simulation, asymmetric policies, event-triggered objectives |
| [Finance: portfolio rebalancing](finance_portfolio.md) | Trading / risk | Deterministic schedules inside state, float state, growth-vs-risk frontiers |
| [Software engineering: sprint planning](software_engineering_sprint.md) | Eng management | The generator + critic two-model relay, validating synthesized code against ground truth |
| [Benchmarking: build a dataset](benchmark_dataset.md) | Evaluation / benchmarks | The instance schema and validation gate, the paired single-shot vs in-world ablation, recipes and reproducibility tiers |
| [Multimodal: the perception boundary](multimodal_perception.md) | Multimodal inputs | Feeding text (and later audio/video) into a symbolic world via a gated, untrusted Perceptor — without touching the verified core |

All scripts run offline (they fall back to `MockLLM` or use hand-written
dynamics); the healthcare and software engineering ones do live code synthesis
when an Ollama server is reachable:

```bash
python tutorials/healthcare_triage.py qwen2.5:7b
python tutorials/legaltech_settlement.py
python tutorials/finance_portfolio.py
python tutorials/software_engineering_sprint.py qwen2.5:7b qwen2.5:3b
python tutorials/benchmark_dataset.py
python tutorials/from_scikit_learn.py
python tutorials/multimodal_perception.py
```

## The shape every tutorial follows

1. **Declare the world** — symbolic state, action names, plain-language rules.
2. **Get dynamics** — `world.compile()` (the LLM writes verified code) or
   `FunctionTransition` (you write it).
3. **Add agents** — LLM planners with goals, or deterministic `policy` functions.
4. **Declare objectives** — scoring functions weighted by fixed floats or a `Dial`.
5. **Sweep the dial** — trace the Pareto frontier between competing objectives
   and pick the operating point you can defend.
