# E119 strategy — the MSA (arXiv 2507.12547) Bayesian ideas, made concrete in OpenWorld

**Paper:** *Modeling Open-World Cognition as On-Demand Synthesis of Probabilistic Models*
(Wong, Collins, Ying, Zhang, … Tenenbaum, Brooke-Wilson). The **Model Synthesis Architecture (MSA)**:
a language model, facing a novel situation, does not answer directly — it **synthesizes a bespoke
probabilistic *program*** as a world model and does **coherent (Bayesian) inference** over it.

E119 applies two MSA ideas, **adapted** to the replay-only ARC-AGI-3 env and OpenWorld's
verified-code-world-model stance:

1. **On-demand model synthesis** → the SLM synthesizes the *goal* of a level as a small
   **predicate-program** (`reach` / `count` / `align`), compiled to an executable grader.
2. **Posterior over hypotheses with abstention** → instead of inference *inside* one program, we put a
   **posterior over candidate synthesized programs**: sample best-of-N, cluster by *observed behavioral
   effect*, weight by cluster mass, and **abstain below τ**. Uncertainty is principled: no confident
   hypothesis ⇒ no claim.

The MSA twist that makes it safe: the synthesized model is used **only to order search** — the
**replay-only env is ground truth**. The SLM can make search *faster*, never *wrong*.

```
                              E119  ·  MSA Bayesian ideas inside an OpenWorld solver
                              ════════════════════════════════════════════════════

  ARC-AGI-3 env                ┌──────────  P E R C E P T O R S  (perceive→world boundary) ──────────┐
  ┌───────────────┐            │  status_mask ─ zero the always-changing status bar (>0.95 freq)     │
  │ 64×64×16 grid │   frame    │  state_key σ(s) ─ masked-frame hash  = DISCRETE STATE IDENTITY      │
  │ levels_completed ├────────►│  object_json ─ RELATIONAL scene (objects vs. the largest object)    │
  │  (replay-only)│            │  click_candidates ─ pixel-inferred sprite targets (small/rare comps) │
  └───────┬───────┘            │  probe ─ one-step (action → before/after/Δlevels) transitions        │
          │                    └─────────────────────────────┬──────────────────────────────────────┘
          │                                                  │ σ(s), object_json, probe frames
          │                                                  ▼
          │             ┌─────────  B A Y E S I A N   S U B G O A L   (the MSA core) ─────────────┐
          │             │  synthesize:  SLM samples N goal PROGRAMS  πᵢ ~ p(π | object_json)       │
          │             │       π ∈ { reach(c) | count(c,op,k) | align(a,b) } → compile_predicate  │
          │             │                                                                          │
          │             │  posterior by BEHAVIOR (not text):                                       │
          │             │     sig(π) = ( satisfiable(π, frames) , canonical(π) )                   │
          │             │     ███████  reach(5)     cluster mass ≈ p̂(π|obs)   ◄─ argmax            │
          │             │     ███      count(2,≥,3)                                                 │
          │             │     ▌        align(1,4)                                                   │
          │             │  τ-GATE:  max cluster mass ≥ τ ?  ──no──►  ABSTAIN (search runs unguided) │
          │             └────────────────────────────────┬─────────────────────────────────────────┘
          │                                               │ winning predicate π*  →  score_fn(frame)
          │                                               ▼
          │             ┌──────  E N V - G R O U N D - T R U T H   S E A R C H  (correctness here) ─┐
          ├────────────►│  search_level: BFS, or BEST-FIRST ordered by π*'s score_fn                │
          │  replay      │  every node = action prefix, REPLAYED from reset()  (env is ground truth) │
          │              │  a child that raises levels_completed  ⇒  level solved                     │
          │             └────────────────────────────────┬─────────────────────────────────────────┘
          │                                               │ action sequence
          ▼                                               ▼
  ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
  │  REPLAY-VERIFY → BANK (monotonic solved.json + JSONL log)  →  EMIT as openworld.World           │
  │  state = σ(s);  learned (σ(s), a) → (σ(s′), levels) table = FunctionTransition;                  │
  │  to_spec → preview.graph = the MAP  ·  render_card = the atlas  ·  serve /view  (emit boundary)  │
  └──────────────────────────────────────────────────────────────────────────────────────────────┘

  INVARIANT:  the SLM-synthesized model only ORDERS search; the replay-only env decides correctness.
              ⇒ a wrong/abstained subgoal costs speed, never a false solve.   (search ⊒ control rung)
```

**Rung structure (the experiment).** `--mode search` is the control (no SLM; blind BFS) and
`--mode slm` adds the Bayesian subgoal as a search-ordering prior. Identical reachable set, identical
replay-verification — the SLM only changes *expansion order*, isolating "does on-demand model synthesis
make search faster?" as the measured effect.
