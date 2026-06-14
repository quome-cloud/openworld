# E45 - Next-token world models: exact length generalization on the LLM field's home turf

**Date:** 2026-06-13
**Status:** approved (design); pending spec review
**Supersedes:** the dropped real-repository-induction experiment (too trivial -
one-step prediction of an accounting identity).

## Goal

Land the paper's exact / OOD-transfer / auditability thesis on the LLM field's
native task - **next-token prediction** - and its native, well-documented failure
mode: **length generalization** on algorithmic sequences. The pitch is
"synthesize the rule, don't *be* the rule": the SAME local model, used to predict
the next character directly, degrades as sequences grow; used to synthesize a
verified program (a world model of the sequence's generator, induced from short
examples and verified by reproduction), produces a predictor that is **exact at
any length**. The framework turns an unreliable next-token predictor into an
exact, auditable, length-invariant one - wherever the next-token process has
exact structure.

## Tasks (deterministic next-char rules; each needs memory a fixed window cannot hold)

A scored position's next character is a deterministic function of the prefix;
"filler" input characters are seeded-random and not scored.

1. **parity** - running XOR of the input bits seen so far, emitted at query
   positions. Needs 1 bit of state over an UNBOUNDED prefix - a fixed window
   cannot compute it for long prefixes.
2. **dyck** - bracket-nesting depth, emitted as a digit at query positions.
   Needs an unbounded counter (push/pop).
3. **modk** - running count of a marker symbol modulo k, emitted at query
   positions. Needs a modular counter over the full history.
4. **incr** - a binary counter printed and incremented step by step
   (`1 10 11 100 101 ...`); predict the next number's bits. Needs carry
   propagation across unbounded length (the classic transformer addition /
   length-generalization failure).

Each task is generated deterministically from a fixed seed. Training sequences
use lengths up to `L_train`; evaluation sweeps a length axis out to ~10x
`L_train` (the OOD regime).

## Methods (all predict the next char at each scored position from its prefix)

- **symbolic (ours)** - the framework's synthesis path (E37): the LLM is given
  `(prefix -> next_char)` examples and writes `transition(state, action)` reading
  `state["prefix"]`, returning `{"next": <char>}`; accepted only if it reproduces
  the training pairs (verification). Because the program recomputes its state from
  the whole prefix, it generalizes to any length. The synthesized program is
  recorded (auditability). Reuses `extract_code` + `run_transition_code`.
- **n-gram** - next char from the last `n` chars (count table + backoff). Cannot
  represent rules needing more than `n` of memory.
- **window-MLP** - one-hot of the last `w` chars -> next-char classifier (the
  numpy MLP from E12). Fixed receptive field; fails beyond the window and OOD.
- **LLM-direct** - the SAME local model prompted to emit only the next character
  given the prefix. The punchline arm: good on short prefixes, decays with length.

## Metric

Next-char **exact accuracy** at scored positions, as a function of sequence
length (the length-generalization curve). Reported per task and averaged.
Expected shape: symbolic flat at ~1.0 across all lengths; LLM-direct high at
short lengths decaying with length; n-gram / window-MLP low (fixed memory cannot
represent unbounded-memory rules), collapsing OOD.

## Self-checks (asserts)

- symbolic mean OOD accuracy ~= 1.0 (the induced program is exact),
- symbolic OOD >> LLM-direct OOD, and >> n-gram and window-MLP OOD,
- (sanity) at least one baseline is decent in-distribution at short lengths
  (so the contrast is "collapses with length", not "never worked").

## Reproducibility

- Tasks + n-gram + window-MLP are deterministic (seeded), so those rerun offline.
- The symbolic synthesis and LLM-direct arms use a local Ollama model
  (qwen2.5:7b for direct prediction; a code model for synthesis); their results
  are recorded in the committed JSON (E37/E38 pattern), so paper assets rebuild
  offline.

## Deliverables

- `experiments/e45_next_token.py` (+ `results/e45_next_token.json`).
- Figure: next-char accuracy vs sequence length, one panel per task or a faceted
  panel, symbolic flat while the others decay; plus a bar of mean OOD accuracy by
  method. Table: per-task in-dist vs OOD accuracy by method. In
  `scripts/make_paper_assets.py`.
- Paper subsection bridging to the LLM field (length generalization / structured
  generation); `\NumExperiments` 43 -> 44.
- PR based on the E44 branch (merge after PR #26).

## Honest boundaries

- These are deterministic FORMAL sequences. Natural-language next-token is not a
  finite automaton, so the framework cannot symbolically capture English; the
  claim is scoped to next-token processes with exact structure (code, formal
  languages, grammar/JSON-constrained generation, tool/protocol outputs). That
  scoping is the honest bridge to a real LLM application (constrained decoding),
  not an overclaim about replacing language models.
- The symbolic win requires the rule to be in the synthesizable class; tasks the
  model cannot synthesize would be reported as such, not hidden.
