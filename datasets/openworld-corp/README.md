# openworld-corp

A **synthetic-but-realistic corporate transcript dataset** for driving a company
world model: LLM-generated meeting transcripts (all-hands, division reviews,
standups, 1:1s) and Slack threads for a DigitalOcean-style PaaS company, each
**grounded in ground-truth division metrics** so a perceptor can read division
health back out of the prose.

It is meant to be *used*, not admired: drop in your own org and metrics, generate
a corpus with a local model or a hosted API, or replace the corpus with your
company's real transcripts (same schema) and run the world model on a live org.

## Files

- `org.py` — the organization seed: CEO, five divisions, named people with
  levels, and per-division metrics (revenue, growth, headcount, open roles, and
  the latent productivity `a` the world model uses). **Edit this to model your
  company.**
- `generate_corpus.py` — the LLM generator. Pluggable backend via `BaseLLM`.
- `corpus.json` — a committed, real LLM-generated sample so the dataset is usable
  offline with no model.
- `CARD.md` — dataset card (provenance, intended use, limitations).

## Record schema (`corpus.json` → `records[]`)

Meeting record:
```json
{
  "id": "division_review-database-01",
  "type": "all_hands | division_review | standup | one_on_one",
  "period": "2026-Q2",
  "scope": "company | <division>",
  "participants": [{"name": "...", "role": "...", "level": "...", "division": "..."}],
  "ground_truth": { "revenue": 200, "growth": 0.42, "headcount": 10,
                    "open_roles": 4, "a": 2.0 },
  "transcript": "Priya Raman: ...\nSam Cole: ..."
}
```
Slack record: same head, with `"type":"slack"`, `"channel"`, and `"messages"`
instead of `transcript`. `all_hands.ground_truth` carries every division's
metrics (`{total_revenue, divisions:{...}}`).

## Generate your own

Local (Ollama):
```bash
python datasets/openworld-corp/generate_corpus.py --model qwen2.5:7b
```

Hosted API — implement one method and pass it in (no keys live in the repo):
```python
from openworld import BaseLLM
from datasets.openworld_corp.generate_corpus import generate  # or import the module

class ApiLLM(BaseLLM):
    def chat(self, messages, **opts):
        # call Anthropic / OpenAI / etc. here, return the assistant text
        ...

corpus = generate(ApiLLM())          # same grounded prompts, your model
```

## Use your company's real data

Skip generation entirely: write your real transcripts into `corpus.json` in the
schema above (set `ground_truth` to the metrics each meeting concerned). The
world model and the `TextPerceptor` consume them unchanged — that is the "drop
in" path to running this on an actual organization.

## How the world model uses it

`experiments/e48_corporate_world.py` builds the org as a nested `CompositeWorld`
and uses the corpus as the **perception boundary**: each role extracts division
health from the prose it actually sees (CEO ← all-hands, director ← its review,
IC ← 1:1) via `openworld.TextPerceptor`, then acts on that perceived state. The
experiment measures the cost of acting on transcript-derived state vs ground
truth — which is largest for the CEO, who relies on coarse all-hands aggregates.
