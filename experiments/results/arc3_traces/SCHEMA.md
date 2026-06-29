# `runs.jsonl` schema

One JSON object per line = one **run** (one method's attempt at one game). Fields:

## Identity & routing
| Field | Type | Meaning |
|-------|------|---------|
| `run_id` | str | `<game>__<tier>__<UTC-ISO8601>`; unique, immutable. |
| `game` | str | ARC-AGI-3 game id (e.g. `g50t`). |
| `tier` | str | `cheap` (fixed pixel search) or `agent` (live coding agent). |
| `method` | str | Human-readable solver name. |
| `source_free` | bool | Always true (the protocol). |
| `fairness` | str | `by-construction (...)` (agent/sandbox) or `by-audit (...)` (fixed solver). |

## Timing
| Field | Type | Meaning |
|-------|------|---------|
| `started_at` / `ended_at` | str | UTC ISO-8601 wall-clock bounds of the run. |
| `finalized_at` | str | When the verified outcome was computed. |
| `exit_code` | int | Process exit code of the run (agent tier). |

## Model config (artifact isolation key)
`model_config`:
| Field | Type | Meaning |
|-------|------|---------|
| `requested_model` | str/null | The `--model` passed (e.g. `claude-opus-4-8`); null for cheap tier. |
| `resolved_model` | str/null | The model id reported by the session (e.g. `claude-opus-4-8`). |
| `effort` | str/null | Pinned reasoning effort (`--effort`, e.g. `high`); null for cheap. |
| `fallback_model` | str/null | `--fallback-model` if set. |
| `claude_code_version` | str/null | CLI version (e.g. `2.1.193`). |
| `fast_mode_state` | str/null | `on`/`off`. |
| `permission_mode` | str/null | e.g. `bypassPermissions`. |

## Prompt & transcript
| Field | Type | Meaning |
|-------|------|---------|
| `prompt_file` | str/null | `prompts/<run_id>.md` — exact prompt (agent only). |
| `prompt` | obj | `{chars, lines, sha256, approx_tokens}`. |
| `solution_file` | str | `solutions/<run_id>.json` — the produced action trace. |
| `transcript_file` | str/null | `transcripts/<run_id>.jsonl` — full stream-json (gitignored; on disk). |
| `transcript_sha256` | str/null | SHA-256 of the transcript file. |
| `transcript` | obj | See below. |

### `transcript` sub-fields
`{session_id, num_turns, n_messages, n_tool_calls, tool_calls_by_name, n_text_blocks, n_thinking_blocks, n_user_msgs, is_error, api_error_status, duration_ms, duration_api_ms, ttft_ms}` plus token/cost:
| Field | Meaning |
|-------|---------|
| `tokens` | `{input, output, cache_creation, cache_read, total, source}`. |
| `tokens.source` | `result_block` (authoritative, from the `claude -p` result line) · `reconstructed_from_messages` (run was cut mid-stream → summed from each assistant message's `usage`) · `none` (empty transcript). |
| `cost_usd` | Authoritative total cost from the result block (null if the run was cut before emitting it). |
| `cost_usd_estimated` | Present only when `cost_usd` is null: estimated from `tokens` × `pricing_assumed`. |
| `cost_basis` | `result_block` · `estimated_from_tokens` · `unknown`. |
| `pricing_assumed` | The `$/Mtok` rates used for the estimate (`{input, output, cache_write, cache_read, model}`) — recorded so estimates are transparent and correctable. |
| `usage` | Raw usage object (authoritative or reconstructed). |
| `params` | obj | Cheap tier: `{solver, budget, max_steps, seed}`. |

## Provenance
| Field | Type | Meaning |
|-------|------|---------|
| `host` | obj | `{hostname, platform, system, machine, python}`. |
| `git` | obj | `{commit, branch, dirty, remote}` at run time. |
| `pipeline` | obj | `{<script>: {sha256, bytes, mtime}}` for the scripts that produced the run. |
| `benchmark` | obj | `{name: ARC-AGI-3, grid: 64x64, colors: 16, actions}`. |
| `dataset_version` | str | Schema/dataset version. |

## Source-free integrity
| Field | Type | Meaning |
|-------|------|---------|
| `source_free` | bool | The protocol (always true). |
| `fairness` | str | `by-construction` (agent sandbox) or `by-audit` (fixed solver). |
| `knowledge_audit` | obj | `{clean, findings, scanned}` — scan of the agent's loaded **memory notes + CLAUDE.md** for source-*derived* content (e.g. `environment_files`, a `<gameid>.py` reference, "source-derived/faithful", "read … source"). Computed per run, so memory/CLAUDE.md contamination is self-detecting, not caught only by manual review. |
| `memory_tainted` | bool | True if `knowledge_audit` found source-derived knowledge (or a run was manually flagged). **`bank_from_runs` excludes these from the fair count.** Kept in the dataset for transparency rather than deleted. |
| `memory_tainted_reason` | str | Present when manually flagged (e.g. the 2026-06-26 source-derived-note purge). |

## Outcome (independently recomputed)
`outcome`:
| Field | Type | Meaning |
|-------|------|---------|
| `levels` | int | Levels completed by the trace (real-engine replay). |
| `win` | int | Total levels in the game. |
| `full_solve` | bool | audit-clean ∧ replay-verified ∧ round-trip-pass ∧ `levels>=win`. |
| `replay_verified` | bool | Trace raises `levels_completed` to the claimed depth in the real engine. |
| `audit` | obj | `{mode, dir|files, clean, findings}` — source-free audit. |
| `openworld_roundtrip` | obj/null | `{depth_through_world, depth_real, misses, spec_valid, card_renders, n_states, n_transitions, indexed_fallback, pass}`. |
| `actions` | list | The verified action trace (`[a]` directional or `[6,x,y]` click). |
| `action_stats` | obj | `{n_actions, n_click, n_directional}`. |

`provenance` (in the banked archive `arc3_fullgame_sourcefree.json`, not per-run) ties each banked game to
the winning `run_id`, tier, model, and effort.
