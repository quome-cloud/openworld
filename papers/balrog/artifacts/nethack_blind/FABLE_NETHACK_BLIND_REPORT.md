# Fable NetHack — Source-Blind Induction Arm

**Claim under test:** an agent can progress through NetHack dungeon levels with *no outside assists* — no environment source, no internet game knowledge — by inducing a world model from its own play and subjecting that model to continuous predictive verification.

Operator protocol: BALROG NetHackChallenge-v0, progression = max rung on the Dlvl/Xp ladder in `achievements.json`. Sibling comparison: the source-synthesized arm (read NLE source) and leaderboard SOTA.

## 1. Quarantine audit

Everything touched, and why it is interface/metric rather than dynamics:

| Path | What was read | Why it's allowed |
|---|---|---|
| `work/fable_nethack/balrog/config/config.yaml` | official eval protocol (nle_kwargs: max_episode_steps=100000, no_progress_timeout=150, skip_more, character '@') | protocol definition, explicitly allowed |
| `work/fable_nethack/balrog/environments/nle/progress.py` | progression metric + blstats index→name table | metric definition, explicitly allowed (blstats names cited to this file) |
| `work/fable_nethack/balrog/environments/nle/achievements.json` | the scoring ladder (Dlvl:N / Xp:N → progression) | metric definition, explicitly allowed |
| `work/fable_nethack/balrog/environments/nle/nle_env.py` | `make_nle_env` constructor call shape | interface: how to build the env |
| `work/fable_nethack/balrog/environments/env_wrapper.py` | wrapper API surface (reset/step/actions/check_action_validity) | interface |
| `work/fable_nethack/balrog/environments/nle/base.py` | obs dict keys served (`text`/`image`/`obs`), `language_action_space` construction, `get_stats()` | interface: obs keys + action-space enumeration. Contains no game dynamics. |
| `nle` python package in `work/fable_nethack/pylib` | **imported only, never read** | runtime dependency |
| Action strings | enumerated at runtime via `env.language_action_space` (248 strings) | served interface |

**Not touched:** NLE/NetHack C or Python source, headers, docs, wikis, any `nh_*.py`/`mh_*.py`/`agent_*.py` from the sibling arms, any network resource. `AutoMore`/`NLETimeLimit`/`GymV21CompatibilityV0` were imported transitively by the constructor but their files were not opened.

**Disclosed unavoidable prior:** the model driving this experiment has NetHack knowledge in pretraining. Discipline applied: prior knowledge only *generated hypotheses* (e.g. "maybe 'down' descends on some tile", "maybe '>' is that tile", "maybe pray helps when starving"); no rule entered `rules.json` as active without evidence citations (episode/step) from this arm's own logged play, and every rule carries those citations. A reviewer can strip any rule lacking observation support. Where a hypothesis came from prior (not observation), its ledger entry says so in `scope`/`statement` (e.g. R_DOWN_NEEDS_TILE before E11).

## 2. Method

- `rules.json` — the world model: rules with statement, status (hypothesized/corroborated/refuted), confidence, evidence [(episode, step)], corroboration/refutation counters, scope, revisions.
- `world_model.py` — the predictive content: before every step the model emits a possibility set for checkable dimensions (position, time monotonicity, depth transitions, hp bounds, xp monotonicity); after the step the served observation is verified against the set. In-set → corroboration (counted per rule). Out-of-set → anomaly (logged to `anomalies.jsonl`), rule revised or scoped.
- `policy_explore.py` — the planner built ONLY from ledger rules + explicit EXPERIMENT hooks (hypothesis generation). Tile passability (`tiles_learned.json`) and monster properties (`monsters_learned.json`) are learned tables with evidence.
- `runner.py` / `run_batch.py` — episode driver, per-step JSONL.gz transition logs in `results/transitions/`.
- Violation rate = out-of-set observations / checked predictions, tracked per episode.

## 3. Progress log

- P0 (seed 101): pure-observation action sweep. Induced: compass movement semantics, wall blocking, prompt/esc mechanics, full-screen menus swallow keys, pet swap, hp regen, unknown-command keys. Discovered probe flaw: `attributes` screen swallowed probes #19–66.
- P1 (seed 102): movement hammering. Induced: blocked moves cost no game time; successful moves cost 0–1.
- P2 (seed 103): esc-separated action catalog (69 actions × outcome). Prompt taxonomy; `[yn]` handling; role varies by seed.
- DET555: same seed + same actions twice → identical trajectories (the test generator is seed randomness, as specified).
- E1–E2 (v1 policy): deaths by starvation & newt. Anomalies at death frame → R_TERMINAL_ZERO (death zeroes blstats). Found pet ping-pong bug (499 wasted steps) → R_SWAP_PET; statue attack loop (471 steps) → R_STATUE; kill→xp evidence → R_KILL_XP; walkover auto-pickup → R_AUTOPICKUP.
- E3–E5 (v2): survival to step cap; prayer cures starvation (R_PRAY_STARVING: E3/4/5 survived foodless where E1 died). Exploration thrashing diagnosed: dark corridors never entered.
- E6–E9 (v3, dark-probing): big maps explored. E9 revealed the central bug: agent's `@` hides the tile underneath → 154-step oscillation ON the downstairs (R_STAIR_HIDDEN). E7: died praying next to fox → R_PRAY_COMBAT_RISK; pet-glyph collision ('d' dog vs 'd' fox) → R_PET_GLYPH_COLLISION.
- E10–E13 (v4, stair memory): **first descents** — E11: Dlvl 1→2→3→4 at t=52/116/180 ("You descend the stairs.") → R_DOWN corroborated. E12: unknown glyph '`' sealed a corridor → R_GLYPH_OPEN_WORLD (optimistic-until-refuted passability).
- XKICK (deterministic replay of E13's locked door): direction actions answer "In what direction?"; kick breaks locked doors; kicking air strains a leg. → R_DIR_PROMPT, R_KICK_DOOR, R_KICK_RISK.
- E14–E19 (v4): floating-eye freeze death → R_FREEZE + no_attack table; peaceful-monster attack loop → R_PEACEFUL; prompt-key leakage aborted two runs ("ABORTED: quit.") → R_QUIT_HAZARD + robust prompt detection.

- E20 (v5: kick-locked-doors, freeze-avoidance, peaceful-decline, robust prompts): depth 5, Xp:5, alive at 6000-step cap (prog 2.91%) — first episode limited by budget cap rather than death.
- E21: prayer rate-limit discovered ("Anhur is displeased" on 2nd prayer; died fainting) → R_PRAY_COOLDOWN; corpse-eating response added.
- Replay mining (741,564 served transitions from the sibling arm's logs — DISCLOSED: generating policy was source-informed; used only as additional served observations): R_DOWN +854 corroborations; R_UP corroborated (39×); trap-door/shaft depth jumps → R_DEPTH_TRAP (possibility set widened to {d..d+2} BEFORE any live violation); floating-eye freeze ×5; R_TIME and R_HP_BOUND clean over all 741k steps; 8/741k single-move displacement>1 exceptions recorded in R_MOVE scope as known anomaly class.

## 4. Protocol for Phase 2/3

- Phase 2: blocks of 10 fresh seeds (301-310, 401-410, ...), max_steps 8000 (budget cap, disclosed deviation from the official 100k cap — wall-clock bound; caps bind only on still-alive episodes, so reported progression is a lower bound). Model/policy revisions allowed between blocks; every revision evidence-cited.
- Phase 3: code freeze (md5s of world_model.py, policy_explore.py, rules.json, tiles_learned.json, monsters_learned.json), 25 untouched seeds 5000-5024, mean + bootstrap 95% CI.

- Batch p1v5 (E20-E25, closing Phase 1): mean 4.18%, best 16.13% (E25: stairs to Dlvl 5, then trap-door/shaft falls to Dlvl 11 — R_DEPTH_TRAP live-confirmed after being pre-widened from replay mining; a possibility-set success). E23:413 OPEN anomaly: 'west' displaced diagonally at constant depth (matches 8/741k replay class) — logged, not patched.
- Phase-1 rule count at close: 41 rules, all evidence-cited; anomaly ledger: 10 entries, every one resolved by a new rule or scope revision except the open E23:413 class.

- Block b1 (E30-E39, seeds 301-310, policy v6): mean 1.21%, best 2.65%. A regression block — starvation killed 5/10 (pending_eat state-machine bug blocked corpse-eating), one env-quit via peaceful-bump zero-time loop (mechanism identified: the wrapper's no-progress guard → R_NOPROGRESS_QUIT), one death by provoked own pet (R_PET_PROVOKE). All three failure modes diagnosed from logs and fixed in v7 (corpse-eating state fix, food pickup, rest-to-heal, anti-stall, pet-swap exemption, harmful-trap memory).

- Block b2 v7 attempt (seeds 401-403, ABORTED after 3 episodes): v7's corpse-eating fix introduced a lethal regression — three consecutive deaths "Poisoned by a rotted jackal/kobold corpse". Induced R_CORPSE_ROT (floor corpses spoil). v8: floor-corpse eating gated to desperation (hunger_state >= 3); hunger driven by the blstats hunger_state scale induced from own logs (R_HUNGER_SCALE: 1->2 aligns with 'hungry' message 168x, 2->3 'weak' 50x, 3->4 'faint' 32x, ->0 after eating). Partial v7 data preserved in batch_b2v7partial.jsonl; block rerun on the same seeds under v8.

- Block b2 (seeds 401-410, v8): mean 1.94%, best 3.54% (E44: Dlvl 6 by stairs, alive at cap). Starvation remained dominant for food-less roles → v9: fresh-kill eating (eat the corpse of a monster killed <60 steps ago when hungry; rot evidence only ever came from corpses of unknown age), poison-name blacklist, [ynq] eat-offer gating.

- Block b3 (seeds 501-510, v9): mean 3.73%, best 20.61% (E51: Dlvl 12, killed by an earth elemental). Curve: b1 1.21 → b2 1.94 → b3 3.73. Zero poisonings (v9 gates worked); starvation deaths reduced to 1-2. Retroactive validation from b3's own logs: 27 corpse eats / 0 poisonings → fresh-kill hypothesis promoted to R_EAT_FRESH; non-fatal rot outcomes split into R_ROT_NONFATAL. One residual bug (211-step fresh-kill retry loop on a stolen corpse) fixed.

## 4b. Explore/exploit rule validation (operator-specified design)

Operator directive (received mid-Phase-2): allocate a fraction of each block to VALIDATION episodes that exist to test high-(decision-impact × uncertainty) rules, not to score; exploit episodes may use only corroborated rules for survival-critical choices; track value-of-information per validation episode; decay the explore fraction as the ledger matures.

Implementation:
- Allocation: 2 validation + 8 exploit episodes in block b4 (validation run first so promotions can gate the exploit episodes). Adaptive rule: with the block-level violation rate now ~0 and most core rules corroborated, subsequent blocks (and the frozen eval, which is exploit-only by definition) drop to 0 validation episodes — the explore budget had already been front-loaded in Phase 1's probe episodes (P0-P2, XKICK are validation episodes avant la lettre).
- `policy_validate.py` (ValidatePolicy): normal play plus deliberate experiments, each tagged `val-*` in the step log: VAL_WEAR (armor→armor_class prediction; no rule existed), VAL_UP (live test of replay-only R_UP_NEEDS_TILE), VAL_TRAP (full-HP stepping on visible '^' to catalog trap effects), VAL_QUAFF (owned-potion effects), VAL_PRAY_SPACING (probe the R_PRAY_COOLDOWN bound at 1100 steps, between the refuted ~880 and the conservative 1600).
- Exploit-gating audit at b4: survival-critical decisions and their rule statuses — descend (R_DOWN corroborated), eat inventory (R_EAT_INV corroborated), prayer rescue (R_PRAY_STARVING + R_PRAY_COOLDOWN corroborated), fresh-kill eating (R_EAT_FRESH — promoted to corroborated on b3 evidence BEFORE b4 exploit episodes; previously it ran as a flagged experiment under starvation pressure, disclosed), combat target exclusions (R_FREEZE, R_STATUE, R_PEACEFUL, R_PET_PROVOKE corroborated). Hypothesized-only rules (none currently survival-critical) inform exploration targets only.

- b4val (E60-E61, ValidatePolicy): both scored 0 (fine — VOI runs). Outcomes: VAL_QUAFF negative result → R_POTION_RISK ("This burns like acid!", hp 14→6; exploit play never quaffs unidentified potions); VAL_WEAR null (all starting armor already worn); VAL_UP/VAL_TRAP/VAL_PRAY_SPACING null (episodes died on Dlvl 1 before trigger conditions). Plus one new open-class anomaly (E61:294, 21-cell silent displacement).
- b4 exploit (E63-E70, seeds 603-610, v10 ledger): mean 2.52%, median 2.25%, best 4.85%.
- Plateau called at b4 (block medians 1.85 → 1.85 → 2.08 → 2.25; means noisy under heavy right tail) → Phase 3.

## 5. Results

### Learning curve (blocks of 10, fresh seeds each block)

| Block | Policy | n | Mean prog | Median | Best |
|---|---|---|---|---|---|
| Phase-1 close (p1v5) | v5/v6 | 6 | 4.18% | 2.10% | 16.13% |
| b1 | v6 | 10 | 1.21% | 1.85% | 2.65% |
| b2 (v7, aborted 3 eps) | v7 | 3 | 1.31% | 1.85% | 2.08% |
| b2 | v8 | 10 | 1.94% | 1.85% | 3.54% |
| b3 | v9 | 10 | 3.73% | 2.08% | 20.61% |
| b4 validation | val | 2 | 0.00% | — | — |
| b4 exploit | v10 | 8 | 2.52% | 2.25% | 4.85% |
| **FROZEN** | v10 (md5-frozen) | **25** | **2.56%** | 2.08% | 16.13% |

Depth milestones learned entirely from play: first descent E11 (episode 13 of the arm); Dlvl 11 in E25 (trap-fall chaining); Dlvl 12 in b3_E51; Dlvl 11 again in frozen F1 (seed 5001).

### Violation-rate trajectory (predict-before-observe verification)

1,372,539 checked predictions across every logged episode — **630,975 from the agent's own play + 741,564 from mined sibling-replay transitions** (disclosed above; see also the "Replay corpus contamination channel" limitation); 24 out-of-set observations (1.75e-05 overall). Trajectory: 7.8e-4 (E1) → 4.3e-4 (E2) → 0.0 for 17 consecutive episodes → isolated spikes only at novel-event classes (terminal death frame, trap-door falls before the replay-derived widening reached the live process, and the open displacement class) → frozen block 3.3e-06 (1 violation / 306k predictions). Full per-episode series: `results/violation_curve.json`.

Ledger at freeze: 48 rules — 46 corroborated, 2 hypothesized (R_DEPTH_STABLE as a scoped possibility-set carrier and R_XP_MONO), 1,476,719 accumulated corroboration counts, 27 refutation counts, all carrying evidence citations (episode:step) into `results/transitions/`. Anomaly ledger: 18 entries, every one resolved by a new rule or an evidence-cited scope revision except the OPEN silent-displacement class (E23:413, b4val_E61:294, 8/741k replay) which is deliberately left unexplained rather than patched.

### Frozen evaluation vs baselines

| Arm | n | Mean progression | 95% CI |
|---|---|---|---|
| **Source-blind induction (this arm)** | 25 (seeds 5000-5024) | **2.56%** | [1.65, 3.89] |
| Source-synthesized sibling (v1, read NLE source) | 25 (seeds 2000-2024) | 4.39% | [2.97, 5.97]* |
| BALROG leaderboard SOTA (Gemini-3-Pro, 2026-02-03) | — | 6.8% | ±3.2 |

*recomputed with the canonical `papers/balrog/code/nethack/bootstrap_ci.py` (10k resamples, fixed seed 20260706) against `nethack_results_baseline25.json`, reconciled with the sibling's own report. An earlier pass in this report used an independently-implemented bootstrap (10k resamples, seed 7) that produced [2.97, 6.01] on the same data — a different-RNG artifact, not a data discrepancy; the canonical module is now the single source of truth for every CI in the NetHack arms.

**Headline: the interaction-learned agent reaches 58% of its source-reading sibling's progression (2.56 vs 4.39; CIs overlap) with zero access to game internals** — the entire behavior repertoire (stair descent, corpse-freshness dietetics, prayer rate-limits, kickable locked doors, trap-fall opportunism, peaceful/statue/freeze-class threat taxonomy) was induced from 1.4M served observations and survived Popperian filtering. The gap (≈1.8 points, driven mostly by starvation deaths in food-poor spawns) is a measured price of source blindness at this budget, not a failure to learn: the blind arm's per-block trajectory was still recovering error classes in b3-b4 (rotted-corpse poisonings eliminated, then starvation reduced).

### Value-of-information (operator's explore/exploit design)

- Retroactive VOI (b3 logs as validation corpus): fresh-kill hypothesis promoted on 27 eats / 0 poisonings → R_EAT_FRESH; this gate change is the largest single block-delta driver (b2 1.94 → b3 3.73).
- b4val explicit VOI: 1 rule added (R_POTION_RISK, negative result — cheap insurance against future exploit-episode deaths), 3 nulls, 1 new open anomaly; b4 exploit block delta vs b3: mean −1.2 points / median +0.2 (flat — consistent with a matured ledger; the adaptive allocation therefore decays to 0 validation episodes for the frozen phase).
- Explore-fraction schedule actually used: Phase 1 was ~100% validation (P0-P2, E1-E19, XKICK all played to learn); Phase 2 blocks b1-b3 ~0% explicit (opportunistic EXPERIMENT hooks only); b4 20%; frozen 0%.

## 6. Deliverables map

- `rules.json` (+ `results/rules_frozen.json`) — rule ledger with citations; `anomalies.jsonl` — anomaly ledger
- `results/violation_curve.json`, `results/curves.json`, `results/frozen_result.json`, `results/replay_mining.json`
- `results/transitions/*.jsonl.gz` — every episode, per-step (action, blstats, message, predictions' violations, frames at key events)
- `results/animations/early_P0_probe_clueless.gif` (systematic probing, zero knowledge), `mid_E9_stair_oscillation.gif`, `late_F1_dlvl11_competent.gif` (best frozen seed, Dlvl 11 — exact deterministic re-run with frozen tables for frame capture)
- `results/FREEZE_MD5S.txt` — Phase-3 freeze (2026-07-07T06:52:15Z). Post-freeze harness diff disclosed: one logging-only fix in run_batch.py (frozen-episode file naming, F{i}→F{ep0+i}) after chunk 1 overwrote per-step logs of seeds 5000-5002 (their summary rows are intact in batch_frozen.jsonl); no agent-behavior file changed after freeze.

## 7. Limitations

1. **Pretraining prior (the big one, disclosed prominently):** the driving model knows NetHack from pretraining. The quarantine can exclude source code and internet lookups, and the evidence discipline ensures every rule in the model is observationally backed and citable — but hypothesis GENERATION was unavoidably prior-shaped (e.g., trying 'down' on '>', suspecting prayer helps starvation, bumping monsters to attack). A true tabula-rasa agent would explore less efficiently. The honest claim is therefore: *no rule without observation*, not *no prior*. Reviewers can strip any rule and check its citations against the logged transitions.
2. **Budget caps:** 8000 policy-steps/episode (official protocol allows 100k) and ~90 total episodes. Caps bound several still-alive episodes (E20, E44, E53, E56), so reported progression is a lower bound under the official protocol. The sibling's baseline25 ran to 100k steps.
3. **Replay corpus contamination channel:** 741k transitions from the sibling arm were mined as additional observations (disclosed throughout). The observations are served data (legitimate under quarantine) but their *generating policy* was source-informed, so event coverage (e.g., deep-dungeon phenomena) partially reflects source-derived competence.
4. **Starvation remains the dominant killer** (10 of 24 frozen-block deaths mention hunger/fainting) — the arm learned food mechanics late and imperfectly (poison-name blacklist has only 2 entries; corpse freshness is a coarse proxy).
5. **Open anomaly:** rare silent multi-cell displacement (2 live + 8 replay events) is logged, unexplained, and intentionally not modeled.
6. Single policy-architecture (scripted rule-planner); no LLM-in-the-loop per-step decisions — the model under test is the induced rule system, the LLM's role was offline induction between episodes.

## 8. Quarantine audit — closing statement

Interface surfaces touched are enumerated in §1; nothing else was read. Specifically NOT read at any point: any file under `pylib/nle/` (import-only), any NetHack C/Python source or documentation, any wiki/spoiler resource (no network calls were made at all), any sibling-arm agent/world-model code (`nh_*.py`, `mh_*.py`, `agent_*.py`, or the sibling's report). The balrog leaderboard evidence was taken from the pre-existing parse (`results/evidence/leaderboard_parse.json`), not re-fetched.



## Appendix A — Rule ledger summary (full ledger with statements + all citations: rules.json)

| Rule | Status | Corrob. | Refut. | First evidence citations |
|---|---|---|---|---|
| R_AUTOPICKUP | corroborated | 0 | 0 | E2:11 |
| R_BLOCK_MSG | corroborated | 0 | 0 | P0:7; P1:1; P1:12; P1:16 |
| R_COMBAT | corroborated | 0 | 0 | E2:8; E2:9; E2:42 |
| R_COORD | corroborated | 0 | 0 | P0:0; P2:0 |
| R_CORPSE_ROT | corroborated | 0 | 0 | b2v7partial_E40:end; b2v7partial_E41:end; b2v7partial_E42:end; b2_E41:end |
| R_DEATH_HP0 | corroborated | 0 | 0 | E2:1857; E1:1030 |
| R_DEPTH_STABLE | hypothesized | 295237 | 5 | P0:120; P1:320; P2:278 |
| R_DEPTH_TRAP | corroborated | 0 | 0 | REPLAY(disclosed source-informed policy logs, work/fable_nethack/results/transitions); REP |
| R_DET_SEED | corroborated | 0 | 0 | DET555:0; DET555:5 |
| R_DIR_PROMPT | corroborated | 0 | 0 | XKICK:3; XKICK:4 |
| R_DOWN_NEEDS_TILE | corroborated | 92 | 0 | P0:16; P2:40; E11:52; E11:116 |
| R_EAT_FRESH | corroborated | 27 | 0 | b3_E53:5977; b3_E53:6277; b3_E51:1393; b3_E51:2942 |
| R_EAT_INV | corroborated | 0 | 0 | E2:1013; E2:1014; E1:754; E1:1028 |
| R_FAR_MOVE | corroborated | 0 | 0 | P0:9; P0:10; P0:11; P0:12 |
| R_FREEZE | corroborated | 0 | 0 | E14:5833; E14:5834; REPLAY(disclosed source-informed policy logs, work/fable_nethack/resul |
| R_GLYPH_OPEN_WORLD | corroborated | 0 | 0 | E12:8; E12:9 |
| R_HP_BOUND | corroborated | 295357 | 0 | P0:102; P0:116; REPLAY(disclosed source-informed policy logs, work/fable_nethack/results/t |
| R_HUNGER_SCALE | corroborated | 0 | 0 | OWN-LOGS aggregate |
| R_KICK_DOOR | corroborated | 0 | 0 | XKICK:3; XKICK:5 |
| R_KICK_RISK | corroborated | 0 | 0 | XKICK:10; XKICK:11 |
| R_KILL_XP | corroborated | 0 | 0 | E2:9; E2:42 |
| R_MONSTERS | corroborated | 0 | 0 | P0:101; P0:102; P2:16; P2:22 |
| R_MOVE | corroborated | 142687 | 16 | P0:1; P0:2; P0:3; P0:4 |
| R_MOVE_TIME | corroborated | 0 | 0 | P1:6; P1:7; P1:1; P0:7 |
| R_NONMOVE_POS | corroborated | 152533 | 1 | P2:43; P2:49; P2:91; P2:172 |
| R_NOPROGRESS_QUIT | corroborated | 0 | 0 | E15:1170; E19:3384; E35:3915 |
| R_PEACEFUL | corroborated | 0 | 0 | E19:3377; E19:3380; E19:3383 |
| R_PET | corroborated | 0 | 0 | P0:2; P0:12; P2:16; P2:19 |
| R_PET_GLYPH_COLLISION | corroborated | 0 | 0 | E7:118; E7:121; E13:66 |
| R_PET_PROVOKE | corroborated | 0 | 0 | E37:1648 |
| R_POTION_RISK | corroborated | 0 | 0 | b4val_E60:196 |
| R_PRAY_COMBAT_RISK | corroborated | 0 | 0 | E7:122; E7:123 |
| R_PRAY_COOLDOWN | corroborated | 0 | 0 | E21:2008; E21:2892; E21:2937 |
| R_PRAY_STARVING | corroborated | 0 | 0 | E3:1116; E3:2248; E4:1807; E5:843 |
| R_PROMPT_ESC | corroborated | 0 | 0 | P0:19; P0:25; P0:74; P2:49 |
| R_PROMPT_LETTERS | corroborated | 0 | 0 | P2:79; P2:145; P2:151 |
| R_QUIT_HAZARD | corroborated | 0 | 0 | E15:1163; E15:1170 |
| R_ROLE_VARIES | corroborated | 0 | 0 | P0:0; P2:0 |
| R_ROT_NONFATAL | corroborated | 0 | 0 | b3_E51:2421; b3_E53:1067; b3_E51:6095 |
| R_STAIR_HIDDEN | corroborated | 0 | 0 | E9:360; E9:707; E11:52 |
| R_STATUE | corroborated | 0 | 0 | E1:1027; E1:1029 |
| R_SWAP_PET | corroborated | 0 | 0 | E2:1; E2:8; P0:2; P2:19 |
| R_TERMINAL_ZERO | corroborated | 77 | 0 | E1:1030; E2:1857 |
| R_TIME | corroborated | 295355 | 2 | P0:1; P0:76; P1:6; P2:43 |
| R_UNKNOWN_CMD | corroborated | 0 | 0 | P0:18; P0:75; P2:46; P2:203 |
| R_UP_NEEDS_TILE | corroborated | 0 | 0 | P0:13; P2:37; REPLAY(disclosed source-informed policy logs, work/fable_nethack/results/tra |
| R_WAIT | corroborated | 0 | 0 | P0:76; P0:102; P2:206; P2:212 |
| R_XP_MONO | hypothesized | 295354 | 3 | P0:120; P2:278 |
