# FABLE_CRAFTER_REPORT — world-model synthesis + classical search on BALROG Crafter (first stochastic arm)

**Synthesis model:** Fable 5 (max reasoning). Runtime is LLM-free: pure code, no API calls, nothing in the scored loop touches a network or a model.
**Claim under test:** the world-model-synthesis + classical-search recipe from the Baba Is AI arm (100.0% there) extends to a STOCHASTIC, partially observable domain via closed-loop replanning. Crafter is the pilot; the design decisions here feed the MiniHack/NetHack arms.
**Protocol:** CLEAN ONLY (operator directive mid-run): the scored agent consumes exactly what BALROG's Crafter wrapper serves a text agent, acts via reset()/step() only. A privileged full-state arm existed as a development diagnostic before the directive and is disclosed in §8; no scored number comes from it.

## Headline

| | progression | SE | mean achievements | deaths |
|---|---|---|---|---|
| **Condition A — memoryless, clean (leaderboard-comparable)** | **47.3%** | ±5.3 | 10.4 / 22 | 10/10 |
| **Condition B — cross-episode memory, clean** | **48.6%** | ±3.0 | 10.7 / 22 | 10/10 |
| BALROG Crafter SOTA (Grok-4; Gemini-3-Pro ties) | 57.3% | ±3.9 / ±4.4 | ~12.6 / 22 | — |
| Claude-Opus-4.5 | 49.5% | ±3.1 | — | — |
| Gemini-3.1-Pro-Thinking | 55.0% | ±6.4 | — | — |
| Gemini-3.1-Pro | 46.8% | ±4.2 | — | — |

SOTA was **not** beaten in this arm: 47.3% sits at Claude-Opus-4.5's level, ~10pp below Grok-4/Gemini-3-Pro. The memory condition's main effect is **variance, not mean**: SE halves (5.3→3.0) and the catastrophic tail disappears (worst episode 36.4% vs 13.6%). Death, not tech-tree competence, is the binding constraint in both conditions (10/10 episodes end in death; §6).

Leaderboard note: column order on balrogai.com is `BabyAI | Crafter | TextWorld | BabaIsAI`. The Baba arm's recorded "75.7% BabaIsAI SOTA (Gemini-3.1-Pro-Thinking)" is actually that model's **TextWorld** number; BabaIsAI SOTA is 90.0% (Gemini-3.1-Pro). The Baba arm's 100.0% still exceeds it; its report should be corrected.

## 1. BALROG Crafter protocol (discovered from the repo/package)

- Single task "default": `crafter.Env(area=(64,64), view=(9,9), size=(256,256), reward=True)` wrapped in `CrafterLanguageWrapper(max_episode_steps=2000, unique_items=True, precise_location=False, skip_items=[], edge_only_items=[])`. The wrapper's `reset()` takes one hidden Noop step; the evaluator then loops `for step in range(env.max_steps)` = 2000 (not crafter's native 10k).
- **10 episodes**, per-episode progression = achievements_unlocked / 22 (`get_stats()`), suite score = mean × 100.
- Observation actually served to a text agent: `obs["text"]["long_term_context"]` (status line; "You see: closest-instance-per-type with Manhattan distance + coarse direction, within the 9×7 view; You face X at your front") and `obs["text"]["short_term_context"]` (vitals + inventory). `unique_items=True` collapses each type to its closest instance; `precise_location=False` reports "N steps to your south-west" (distance + signed direction class, not exact offsets).
- **Harness findings:**
  1. `env.reset(seed=...)` is a **silent no-op** for Crafter: `GymV21CompatibilityV0.reset` calls the deprecated `gym.Wrapper.seed`, which never reaches `crafter.Env._seed`; BALROG leaves the constructor seed null ⇒ official episodes are unreproducible (same bug class as Baba's `Game(seed=...)` no-op). We seed the constructor per episode and record it.
  2. Even constructor-seeded, crafter is only reproducible **within a process**: chunk balancing iterates Python sets of live objects (id-ordered), so mob spawn/despawn choices vary across processes with identical seeds.
- SOTA recorded from balrogai.com (2026-07-06): Crafter column top = 57.3 (Grok-4 57.3±3.9, Gemini-3-Pro 57.3±4.4).

## 2. World-model synthesis (crafter_model.py) and validation

**(a) Deterministic core — exact.** Mirrors data.yaml + objects.py/env.py semantics: 12 materials + walkability, all collect/place/make recipes with tool gates and the 3×3 `nearby` rule, movement/facing coupling, lava death, the four hidden vitals timers (hunger 1/tick → food−1 at >25; thirst 1/tick → drink−1 at >20; fatigue → energy∓1 at >30/<−10; recover → health±1 at >25/<−15, all halved/modified while sleeping), inventory clamping, the daylight clock `1−|cos(π((t/300)%1+0.3))|³`, and the load-bearing update-order fact: **the player is always the first object updated**, so obs→action outcomes for the player are deterministic given the served pre-step state (mobs seen in the obs cannot pre-empt the player's next action).

**Validation (offline, validate_model.py — dev tool, never in scored runs):** one-step lock-stepped prediction of the player-controlled state component (pos, facing, sleeping, full inventory, achievements, hidden timers, terrain edits) from full pre-step state over a 20-episode mixed random/boosted-inventory sweep: **0 mismatches in 3,399 compared steps** (1,068 steps hostile-excluded by rule, 633 sapling draws excluded as the one stochastic recipe). Daylight formula exact 3,399/3,399.

**(b) Stochastic transition spec — distributions, not points** (`STOCHASTIC_SPEC`): sapling Bernoulli(0.1); cow random-walk (move attempt w.p. 0.5); zombie chase (w.p. 0.9 toward player within 8, melee 2 — 7 on sleepers — cooldown 5); skeleton retreat/shoot/approach bands (arrows: deterministic 1 cell/tick flight, 2 dmg, destroy tables/furnaces); chunk spawn/despawn every 10th step with daylight-dependent targets (zombie pressure ⇒ unbounded night exposure, day despawn). Distributional validation: sapling **0.098** observed (n=633) vs 0.1; cow move 0.442 vs 0.5×P(free dir); zombie chase 0.81 among successful moves (consistent with 0.9 chase share once blocked chase steps and the 10% random-move stream are accounted); plant growth-gating (frozen beyond Manhattan 18) taken from source, not exercised in the sweep.

**What cannot be lock-stepped and why:** mob action choices, spawns, and the sapling draw consume the env's private `np.RandomState`, whose call stream interleaves with *rendering* (night vignette noise consumes RNG draws). Bit-mirroring it would mean reimplementing the whole env including the renderer's RNG consumption — at which point the "model" is the env. Hence: exact core validated pointwise, stochastic components validated distributionally and handled by planning structure (§3), the same split we propose as the recipe's general form for stochastic domains.

## 3. Recipe evolution for stochasticity (the research contribution)

Baba arm: synthesize exact model → search offline for a full action sequence → execute open-loop. Sound because nothing else moves. In Crafter that is unsound on three axes: stochastic mobs/spawns, a stochastic recipe, and (clean protocol) partial observability. The recipe evolved into:

**3.1 Three-layer closed-loop planner, replanned every step (brain.py).**
- **L1 reactive safety** — uses *worst-case bounds* from the stochastic spec, which are exact even though transitions are random: zombies close ≤1 cell/tick and deal 2 per 6 ticks adjacent; arrows fly 1/tick in-line for 2; lava kills. Adjacent zombies are always fought (our DPS 2–5/hit beats their bounded 2/6-ticks; fleeing keeps them adjacent 90% of ticks — measured death spiral).
- **L2 vitals scheduler** — the vitals clocks are deterministic, so deadlines are computable: thresholds trigger drink/eat/sleep with hysteresis (top up to full once engaged) sized against worst-case travel times.
- **L3 achievement DAG** — subgoals in tech-tree order (wood→table→wood tools→stone→stone tools+furnace→coal/iron→iron tools→diamond) interleaved with opportunistic goals (sapling, plant, cow, mob kills, sleep). Each subgoal compiles to *navigate + face + primitive* via A\* over the believed map with mine-through costs. Search runs on the synthesized model only — the env is never simulated or cloned.
- **Determinization:** mobs are planned around as static obstacles with graded danger costs; per-step replanning absorbs their actual motion; their attack dynamics live in L1's worst-case bounds. This determinize-and-replan + worst-case-safety split is the transferable pattern for MiniHack/NetHack.

**3.2 Belief state under the honest observation (belief.py).** The text obs is lossy (closest-per-type only, coarse directions). Three synthesized-model facts make it workable:
- **Distance-1 exactness lemma:** any adjacent cell's material must belong to {types reported at distance exactly 1} (else that type's closest would be closer than reported), and distance-1 reports are always pure-cardinal hence exact cells. Corollary: *adjacent lava is always detectable* → the executor can hard-guarantee never stepping into lava; walkability of the next cell is usually certain.
- **Dead reckoning is deterministic:** spawn is always the area centre (32,32); the player updates before mobs, so move success is predictable whenever the target cell's content is known; ambiguous moves (unknown cell, mixed-walkability candidate set) fork a bounded 2-hypothesis tracker scored against subsequent reports. Dev verification: **0 position errors and 0 wrong map cells (302 audited) over a full episode**; scored-run diagnostics show ~7–15 ambiguous steps/episode, 0 relocalizations.
- **'floor' cells:** every visited cell is walkable even if its type is unknown — marking these was worth more to pathfinding than any other single change.
Map knowledge accrues at ~1 exact cell/step (faced cell + pure-cardinal reports + sharpened diagonals); ore cells seen through rock (the 9×7 semantic window is X-ray) are remembered as hints.

**3.3 Emergent-constraint discoveries** (found by closed-loop failure analysis, all traceable to source semantics):
- **Facing follows movement**: you cannot turn toward a walkable cell without stepping onto it. Consequence: "face open cell + place stone" is not a primitive — you can only reliably place on the cell you already face. This kills naive walled-box shelters.
- **The dig-and-cork burrow** is the only self-sealing shelter the action set allows: mine 2 cells into a stone face, step in, step back (you now face the entrance), place one stone on it. Net stone +1 (2 mined, 1 placed). Flank soundness is verified *while digging* via the distance-1 lemma, aborting and blacklisting leaky corridors.
- **Table corks are tombs**: data.yaml has no collect entry for 'table' — players can never remove a placed table; only skeleton arrows destroy them. A table-corked burrow is self-entombment. (Cork = stone only.)
- **No zero-tech shelter exists**: any cork requires stone ⇒ wood pickaxe ⇒ table ⇒ wood. Night 1 is therefore a hard tech race — the great filter of Crafter — and open-field night fighting is mathematically losing (spawn-queue DPS ≥ 2×2/6-ticks sustained exceeds kill-rate + regen).
- **Sleep is a commitment**: a sleeping player cannot act until energy is full or they are hurt (7-damage zombie hits). "Nap in a calm moment" is a death sentence; sleep only corked (night) or in verified-clear daylight (zombie day-despawn makes morning naps safe).
- **Home-base persistence**: the first verified burrow is reused nightly with a return-by-dusk constraint (distance-to-home budgeted against the daylight clock), converting per-night shelter search variance into a one-time cost. Episodes that established a home lived 2–4× longer (paired data in results/).

## 4. Deliverables map

```
fable_crafter/
  crafter_model.py      synthesized world model (exact core + stochastic spec)
  belief.py             TextBelief (clean protocol belief state)  [scored path]
  brain.py              3-layer closed-loop planner               [scored path]
  run_suite.py          clean suite runner, checkpoints, logs     [scored path]
  memory.py             condition-B ledger + provenance           [scored path]
  balrog_text_env.py    vendored BALROG wrapper (the obs channel) [scored path]
  dev_privileged.py     privileged belief — DEV/DIAGNOSTIC ONLY
  validate_model.py     offline model-fidelity sweep (dev only)
  explore_data.py       induction-dataset exploration collector
  results/
    condition_A/ ep_*.json   condition_B/ ep_*.json   (full action logs inside)
    summary_A.json summary_B.json  memory_ledger.json  model_validation.json
    transitions/ *.jsonl.gz  (38 files, 9,696 transitions, one per episode)
    animations/  *.mp4 (21) + highlights/ (named picks)
```

## 5. Results

### Condition A (memoryless, clean; official 10-episode suite, fresh seeds 9001–9010 — dev-tuning used disjoint seeds 7001–7010)

**47.27% ± 5.34** (mean achievements 10.4/22; per-episode: 13.6, 50.0, 59.1, 27.3, 54.5, 50.0, 40.9, 72.7, 45.5, 59.1; best episode 16/22). All 10 episodes end in death (median lifetime ~260 of 2000 steps). Achievement reliability across A episodes: collect_drink/collect_wood/defeat_zombie 10/10; eat_cow, wood tools, table 9/10; sapling 8/10, place_plant 8/10, collect_stone 7/10, place_stone 6/10; wake_up + stone_pickaxe 3/10; furnace 2/10; coal 1/10; stone_sword 1/10; **never in A**: iron chain (collect_iron, iron tools), diamond, eat_plant, defeat_skeleton.

Death causes (classified from the agent's own belief at death): 3 zombie-at-night-without-home, 3 starvation, 1 skeleton_arrows, 1 zombie_day, 1 zombie_night (home existed, caught outside), 1 unknown.

### Condition B (cross-episode memory, clean; same seeds, ledger seeded from A's episode files then updated after each B episode)

**48.64% ± 2.96** (10.7/22; per-episode: 45.5, 45.5, 59.1, 50.0, 63.6, 59.1, 45.5, 45.5, 36.4, 36.4).

Memory entries fired (all four rules, provenance = A episode files in results/condition_A/):
- `earlier_shelter_prep` (cites 3 zombie_night_no_home deaths) → prep_start 108→92, cork-stone urgency on;
- `bigger_vitals_margin` (3 starvation deaths) → vitals floor 4→6;
- `wider_skeleton_avoidance` (1 skeleton death) → zone radius 3→4, hunt gate hp 7→9;
- `earlier_plant_pipeline` (long lives without eat_plant) → sapling budget 30→45.

Pass-by-pass: 45→45→59→50→64→59→45→45→36→36. The curve *rises* through mid-pass then **falls as the ledger over-adapts**: each additional night death ratchets prep_start earlier (92→…→60 by ep9), consuming up to a third of each day on shelter logistics for worlds where the true problem was corridor availability, not timing. Paired per-seed: B wins 5, ties 1, loses 4; B's floor is dramatically better (36.4 vs 13.6; A's seed-9001 starvation death at 3/22 becomes 10/22 under the vitals rule — see highlights video), B's ceiling is worse (72.7→45.5 on seed 9008).

**Findings on the operator hypothesis** ("consecutive play with long-term memory → right answers with fewer attempts"): supported for *tail risk* (variance halves, catastrophic first-attempt failures eliminated), not for the mean (+1.4pp, within noise). And a clean negative result: **monotone adaptation rules over-fire** — the ledger needs regression-aware rules (back off when a rule's parameter change doesn't reduce the cited death cause) rather than ratchets.

**Source-derived vs experience-derived knowledge:** the source-synthesized model already fixes everything classic "world-model learning" would learn (mob HP/damage/cooldowns, recipe gates, vitals clocks, daylight). What was left for memory was *policy calibration against environment statistics* (how early shelter prep must start in practice, real vitals burn rates including detours, skeleton-zone cost). That residue is real but small at this policy's capability level — the honest conclusion is that **source-synthesis crowds out most of what episodic memory could contribute**, and the remaining deaths are a capability gap (guaranteed shelter under partial observability), not a parameter gap.

## 6. Why we die (and why this is the interesting number)

10/10 deaths in both conditions, median lifetime ~13–15% of the episode budget. The tech tree through stone tools is reliable (§5) — the score is *survival-limited*: every death forfeits the iron tier (needs furnace+coal+iron+table logistics), diamond, eat_plant (301 growth ticks near the plant), and usually wake_up. The privileged development diagnostic (disclosed, unscored) plateaued at ~52–57% with the *same* brain and full observability, and its best episodes reached 18–20/22 — so roughly: ~5pp of the gap to SOTA is observability (map/threat knowledge), and the rest is night-1-shelter reliability, which is a *planning-under-partial-observability* problem: the burrow needs a known 2-deep stone corridor by dusk, and the honest map often hasn't seen one yet. LLM SOTA agents (57.3%) die too — Crafter's leaderboard ceiling is survival-shaped; nobody is iron-tier-consistent at 10 episodes.

## 7. Induction dataset (for the source-blind leg)

38 gzip JSONL files, **9,696 transitions** (~0.7 MB): 3,228 condition-A + 3,020 condition-B (competent play) + 3,448 exploration. Each record: served text obs, action, reward, done, and info-as-served (inventory, achievements, discount, player_pos, full 64×64 semantic map base64). Exploration set: 3 uniform-random episodes, noop-only vitals-decay episode, systematic action×facing probing with empty inventory (failed-precondition no-ops), boosted-inventory random (recipe/placement dynamics), deliberate lava death, open-air night-sleep (7-damage sleeping-zombie demo), skeleton-fire death, plant lifecycle ×4, ore tours, iron smithing.

**Coverage:** all 22 achievement mechanics appear in the union of logs (verified by scan): the A/B runs cover the wood/stone economy, combat, sleep; exploration adds collect_iron, collect_diamond, eat_plant, make_iron_pickaxe, make_iron_sword. **Caveats for the induction agent:** (1) two demos are *assisted* — boosted initial inventories (disclosed in file meta; the boost appears only as the initial condition, never as a mid-episode discontinuity) and one plant demo fast-forwards `plant.grown` to 295 after placement (meta-disclosed; the eat-ripe-plant transition itself is authentic). (2) Rare mechanics appear once each (diamond, iron recipes, eat_plant) — thin but present. (3) Arrows destroying tables/furnaces, multi-mob pile-ups, and deep-cave dynamics appear only incidentally. (4) The night-render darkening affects only the image channel, not the text/semantic channels logged.

## 8. Source-leak audit (clean-protocol accounting)

**Consumed by the scored agent, per step:** `obs["text"]["long_term_context"]` and `obs["text"]["short_term_context"]` — nothing else. Both are exactly what BALROG prompts a text LLM agent with. The agent additionally knows the published observation-format semantics and the offline world model (below).

**Audit method:** grep of the scored import path (`run_suite.py → belief.py, brain.py, crafter_model.py, memory.py, balrog_text_env.py`) for `env.`, `_world`, `_player`, `_mat`, `_sem` access. Result: **zero env-internal touchpoints in the scored path.** All privileged access lives in `dev_privileged.py` (development diagnostics) and `validate_model.py` (offline model fidelity), neither imported by `run_suite.py`; `explore_data.py` uses env internals for *policy targeting* in the induction dataset only (per-file meta discloses the policy; logged data is as-served). Harness-side (not agent input): `wrapper.get_stats()` and the step `info` dict for scoring/logging — the same channels BALROG's own evaluator uses; `obs["image"]` for the mp4 renders.

**Episode-independent offline knowledge (the disclosed asterisk, as in the Baba arm):** `crafter_model.py` was synthesized by Fable 5 from reading the crafter 1.8.3 source before any scored episode. It contains only episode-independent rules (recipes, dynamics, constants like spawn-at-centre and the daylight formula). Nothing episode-specific flows at test time: no worldgen seeds peeked, no terrain lookups outside the served window, no plan verification against the env (no clone, no replay — there is nothing open-loop to verify anyway).

**Provenance disclosure (per operator rules):** (1) Condition-B memory entries cite only clean episode files present in `results/condition_A/` (mechanically checkable in `memory_ledger.json` + per-episode `memory_entries_fired`). (2) Before the clean-only directive, a privileged full-state arm ran as the development diagnostic (seeds 7001–7010; scores ~50–57%); those runs are excluded from all headline numbers and ledgers; official runs use disjoint seeds 9001–9010. (3) Honest residue: the *author* of this code iterated on policy thresholds while privileged diagnostics were visible; parameters therefore fall into three provenance classes — (a) source-model-derived (mob damage/HP, cooldowns, night window, ripeness age, recipe costs: fixed, episode-independent), (b) dev-tuned scalars (vitals floors, hp gates, phase offsets: tuned during development, partly against privileged traces — disclosed, and exactly the parameters condition B re-derives from clean data), (c) clean-data-derived (all condition-B ledger adjustments). The quarantined source-blind induction leg (dataset in §7) removes the remaining asterisk.

## 9. Animations (results/animations/, highlights/)

One mp4 per official episode (served RGB frames + overlay strip: step, day/night, subgoal, action, achievement ticker). Curated picks in `highlights/`: `best_run_16of22_A_seed9008.mp4` (wood→stone→furnace arc, 72.7%), `death_starvation_A_seed9001.mp4`, `death_zombie_night_no_home_A_seed9005.mp4`, `death_skeleton_arrows_A_seed9009.mp4`, and `memory_fixed_starvation_B_seed9001.mp4` (same world as the starvation death; the `bigger_vitals_margin` ledger entry visibly changes behavior — early water/food topping — and the episode scores 10/22 instead of 3/22).

## 10. Limitations & reviewer-grade caveats

1. **SOTA not beaten** (47.3% vs 57.3%). The Baba-arm's "exact model + search ⇒ ceiling" story does not transfer intact to stochastic, partially observable, survival-gated domains; what transfers is the *method decomposition* (exact core / stochastic bounds / closed-loop replanning). We report the gap and its decomposition (§6) rather than a privileged number that would look better.
2. **Small samples**: 10 episodes/condition (benchmark protocol); A-vs-B differences except variance reduction are within noise. The B curve confounds learning with world difficulty ordering (mitigated by seed pairing with A).
3. **Env nondeterminism**: even constructor-seeded, crafter varies across processes (set-iteration order in chunk balancing). Suites are single-process; exact replays of these runs require same-process execution; transition logs are the ground-truth record.
4. **Author-level provenance residue** on dev-tuned scalars (§8.3) — disclosed; removable only by the source-blind leg.
5. **Achievement self-tracking** in the honest agent is imperfect (3 false-positives / 8 false-negatives vs 104 true across A) — it only mis-schedules subgoals; scoring uses the env's own ledger via the harness.
6. **Memory rules are hand-written** (four ratchets). Their over-adaptation is itself a finding, but a principled version (regression-tested adaptations) is future work.
7. The BALROG leaderboard comparison is text-protocol-faithful but not submission-identical (their agents also get invalid-action feedback and message history; we use neither).

## 11. What the MiniHack/NetHack arms should inherit

1. The **exact-core / stochastic-bounds split**: lock-step-validate everything the player controls; write distributions + worst-case bounds for everything else; plan on the core, guard with the bounds, replan every step.
2. **Partial-observation exactness lemmas**: derive what the obs format *guarantees* (here: distance-1 exactness) and build the belief update on guarantees, not heuristics — 0 localization errors came from that.
3. **Survival scheduling as a first-class plan layer**: deterministic resource clocks + a *guaranteed-shelter invariant* (here: cork-by-dusk). In NetHack the analogues are prayer/food clocks and escape items.
4. **Failure-ledger memory** helps tails, not means, when the world model is already source-exact; spend memory capacity on *capability* gaps (e.g., cached shelter geometries per biome), not scalar ratchets.
5. Expect **emergent action-interface constraints** (facing-follows-movement, irreversible placements): enumerate irreversibilities from the model before trusting any "build structure" plan.

## Run log
- 15:46 dirs created; crafter 1.8.3 + gym 0.25 installed; BALROG cloned; protocol + seeding no-op confirmed; SOTA column verified from raw leaderboard HTML (57.3).
- 16:02 crafter_model.py synthesized (exact core + stochastic spec).
- 16:0x–18:35 planner/belief development loop on dev seeds 7001–7010 (privileged diagnostics until the clean-only directive; honest arm thereafter): burrow discovery, table-tomb discovery, sleep-commitment rule, home base, vitals hysteresis; locked at honest ~52% dev-mean.
- 18:44–18:46 **Condition A official** (clean, fresh seeds): 47.27% ± 5.34, checkpointed per episode + transitions + mp4s.
- 18:47–18:52 model validation sweep: 0/3,399 mismatches; distributional checks pass. Ledger seeded from A; **Condition B official**: 48.64% ± 2.96, all four memory rules fired with file-level provenance.
- 18:5x exploration dataset (16 episodes incl. forced demos); coverage scan: 22/22 achievement mechanics demonstrated; highlights curated; this report.

---

# V2 CYCLE (T386 "fix the crafter") — powered blocks, an ablation inversion, and the shipped v1.1 agent

**Operator directive:** survival-first scheduling, threat-aware pathing, recalibrating memory; success bar >57.3 clean memoryless on a 25-episode untouched-seed block with bootstrap CI (NetHack-arm discipline: code-freeze md5s, the powered block is the number we trust).

## V2 headline (what actually happened)

| agent (frozen) | 25-ep untouched block | bootstrap 95% CI | verdict vs SOTA 57.3 |
|---|---|---|---|
| v1 (the "not beaten" baseline) | **55.45** (seeds 11001-11025) | [50.73, 60.18] | **at-SOTA (straddles)** |
| v2 (full survival-first rebuild) | **44.91** (same seeds 11001-11025) | [39.45, 50.00] | below SOTA — **and below v1: CIs disjoint** |
| **v1.1 (SHIPPED: v1 + 2 anti-churn bugfixes)** | **56.00** (fresh seeds 12001-12025) | **[50.91, 61.09]** | **at-SOTA (straddles)** |

The success bar (>57.3 decisively) is **not met**: the shipped agent is statistically indistinguishable from SOTA (Grok-4/Gemini-3-Pro 57.3±3.9/±4.4 at n=10), with a point estimate 1.3pp under it. Official 10-ep protocol numbers for continuity: v1.1 A (seeds 9001-9010) 45.0±4.3, v1 47.3±5.3, v2 47.7±3.5 — that seed block is simply harsh; all three agents are within noise of each other on it, which is itself the n=10 lesson again.

## The ablation inversion (main scientific result of the cycle)

v2 implemented the directive faithfully: a dusk-horizon scheduler with shelter-ETA preemption of tech goals, prospect-for-stone stages, funnel fallbacks, continuous night-scaled threat fields with skeleton line-of-fire ridges, interior (depth-shifted) burrow corridors, extension-on-abort, and windowed recalibrating memory. Dev evaluations (n=10-20, seeds 7001-7020) suggested progress (peaks of 58.6%). The powered block then showed **v2 is 10.5pp WORSE than v1 on identical untouched seeds, with non-overlapping CIs.**

Diagnosis, in order of confidence:
1. **Day-1 banking is the dominant score term, and survival-first taxes it.** Every arm still dies 25/25; at this capability level night-1 survival is largely world-determined (corridor availability, skeleton adjacency, spawn pressure). v2 spent 20-40 day-1 steps on shelter ETA logistics, prospecting marches and funnel pre-positioning — steps v1 spent banking wood/stone-tier achievements before the same death. "Death binds, not tech" (v1 §6) was the right *observation* but the wrong *lever*: reallocating time from tech to survival bought too little survival probability to pay for the lost banking.
2. **Dev-eval noise at n≤20 (sd≈12pp, SE≈3) cannot support greedy iterative policy tuning.** Successive v2 iterations moved the dev mean 47.7→58.6→49.5→54.3→52.0 with changes whose true effects were fractions of the noise band; the loop selected for seed-block luck. The NetHack arm's n=5 leaderboard finding recurs one level down, in the development loop itself.
3. Several individually-plausible mechanisms (funnel-holds, bearing-march during vitals foraging, line-of-fire detours) each add small step taxes that compound.

**v1.1** keeps v1's policy untouched and adds only two mechanism-verified bugfixes found during v2 forensics: (a) phantom-station revalidation — arrows destroy tables/furnaces (objects.py Arrow.update), so a believed station can be a phantom; after a no-op make attempt the agent now faces the station cell so the served report corrects the map (a 42-step craft churn was observed); (b) furnace anti-orbit fallback (place anywhere legal after 25 stuck attempts). Dev n=20: 56.4±3.0; powered block: 56.00 [50.91, 61.09] on fresh untouched seeds.

## Memory arm (recalibrating ledger, directive lever 3)

The v1 monotone ratchets were replaced by windowed recalibration: each rule reads only the last 6 clean episodes, tightens while its cited death cause persists, relaxes back toward defaults when it disappears, plus a window-over-window regression guard. Under v1.1 on seeds 9001-9010: A 45.0±4.3 vs B 45.5±5.4 — **no mean effect, and v1's variance-halving did not replicate** (seed 9001 scores 13.6% under every configuration tried; its world is simply lethal). Under v2 the same design also showed no gain (46.4±4.5). Combined with the NetHack arm's memory rejection, the program-level picture: when the world model is source-exact, cross-episode scalar adaptation has little to bite on; the residual variance is world-difficulty, not policy miscalibration.

## Freeze discipline & artifacts (results_v2/, results_v11/)

- Code-freeze md5s + post-run verification in `results_v2/RUN_LOG.md` (v2 freeze 22:32:14, ALL 6 FILES UNCHANGED) and `results_v11/RUN_LOG.md` (v1.1 freeze 22:51:30). The v1 ablation block ran byte-identical code extracted from git (`origin/aleph/fable-crafter` @ 22a555e).
- Bootstrap CIs: 10k resamples, fixed RNG seed 20260706 (`bootstrap_ci.py`, NetHack format), written into the summary JSONs.
- Per-episode JSON checkpoints, full transition logs (`results_v11/transitions/`, 45 episodes incl. the 25-block), and overlay mp4s for every v1.1 episode (`results_v11/animations/`, highlights: `best_block25_17of22_seed12005.mp4`, `memory_best_B_seed9002_17of22.mp4`).
- v1 artifacts remain untouched under `results/` as the ablation baseline.

## V2-cycle caveats

1. v1-vs-v2 shares seed block 11001-11025 (paired, CIs disjoint — solid); v1.1's block is a different fresh block (12001-12025), so v1-vs-v1.1 (55.45 vs 56.00) is a cross-block comparison: read it as "both at-SOTA", not as an ordering.
2. Seeds 12001-12025 were untouched before the frozen run; seeds 11001-11025 were burned for the v1/v2 comparison and must not be reused for tuning.
3. The at-SOTA claim compares our n=25 CI against leaderboard n=10 point estimates with their own ±3.9-4.4 SEs; a strict superiority claim would need the leaderboard agents re-run at n=25 under identical seeding — unavailable.
4. All 25 block episodes end in death (median lifetime ~300 steps of 2000); the ceiling identified in v1 (§6) stands: closing the last 1-2 achievements/episode to decisively beat SOTA requires reliable night-1 shelter under partial observability — a capability, not calibration, gap; the v2 cycle demonstrates that scheduling pressure alone does not purchase it.
