"""Cross-episode memory (condition B) with clean provenance.

Every entry cites the clean-condition episode file(s) it was learned from.
Nothing here descends from privileged observations or from reading crafter
source at runtime: the adaptation rules below only consume the per-episode
JSON results the harness writes for CLEAN runs (score, achievements, death
cause classified from the agent's own belief at death, timings).

What is left for memory to learn, given the source-synthesized model already
fixes the rules of the world (mob damage, recipes, daylight clock)?
Environment-level *statistics and policy calibration* that the rules do not
determine: how early shelter preparation must start in practice, how much
vitals margin real nights consume, how costly skeleton zones are, whether
plant-tending fits in a lifetime. These are exactly the parameters below.
"""

import json
import os

LEDGER = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      'results', 'memory_ledger.json')

DEFAULT_PARAMS = dict(
    prep_start=108,        # phase at which dusk burrow prep begins
    vitals_floor=4,        # day food/drink trigger threshold
    skel_zone_rad=3,       # path-cost radius around skeleton sightings
    skel_hunt_min_hp=7,    # hp gate for skeleton hunts
    sapling_budget=30,     # max sapling draws
    home_stone_urgency=0,  # >0: collect first stone before other stone goals
)


def load_ledger():
    if os.path.exists(LEDGER):
        return json.load(open(LEDGER))
    return dict(episodes=[], entries=[])


def save_ledger(ledger):
    os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
    json.dump(ledger, open(LEDGER, 'w'), indent=1)


def derive_params(ledger):
    """Fold adaptation rules over the ledger. Returns (params, fired) where
    fired lists rule activations with the episode files they cite."""
    p = dict(DEFAULT_PARAMS)
    fired = []
    eps = ledger['episodes']

    def cite(pred):
        return [e['file'] for e in eps if pred(e)]

    night_no_home = cite(lambda e: e.get('death_cause') == 'zombie_night_no_home')
    if len(night_no_home) >= 2:
        step = 8 * (len(night_no_home) - 1)
        p['prep_start'] = max(60, DEFAULT_PARAMS['prep_start'] - step)
        p['home_stone_urgency'] = 1
        fired.append(dict(rule='earlier_shelter_prep',
                          effect=f"prep_start->{p['prep_start']}, "
                                 f"home_stone_urgency->1",
                          cites=night_no_home))
    starve = cite(lambda e: e.get('death_cause') == 'starvation')
    if starve:
        p['vitals_floor'] = min(6, DEFAULT_PARAMS['vitals_floor'] + len(starve))
        fired.append(dict(rule='bigger_vitals_margin',
                          effect=f"vitals_floor->{p['vitals_floor']}",
                          cites=starve))
    skel = cite(lambda e: e.get('death_cause') == 'skeleton_arrows')
    if skel:
        p['skel_zone_rad'] = min(6, DEFAULT_PARAMS['skel_zone_rad'] + len(skel))
        p['skel_hunt_min_hp'] = 9
        fired.append(dict(rule='wider_skeleton_avoidance',
                          effect=f"skel_zone_rad->{p['skel_zone_rad']}, "
                                 f"hunt_min_hp->9",
                          cites=skel))
    # plant never eaten but lives long enough: spend more on saplings early
    long_lives = [e for e in eps if e.get('steps', 0) >= 500]
    if len(long_lives) >= 2 and not any(
            'eat_plant' in e.get('achievements', []) for e in eps):
        p['sapling_budget'] = 45
        fired.append(dict(rule='earlier_plant_pipeline',
                          effect='sapling_budget->45',
                          cites=[e['file'] for e in long_lives]))
    return p, fired


def record_episode(ledger, result, path):
    ledger['episodes'].append(dict(
        file=os.path.basename(path), seed=result['seed'],
        condition=result.get('condition', 'A'),
        score=result['score'], steps=result['steps'],
        achievements=result['achievements'],
        death_cause=result.get('death_cause'),
        death_phase=(result['steps'] % 300) if result['died'] else None))
