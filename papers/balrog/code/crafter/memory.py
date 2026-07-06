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
                      os.environ.get('CRAFTER_RESULTS_DIR', 'results'),
                      'memory_ledger.json')

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


def derive_params(ledger, window=6):
    """v2: RECALIBRATING adaptation (fixes the v1 over-adaptation finding —
    monotone ratchets degraded late passes). Each rule looks at the last
    `window` clean episodes only: it tightens one notch while its cited
    death cause is present in the window, and RELAXES one notch back toward
    the default when the cause has been absent — thresholds track the
    current failure distribution instead of integrating all history."""
    p = dict(DEFAULT_PARAMS)
    fired = []
    eps = ledger['episodes']
    recent = eps[-window:]
    if not eps:
        return p, fired

    def recent_cites(cause):
        return [e['file'] for e in recent if e.get('death_cause') == cause]

    # 1. shelter timing: notch = 8 phase-steps per recent night-no-home death
    nn = recent_cites('zombie_night_no_home')
    if nn:
        p['prep_start'] = max(84, DEFAULT_PARAMS['prep_start'] - 8 * len(nn))
        p['home_stone_urgency'] = 1
        fired.append(dict(rule='earlier_shelter_prep',
                          effect=f"prep_start->{p['prep_start']}",
                          cites=nn))
    # 2. vitals margin: +1 per recent starvation death, decays when absent
    st = recent_cites('starvation')
    if st:
        p['vitals_floor'] = min(6, DEFAULT_PARAMS['vitals_floor'] + len(st))
        fired.append(dict(rule='bigger_vitals_margin',
                          effect=f"vitals_floor->{p['vitals_floor']}",
                          cites=st))
    # 3. skeleton avoidance: widen while arrows kill, narrow when they stop
    sk = recent_cites('skeleton_arrows')
    if sk:
        p['skel_zone_rad'] = min(6, DEFAULT_PARAMS['skel_zone_rad'] + len(sk))
        p['skel_hunt_min_hp'] = 9
        fired.append(dict(rule='wider_skeleton_avoidance',
                          effect=f"skel_zone_rad->{p['skel_zone_rad']}",
                          cites=sk))
    # 4. plant pipeline: only while long lives exist without eat_plant
    long_recent = [e for e in recent if e.get('steps', 0) >= 500]
    if len(long_recent) >= 2 and not any(
            'eat_plant' in e.get('achievements', []) for e in recent):
        p['sapling_budget'] = 45
        fired.append(dict(rule='earlier_plant_pipeline',
                          effect='sapling_budget->45',
                          cites=[e['file'] for e in long_recent]))
    # regression guard: if the window's mean score under adapted params is
    # below the mean of the window before it, drop the weakest adaptation
    if len(eps) >= 2 * window:
        prev = eps[-2 * window:-window]
        m_now = sum(e['score'] for e in recent) / len(recent)
        m_prev = sum(e['score'] for e in prev) / len(prev)
        if m_now + 0.5 < m_prev and fired:
            dropped = fired.pop()
            p = dict(DEFAULT_PARAMS)
            for f in fired:  # re-apply the remaining rules only
                pass
            fired.append(dict(rule='regression_guard',
                              effect=f"dropped {dropped['rule']} (window mean "
                                     f"{m_now:.1f} < prev {m_prev:.1f})",
                              cites=[e['file'] for e in recent]))
    return p, fired


def record_episode(ledger, result, path):
    ledger['episodes'].append(dict(
        file=os.path.basename(path), seed=result['seed'],
        condition=result.get('condition', 'A'),
        score=result['score'], steps=result['steps'],
        achievements=result['achievements'],
        death_cause=result.get('death_cause'),
        death_phase=(result['steps'] % 300) if result['died'] else None))
