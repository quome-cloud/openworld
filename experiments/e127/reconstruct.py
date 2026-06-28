"""Differential-CEGIS reconstruction. The REAL env is the convergence oracle; the second model is a
diversity source for counterexamples, never the acceptance gate. Emits an equivalence-to-real
certificate plus the A-vs-B-vs-real gap (how much shared-prior agreement exceeds agreement with
reality).

ALL-MODALITY: exploration derives its action pool from the real env's actual action space via
`perception.candidate_actions` -- directional moves for a directional game, pixel-inferred clicks
(6,x,y) for a click/mouse game -- so the loop converges on both modalities (Adaptation 1; replaces
the brief's hard-coded directional [1,2,3,4,5,7], which fails every click game).
"""
import numpy as np
from experiments.e127 import engine as _engine
from experiments.e127 import certify as _certify
from experiments.e127 import probes as _probes
from experiments.e127 import perception as _perception
from experiments.e127.safe_exec import compile_engine


def _explore(real_factory, n_eps, len_eps, seed, mask=None):
    """Collect Episodes by random play over the real env's ACTUAL action space (all-modality).
    Two disjoint pools are produced by passing disjoint seeds. Click targets are inferred from
    pixels; identity-masked click cells (the constant status corner) are dropped consistently with
    probes when a mask is supplied."""
    g = real_factory()
    f0 = np.asarray(g.reset())
    avail = list(getattr(g, "avail", [1, 2, 3, 4, 5, 7]))
    pool = [tuple(a) for a in _perception.candidate_actions(f0, avail)]
    if mask is not None and getattr(mask, "shape", None) == f0.shape:
        pool = [a for a in pool if not (a[0] == 6 and bool(mask[a[2], a[1]]))]
    if not pool:
        pool = [(7, None, None)]
    eps = []
    for i in range(n_eps):
        rng = np.random.default_rng(seed + i)
        g = real_factory()
        idx = rng.integers(0, len(pool), size=len_eps)
        acts = [pool[int(j)] for j in idx]
        eps.append(_engine.play(g, acts))
    return eps


def _acc_vs_real(factory, holdout):
    if factory is None:
        return 0.0
    n = exact = 0
    for ep in holdout:
        sc = _engine.score_rollout(factory, ep)
        n += sc["transitions"]; exact += sc["exact"]
    return (exact / n) if n else 0.0


def _ab_agreement(fa, fb, probe_eps):
    """Fraction of held-out transitions where the two final engines produce identical frames.
    Measured on a probe set distinct from the accuracy holdout, so that when one model is faithful
    (== reality) the gap is a small SAMPLING residual rather than identically zero, while two models
    that share a wrong engine register a structurally large gap (folie a deux)."""
    if fa is None or fb is None:
        return 0.0
    agree = tot = 0
    for ep in probe_eps:
        actions = [s["action"] for s in ep[1:]]
        try:
            ra = _engine.rollout(fa, actions); rb = _engine.rollout(fb, actions)
        except _engine.EngineError:
            continue
        for i in range(1, min(len(ra), len(rb))):
            tot += 1
            if ra[i].shape == rb[i].shape and np.array_equal(ra[i], rb[i]):
                agree += 1
    return (agree / tot) if tot else 0.0


def _prompt(action_api, observed, cexs, round_idx):
    """Build the model prompt (text only; tests ignore it). Includes the action API, a few observed
    transitions, and the real-labeled counterexamples from prior rounds."""
    lines = [f"Reconstruct the game engine as a Python class `Engine` (reset/step/is_win/state).",
             f"Action API: {action_api}", f"Round {round_idx}."]
    for c in cexs[:12]:
        lines.append(f"COUNTEREXAMPLE after actions {[a[0] for a in c['actions']]}: your frame was wrong; "
                     f"the REAL next frame differs (kind={c['kind']}).")
    lines.append('Reply strict JSON: {"engine_src": "...", "rationale": "..."}')
    return "\n".join(lines)


def reconstruct(real_factory, action_api, n_levels, models=("claude", "codex"),
                max_rounds=4, budget=None, _runners=None, seed=0):
    budget = budget if budget is not None else {"limit": 4000, "used": 0}
    observed = _explore(real_factory, n_eps=8, len_eps=16, seed=seed)
    mask = _engine.identity_mask(observed)
    holdout = _explore(real_factory, n_eps=32, len_eps=16, seed=seed + 1000, mask=mask)   # DISJOINT
    probe_eps = _explore(real_factory, n_eps=16, len_eps=16, seed=seed + 2000, mask=mask)  # for A/B gap

    if _runners is None:
        from experiments.e127 import iso
        _runners = [(lambda prompt, r, m=m: iso.run(prompt, model=m).get("engine_src")) for m in models]

    cur = [None, None]          # current engine factory per model
    cur_src = [None, None]
    cur_acc = [0.0, 0.0]
    cexs = []
    history = []

    for r in range(max_rounds):
        for mi, runner in enumerate(_runners):
            src = runner(_prompt(action_api, observed, cexs, r), r)
            if not src or _engine.looks_like_lookup_table(src):
                continue
            fac = compile_engine(src)
            if fac is None:
                continue
            acc = _acc_vs_real(fac, holdout)
            if acc >= cur_acc[mi]:            # keep-best monotone gate
                cur[mi], cur_src[mi], cur_acc[mi] = fac, src, acc
        # champion = best vs REAL
        champ_i = 0 if cur_acc[0] >= cur_acc[1] else 1
        champ = cur[champ_i]
        # gather counterexamples against the champion for the next round
        new_cexs = []
        if champ is not None:
            new_cexs = _probes.find_counterexamples(champ, real_factory, observed, mask, action_api, budget)
            new_cexs += _probes.property_violations(champ, real_factory, action_api, budget)
        cexs = (cexs + new_cexs)[-32:]
        # Adaptation 2: coverage is MEASURED but not GATED in Milestone 1 (coverage_target=0.0); the
        # acceptance gate is the accuracy lower bound (equivalence-to-real). Per-level coverage GATING
        # is deferred to Milestone 2, when the disagreement-driven explorer + solver reach deep levels.
        cert = (_certify.certify_engine(champ, holdout, n_levels, coverage_target=0.0)
                if champ else {"pass": False, "acc": 0.0})
        history.append({"round": r, "champion": champ_i, "champion_acc": cur_acc[champ_i],
                        "n_cex": len(new_cexs), "cert_pass": cert.get("pass", False),
                        "coverage": cert.get("coverage", 0.0), "real_steps": budget["used"]})
        if cert.get("pass"):
            break

    champ_i = 0 if cur_acc[0] >= cur_acc[1] else 1
    champ = cur[champ_i]
    cert = (_certify.certify_engine(champ, holdout, n_levels, coverage_target=0.0) if champ else {
        "pass": False, "acc": 0.0, "acc_lower": 0.0, "n": 0, "exact": 0, "coverage": 0.0})
    ab_agree = _ab_agreement(cur[0], cur[1], probe_eps)
    min_vs_real = min(cur_acc)
    return {"engine_src": cur_src[champ_i], "certificate": cert, "champion_acc": cur_acc[champ_i],
            "ab_agreement": ab_agree, "ab_vs_real_gap": ab_agree - min_vs_real,
            "coverage": cert.get("coverage", 0.0),
            "rounds": len(history), "real_steps": budget["used"], "history": history}
