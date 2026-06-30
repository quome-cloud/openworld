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


def _ab_agreement(fa, fb, holdout):
    """Fraction of held-out transitions where the two final engines produce identical frames.
    Measured on the SAME holdout set that drives the vs-real accuracies, so that when one model is
    faithful (== reality) ab_agreement(A,B) identically equals acc_B_vs_real and the gap is exactly 0
    (a clean, interpretable baseline: 0 = no shared-prior bias). Two models that share a wrong engine
    are byte-identical (ab_agreement == 1.0 on any set), so the gap stays robustly positive (folie a
    deux) regardless of the probe set."""
    if fa is None or fb is None:
        return 0.0
    agree = tot = 0
    for ep in holdout:
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


_HEX = "0123456789abcdef"


def _grid(frame):
    """Render a 2D int frame as rows of single hex chars (colors 0-15 -> 0-f)."""
    return "\n".join("".join(_HEX[int(c) & 15] for c in row) for row in np.asarray(frame))


def _changed_cells(a, b, k=60):
    """Compact diff: list the cells that differ between two frames as (y,x):old->new."""
    a = np.asarray(a); b = np.asarray(b)
    if a.shape != b.shape:
        return "[frame shape changed]"
    ys, xs = np.where(a != b)
    if len(ys) == 0:
        return "(no visible change)"
    parts = [f"({int(y)},{int(x)}):{int(a[y, x])}->{int(b[y, x])}" for y, x in list(zip(ys, xs))[:k]]
    extra = "" if len(ys) <= k else f" ...(+{len(ys) - k} more cells)"
    return "; ".join(parts) + extra


def _render_transitions(episodes, k_eps=2, k_steps=10):
    """Render a few observed episodes: the reset board in full, then per-action changed cells."""
    out = []
    for ei, ep in enumerate(episodes[:k_eps]):
        out.append(f"-- episode {ei}: RESET board (levels={ep[0]['levels']}) --\n{_grid(ep[0]['frame'])}")
        for i in range(1, min(len(ep), k_steps + 1)):
            act = ep[i]["action"]
            out.append(f"action {act} -> changed: {_changed_cells(ep[i - 1]['frame'], ep[i]['frame'])}"
                       f"  (levels {ep[i - 1]['levels']}->{ep[i]['levels']})")
    return "\n".join(out)


def _prompt(action_api, observed, cexs, round_idx):
    """Build the model prompt: the stateful-engine CONTRACT + rendered observed play (frames + per-step
    changed cells) + the real-labeled counterexamples to fix. The model only ever sees frames the agent
    perceived by acting -- never game source (source-free)."""
    lines = [
        "You are reverse-engineering the dynamics of a DETERMINISTIC grid game and must output a Python",
        "class `Engine` that reproduces it from observed play. Contract:",
        "  class Engine:",
        "    def __init__(self): ...        # set self.state, a dict that MUST include integer key 'levels'",
        "    def reset(self): ...           # -> numpy 2D int array (colors 0-15); (re)sets self.state",
        "    def step(self, action): ...    # action=(kind,x,y); kind 1=up 2=down 3=left 4=right 5,7=other,",
        "                                   #   6=click at x=col,y=row (x,y are None for non-click). returns next frame",
        "    def is_win(self, prev_frame): ...  # read self.state (procedural progress) -> bool",
        "Only `np` (numpy) and plain Python are available -- NO imports, no file/network access.",
        f"Action space / mechanics hint for THIS game: {action_api}",
        f"(reconstruction round {round_idx})",
        "",
        "OBSERVED PLAY (boards are rows of hex 0-f for colors 0-15; transitions list only the cells that changed):",
        _render_transitions(observed),
    ]
    if cexs:
        lines.append("\nYOUR PREVIOUS ENGINE WAS WRONG at these transitions -- fix them:")
        for c in cexs[:8]:
            acts = [a[0] for a in c.get("actions", [])]
            rf, ef = c.get("real_frame"), c.get("engine_frame")
            if rf is not None and ef is not None:
                diff = _changed_cells(ef, rf)
                lines.append(f"  after actions {acts}: your frame differs from REAL at {diff}")
            else:
                lines.append(f"  after actions {acts}: property violation (kind={c.get('kind')})")
    lines.append('\nReply with STRICT JSON ONLY: {"engine_src": "<full python defining class Engine>", '
                 '"rationale": "<one sentence>"}')
    return "\n".join(lines)


def reconstruct(real_factory, action_api, n_levels, models=("claude", "codex"),
                max_rounds=4, budget=None, _runners=None, seed=0):
    budget = budget if budget is not None else {"limit": 4000, "used": 0}
    observed = _explore(real_factory, n_eps=8, len_eps=16, seed=seed)
    mask = _engine.identity_mask(observed)
    holdout = _explore(real_factory, n_eps=32, len_eps=16, seed=seed + 1000, mask=mask)   # DISJOINT

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
    ab_agree = _ab_agreement(cur[0], cur[1], holdout)
    min_vs_real = min(cur_acc)
    return {"engine_src": cur_src[champ_i], "certificate": cert, "champion_acc": cur_acc[champ_i],
            "ab_agreement": ab_agree, "ab_vs_real_gap": ab_agree - min_vs_real,
            "coverage": cert.get("coverage", 0.0),
            "rounds": len(history), "real_steps": budget["used"], "history": history}
