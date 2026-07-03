"""E148 -- Strategy-space analysis of ARC-AGI-3 solver transcripts (opus vs codex vs fable).

We captured every solver session as a full stream-json transcript (prompts/transcripts/solutions/meta
under experiments/results/arc3_traces/). Claude's extended *thinking* is redacted in the capture, but
the OBSERVABLE record is rich and objective: the agent's text narration, its Bash commands, and the
names/paths of the scripts it writes (l7_probe.py, l7_shaft.py, l7_ladder.py, beam.py, sim6.py, ...).
Those name the STRATEGY directly.

This experiment turns each session into a strategy fingerprint and asks:
  1. Do the three models occupy different regions of strategy space? (PCA map)
  2. HOW does each model solve -- which strategy families does it lean on? (per-model rates)
  3. Which strategies correlate with getting deeper? (success correlation)

Method (deterministic, offline, zero extra deps -- numpy + matplotlib only; no sklearn, no LLM):
  - parse each transcript -> a session doc (text + Bash commands + Write/Edit file paths) + tool counts
  - featurize: STRATEGY LEXICON (regex families) rate per session + structural features
  - z-score, then PCA via numpy SVD (deterministic up to a fixed sign convention) for the 2-D map
  - per-model strategy means; Pearson corr of each strategy with levels-reached

Honest scope: thinking is redacted -> we use observable actions/text (arguably the more objective
signal); N is opus-heavy (opus is the long-running arm) so per-model means are reported with N and the
map plots every session with the small-N arms highlighted. This is exploratory analysis, not a
hypothesis test -- it visualizes how the models' solving strategies differ.

  python experiments/e148_strategy_space.py
"""
import os, sys, re, json, glob, gzip
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import save_results

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACES = os.path.join(ROOT, "experiments/results/arc3_traces")
RESULTS = os.path.join(ROOT, "experiments/results")
INK, BLUE, OCHRE, TEAL, GREY = "#16202e", "#1d4ed8", "#b45309", "#0f766e", "#9aa4b2"
MODEL_COLOR = {"opus": BLUE, "codex": OCHRE, "fable": TEAL}
MODEL_OF = {"claude-opus-4-8": "opus", "gpt-5.5": "codex", "claude-fable-5": "fable"}

# Strategy lexicon: family -> regex (word-ish). Counted over the session's observable text (narration
# + bash + written filenames). Rates are per-session normalized. Deliberately about METHOD, not games.
LEXICON = {
    "simulate":   r"\bsim\b|simulat|\bpredict\b|world.?model|faithful|reproduce|forward.?model|emulat",
    "search":     r"\bbeam\b|\bbfs\b|\bdfs\b|a\*|astar|\bsearch\b|exhaust|brute|enumerate|\bsweep\b|random.?restart|backtrack",
    "state_graph":r"state.?graph|\bgraph\b|\bnode\b|\bedge\b|reachable|\bbfs\b|transition.?table",
    "perceive":   r"\bframe\b|\bgrid\b|\bmask\b|object|component|colou?r|sprite|render|pixel|\bcell\b|bbox|connected",
    "verify":     r"replay|verif|reset|determinist|checkpoint|\bseed\b|round.?trip|reproduc",
    "goal_infer": r"\bwin\b|\bgoal\b|level.?up|objective|win.?condition|\bprotocol\b|procedure|\breward\b",
    "mechanic":   r"parity|lights.?out|\bflood\b|ladder|pocket|portal|elevator|climb|\bride\b|\bwarp\b|conduit|ghost|lever|socket|teleport",
    "memory":     r"\bnotes?\b|workspace|\bprior\b|resume|best.?keeper|toolkit|inventory|\brecall\b|solved\.json|frontier",
    "probe":      r"\bprobe\b|\bexplore\b|\bmap\b|\bdump\b|inspect|\bscan\b|examine|trace\b",
}
STRUCT = ["n_tool", "n_bash", "n_write", "n_read", "n_edit", "n_text", "code_chars", "n_scripts"]


def parse_transcript(path):
    text, bash, paths = [], [], []
    counts = dict(n_tool=0, n_bash=0, n_write=0, n_read=0, n_edit=0, n_text=0, code_chars=0)
    scripts = set()
    try:
        fh = open(path, errors="ignore")
    except Exception:
        return None
    for line in fh:
        try:
            d = json.loads(line)
        except Exception:
            continue
        c = (d.get("message") or {}).get("content")
        if not isinstance(c, list):
            continue
        for b in c:
            if not isinstance(b, dict):
                continue
            t = b.get("type")
            if t == "text":
                counts["n_text"] += 1
                text.append(b.get("text") or "")
            elif t == "tool_use":
                counts["n_tool"] += 1
                nm = b.get("name", ""); inp = b.get("input") or {}
                if nm == "Bash":
                    counts["n_bash"] += 1
                    cmd = inp.get("command") or ""; bash.append(cmd); counts["code_chars"] += len(cmd)
                elif nm == "Write":
                    counts["n_write"] += 1
                    fp = inp.get("file_path") or ""; paths.append(fp)
                    counts["code_chars"] += len(inp.get("content") or "")
                    if fp.endswith(".py"): scripts.add(os.path.basename(fp))
                elif nm == "Edit":
                    counts["n_edit"] += 1; paths.append(inp.get("file_path") or "")
                elif nm == "Read":
                    counts["n_read"] += 1
    counts["n_scripts"] = len(scripts)
    doc = "\n".join(text) + "\n" + "\n".join(bash) + "\n" + "\n".join(paths)
    return {"doc": doc.lower(), "counts": counts}


def parse_codex_log(path):
    """codex exec plaintext (gzipped): strip the echoed TASK prompt (everything before codex's first
    turn), then use the reasoning summaries + `exec` command blocks as the observable doc. Structural
    counts are best-effort (exec blocks ~ commands; `cat >`/apply_patch ~ file writes)."""
    try:
        lines = gzip.open(path, "rt", errors="ignore").read().splitlines()
    except Exception:
        return None
    start = next((i for i, ln in enumerate(lines) if ln.strip() == "codex"), None)   # first codex turn
    body = "\n".join(lines[start:]) if start is not None else "\n".join(lines)
    n_exec = len(re.findall(r"(?m)^exec\b", body))
    n_write = len(re.findall(r"cat >\s|apply_patch|<<'?PY'?|>\s*\S+\.py", body))
    scripts = set(re.findall(r"([\w./-]+\.py)\b", body))
    counts = dict(n_tool=n_exec + n_write, n_bash=n_exec, n_write=n_write, n_read=0, n_edit=0,
                  code_chars=len(body), n_scripts=len(scripts), n_text=body.count("\ncodex\n"))
    return {"doc": body.lower(), "counts": counts}


def _levels_for(rid):
    sol = os.path.join(TRACES, "solutions", rid + ".json")
    if os.path.exists(sol):
        try:
            return int(json.load(open(sol)).get("levels", 0))
        except Exception:
            return 0
    return 0


def load_sessions():
    """One row per session (opus/fable from stream-json .jsonl; codex from .codex.log.gz)."""
    rows = []
    # opus + fable: stream-json transcripts, model from meta
    for meta_f in glob.glob(os.path.join(TRACES, "meta", "*.json")):
        try:
            m = json.load(open(meta_f))
        except Exception:
            continue
        arm = MODEL_OF.get((m.get("model_config") or {}).get("resolved_model") or m.get("model"))
        if arm is None:
            continue
        rid = m.get("run_id") or os.path.basename(meta_f)[:-5]
        tr = os.path.join(TRACES, "transcripts", rid + ".jsonl")
        if not os.path.exists(tr):
            continue
        parsed = parse_transcript(tr)
        if parsed is None or parsed["counts"]["n_tool"] < 3:
            continue
        rows.append({"rid": rid, "arm": arm, "game": m.get("game"), "levels": _levels_for(rid), **parsed})
    # codex: gzipped plaintext logs (identified by the .codex.log.gz transcript)
    for tr in glob.glob(os.path.join(TRACES, "transcripts", "*.codex.log.gz")):
        rid = os.path.basename(tr)[:-len(".codex.log.gz")]
        parsed = parse_codex_log(tr)
        if parsed is None or parsed["counts"]["n_tool"] < 3:
            continue
        game = rid.split("__")[0]
        rows.append({"rid": rid, "arm": "codex", "game": game, "levels": _levels_for(rid), **parsed})
    return rows


def featurize(rows):
    fams = list(LEXICON)
    F = np.zeros((len(rows), len(fams) + len(STRUCT)))
    for i, r in enumerate(rows):
        doc = r["doc"]; n = max(1, len(doc.split()))
        for j, fam in enumerate(fams):
            F[i, j] = len(re.findall(LEXICON[fam], doc)) / n * 1000.0     # hits per 1k tokens
        for k, s in enumerate(STRUCT):
            F[i, len(fams) + k] = r["counts"][s]
    return F, fams + STRUCT


def zscore(F):
    mu = F.mean(0); sd = F.std(0); sd[sd == 0] = 1.0
    return (F - mu) / sd


def pca2(Z):
    Zc = Z - Z.mean(0)
    U, S, Vt = np.linalg.svd(Zc, full_matrices=False)
    for k in range(2):                                            # fix sign: largest-|loading| positive
        if Vt[k][np.argmax(np.abs(Vt[k]))] < 0:
            Vt[k] = -Vt[k]; U[:, k] = -U[:, k]
    coords = U[:, :2] * S[:2]
    evr = (S ** 2 / (S ** 2).sum())[:2]
    return coords, Vt[:2], evr


def main():
    rows = load_sessions()
    if len(rows) < 10:
        raise SystemExit(f"too few sessions with known model: {len(rows)}")
    F, names = featurize(rows)
    nfam = len(LEXICON)
    # Strategy MAP uses the lexicon families ONLY -- comparable word-rate signal across capture formats
    # (codex `exec` vs Claude Bash/Read/Write tool counts are format-dependent, so they are reported
    # separately, not mixed into the cross-model projection).
    coords, load, evr = pca2(zscore(F[:, :nfam]))
    names = names[:nfam]
    arms = np.array([r["arm"] for r in rows])
    levels = np.array([r["levels"] for r in rows], float)
    fams = list(LEXICON)

    # per-model strategy means (family rates, per 1k tokens)
    per_model = {}
    for a in ("opus", "codex", "fable"):
        idx = arms == a
        means = {fam: round(float(F[idx, j].mean()), 3) for j, fam in enumerate(fams)} if idx.any() else {}
        per_model[a] = {"n": int(idx.sum()), **means}

    # which strategies correlate with levels reached (Pearson, across all sessions)
    corr = {}
    for j, fam in enumerate(fams):
        x = F[:, j]
        corr[fam] = round(float(np.corrcoef(x, levels)[0, 1]), 3) if x.std() > 0 else 0.0

    # top PCA loadings per axis (what the axes mean)
    axis_top = []
    for k in range(2):
        order = np.argsort(-np.abs(load[k]))[:4]
        axis_top.append([(names[o], round(float(load[k][o]), 2)) for o in order])

    # ---------------- figures ----------------
    _plots(rows, coords, evr, axis_top, arms, levels, per_model, corr, fams)

    payload = {
        "description": "Strategy-space analysis of ARC-AGI-3 solver transcripts across opus/codex/fable: "
                       "each session fingerprinted by observable strategy (narration + bash + written "
                       "script names); PCA map + per-model strategy rates + success correlation.",
        "n_sessions": len(rows),
        "by_model": {a: int((arms == a).sum()) for a in ("opus", "codex", "fable")},
        "note_thinking_redacted": True,
        "pca_explained_variance": [round(float(x), 3) for x in evr],
        "pca_axis_top_loadings": {"PC1": axis_top[0], "PC2": axis_top[1]},
        "per_model_strategy_rate_per_1k_tokens": per_model,
        "strategy_corr_with_levels_reached": corr,
    }
    save_results("e148_strategy_space", payload)

    print(f"E148 OK  sessions={len(rows)}  by_model={payload['by_model']}")
    print(f"  PCA var explained: {evr[0]:.2f}, {evr[1]:.2f}  | PC1~{axis_top[0][:2]}  PC2~{axis_top[1][:2]}")
    for a in ("opus", "codex", "fable"):
        pm = per_model[a]
        if pm.get("n"):
            top = sorted(((k, v) for k, v in pm.items() if k in fams), key=lambda kv: -kv[1])[:3]
            print(f"  {a:5} (n={pm['n']}): top strategies {top}")
    print(f"  strategy~levels corr: " + ", ".join(f"{k}:{corr[k]:+.2f}" for k in fams))


def _plots(rows, coords, evr, axis_top, arms, levels, per_model, corr, fams):
    plt.rcParams.update({"font.size": 10, "figure.facecolor": "white", "savefig.facecolor": "white",
                         "axes.edgecolor": "#c7ccd4"})
    # 1) strategy-space PCA map, colored by model (opus faded, small-N arms highlighted)
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.2))
    for a in ("opus", "codex", "fable"):
        idx = arms == a
        if not idx.any(): continue
        al = 0.18 if a == "opus" else 0.9
        sz = 12 if a == "opus" else 42
        ax[0].scatter(coords[idx, 0], coords[idx, 1], s=sz, alpha=al, c=MODEL_COLOR[a],
                      edgecolor="none", label=f"{a} (n={int(idx.sum())})")
    ax[0].legend(frameon=False, fontsize=9, loc="best")
    ax[0].set_title("A. Strategy space by model (PCA of session fingerprints)", fontsize=11, loc="left", fontweight="bold")
    ax[0].set_xlabel(f"PC1 ({evr[0]*100:.0f}%): " + ", ".join(n for n, _ in axis_top[0][:2]), fontsize=9)
    ax[0].set_ylabel(f"PC2 ({evr[1]*100:.0f}%): " + ", ".join(n for n, _ in axis_top[1][:2]), fontsize=9)
    # 2) same map, colored by outcome (levels reached)
    sc = ax[1].scatter(coords[:, 0], coords[:, 1], s=20, c=levels, cmap="viridis", alpha=0.7, edgecolor="none")
    plt.colorbar(sc, ax=ax[1], label="levels reached")
    ax[1].set_title("B. Same map, colored by depth reached", fontsize=11, loc="left", fontweight="bold")
    ax[1].set_xlabel("PC1", fontsize=9); ax[1].set_ylabel("PC2", fontsize=9)
    for a in ax:
        for s in ("top", "right"): a.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "e148_strategy_space.png"), dpi=170); plt.close(fig)

    # 3) per-model strategy-family rates (grouped bars) + success correlation
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.5, 1]})
    x = np.arange(len(fams)); w = 0.26
    for i, a in enumerate(("opus", "codex", "fable")):
        pm = per_model[a]
        if not pm.get("n"): continue
        vals = [pm.get(f, 0) for f in fams]
        ax[0].bar(x + (i - 1) * w, vals, w, color=MODEL_COLOR[a], label=f"{a} (n={pm['n']})")
    ax[0].set_xticks(x); ax[0].set_xticklabels(fams, rotation=35, ha="right", fontsize=9)
    ax[0].set_ylabel("hits / 1k tokens", fontsize=9); ax[0].legend(frameon=False, fontsize=9)
    ax[0].set_title("C. How each model solves (strategy-family usage)", fontsize=11, loc="left", fontweight="bold")
    cvals = [corr[f] for f in fams]
    cols = [TEAL if v > 0 else OCHRE for v in cvals]
    ax[1].barh(x, cvals, color=cols); ax[1].set_yticks(x); ax[1].set_yticklabels(fams, fontsize=9)
    ax[1].axvline(0, color="#888", lw=0.8); ax[1].set_xlabel("Pearson r", fontsize=9)
    ax[1].set_title("D. Strategy vs depth reached", fontsize=11, loc="left", fontweight="bold")
    for a in ax:
        for s in ("top", "right"): a.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "e148_strategy_bars.png"), dpi=170); plt.close(fig)


if __name__ == "__main__":
    main()
