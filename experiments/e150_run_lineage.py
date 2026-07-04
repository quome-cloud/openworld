"""E150 -- Run lineages: nine figures that characterize HOW the source-free ARC-AGI-3 agents solve,
mined from the captured transcripts (opus + fable arms). The agents' chain-of-thought is redacted, so the
rich signal is what they DO: the ~70 KB of code each run writes, the tool/timestamp trajectory, the banked
action sequences, and level-up events. These figures turn that into a portrait of the solving process.

Figures (papers/assets/figs/e150_*.png):
  1 cardiogram      -- tool-mix ribbon over time per game, with level-ups (anatomy of a solve)
  2 concepts        -- games x code-identifier heatmap (the vocabulary each game forces the agent to invent)
  3 signatures      -- every banked solution as a colored action strip (visual solution taxonomy)
  4 activity        -- windowed tool-type entropy over time (exploration -> exploitation collapse)
  5 surprise        -- self-correction marker rate over time, vs. when levels are cleared
  6 buildup         -- cumulative code bytes over time + final code size vs levels (modeling burden)
  7 clicks          -- click-target density on the 64x64 board for click games (inferred sprite structure)
  8 galaxy          -- per-run code-token PCA scatter (solution fingerprints; same game, different routes)
  9 contrast        -- opus vs fable per shared game: turns and code bytes (how the models differ)

Reads only the immutable captured records; writes figures + experiments/results/e150_run_lineage.json.
Run where the transcripts live (they are gitignored):  python experiments/e150_run_lineage.py
"""
import os, re, json, glob, math, collections
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent.parent
TR = ROOT / "experiments" / "results" / "arc3_traces"
FIGS = ROOT / "papers" / "assets" / "figs"
RES = ROOT / "experiments" / "results"
FIGS.mkdir(parents=True, exist_ok=True)

# ---- paper palette (matches make_arc3_assets.py / the atlas aesthetic) ----
BLUE, OCHRE, TEAL = "#1f4e79", "#c8881f", "#2a8a7f"
STEEL, GRAY = "#5b8fb0", "#9aa0a6"
INK, MUT, GRID = "#232323", "#6b6f76", "#e3e3e3"
TOOLCATS = [("Write", BLUE), ("Edit", STEEL), ("Bash", OCHRE), ("Read", TEAL), ("other", GRAY)]
TOOLCOLOR = dict(TOOLCATS)

plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": MUT, "axes.linewidth": 0.6,
                     "xtick.color": MUT, "ytick.color": MUT, "text.color": INK,
                     "axes.labelcolor": INK, "figure.dpi": 200})


def _spines(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ---------- data loading ----------
GAMES = ("ar25 bp35 cd82 cn04 dc22 ft09 g50t ka59 lf52 lp85 ls20 m0r0 r11l re86 s5i5 sb26 sc25 sk48 "
         "sp80 su15 tn36 tr87 tu93 vc33 wa30").split()

STOP = set("""the and for you are not but with this that from have has was int str def return import
print range len list dict set self true false none null json numpy np sys os time open file path data
res out tmp val key idx max min sum abs all any map get put col row def class while elif else try except
lambda yield break continue pass assert global lambda arr new old cur tmp foo bar baz str int float bool
action actions step reset frame frames game games level levels win done state states np_ x y i j k n m
plt fig ax append add copy deepcopy shape reshape astype zeros ones array asarray print format""".split())
IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
SURPRISE = re.compile(r"\b(wait|actually|hmm|oops|wrong|mistake|bug|broke|broken|unexpected|surprising|"
                      r"surprise|revise|revised|rethink|doesn'?t|didn'?t|isn'?t|no[,.]|but no|hold on|"
                      r"turns out|in fact|nope|fail|failed|incorrect|not right)\b", re.I)


def parse_iso(s):
    # 2026-07-03T04:38:01.868Z -> seconds; avoid Date-dependence, use manual parse
    m = re.match(r".*T(\d\d):(\d\d):(\d\d)\.?(\d*)Z?", s or "")
    if not m:
        return None
    h, mi, se, fr = m.groups()
    return int(h) * 3600 + int(mi) * 60 + int(se) + (float("0." + fr) if fr else 0.0)


def arm_of(model):
    m = str(model).lower()
    return "fable" if "fable" in m else ("opus" if ("opus" in m or "claude" in m) else None)


def load_runs(limit=None):
    """One compact record per source-free Claude-arm run."""
    runs = []
    metas = sorted(glob.glob(str(TR / "meta" / "*.json")))
    for mp in metas:
        try:
            md = json.load(open(mp))
        except Exception:
            continue
        if not md.get("source_free"):
            continue
        arm = arm_of((md.get("model_config") or {}).get("requested_model")
                     or (md.get("model_config") or {}).get("model"))
        if arm is None:
            continue
        game = md.get("game")
        tf = TR / (md.get("transcript_file") or "")
        if game not in GAMES or not str(tf).endswith(".jsonl") or not tf.exists():
            continue
        r = parse_transcript(tf)
        if r is None:
            continue
        win = (WIN or {}).get(game, 12)                     # cap at the game's true level ceiling
        r["levels"] = [(t, l) for t, l in r["levels"] if l <= win]
        r["max_level"] = min(r["max_level"], win)
        r.update(game=game, arm=arm, rid=md.get("run_id"))
        runs.append(r)
        if limit and len(runs) >= limit:
            break
    return runs


def parse_transcript(path):
    t0 = None
    cur = 0.0           # running wall-clock (s since t0); carried onto events whose own line has no ts
    tools = []          # (t, cat)
    lv = []             # (t, maxlevel)
    surprise = []       # (t, count)
    codebytes = []      # (t, cum_bytes)
    tokens = collections.Counter()
    cum = 0
    maxlv = 0
    for line in open(path, errors="ignore"):
        if '"timestamp"' not in line and '"tool_use"' not in line and "levels" not in line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        ts = parse_iso(d.get("timestamp"))
        if ts is not None:
            if t0 is None:
                t0 = ts
            cur = ts - t0
            if cur < 0:
                cur += 86400        # midnight wrap
        t = cur                     # tool_use lines carry no ts of their own -> inherit the last seen time
        msg = d.get("message", {})
        content = msg.get("content") if isinstance(msg.get("content"), list) else []
        for blk in content:
            if not isinstance(blk, dict):
                continue
            bt = blk.get("type")
            if bt == "tool_use":
                name = blk.get("name", "other")
                cat = name if name in TOOLCOLOR else "other"
                tools.append((t, cat))
                inp = blk.get("input") or {}
                code = ""
                if name in ("Write", "Edit"):
                    code = str(inp.get("content") or inp.get("new_string") or "")
                elif name == "Bash":
                    code = str(inp.get("command") or "")
                if code:
                    cum += len(code)
                    for tok in IDENT.findall(code.lower()):
                        if tok not in STOP and not tok.isdigit():
                            tokens[tok] += 1
                    nsurp = len(SURPRISE.findall(code))
                    if nsurp:
                        surprise.append((t, nsurp))
                codebytes.append((t, cum))
            elif bt == "text":
                txt = blk.get("text", "")
                nsurp = len(SURPRISE.findall(txt))
                if nsurp:
                    surprise.append((t, nsurp))
            elif bt == "tool_result":
                txt = str(blk.get("content") or "")
                for mm in re.findall(r"levels?[=:\s]+(\d+)", txt):
                    v = int(mm)
                    if v == maxlv + 1:        # accept only the NEXT sequential level (filters spurious
                        maxlv = v             # "levels=10" the agent writes in its own code)
                        lv.append((t, maxlv))
    if not tools:
        return None
    dur = max([t for t, _ in tools] + [t for t, _ in lv] + [t for t, _ in codebytes]) or 1.0
    return dict(dur=dur, tools=tools, levels=lv, surprise=surprise, codebytes=codebytes,
                tokens=tokens, max_level=maxlv, total_bytes=cum, n_turns=len(tools))


def load_solutions():
    """{(arm, game): actions_list} from the two source-free archives (best banked path)."""
    out = {}
    for f, arm in (("arc3_fullgame_sourcefree_fable.json", "fable"),
                   ("arc3_fullgame_sourcefree.json", "opus")):
        p = RES / f
        if not p.exists():
            continue
        d = json.load(open(p))
        for g, s in (d.get("solutions") or {}).items():
            if isinstance(s, list) and s:
                out[(arm, g)] = s
    return out


def load_winmap():
    """{game: total levels (win_levels)} from an archive's per_game -- the true level ceiling per game."""
    for f in ("arc3_fullgame_sourcefree_fable.json", "arc3_fullgame_sourcefree.json"):
        p = RES / f
        if p.exists():
            pg = json.load(open(p)).get("per_game", {})
            wm = {g: v.get("win") for g, v in pg.items() if isinstance(v, dict) and v.get("win")}
            if wm:
                return wm
    return {}


WIN = None      # populated in main()


# ---------- helpers ----------
def _windows(pairs, dur, nb=24, agg="count"):
    """Bin (t, v) pairs into nb windows over [0,dur]. agg: count | sum | last."""
    edges = np.linspace(0, dur, nb + 1)
    out = np.zeros(nb)
    counts = np.zeros(nb)
    last = 0.0
    bi = 0
    for t, v in pairs:
        b = min(nb - 1, int(t / dur * nb)) if dur else 0
        if agg == "count":
            out[b] += 1
        elif agg == "sum":
            out[b] += v
        elif agg == "last":
            out[b] = v
    if agg == "last":                      # forward-fill cumulative
        run = 0.0
        for i in range(nb):
            run = out[i] if out[i] else run
            out[i] = run
    return out


# ---------- FIG 1: cardiogram ----------
def fig_cardiogram(runs):
    # one exemplar (deepest, then longest) run per chosen game
    pick = ["g50t", "dc22", "ka59", "sk48", "vc33", "tr87"]
    best = {}
    for r in runs:
        if r["game"] in pick:
            key = r["game"]
            cur = best.get(key)
            if cur is None or (r["max_level"], r["n_turns"]) > (cur["max_level"], cur["n_turns"]):
                best[key] = r
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.2))
    nb = 26
    for ax, g in zip(axes.flat, pick):
        r = best.get(g)
        _spines(ax)
        if not r:
            ax.set_visible(False); continue
        bottoms = np.zeros(nb)
        xs = np.linspace(0, 1, nb)
        for cat, col in TOOLCATS:
            v = _windows([(t, 1) for t, c in r["tools"] if c == cat], r["dur"], nb, "count")
            ax.fill_between(xs, bottoms, bottoms + v, color=col, lw=0, step="mid")
            bottoms += v
        # level-up markers
        for t, lvl in r["levels"]:
            x = min(1.0, max(0.0, t / r["dur"] if r["dur"] else 0))
            ax.axvline(x, color=INK, lw=0.7, alpha=0.55)
            ax.text(x, bottoms.max() * 1.02, f"L{lvl}", fontsize=6, color=INK, ha="center", va="bottom")
        ax.set_title(f"\\texttt{{{g}}}  (reached L{r['max_level']})".replace("\\texttt{", "").replace("}", ""),
                     fontsize=8.5, loc="left", color=INK)
        ax.set_xlim(0, 1); ax.set_ylim(0, bottoms.max() * 1.14 + 1)
        ax.set_yticks([]); ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["start", "", "end"])
    handles = [Patch(facecolor=c, label=n) for n, c in TOOLCATS]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Anatomy of a solve: tool activity over time, with level-ups", y=1.06, fontsize=10.5,
                 color=INK, x=0.5)
    fig.text(0.5, -0.01, "normalized wall-clock (start→end); band height = tool calls per window",
             ha="center", fontsize=7.5, color=MUT)
    fig.tight_layout()
    fig.savefig(FIGS / "e150_cardiogram.png", bbox_inches="tight")
    plt.close(fig)
    return {g: (best[g]["max_level"] if g in best else None) for g in pick}


# ---------- FIG 2: code-concept heatmap ----------
def fig_concepts(runs, topn=28):
    per_game = {g: collections.Counter() for g in GAMES}
    for r in runs:
        per_game[r["game"]].update(r["tokens"])
    # discriminative concepts: appear in several games but not all; rank by game-document-frequency in a mid band
    gdf = collections.Counter()
    for g in GAMES:
        for tok in per_game[g]:
            if per_game[g][tok] >= 3:
                gdf[tok] += 1
    cand = [t for t, df in gdf.items() if 2 <= df <= 18]
    # score by total salience
    tot = collections.Counter()
    for g in GAMES:
        for t in cand:
            tot[t] += per_game[g][t]
    concepts = [t for t, _ in tot.most_common(topn)]
    # matrix: rate (per game, row-normalized to max)
    M = np.zeros((len(GAMES), len(concepts)))
    for i, g in enumerate(GAMES):
        tot_g = sum(per_game[g].values()) or 1
        for j, c in enumerate(concepts):
            M[i, j] = per_game[g][c] / tot_g
    # normalize columns to [0,1] for visibility
    Mn = M / (M.max(axis=0, keepdims=True) + 1e-9)
    # order rows (games) + cols (concepts) by PCA-1 for block structure
    def pca1_order(X):
        Xc = X - X.mean(0)
        try:
            U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
            return np.argsort(U[:, 0] * S[0])
        except Exception:
            return np.argsort(X.sum(1))
    ro = pca1_order(Mn)
    co = pca1_order(Mn.T)
    Mo = Mn[np.ix_(ro, co)]
    fig, ax = plt.subplots(figsize=(11, 6.4))
    im = ax.imshow(Mo, aspect="auto", cmap="mako" if "mako" in plt.colormaps() else "viridis")
    ax.set_xticks(range(len(concepts))); ax.set_xticklabels([concepts[j] for j in co], rotation=60,
                                                            ha="right", fontsize=7)
    ax.set_yticks(range(len(GAMES))); ax.set_yticklabels([GAMES[i] for i in ro], fontsize=7)
    ax.set_title("The vocabulary each game forces the agent to invent (code identifiers)",
                 fontsize=10.5, loc="left", color=INK, pad=10)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("column-normalized identifier rate", fontsize=7.5, color=MUT)
    cb.ax.tick_params(labelsize=6)
    fig.tight_layout()
    fig.savefig(FIGS / "e150_concepts.png", bbox_inches="tight")
    plt.close(fig)
    return {"n_concepts": len(concepts), "concepts": concepts}


# ---------- FIG 3: solution signatures (piano roll) ----------
def fig_signatures(sols):
    # best arm per game (prefer whichever solution is longer/complete); map to (game, actions)
    by_game = {}
    for (arm, g), acts in sols.items():
        if g not in by_game or len(acts) > len(by_game[g][1]):
            by_game[g] = (arm, acts)
    games = [g for g in GAMES if g in by_game]
    maxlen = max(len(by_game[g][1]) for g in games)
    # color: directional actions 1-5,7 on a sequential blue ramp; click(6) = ochre
    dirmap = {a: plt.cm.Blues(0.35 + 0.55 * i / 6) for i, a in enumerate([1, 2, 3, 4, 5, 7])}
    fig, ax = plt.subplots(figsize=(11, 6.6))
    for row, g in enumerate(games):
        arm, acts = by_game[g]
        for x, a in enumerate(acts):
            aid = a[0] if isinstance(a, (list, tuple)) else a
            col = OCHRE if aid == 6 else dirmap.get(aid, GRAY)
            ax.add_patch(plt.Rectangle((x, row + 0.1), 1, 0.8, color=col, lw=0))
        ax.text(-6, row + 0.5, g, fontsize=7, ha="right", va="center", color=INK)
        ax.text(len(acts) + 3, row + 0.5, str(len(acts)), fontsize=6.5, ha="left", va="center", color=MUT)
    ax.set_xlim(-8, maxlen + 14); ax.set_ylim(-0.5, len(games) + 0.5)
    ax.invert_yaxis(); ax.set_yticks([]); ax.set_xlabel("action index in banked solution")
    _spines(ax); ax.spines["left"].set_visible(False)
    handles = [Patch(facecolor=OCHRE, label="click (ACTION6)"),
               Patch(facecolor=plt.cm.Blues(0.7), label="directional (1–5,7)")]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=8)
    ax.set_title("Solution signatures: every banked path as an action strip", fontsize=10.5,
                 loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(FIGS / "e150_signatures.png", bbox_inches="tight")
    plt.close(fig)
    return {"n_games": len(games), "max_len": maxlen}


# ---------- FIG 4: activity entropy ----------
def fig_activity(runs):
    nb = 16
    curves = []
    for r in runs:
        if r["n_turns"] < nb:
            continue
        # bin by TURN index (equal-count chunks) so wall-clock throttle gaps don't leave empty windows
        bins = [collections.Counter() for _ in range(nb)]
        seq = [c for _, c in r["tools"]]
        for i, c in enumerate(seq):
            bins[min(nb - 1, int(i / len(seq) * nb))][c] += 1
        rows = []
        for cnt in bins:
            tot = sum(cnt.values())
            if tot == 0:
                rows.append(np.nan); continue
            p = np.array(list(cnt.values())) / tot
            H = -(p * np.log2(p)).sum() / math.log2(len(TOOLCATS))   # normalized 0..1
            rows.append(H)
        curves.append(rows)
    C = np.array(curves, float)
    xs = np.linspace(0, 1, nb)
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    _spines(ax)
    for row in C:
        ax.plot(xs, row, color=STEEL, alpha=0.06, lw=0.8)
    med = np.nanmedian(C, axis=0)
    q1, q3 = np.nanpercentile(C, 25, axis=0), np.nanpercentile(C, 75, axis=0)
    ax.fill_between(xs, q1, q3, color=BLUE, alpha=0.18, lw=0)
    ax.plot(xs, med, color=BLUE, lw=2.4, label="median")
    ax.set_xlabel("run progress (fraction of tool calls issued)")
    ax.set_ylabel("tool-activity entropy (normalized)")
    ax.set_title(f"Exploration → exploitation: activity diversity over a run (n={len(C)})",
                 fontsize=10, loc="left", color=INK)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "e150_activity.png", bbox_inches="tight")
    plt.close(fig)
    return {"n_runs": int(len(C)), "median_start": float(med[1]), "median_end": float(med[-1])}


# ---------- FIG 5: surprise vs level-ups ----------
def fig_surprise(runs):
    nb = 20
    xs = np.linspace(0, 1, nb)
    srate = []
    lvl_times = []
    for r in runs:
        if r["n_turns"] < 12:
            continue
        s = _windows(r["surprise"], r["dur"], nb, "sum")
        tu = _windows([(t, 1) for t, c in r["tools"]], r["dur"], nb, "count")
        srate.append(s / (tu + 1e-9))
        for t, lvl in r["levels"]:
            lvl_times.append(t / r["dur"] if r["dur"] else 0)
    S = np.array(srate, float)
    med = np.nanmedian(S, axis=0)
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    _spines(ax)
    ax.plot(xs, med, color=OCHRE, lw=2.4, label="median self-correction rate")
    ax.fill_between(xs, 0, med, color=OCHRE, alpha=0.14, lw=0)
    ax2 = ax.twinx()   # allowed: histogram density on a SEPARATE panel-like overlay, not a competing y for the same series
    ax2.hist(lvl_times, bins=nb, range=(0, 1), color=TEAL, alpha=0.28, label="level-ups")
    ax2.set_yticks([]); _spines(ax2); ax2.spines["right"].set_visible(False)
    ax.set_xlabel("normalized wall-clock (start→end)")
    ax.set_ylabel("self-correction markers / tool call")
    ax.set_title("When the model breaks (self-correction) vs. when levels fall", fontsize=10,
                 loc="left", color=INK)
    ax.set_xlim(0, 1)
    h1, l1 = ax.get_legend_handles_labels()
    ax.legend(h1 + [Patch(facecolor=TEAL, alpha=0.4, label="level-ups (all runs)")],
              [*l1, "level-ups (all runs)"], frameon=False, fontsize=7.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGS / "e150_surprise.png", bbox_inches="tight")
    plt.close(fig)
    return {"n_levelups": len(lvl_times)}


# ---------- FIG 6: simulator build-up ----------
def fig_buildup(runs):
    nb = 24
    xs = np.linspace(0, 1, nb)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2))
    _spines(axL); _spines(axR)
    for r in runs:
        if r["total_bytes"] < 500:
            continue
        cb = _windows(r["codebytes"], r["dur"], nb, "last") / 1000.0
        col = OCHRE if r["arm"] == "fable" else BLUE
        axL.plot(xs, cb, color=col, alpha=0.10, lw=0.8)
    axL.set_xlabel("normalized wall-clock"); axL.set_ylabel("cumulative code written (KB)")
    axL.set_title("The world model gets written, live", fontsize=9.5, loc="left", color=INK)
    axL.set_xlim(0, 1)
    axL.legend(handles=[Patch(facecolor=BLUE, label="opus"), Patch(facecolor=OCHRE, label="fable")],
               frameon=False, fontsize=8)
    # right: final code bytes vs levels reached
    for arm, col in (("opus", BLUE), ("fable", OCHRE)):
        pts = [(r["max_level"], r["total_bytes"] / 1000.0) for r in runs if r["arm"] == arm and r["total_bytes"] > 500]
        if pts:
            x, y = zip(*pts)
            axR.scatter(x, y, s=16, color=col, alpha=0.5, edgecolor="white", linewidth=0.4, label=arm)
    axR.set_xlabel("deepest level reached"); axR.set_ylabel("total code written (KB)")
    axR.set_title("Modeling burden per solve", fontsize=9.5, loc="left", color=INK)
    axR.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "e150_buildup.png", bbox_inches="tight")
    plt.close(fig)
    return {"n_runs": sum(1 for r in runs if r["total_bytes"] > 500)}


# ---------- FIG 7: click-target heatmaps ----------
def fig_clicks(sols):
    # games whose solutions are click-heavy
    frac = {}
    for (arm, g), acts in sols.items():
        c = sum(1 for a in acts if (a[0] if isinstance(a, (list, tuple)) else a) == 6)
        frac.setdefault(g, []).append((c / len(acts), arm, acts))
    clicky = sorted([g for g, v in frac.items() if max(x[0] for x in v) > 0.5])[:6]
    if not clicky:
        return {"n_click_games": 0}
    n = len(clicky)
    fig, axes = plt.subplots(1, n, figsize=(2.05 * n, 2.5))
    if n == 1:
        axes = [axes]
    for ax, g in zip(axes, clicky):
        _, _, acts = max(frac[g], key=lambda x: x[0])
        H = np.zeros((64, 64))
        for a in acts:
            if isinstance(a, (list, tuple)) and a and a[0] == 6 and len(a) >= 3:
                x, y = int(a[1]), int(a[2])
                if 0 <= y < 64 and 0 <= x < 64:
                    H[y, x] += 1
        ax.imshow(H, cmap="rocket" if "rocket" in plt.colormaps() else "magma",
                  interpolation="nearest", origin="upper")
        ax.set_title(g, fontsize=8.5, color=INK)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Where the agent learned to click (inferred sprite targets, 64×64 board)",
                 fontsize=10, y=1.08, color=INK)
    fig.tight_layout()
    fig.savefig(FIGS / "e150_clicks.png", bbox_inches="tight")
    plt.close(fig)
    return {"n_click_games": n, "click_games": clicky}


# ---------- FIG 8: strategy galaxy ----------
def fig_galaxy(runs):
    vocab = collections.Counter()
    for r in runs:
        vocab.update(r["tokens"])
    terms = [t for t, _ in vocab.most_common(300)]
    tix = {t: i for i, t in enumerate(terms)}
    # tf-idf-ish: log tf, idf over runs
    df = collections.Counter()
    for r in runs:
        for t in set(r["tokens"]) & set(tix):
            df[t] += 1
    N = len(runs)
    idf = {t: math.log((N + 1) / (df[t] + 1)) + 1 for t in terms}
    X = np.zeros((N, len(terms)))
    for i, r in enumerate(runs):
        for t, c in r["tokens"].items():
            if t in tix:
                X[i, tix[t]] = math.log1p(c) * idf[t]
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    Xc = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    Z = U[:, :2] * S[:2]
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    _spines(ax)
    for arm, col in (("opus", BLUE), ("fable", OCHRE)):
        idx = [i for i, r in enumerate(runs) if r["arm"] == arm]
        ax.scatter(Z[idx, 0], Z[idx, 1], s=18, color=col, alpha=0.55, edgecolor="white",
                   linewidth=0.4, label=arm)
    # annotate a few game centroids
    cent = collections.defaultdict(list)
    for i, r in enumerate(runs):
        cent[r["game"]].append(Z[i])
    for g, pts in cent.items():
        if len(pts) >= 4:
            c = np.mean(pts, 0)
            ax.text(c[0], c[1], g, fontsize=6.5, color=INK, alpha=0.65, ha="center", va="center")
    ax.set_xlabel("code-token PC1"); ax.set_ylabel("code-token PC2")
    ax.set_title("Solution fingerprints: each run embedded by the code it wrote", fontsize=10,
                 loc="left", color=INK)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "e150_galaxy.png", bbox_inches="tight")
    plt.close(fig)
    return {"n_runs": N, "var_pc12": float((S[:2] ** 2).sum() / (S ** 2).sum())}


# ---------- FIG 9: model contrast ----------
def fig_contrast(runs):
    # per (arm, game): median turns and median code KB, over games both arms attempted
    agg = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in runs:
        agg[r["game"]][r["arm"]].append(r)
    shared = [g for g in GAMES if agg[g].get("opus") and agg[g].get("fable")]
    def med(rs, key):
        return float(np.median([x[key] for x in rs]))
    turns = {g: (med(agg[g]["opus"], "n_turns"), med(agg[g]["fable"], "n_turns")) for g in shared}
    kb = {g: (med(agg[g]["opus"], "total_bytes") / 1000, med(agg[g]["fable"], "total_bytes") / 1000)
          for g in shared}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5.6))
    for ax, data, lab in ((a1, turns, "median tool calls / run"), (a2, kb, "median code written (KB)")):
        _spines(ax)
        order = sorted(shared, key=lambda g: data[g][0])
        for row, g in enumerate(order):
            o, f = data[g]
            ax.plot([0, 1], [o, f], color=GRID, lw=1.0, zorder=1)
            ax.scatter([0], [o], s=22, color=BLUE, zorder=3)
            ax.scatter([1], [f], s=22, color=OCHRE, zorder=3)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["opus", "fable"])
        ax.set_ylabel(lab); ax.set_xlim(-0.3, 1.3)
        ax.set_title(lab, fontsize=9.5, loc="left", color=INK)
    fig.suptitle(f"Same games, different routes: opus vs. fable ({len(shared)} shared games)",
                 fontsize=10.5, y=1.02, color=INK)
    fig.tight_layout()
    fig.savefig(FIGS / "e150_contrast.png", bbox_inches="tight")
    plt.close(fig)
    return {"n_shared": len(shared)}


def main():
    global WIN
    WIN = load_winmap()
    print("loading runs (parsing transcripts)...", flush=True)
    runs = load_runs()
    sols = load_solutions()
    print(f"  {len(runs)} runs parsed ({sum(1 for r in runs if r['arm']=='opus')} opus, "
          f"{sum(1 for r in runs if r['arm']=='fable')} fable); {len(sols)} banked solutions", flush=True)
    summary = {"n_runs": len(runs),
               "n_opus": sum(1 for r in runs if r["arm"] == "opus"),
               "n_fable": sum(1 for r in runs if r["arm"] == "fable"),
               "median_code_kb": float(np.median([r["total_bytes"] / 1000 for r in runs if r["total_bytes"] > 500]))}
    for name, fn, arg in [
        ("cardiogram", fig_cardiogram, runs), ("concepts", fig_concepts, runs),
        ("signatures", fig_signatures, sols), ("activity", fig_activity, runs),
        ("surprise", fig_surprise, runs), ("buildup", fig_buildup, runs),
        ("clicks", fig_clicks, sols), ("galaxy", fig_galaxy, runs),
        ("contrast", fig_contrast, runs)]:
        try:
            summary[name] = fn(arg)
            print(f"  wrote e150_{name}.png -> {summary[name]}", flush=True)
        except Exception as e:
            import traceback; traceback.print_exc()
            summary[name] = {"error": str(e)}
    json.dump(summary, open(RES / "e150_run_lineage.json", "w"), indent=1)
    print("wrote e150_run_lineage.json")


if __name__ == "__main__":
    main()
