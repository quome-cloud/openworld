"""Paper figure for E120 (expert-consensus world) + E121 (surprise-triggered self-rebuilding world).

Three-panel story:
  A) the expert-CONSENSUS atlas card  (N expert lenses -> N OpenWorld worlds -> ConsensusTransition vote)
  B) SURPRISE vs ground-truth rule-changes: the masked-board delta over a solved trace, with surprise-
     detected regime boundaries (lines) landing on the level-ups (markers) -- recall 1.0, detected from
     frames alone.
  C) the self-rebuilding regime atlas card (openworld.PhasedTransition over the detected regimes).

    DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib ~/.arcv/bin/python scripts/make_arc3_selfrebuild_fig.py
Writes papers/assets/figs/arc3_selfrebuild.png
"""
import os, sys, io, json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cairosvg
from PIL import Image

ROOT = Path("/Users/jim/Desktop/openworld")
sys.path.insert(0, str(ROOT / "experiments"))
import arc_agi
import e121_surprise_regimes as E121
import e119.perceive as P

MAPS = ROOT / "papers/arc-3/maps"
FIGS = ROOT / "papers/assets/figs"; FIGS.mkdir(parents=True, exist_ok=True)
GAME_B = "ka59"   # the alignment panel uses the deepest game (6 rule changes)


def card_img(name, width=1400):
    svg = MAPS / f"{name}.svg"
    if not svg.exists():
        return None
    png = cairosvg.svg2png(bytestring=svg.read_text(encoding="utf-8").encode("utf-8"), output_width=width)
    return Image.open(io.BytesIO(png))


def surprise_series(game):
    arc = arc_agi.Arcade(); env = arc.make(game)
    sol = json.loads((ROOT / "experiments/results/arc3_fullgame_sourcefree.json").read_text())["solutions"][game]
    frames, acts, levels = E121.replay_with_levels(env, sol)
    mask = P.status_mask(frames)
    sigs, novelty, contra, delta, idmap = E121.surprise_signals(frames, acts, mask)
    bounds = E121.detect_boundaries(delta)
    ups = [t for t in range(1, len(levels)) if levels[t] > levels[t - 1]]
    sc = E121.score(bounds, levels)
    return delta, bounds, ups, sc


def recall_distribution():
    """Per-game retrospective recall from the committed E121 run (honest generalization, n=25)."""
    d = json.loads((ROOT / "experiments/results/e121_surprise_regimes.json").read_text())["games"]
    pairs = [(g, v["recall"]) for g, v in d.items() if v.get("recall") is not None]
    pairs.sort(key=lambda kv: kv[1])              # worst first, so failures are visible on the left
    return pairs


def main():
    delta, bounds, ups, sc = surprise_series(GAME_B)
    cardA = card_img(f"{GAME_B}_consensus")
    cardC = card_img(f"{GAME_B}_regimes")
    dist = recall_distribution()

    fig = plt.figure(figsize=(15.5, 12.4))
    gs = GridSpec(3, 2, height_ratios=[1.05, 0.92, 0.66], width_ratios=[1, 1], hspace=0.34, wspace=0.10)

    # ---- A: consensus card ----
    axA = fig.add_subplot(gs[0, 0]); axA.axis("off")
    if cardA is not None:
        axA.imshow(cardA, aspect="auto")
    axA.set_title("A  ·  expert-CONSENSUS world  (lenses → worlds → ConsensusTransition vote)",
                  fontsize=11, color="#1e293b", loc="left", pad=6)

    # ---- C: self-rebuilding regime card ----
    axC = fig.add_subplot(gs[0, 1]); axC.axis("off")
    if cardC is not None:
        axC.imshow(cardC, aspect="auto")
    axC.set_title("C  ·  self-rebuilding world  (PhasedTransition over surprise-detected regimes)",
                  fontsize=11, color="#1e293b", loc="left", pad=6)

    # ---- B: surprise vs rule-changes (full width bottom) ----
    axB = fig.add_subplot(gs[1, :])
    x = np.arange(len(delta))
    axB.fill_between(x, 0, delta, color="#2563eb", alpha=0.18, zorder=1)
    axB.plot(x, delta, color="#1d4ed8", lw=1.3, zorder=2, label="surprise signal  (masked-board Δ per step)")
    for j, b in enumerate(bounds):
        if b == 0:
            continue
        axB.axvline(b, color="#ea580c", ls="--", lw=1.6, alpha=0.9, zorder=3,
                    label="surprise-detected regime boundary" if j == 1 else None)
    for j, u in enumerate(ups):
        axB.scatter([u], [delta[min(u, len(delta) - 1)]], s=130, marker="v", color="#059669",
                    edgecolor="white", linewidth=1.1, zorder=5,
                    label="ground-truth level-up (rule change)" if j == 0 else None)
    axB.set_xlabel("step along the verified solution trace", fontsize=10.5)
    axB.set_ylabel("board cells changed (Δ)", fontsize=10.5)
    axB.set_title(f"B  ·  one game in detail ({GAME_B}): surprise detects every rule change from frames "
                  f"alone  —  recall={sc['recall']}, precision={sc['precision']} "
                  f"({sc['matched']}/{sc['level_ups']} level-ups; the level signal is used only to score, "
                  f"never to detect)", fontsize=11, color="#1e293b", loc="left", pad=6)
    axB.legend(loc="upper right", fontsize=9.5, framealpha=0.95)
    axB.grid(True, alpha=0.25); axB.margins(x=0.01)
    for s in ("top", "right"):
        axB.spines[s].set_visible(False)

    # ---- D: honest generalization -- per-game recall across all 25 banked traces ----
    axD = fig.add_subplot(gs[2, :])
    games = [g for g, _ in dist]; rec = [r for _, r in dist]
    mean_r = sum(rec) / len(rec); med_r = sorted(rec)[len(rec) // 2]
    colors = ["#dc2626" if r < 0.5 else "#f59e0b" if r < 1.0 else "#059669" for r in rec]
    axD.bar(range(len(games)), rec, color=colors, edgecolor="white", linewidth=0.6)
    axD.axhline(mean_r, color="#1d4ed8", ls="--", lw=1.4, label=f"mean {mean_r:.2f}")
    axD.axhline(med_r, color="#7c3aed", ls=":", lw=1.4, label=f"median {med_r:.2f}")
    axD.set_xticks(range(len(games))); axD.set_xticklabels(games, rotation=90, fontsize=7.5)
    axD.set_ylim(0, 1.05); axD.set_ylabel("rule-change recall", fontsize=10.5)
    axD.set_title(f"D  ·  honest generalization across all {len(games)} banked games  —  median "
                  f"{med_r:.2f}, mean {mean_r:.2f}, {sum(1 for r in rec if r==1.0)}/{len(rec)} perfect; "
                  f"failures (red) are games whose level-ups do NOT reload the board (e.g. cd82, bp35)",
                  fontsize=11, color="#1e293b", loc="left", pad=6)
    axD.legend(loc="lower right", fontsize=9.5, framealpha=0.95)
    axD.grid(True, axis="y", alpha=0.25)
    for s in ("top", "right"):
        axD.spines[s].set_visible(False)

    fig.suptitle("Multi-world ARC-AGI-3: expert world models that VOTE, and a world model that NOTICES when "
                 "the rules change and REBUILDS itself (honest cross-game generalization)",
                 fontsize=13.5, fontweight="bold", y=0.999)
    out = FIGS / "arc3_selfrebuild.png"
    fig.savefig(out, dpi=155, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}  (cardsA/C {'ok' if cardA is not None else 'MISSING'}, "
          f"recall={sc['recall']})")


if __name__ == "__main__":
    main()
