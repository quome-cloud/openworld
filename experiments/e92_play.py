"""E92 -- Claude as an INTERACTIVE ARC-AGI-3 player (the shot at actually solving a level).

Unlike E89's one-shot goal_score (a committed guess), here Claude PLAYS: it sees the object-graph
state + the reward feedback, proposes a short plan + an evolving goal note, executes in the real env,
observes what changed and whether a level completed, and adapts -- the way a human discovers an
ARC-3 game. Local only (claude -p + CPU).

  python3 e92_play.py --game sp80 --turns 30 --plan 6
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import arc_agi
from arcengine import GameAction

import e86_arc3 as E
import arc3_graph as GR

HERE = Path(__file__).resolve().parent
ACTS = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4,
        GameAction.ACTION5, GameAction.ACTION6, GameAction.ACTION7]


def grid(obs):
    a = np.asarray(obs.frame)
    return a[-1].reshape(64, 64) if a.ndim == 3 else a.reshape(64, 64)


PROMPT = """You are PLAYING an ARC-AGI-3 grid game. Your job: complete a LEVEL (reach the game's
win condition). You discover the rules by playing and watching what changes.

Current state (objects = connected color regions):
{state}

Available actions (integers): {avail}
Levels completed so far: {levels}   (raising this is the ONLY success signal)

Recent turns (your action plan -> what changed -> level gained):
{history}

Your running notes / goal hypothesis from before:
{notes}

Think step by step about what completes a level, using what changed last turn as evidence. Then output EXACTLY:
PLAN: <comma-separated action integers, 1 to {planmax} of them>
NOTES: <one updated hypothesis about the goal + what each action seems to do, to remember next turn>"""


def parse(resp, avail, planmax):
    pm = re.search(r"PLAN:\s*([0-9,\s]+)", resp)
    nm = re.search(r"NOTES:\s*(.+)", resp, re.S)
    plan = []
    if pm:
        for tok in pm.group(1).split(","):
            tok = tok.strip()
            if tok.isdigit() and int(tok) in avail:
                plan.append(int(tok))
    notes = nm.group(1).strip()[:600] if nm else ""
    return (plan[:planmax] or [avail[0]]), notes


def play(game, turns, planmax, seed):
    arc = arc_agi.Arcade(); env = arc.make(game); obs = env.reset()
    avail = list(obs.available_actions); g = grid(obs); win = obs.win_levels
    best = obs.levels_completed; notes = "(none yet)"; history = []; solved_turn = None
    for t in range(turns):
        state = GR.graph_repr(g)
        hist = "\n".join(history[-4:]) or "(none yet)"
        prompt = PROMPT.format(state=state, avail=avail, levels=best, history=hist,
                               notes=notes, planmax=planmax)
        try:
            resp = E.claude_cli(prompt, timeout=300)
        except Exception as e:  # noqa: BLE001
            print(f"  turn {t}: claude error {e}", flush=True); continue
        plan, notes = parse(resp, avail, planmax)
        before = best
        changes = []
        for a in plan:
            obs = env.step(ACTS[a - 1])
            if obs is None or getattr(obs, "frame", None) is None:
                changes.append(f"a{a}:GAME-OVER"); obs = env.reset(); g = grid(obs)
                avail = list(obs.available_actions); break
            ng = grid(obs); d = GR.graph_diff(g, ng); g = ng
            lvl = obs.levels_completed
            mv = d["moved"][:2]
            changes.append(f"a{a}:moved{mv}" + (f" LEVEL+{lvl - before}" if lvl > before else ""))
            best = max(best, lvl)
            if str(obs.state) != "GameState.NOT_FINISHED":
                changes.append("(reset)"); obs = env.reset(); g = grid(obs); avail = list(obs.available_actions)
        history.append(f"turn{t} plan={plan} -> {'; '.join(changes)} | levels now {best}")
        print(f"  turn {t}: plan={plan} levels={best}/{win} | {notes[:80]}", flush=True)
        if best > before and solved_turn is None:
            solved_turn = t
            print(f"  *** LEVEL COMPLETED at turn {t} (levels {best}/{win}) ***", flush=True)
    return {"best_levels": best, "win_levels": win, "solved_turn": solved_turn,
            "turns": turns, "final_notes": notes}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="sp80")
    ap.add_argument("--turns", type=int, default=30)
    ap.add_argument("--plan", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    print(f"[e92/{args.game}] interactive play, {args.turns} turns x up to {args.plan} actions", flush=True)
    res = play(args.game, args.turns, args.plan, args.seed)
    res["game"] = args.game
    res["solved"] = res["best_levels"] > 0
    print(f"[e92/{args.game}] DONE best {res['best_levels']}/{res['win_levels']} "
          f"solved_turn={res['solved_turn']}", flush=True)
    out = Path(args.out) if args.out else HERE / "results" / f"e92_arc3_{args.game}.json"
    out.write_text(json.dumps(res, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
