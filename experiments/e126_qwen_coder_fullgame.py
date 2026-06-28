"""E126 -- qwen3-coder:30b full-game sweep (copy of E118, qwen3-coder default + num_ctx 8192 per the GPU-swap gotcha; do not alter E118). QWEN-CODER agentic full-game solver: the SAME pipeline as the Claude live coding agent
(explore -> write executable world-model code -> reason the goal -> plan -> verify), but driven by a
LOCAL model (qwen3-coder:30b via Ollama) inside a ReAct loop we build ourselves (qwen is not an agentic
harness). EVERY prompt and response is logged to JSONL for reproducibility.

Head-to-head context: the Claude live agent solves many games; per E86b the agentic harness is
CAPABILITY-GATED, so we expect qwen to do far worse -- this experiment measures that gap with full logs.

  # load the model first:  ollama pull qwen3-coder:30b
  python e118_qwen_agent.py --game ka59 --rounds 14
  python e118_qwen_agent.py --model qwen2.5:3b --game vc33   # quick scaffolding test on a small model
"""
import argparse, json, re, subprocess, time, logging, contextlib, io
from pathlib import Path
import openworld as O
ROOT=Path("/Users/jim/Desktop/openworld")
HARNESS=ROOT/"scratch_arc"/"agent"/"arc3_harness.py"
LOGDIR=ROOT/"experiments"/"results"/"e126_qwen_logs"; LOGDIR.mkdir(parents=True,exist_ok=True)
VENV="/Users/jim/.arcv/bin/python"

TASK = """You are solving the interactive ARC-AGI-3 game **{game}** by writing Python. Each round you
output ONE self-contained Python script; we run it and give you its stdout; you iterate. Goal: complete
EVERY level (g.levels == g.win), saving progress to solved.json.

The script may use the harness (already importable):
    from arc3_harness import Game
    g = Game("{game}"); g.reset()
    g.frame  # 64x64 numpy int array (colors 0-15);  g.levels, g.win, g.avail, g.done
    g.step(a)        # directional a in 1..5,7
    g.step(6, x, y)  # ACTION6 = CLICK at column x, row y (0..63)
The env is DETERMINISTIC: replaying actions from reset() reproduces frames, so explore then replay-verify.
Clicks often work ONLY on specific cells (try distinct/non-background cells, not (0,0)).

Recipe: (1) explore to learn what actions do; (2) write predict(frame,action) reproducing transitions;
(3) reason what raises g.levels; (4) plan a sequence; (5) when g.levels rises, write
solved.json = {{"game":"{game}","actions":[[1],[6,60,32],...],"levels":N,"win":W}} (each action is [a]
or [6,x,y]); keep updating it to your deepest progress. PRINT what you learned each round.

Output ONLY a single ```python code block (no prose).

[round {r} -- TRANSCRIPT of your previous rounds: each shows the script YOU wrote and its stdout.
BUILD ON what you already learned -- do NOT repeat the same exploration; extend toward raising g.levels]:
{history}
"""

def build_history(log, budget=24000):
    """Accumulate a rolling transcript of prior rounds (your code + its stdout), newest kept first,
    trimmed to a char budget so it fits num_ctx. This is the cross-round memory the agent needs to
    iterate a world model instead of restarting from scratch every round."""
    if not log: return "(none yet -- start by exploring)"
    parts=[]
    for rec in log:
        code=rec["script"][:1800]
        out=rec["stdout"][-1800:]
        parts.append(f"===== round {rec['round']} (best_levels after this round = {rec.get('best_after',0)}) =====\n"
                     f"--- YOUR SCRIPT ---\n{code}\n--- ITS STDOUT ---\n{out}")
    kept=[]; total=0
    for p in reversed(parts):           # keep the most recent rounds that fit the budget
        total+=len(p)
        if total>budget and kept: break
        kept.append(p)
    return "\n\n".join(reversed(kept))

def extract_code(text):
    m=re.search(r"```(?:python)?\s*(.*?)```", text, re.S)
    return m.group(1).strip() if m else text.strip()

def run_script(code, wd):
    (wd/"arc3_harness.py").write_text(HARNESS.read_text())
    f=wd/"play.py"; f.write_text(code)
    try:
        r=subprocess.run([VENV,str(f)],cwd=str(wd),capture_output=True,text=True,timeout=300)
        out=(r.stdout or "")+("\n[stderr]\n"+r.stderr if r.stderr else "")
    except subprocess.TimeoutExpired: out="[TIMEOUT after 300s]"
    except Exception as e: out=f"[exec error: {e}]"
    return out[-4000:]

def solve(game, model, rounds, num_ctx, timeout):
    llm=O.OllamaLLM(model=model, options={"num_ctx":num_ctx}, timeout=timeout)
    wd=ROOT/"scratch_arc"/f"qwen3c_{game}"; wd.mkdir(parents=True,exist_ok=True)
    (wd/"arc3_harness.py").write_text(HARNESS.read_text())
    log=[]; history="(none yet -- start by exploring)"
    best=0; win=0
    for r in range(rounds):
        prompt=TASK.format(game=game, r=r, history=history)
        t=time.time()
        try: resp=llm.ask(prompt)
        except Exception as e: resp=f"<LLM ERROR: {e}>"
        dt=time.time()-t
        code=extract_code(resp); out=run_script(code, wd) if code else "(no code)"
        rec={"round":r,"game":game,"model":model,"prompt":prompt,"response":resp,
             "script":code,"stdout":out,"llm_seconds":round(dt,1)}
        log.append(rec)
        # read solved.json if the script wrote one
        sj=wd/"solved.json"
        if sj.exists():
            try:
                d=json.loads(sj.read_text()); best=max(best,int(d.get("levels",0))); win=int(d.get("win",win) or win)
            except Exception: pass
        rec["best_after"]=best
        history=build_history(log)
        print(f"  [{game}] round {r}: {dt:.0f}s, best_levels={best}, code={len(code)}ch, hist={len(history)}ch",flush=True)
        if win and best>=win: break
    (LOGDIR/f"{game}.jsonl").write_text("\n".join(json.dumps(x) for x in log))
    return {"game":game,"model":model,"best_levels":best,"win":win,"rounds":len(log),
            "log":str(LOGDIR/f"{game}.jsonl")}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",default="qwen3-coder:30b")
    ap.add_argument("--game",default=""); ap.add_argument("--games",default="")
    ap.add_argument("--rounds",type=int,default=24); ap.add_argument("--num_ctx",type=int,default=8192)
    ap.add_argument("--timeout",type=int,default=1800); ap.add_argument("--out",default="results/e126_qwen_coder_fullgame.json")
    a=ap.parse_args()
    games=[a.game] if a.game else (a.games.split(",") if a.games else [])
    if not games: print("specify --game or --games"); return
    print(f"[e126] qwen-coder agentic full-game solve: model={a.model} games={games}",flush=True)
    print(f"       logging all prompts+responses to {LOGDIR}/",flush=True)
    res={}
    for g in games:
        try: res[g]=solve(g,a.model,a.rounds,a.num_ctx,a.timeout)
        except Exception as e:
            import traceback; traceback.print_exc(); res[g]={"game":g,"error":str(e)[:120]}
        r=res[g]; print(f"  => {g}: best {r.get('best_levels',0)}/{r.get('win','?')} levels",flush=True)
    Path(a.out).write_text(json.dumps({"model":a.model,"results":res},indent=2))
    print(f"[e126] done. logs in {LOGDIR}/, summary in {a.out}",flush=True)

if __name__=="__main__": main()
