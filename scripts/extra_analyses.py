"""Additional analyses computed from the saved forecasts (no model is run):
  horizon.csv      scaled quantile loss by forecast step, every model, on the LLM-subset rows
  perorigin.csv    mean SQL per origin, every model, on the LLM-subset rows
  reliability.csv  empirical frequency P(y <= q_tau) for tau in {0.1, 0.5, 0.9}, every model
  sharpness.csv    coverage of the 80% interval and mean scaled interval width, every model
  selection.csv    LLM-selector choice vs the hindsight-best candidate (confusion counts) and regret
  agent_contrasts.csv  selector-vs-control contrasts parsed from the comparison log
python -m scripts.extra_analyses   ->  results/summary/extra/*.csv
"""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np, pandas as pd
from scripts.significance import pinball
from src.eval import run, tasks
from src.models.agent import CAND

ROOT = Path(__file__).resolve().parents[1]
FC, OUT = ROOT / "results" / "forecasts", ROOT / "results" / "summary" / "extra"
OUT.mkdir(parents=True, exist_ok=True)
TASKS = ["flu", "fred", "wiki", "eia"]
Q = [("q10", 0.1), ("q50", 0.5), ("q90", 0.9)]


def sql_rows(s: pd.DataFrame) -> pd.Series:
    return sum(pinball(s.y_true, s[c], t) for c, t in Q) / 3 / s.scale


hz, po, rl, sh, sel = [], [], [], [], []
for name in TASKS:
    t = tasks.BUILDERS[name]()
    ag = pd.read_parquet(FC / f"{name}_agentc_qwen3_8b.parquet")
    keys = set((o, u) for o, u in zip(ag.origin, ag.unique_id))
    files = {p.stem.split("_full_", 1)[1]: p for p in sorted(FC.glob(f"{name}_full_*.parquet"))}
    files = {m: p for m, p in files.items() if m not in ("theta@52", "ets@12")}
    for m in ("qwen3_8b", "mistral7b"):
        files[m] = FC / f"{name}_llm_{m}.parquet"
        files[f"agent_{m}"] = FC / f"{name}_agentc_{m}.parquet"
    scored = {}
    for m, p in files.items():
        d = pd.read_parquet(p)
        d = d[[(o, u) in keys for o, u in zip(d.origin, d.unique_id)]]
        s = run.score(t, d.assign(model=m))
        s["loss"] = sql_rows(s)
        s["width"] = (s.q90 - s.q10) / s.scale
        scored[m] = s
        h = t.horizon
        bins = s.step if name != "eia" else ((s.step - 1) // 24 + 1)      # EIA: day of the 7-day horizon
        g = s.assign(hstep=bins).groupby("hstep").loss.mean().reset_index()
        hz.append(g.assign(task=name, model=m, unit="day" if name == "eia" else "step"))
        po.append(s.groupby("origin").loss.mean().reset_index().assign(task=name, model=m))
        rl.append(dict(task=name, model=m, **{f"f{int(tau * 100)}": float((s.y_true <= s[c]).mean()) for c, tau in Q}))
        sh.append(dict(task=name, model=m, coverage=float(((s.y_true >= s.q10) & (s.y_true <= s.q90)).mean()),
                       width=float(s.width.mean()), sql=float(s.loss.mean())))
    # selector choice vs hindsight best candidate (per row)
    cand = CAND[name]
    cl = pd.DataFrame({m: scored[m].groupby(["origin", "unique_id"]).loss.mean() for m in cand})
    best = cl.idxmin(axis=1)
    for m in ("qwen3_8b", "mistral7b"):
        ch = pd.read_parquet(FC / f"{name}_agentc_{m}.choices.parquet")
        ch["best"] = [best.loc[(o, u)] for o, u in zip(ch.origin, ch.unique_id)]
        ch["regret"] = [cl.loc[(o, u), c] - cl.loc[(o, u), b] for o, u, c, b in zip(ch.origin, ch.unique_id, ch.chosen, ch.best)]
        sel.append(ch.groupby(["chosen", "best"]).agg(n=("regret", "size"), regret=("regret", "mean")).reset_index().assign(task=name, llm=m))
    print(name, "done", flush=True)

pd.concat(hz).to_csv(OUT / "horizon.csv", index=False)
pd.concat(po).to_csv(OUT / "perorigin.csv", index=False)
pd.DataFrame(rl).to_csv(OUT / "reliability.csv", index=False)
pd.DataFrame(sh).to_csv(OUT / "sharpness.csv", index=False)
pd.concat(sel).to_csv(OUT / "selection.csv", index=False)

# selector contrasts from the comparison log
txt = (ROOT / "results" / "summary" / "compare_agentc_log.txt").read_text(encoding="utf-8", errors="replace")
rows, cur = [], None
for line in txt.splitlines():
    m = re.match(r"== (\w+) / agent LLM (\w+): (\d+) rows", line)
    if m:
        cur = (m.group(1), m.group(2)); continue
    m = re.match(r"\s+agent vs (.+?): mean loss diff ([+-][\d.]+) \[([+-][\d.]+),([+-][\d.]+)\] p=([\d.]+) over (\d+) origins", line)
    if m and cur:
        ref = re.sub(r"\s*\(.*$", "", m.group(1)).strip()
        rows.append(dict(task=cur[0], llm=cur[1], reference=ref, diff=float(m.group(2)), lo=float(m.group(3)),
                         hi=float(m.group(4)), p=float(m.group(5)), origins=int(m.group(6))))
pd.DataFrame(rows).to_csv(OUT / "agent_contrasts.csv", index=False)
print("wrote results/summary/extra/")
