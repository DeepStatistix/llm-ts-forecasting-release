"""Run the light agent and its non-LLM baselines on the same fixed subsets as the prompted LLMs.

python -m scripts.run_agent <llm> [task ...] [--n N]
Writes results/forecasts/<task>_agent_<llm>.parquet (+ .choices.parquet, .cost.json) and the
baselines <task>_agentbase_{ensemble,random,oracle}.parquet on the same rows.
Baselines: ensemble = mean of the candidates' quantiles; random = seeded uniform candidate pick;
oracle = hindsight best candidate per item (uses the truth: upper bound only, not a competitor).
"""
import json
import sys

import numpy as np
import pandas as pd

from scripts.significance import pinball
from src.eval import metrics, run, tasks
from src.models import agent, llm

ROOT = run.ROOT
Q = ["q10", "q50", "q90"]


def item_loss(s: pd.DataFrame) -> pd.Series:
    l = sum(pinball(s.y_true, s[c], t) for c, t in zip(Q, (.1, .5, .9))) / 3 / s.scale
    return l.groupby([s.origin, s.unique_id]).mean()


def baselines(task, name, fc, out):
    cands = agent.load_candidates(name, ROOT)
    mn = list(cands)
    keys = sorted({(o, u) for o, u in zip(fc.origin, fc.unique_id)}, key=lambda k: (str(k[0]), str(k[1])))
    base = pd.concat([pd.DataFrame({"unique_id": u, "ds": g_ds, "origin": o})
                      for (o, u), g_ds in ((k, fc[(fc.origin == k[0]) & (fc.unique_id == k[1])].ds.to_numpy())
                                           for k in keys)], ignore_index=True)
    arr = np.stack([np.concatenate([cands[m][(str(o), u)] for o, u in keys]) for m in mn])   # (m, N, 3)
    N = arr.shape[1]
    h = task.horizon
    per_item = {}
    for j, m in enumerate(mn):
        d = base.assign(**{c: arr[j][:, i] for i, c in enumerate(Q)}, model=m)
        per_item[m] = item_loss(run.score(task, d))
    L = pd.DataFrame(per_item)
    best = L.idxmin(axis=1)
    rng = np.random.default_rng(1)
    pick_r = rng.integers(0, len(mn), len(keys))
    pick_o = np.array([mn.index(best.loc[(o, u)]) for o, u in keys])
    for nm, pick in (("random", pick_r), ("oracle", pick_o)):
        sel = np.repeat(pick, h)
        d = base.assign(**{c: arr[sel, np.arange(N), i] for i, c in enumerate(Q)})
        run.apply_floor(task, d).assign(model="agentbase_" + nm).to_parquet(out[nm], index=False)
    d = base.assign(**{c: arr.mean(0)[:, i] for i, c in enumerate(Q)})
    run.apply_floor(task, d).assign(model="agentbase_ensemble").to_parquet(out["ensemble"], index=False)


if __name__ == "__main__":
    args = sys.argv[1:]
    n = int(args[args.index("--n") + 1]) if "--n" in args else None
    if n is not None:
        del args[args.index("--n"):args.index("--n") + 2]
    model, names = args[0], args[1:] or ["flu", "fred", "wiki", "eia"]
    use_ctx = "--ctx" in args           # text-context run on the same rows as the main agent run
    if use_ctx:
        args.remove("--ctx")
        model, names = args[0], args[1:] or ["flu", "fred", "wiki", "eia"]
    # constrained decoding (the final protocol) writes agentc*/agentcctx*; the first free-text runs are "agent*"
    tag = "agentcctx" if use_ctx else ("agentcsmoke" if n else "agentc")
    fdir = ROOT / "results" / "forecasts"
    for name in names:
        task = tasks.BUILDERS[name]()
        items = llm.subset(task, n or llm.CFG[name]["n_eval"])
        fc, ch, cost = agent.run_agent(model, task, items, ROOT, with_ctx=use_ctx)
        fc = run.apply_floor(task, fc.assign(model="agent_" + model))
        fc.to_parquet(fdir / f"{name}_{tag}_{model}.parquet", index=False)
        ch.to_parquet(fdir / f"{name}_{tag}_{model}.choices.parquet", index=False)
        (fdir / f"{name}_{tag}_{model}.cost.json").write_text(json.dumps(cost, indent=1))
        if not use_ctx:
            baselines(task, name, fc, {k: fdir / f"{name}_agentbase_{k}.parquet" for k in ("ensemble", "random", "oracle")})
        r = metrics.summarize(run.score(task, fc)).iloc[0]
        print(f"== {name} agent {model}: n={len(items)} MASE={r.MASE:.3f} WQL={r.WQL:.3f} cov80={r.cov80:.3f} "
              f"fallback={cost['fallback_rate']} {cost['sec_per_item']}s/item "
              f"choices={ch.chosen.value_counts().to_dict()} widths={ch.width.value_counts().to_dict()}", flush=True)
