"""Does the LLM add value over the tools it is given?  Same-row comparison for the light agent.

For each task and agent LLM: every row below is scored on exactly the agent's (origin, series) rows.
  agent_<llm>        the LLM selector (letter + width)
  candidates         naive / snaive / classical / chronos_bolt_small (what the agent chooses among)
  agentbase_*        ensemble (mean of candidates), random pick, oracle (hindsight; upper bound)
  context            best foundation model and the prompted LLM, for scale
Paired origin-level block bootstrap of the agent vs: best single candidate (chosen in hindsight on
these rows, i.e. a conservative reference), ensemble, random pick.

python -m scripts.compare_agent   ->  results/summary/agent_compare_<task>.csv
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd
from scripts.significance import FC, OUT, BLOCK, boot_p, pinball
from src.models.agent import CAND

AGENTS = ["qwen3_8b", "mistral7b"]
CONTEXT = ["chronos2", "timesfm25"]


def main(names):
    from src.eval import metrics, run, tasks
    rng = np.random.default_rng(0)
    for name in names:
        t = tasks.BUILDERS[name]()
        for a in AGENTS:
            ap = FC / f"{name}_agentc_{a}.parquet"
            if not ap.exists():
                continue
            ag = pd.read_parquet(ap)
            keep = set(zip(ag.origin.astype(str), ag.unique_id))
            fcs = {f"agent_{a}": ag}
            for m in CAND[name] + CONTEXT:
                fcs[m] = pd.read_parquet(FC / f"{name}_full_{m}.parquet")
            for b in ("ensemble", "random", "oracle"):
                fcs[f"base_{b}"] = pd.read_parquet(FC / f"{name}_agentbase_{b}.parquet")
            lp = FC / f"{name}_llm_{a}.parquet"
            if lp.exists():
                fcs[f"prompted_{a}"] = pd.read_parquet(lp)
            scored = {}
            for m, d in fcs.items():
                d = d[[k in keep for k in zip(d.origin.astype(str), d.unique_id)]]
                s = run.score(t, d.assign(model=m))
                s["loss"] = sum(pinball(s.y_true, s[c], tau) for c, tau in
                                (("q10", .1), ("q50", .5), ("q90", .9))) / 3 / s.scale
                scored[m] = s
            # control: best FIXED (candidate, width) over all rows, chosen in hindsight -> separates the
            # LLM's per-item choice from the effect of simply widening/choosing one global setting
            from src.models.agent import WIDTH
            best = (9e9, None)
            for m in CAND[name]:
                dm = fcs[m][[k in keep for k in zip(fcs[m].origin.astype(str), fcs[m].unique_id)]]
                for w in WIDTH.values():
                    s = run.score(t, run.widen(t, dm, w).assign(model=m))
                    s["loss"] = sum(pinball(s.y_true, s[c], tau) for c, tau in
                                    (("q10", .1), ("q50", .5), ("q90", .9))) / 3 / s.scale
                    if s.loss.mean() < best[0]:
                        best = (s.loss.mean(), (m, w), s.assign(model=f"ctrl_best_fixed({m},{w}x)"))
            scored["ctrl_best_fixed"] = best[2]
            summ = metrics.summarize(pd.concat(scored.values())).sort_values("WQL")
            summ["n_rows"] = len(keep)
            summ.to_csv(OUT / f"agent_compare_{name}_{a}.csv", index=False)
            print(f"\n== {name} / agent LLM {a}: {len(keep)} rows")
            print(summ[["model", "MASE", "WQL", "cov80"]].round(3).to_string(index=False))
            per_origin = lambda s: s.groupby(["origin", "unique_id"]).loss.mean().groupby(level=0).mean()
            me = per_origin(scored[f"agent_{a}"])
            cand = {m: per_origin(scored[m]) for m in CAND[name]}
            best_c = min(cand, key=lambda m: cand[m].mean())
            for lab, ref in ((f"best single candidate ({best_c})", cand[best_c]),
                             (f"best fixed (candidate, width) {best[1]}", per_origin(scored["ctrl_best_fixed"])),
                             ("ensemble", per_origin(scored["base_ensemble"])),
                             ("random pick", per_origin(scored["base_random"]))):
                d = (me - ref.reindex(me.index)).dropna().to_numpy()
                if len(d) >= 5:
                    p, ci = boot_p(d, BLOCK[name], rng)
                    print(f"  agent vs {lab}: mean loss diff {d.mean():+.4f} [{ci[0]:+.4f},{ci[1]:+.4f}] "
                          f"p={p:.4f} over {len(d)} origins")


if __name__ == "__main__":
    main(sys.argv[1:] or ["flu", "fred", "wiki", "eia"])
