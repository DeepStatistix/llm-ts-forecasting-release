"""Same-row comparison of the prompted LLMs against every other model.

Each LLM was run on a fixed seeded subset of (origin, series) rows. For each LLM, every other
model is restricted to exactly those rows (and, when several LLMs exist, to the rows ALL of them
completed), then scored. Also: paired origin-level block bootstrap, LLM vs the best non-LLM model.

python -m scripts.compare_llm   ->  results/summary/llm_compare_<task>.csv
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd
from scripts.significance import FC, OUT, BLOCK, B, boot_p, pinball

LLMS = ["qwen3_8b", "mistral7b"]   # Gemma 4 E4B: partial flu only (too slow), reported separately


def keys_of(df):
    return set(zip(df.origin.astype(str), df.unique_id.astype(str)))


def main(names):
    from src.eval import metrics, run, tasks
    rng = np.random.default_rng(0)
    for name in names:
        llm_fc = {m: pd.read_parquet(p) for m in LLMS if (p := FC / f"{name}_llm_{m}.parquet").exists()}
        if not llm_fc:
            continue
        keep = set.intersection(*[keys_of(d) for d in llm_fc.values()])
        t = tasks.BUILDERS[name]()
        allfc = dict(llm_fc)
        for p in sorted(FC.glob(f"{name}_full_*.parquet")):
            allfc[p.stem.split("_full_", 1)[1]] = pd.read_parquet(p)
        scored = {}
        for m, d in allfc.items():
            d = d[[k in keep for k in zip(d.origin.astype(str), d.unique_id.astype(str))]]
            s = run.score(t, d.assign(model=m))
            s["loss"] = sum(pinball(s.y_true, s[c], tau) for c, tau in
                            (("q10", .1), ("q50", .5), ("q90", .9))) / 3 / s.scale
            scored[m] = s
        summ = metrics.summarize(pd.concat(scored.values())).sort_values("WQL")
        summ["n_rows"] = len(keep)
        summ.to_csv(OUT / f"llm_compare_{name}.csv", index=False)
        print(f"\n== {name}: {len(keep)} shared rows, {len(llm_fc)} LLM(s)")
        print(summ[["model", "MASE", "WQL", "cov80"]].round(3).to_string(index=False))
        others = {m: s for m, s in scored.items() if m not in LLMS}
        best = min(others, key=lambda m: others[m].loss.mean())
        for m in llm_fc:
            a = scored[m].groupby(["origin", "unique_id"]).loss.mean().groupby(level=0).mean()
            b = others[best].groupby(["origin", "unique_id"]).loss.mean().groupby(level=0).mean()
            d = (a - b.reindex(a.index)).dropna().to_numpy()
            if len(d) >= 5:
                p, ci = boot_p(d, BLOCK[name], rng)
                print(f"  {m} vs best other ({best}): mean loss diff {d.mean():+.3f} "
                      f"[{ci[0]:+.3f},{ci[1]:+.3f}] p={p:.4f} over {len(d)} origins")


if __name__ == "__main__":
    main(sys.argv[1:] or ["flu", "fred", "wiki", "eia"])
