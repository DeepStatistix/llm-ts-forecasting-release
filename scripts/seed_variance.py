"""Seed variance of the stochastic models.
 * Prompted LLMs (Qwen3-8B, Mistral-7B): sampling-seed repeats (main run + seeds 1, 2) on the same strided rows.
 * Deep models (NHITS, PatchTST): training seeds 1 (main), 2, 3 on the full tasks (from seeds_deep.csv).
Foundation models are deterministic at inference.  Reports per-seed WQL, the range, and the range relative to the
gap between that model and the best foundation model (is seed noise small next to the model differences?).

python -m scripts.seed_variance  ->  results/summary/seed_variance.csv
"""
import numpy as np, pandas as pd
from scripts.significance import FC, OUT
from src.eval import metrics, run, tasks

if __name__ == "__main__":
    rows = []
    for name in ["flu", "fred", "wiki"]:
        t = tasks.BUILDERS[name]()
        for m in ("qwen3_8b", "mistral7b"):
            files = {0: FC / f"{name}_llm_{m}.parquet"}
            for s in (1, 2):
                files[s] = FC / f"{name}_llmseed{s}_{m}.parquet"
            if not all(p.exists() for p in list(files.values())[1:]):
                continue
            seeds = {s: pd.read_parquet(p) for s, p in files.items() if s}
            keys = set.intersection(*[set(zip(d.origin.astype(str), d.unique_id)) for d in seeds.values()])
            for s, p in files.items():
                d = pd.read_parquet(p)
                d = d[[k in keys for k in zip(d.origin.astype(str), d.unique_id)]]
                r = metrics.summarize(run.score(t, d.assign(model=m))).iloc[0]
                rows.append(dict(task=name, model=m, kind="prompted_llm", seed=s, rows=len(keys), MASE=r.MASE,
                                 WQL=r.WQL, cov80=r.cov80))
    sd = OUT / "seeds_deep.csv"
    if sd.exists():
        d = pd.read_csv(sd)
        d["kind"] = "deep"
        rows += d[["task", "model", "kind", "seed", "MASE", "WQL", "cov80"]].to_dict("records")
    r = pd.DataFrame(rows)
    g = r.groupby(["task", "model", "kind"]).WQL.agg(["mean", "min", "max", "std", "count"]).reset_index()
    g["range_pct_of_mean"] = (g["max"] - g["min"]) / g["mean"] * 100
    r.to_csv(OUT / "seed_variance_runs.csv", index=False)
    g.to_csv(OUT / "seed_variance.csv", index=False)
    print(g.round(4).to_string(index=False))
