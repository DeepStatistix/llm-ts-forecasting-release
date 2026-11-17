"""Training-seed variance of the deep models (NHITS, PatchTST): the main runs use seed 1; repeat with
seeds 2 and 3 on every task and report WQL/MASE/coverage per seed.

python -m scripts.seeds_deep  ->  results/summary/seeds_deep.csv   (+ results/forecasts/<task>_seeddeep_<arch>_s<k>.parquet)
Foundation models (Chronos-Bolt/2, TimesFM, Moirai) are deterministic at inference, so they have no seed variance.
"""
from __future__ import annotations
import pandas as pd
from src.eval import metrics, run, tasks
from src.models import deep

SEEDS = [2, 3]

if __name__ == "__main__":
    rows = []
    for name in ["fred", "flu", "wiki", "eia"]:
        task = tasks.BUILDERS[name]()
        for arch in ("nhits", "patchtst"):
            main = pd.read_parquet(run.ROOT / "results" / "forecasts" / f"{name}_full_{arch}.parquet")
            r = metrics.summarize(run.score(task, main.assign(model=arch))).iloc[0]
            rows.append(dict(task=name, model=arch, seed=1, MASE=r.MASE, WQL=r.WQL, cov80=r.cov80))
            for s in SEEDS:
                fc, cost = deep.forecast_task(arch, task, seed=s)
                fc = run.apply_floor(task, fc.assign(model=arch))
                fc.to_parquet(run.ROOT / "results" / "forecasts" / f"{name}_seeddeep_{arch}_s{s}.parquet", index=False)
                r = metrics.summarize(run.score(task, fc)).iloc[0]
                rows.append(dict(task=name, model=arch, seed=s, MASE=r.MASE, WQL=r.WQL, cov80=r.cov80,
                                 train_s=cost["train_seconds"]))
                print(rows[-1], flush=True)
            pd.DataFrame(rows).to_csv(run.ROOT / "results" / "summary" / "seeds_deep.csv", index=False)
