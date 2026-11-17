"""Run baselines on the FULL task sets. Cached per (task, model): reruns skip finished work.

python -m scripts.run_baselines [task ...]     (default: all four)
Results: results/forecasts/*.parquet, results/summary/baselines_<task>.csv
"""
from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

# models per task. ARIMA is skipped on the hourly/daily panels (cost); can be added later.
PLAN = {
    "fred": ["naive", "snaive", "ets", "theta", "arima", "lgbm"],
    "flu":  ["naive", "snaive", "ets", "theta", "arima", "lgbm"],
    "wiki": ["naive", "snaive", "ets", "theta", "mstl", "lgbm"],
    "eia":  ["naive", "snaive", "ets", "mstl", "lgbm"],
}


def main(only: list[str]) -> None:
    from src.eval import metrics, run, tasks

    root = Path(__file__).resolve().parents[1]
    out = root / "results" / "summary"
    out.mkdir(parents=True, exist_ok=True)
    for name in (only or list(PLAN)):
        task = tasks.BUILDERS[name]()
        print(f"\n== {name}: origins={len(task.histories)} series-origins="
              f"{task.n_series_origins()} truth_rows={len(task.truth):,}", flush=True)
        scored = []
        for m in PLAN[name]:
            if m in os.environ.get("SKIP", "").split(","):   # e.g. SKIP=arima: run it later
                continue
            t0 = time.time()
            try:
                fc = run.run_baseline(task, m, tag="full")
                s = run.score(task, fc)
                scored.append(s)
                r = metrics.summarize(s).iloc[0]
                print(f"  {m:7s} MASE={r.MASE:.3f} (med {r.MASE_med:.3f})  WQL={r.WQL:.3f} "
                      f"(med {r.WQL_med:.3f})  cov80={r.cov80:.3f}  [{time.time() - t0:.0f}s]",
                      flush=True)
            except Exception as e:   # one failing model must not stop the rest
                print(f"  {m:7s} FAILED: {e!r}", flush=True)
                traceback.print_exc(limit=2)
        if scored:
            summ = metrics.summarize(pd.concat(scored))
            summ.to_csv(out / f"baselines_{name}.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1:])
