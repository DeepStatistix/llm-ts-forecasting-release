"""Run foundation models on the FULL task sets (GPU only, no multiprocessing).

python -m scripts.run_foundation [model ...]  (default: chronos_bolt_small chronos_bolt_base chronos2)
Cached per (task, model); costs are written next to the forecasts (*.cost.json).
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

import pandas as pd


def main(models: list[str]) -> None:
    from src.eval import metrics, run, tasks

    out = Path(__file__).resolve().parents[1] / "results" / "summary"
    out.mkdir(parents=True, exist_ok=True)
    for name in ["fred", "flu", "wiki", "eia"]:
        task = tasks.BUILDERS[name]()
        print(f"\n== {name}: series-origins={task.n_series_origins()}", flush=True)
        rows = []
        for m in models:
            t0 = time.time()
            try:
                fc = run.run_baseline(task, m, tag="full")
                r = metrics.summarize(run.score(task, fc)).iloc[0]
                rows.append(r)
                print(f"  {m:20s} MASE={r.MASE:.3f} (med {r.MASE_med:.3f})  WQL={r.WQL:.3f} "
                      f"(med {r.WQL_med:.3f})  cov80={r.cov80:.3f}  [{time.time() - t0:.0f}s]",
                      flush=True)
            except Exception as e:
                print(f"  {m:20s} FAILED: {e!r}", flush=True)
                traceback.print_exc(limit=2)
        if rows:   # merge with earlier runs of other models
            f = out / f"foundation_{name}.csv"
            new = pd.DataFrame(rows)
            if f.exists():
                old = pd.read_csv(f)
                new = pd.concat([old[~old.model.isin(new.model)], new], ignore_index=True)
            new.to_csv(f, index=False)


if __name__ == "__main__":
    main(sys.argv[1:] or ["chronos_bolt_small", "chronos_bolt_base", "chronos2"])
