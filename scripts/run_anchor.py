"""Run the non-LLM models on the M4 anchor tasks (public benchmark; see tasks.m4_task).

python -m scripts.run_anchor [task ...]     (default: m4_monthly m4_weekly m4_daily)
Cached like every other task: results/forecasts/<task>_full_<model>.parquet.
moirai2 needs scripts/export_task + scripts/run_moirai in its own env first (see run_anchor.bat).
Results: results/summary/anchor_<task>.csv
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

import pandas as pd

MODELS = ["naive", "snaive", "ets", "theta", "arima_fast",
          "chronos_bolt_small", "chronos_bolt_base", "chronos2", "timesfm25", "moirai2"]


def main(names):
    from src.eval import metrics, run, tasks
    out = Path(__file__).resolve().parents[1] / "results" / "summary"
    for name in names or list(tasks.ANCHORS):
        task = tasks.BUILDERS[name]()
        print(f"\n== {name}: series={task.n_series_origins()} truth_rows={len(task.truth):,}", flush=True)
        rows = []
        for m in MODELS:
            if m == "moirai2" and not (run.ROOT / "results" / "moirai_out" / f"{name}.parquet").exists():
                print("  moirai2 skipped (run export_task + run_moirai first)", flush=True)
                continue
            t0 = time.time()
            try:
                fc = run.run_baseline(task, m, tag="full")
                r = metrics.summarize(run.score(task, fc)).iloc[0]
                rows.append(r)
                print(f"  {m:20s} MASE={r.MASE:.3f}  WQL={r.WQL:.3f}  cov80={r.cov80:.3f}  [{time.time() - t0:.0f}s]",
                      flush=True)
            except Exception as e:
                print(f"  {m:20s} FAILED: {e!r}", flush=True)
                traceback.print_exc(limit=2)
        if rows:
            pd.DataFrame(rows).to_csv(out / f"anchor_{name}.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1:])
